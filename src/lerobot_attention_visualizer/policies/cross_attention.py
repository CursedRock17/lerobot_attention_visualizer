"""Action-head cross-attention capture — the *action-driving* visual signal.

The vision-encoder attention captured by `VisionAttentionCapture` answers
"which patches does the (usually frozen) image encoder find salient." That is
*not* what most fine-tuning debugging needs. The signal that actually drives the
autonomous action is the **action head's cross-attention onto the vision
tokens**:

    Groot   FlowmatchingActionHead.get_action:
              sa_embs = [state, future_tokens, action]      # QUERIES
              vl_embs = backbone_features (vision + text)    # KEYS / VALUES
              DiT(hidden_states=sa_embs, encoder_hidden_states=vl_embs)
            The DiT's cross-attending BasicTransformerBlock.attn1 is exactly
            "which VL tokens the denoiser looks at while producing the action."

This module captures that cross-attention. Like SigLIP, the diffusers `Attention`
module runs through `scaled_dot_product_attention` and never materializes the
weights, so we hook the `to_q` / `to_k` projections and recompute
`softmax(QK^T / sqrt(d))` ourselves.

# Differences from the rollout path

- The attention matrix is **rectangular** `(q_len, k_len)` — action queries vs.
  VL tokens — not square, so there is no rollout product across layers. We just
  average.
- The denoiser runs `num_inference_timesteps` times and there are several
  cross-attending layers, so the cross-attention fires many times per chunk. We
  accumulate a running mean over every fire (all steps × all cross-attn layers).
- We reduce over heads and over all query rows to get one importance value per
  VL token. The caller then masks that vector to the vision-token columns,
  splits it per camera, and reshapes to the post-projection patch grid.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class CrossAttentionCapture:
    """Hook `to_q` / `to_k` on a list of cross-attention modules and accumulate
    a mean per-key-token importance vector over every forward.

    Most users want a policy adapter (e.g. `GR00TAttention`) instead — this is
    the underlying primitive.

    Args:
        attn_modules: the cross-attention modules to hook (e.g. each DiT block's
            `attn1`). Pass only the cross-attending ones — self-attention blocks
            have a different key length and would pollute the accumulator.
        q_attr / k_attr: projection attribute names. diffusers `Attention` uses
            `to_q` / `to_k`; HF-style modules use `q_proj` / `k_proj`.
        heads_attr: attribute holding the head count (`heads` for diffusers,
            `num_heads` for HF).

    Usage:
        cap = CrossAttentionCapture(cross_blocks)
        with cap:
            policy.predict_action_chunk(batch)   # denoiser runs N steps
        importance = cap.drain()   # (k_len,) mean importance per VL token, or None
    """

    def __init__(
        self,
        attn_modules: list[torch.nn.Module],
        *,
        q_attr: str = "to_q",
        k_attr: str = "to_k",
        heads_attr: str = "heads",
    ):
        self._modules = attn_modules
        self._q_attr = q_attr
        self._k_attr = k_attr
        self._heads_attr = heads_attr
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        # Live per-module Q/K slots, keyed by position in self._modules.
        self._slots: list[dict[str, torch.Tensor | None]] = []
        # Running sum of reduced importance (k_len,) and the number of fires.
        self._acc: torch.Tensor | None = None
        self._count: int = 0

    def __enter__(self) -> "CrossAttentionCapture":
        self._slots = [{"q": None, "k": None} for _ in self._modules]
        self._acc = None
        self._count = 0
        for idx, mod in enumerate(self._modules):
            heads = getattr(mod, self._heads_attr)
            q_proj = getattr(mod, self._q_attr)
            k_proj = getattr(mod, self._k_attr)
            self._handles.append(q_proj.register_forward_hook(self._make_hook(idx, "q", heads)))
            self._handles.append(k_proj.register_forward_hook(self._make_hook(idx, "k", heads)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, idx: int, which: str, heads: int):
        def _hook(_mod, _inp, out: torch.Tensor):
            slot = self._slots[idx]
            slot[which] = out.detach()
            # to_q fires before to_k within one attn forward; reduce once both
            # are present, then clear the slot for the next denoising step.
            if slot["q"] is not None and slot["k"] is not None:
                self._accumulate(slot["q"], slot["k"], heads)
                slot["q"] = None
                slot["k"] = None

        return _hook

    def _accumulate(self, q: torch.Tensor, k: torch.Tensor, heads: int) -> None:
        """Reduce one (q, k) pair to a (k_len,) importance vector and add it in.

        q: (B, q_len, heads*head_dim)   k: (B, k_len, heads*head_dim)
        """
        b, q_len, inner = q.shape
        k_len = k.shape[1]
        head_dim = inner // heads

        q = q.view(b, q_len, heads, head_dim).transpose(1, 2)  # (B, H, q_len, hd)
        k = k.view(b, k_len, heads, head_dim).transpose(1, 2)  # (B, H, k_len, hd)

        scale = head_dim**-0.5
        logits = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B, H, q_len, k_len)
        probs = F.softmax(logits.float(), dim=-1)
        # Mean over heads then over every query row → per-key importance (B, k_len).
        per_key = probs.mean(dim=1).mean(dim=1)
        vec = per_key[0].detach()  # batch size 1

        if self._acc is None:
            self._acc = torch.zeros_like(vec)
        if vec.shape != self._acc.shape:
            # Key length drifted (e.g. a self-attention block slipped in) —
            # skip rather than corrupt the running mean.
            return
        self._acc = self._acc + vec
        self._count += 1

    def drain(self) -> torch.Tensor | None:
        """Mean importance per key token over all fires, then reset. None if empty."""
        if self._acc is None or self._count == 0:
            return None
        result = self._acc / self._count
        self._acc = None
        self._count = 0
        return result


def find_cross_attention_blocks(
    transformer_blocks: list[torch.nn.Module],
    *,
    attn_attr: str = "attn1",
    q_attr: str = "to_q",
    k_attr: str = "to_k",
) -> list[torch.nn.Module]:
    """Return the attention submodules that perform cross-attention.

    A diffusers `Attention` is cross-attending when its key projection consumes a
    different feature dim than its query projection (`cross_attention_dim` ≠
    `query_dim`). This is a static, version-independent test — more robust than
    relying on the optional `is_cross_attention` flag.
    """
    hits: list[torch.nn.Module] = []
    for block in transformer_blocks:
        attn = getattr(block, attn_attr, None)
        if attn is None:
            continue
        q_proj = getattr(attn, q_attr, None)
        k_proj = getattr(attn, k_attr, None)
        if q_proj is None or k_proj is None:
            continue
        if q_proj.in_features != k_proj.in_features:
            hits.append(attn)
    return hits


def vision_importance_to_grids(
    importance: torch.Tensor,
    vision_mask: torch.Tensor,
    n_cameras: int,
) -> list[torch.Tensor]:
    """Slice a per-VL-token importance vector to vision tokens and reshape per camera.

    Args:
        importance: (k_len,) mean importance per VL token from CrossAttentionCapture.
        vision_mask: (k_len,) bool — True where the VL token is an image patch
            (i.e. input_ids == image_token_index). Image tokens appear in camera
            order, one contiguous block per camera.
        n_cameras: number of cameras (the vision tokens split evenly across them).

    Returns one (side, side) heatmap tensor per camera, or [] if the counts don't
    divide into equal square grids (caller should drop the frame).
    """
    vis = importance[vision_mask]
    n = vis.numel()
    if n == 0 or n % n_cameras != 0:
        return []
    per_cam = n // n_cameras
    side = int(round(per_cam**0.5))
    if side * side != per_cam:
        return []
    return [vis[c * per_cam : (c + 1) * per_cam].view(side, side) for c in range(n_cameras)]
