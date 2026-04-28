"""Capture attention weights from SmolVLA's SigLIP vision encoder.

SmolVLM loads the vision encoder through HuggingFace `AutoModelForImageTextToText`.
Under the hood it uses `torch.nn.functional.scaled_dot_product_attention`, which
does NOT materialize a `softmax(QK^T/sqrt(d))` tensor we can grab. So instead of
trying to wedge `output_attentions=True` in, we install forward hooks on each
layer's `q_proj` / `k_proj` and recompute the attention probabilities manually
on demand. It costs one extra matmul+softmax per layer per image, run only when
we want a heatmap (typically once per RTC chunk, not every control step).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class _LayerCache:
    """Captured Q/K for one attention module for one forward pass."""

    q: torch.Tensor | None = None  # (B, seq, num_heads * head_dim)
    k: torch.Tensor | None = None
    num_heads: int | None = None
    head_dim: int | None = None


class VisionAttentionCapture:
    """Hooks every attention layer in a HuggingFace vision encoder.

    Usage:
        capture = VisionAttentionCapture(policy.model.vlm_with_expert.get_vlm_model().vision_model)
        with capture:
            _ = policy.predict_action_chunk(batch, ...)
        per_layer_attn = capture.compute_attentions()   # list[(B, heads, seq, seq)]
        rollout = capture.attention_rollout()           # (B, seq, seq)
    """

    def __init__(self, vision_model: torch.nn.Module):
        self.vision_model = vision_model
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._layers: list[_LayerCache] = []
        self._attn_modules: list[torch.nn.Module] = self._find_attention_modules(vision_model)

    @staticmethod
    def _find_attention_modules(root: torch.nn.Module) -> list[torch.nn.Module]:
        """Every module that has both `q_proj` and `k_proj` submodules.

        Matches the standard HF ViT / SigLIP / Idefics3 attention layout. We do
        NOT match arbitrary attention layers elsewhere in the policy (e.g.
        expert transformer) — by construction we only look inside the submodule
        tree the caller hands us.
        """
        hits: list[torch.nn.Module] = []
        for mod in root.modules():
            if hasattr(mod, "q_proj") and hasattr(mod, "k_proj"):
                hits.append(mod)
        return hits

    def __enter__(self) -> VisionAttentionCapture:
        # One per-layer slot for Q/K caches.
        self._layers = [_LayerCache() for _ in self._attn_modules]
        # One rollout per completed vision-encoder forward (i.e. per camera).
        self.rollouts: list[torch.Tensor] = []

        for idx, attn in enumerate(self._attn_modules):
            # Introspect head layout — HF attention modules all expose these.
            num_heads = getattr(attn, "num_heads", None) or getattr(attn, "num_attention_heads", None)
            head_dim = getattr(attn, "head_dim", None)
            if head_dim is None and num_heads is not None:
                # Fall back: infer from q_proj weight shape.
                head_dim = attn.q_proj.out_features // num_heads

            self._layers[idx].num_heads = num_heads
            self._layers[idx].head_dim = head_dim

            # Hook the Q/K projections — the forward_hook receives the projected output.
            self._handles.append(
                attn.q_proj.register_forward_hook(self._make_proj_hook(idx, "q"))
            )
            self._handles.append(
                attn.k_proj.register_forward_hook(self._make_proj_hook(idx, "k"))
            )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_proj_hook(self, layer_idx: int, which: str):
        def _hook(_mod, _inp, out: torch.Tensor):
            # Cache only the most recent forward — if the encoder runs on multiple
            # images in sequence, the caller is responsible for reading attention
            # after each image (see running/smolvla.py).
            if which == "q":
                self._layers[layer_idx].q = out.detach()
            else:
                self._layers[layer_idx].k = out.detach()

        return _hook

    def compute_attentions(self) -> list[torch.Tensor]:
        """Return per-layer attention probs shaped (B, heads, seq, seq).

        Recomputed manually from the cached Q / K. Same formula as eager
        attention — softmax(Q @ K^T / sqrt(d_head)), averaged nowhere.
        """
        out: list[torch.Tensor] = []
        for cache in self._layers:
            # Skip layers that didn't fire this forward (safety — shouldn't happen in practice).
            if cache.q is None or cache.k is None:
                continue
            q = cache.q
            k = cache.k
            b, s, _ = q.shape

            # (B, seq, heads, head_dim) -> (B, heads, seq, head_dim)
            q = q.view(b, s, cache.num_heads, cache.head_dim).transpose(1, 2)
            k = k.view(b, s, cache.num_heads, cache.head_dim).transpose(1, 2)

            # Scaled dot-product attention logits, matching eager HF attention.
            scale = cache.head_dim**-0.5
            logits = torch.matmul(q, k.transpose(-1, -2)) * scale
            probs = F.softmax(logits.float(), dim=-1)
            out.append(probs)
        return out

    def snapshot(self, add_residual: bool = True) -> torch.Tensor:
        """Compute rollout for the most recent vision-encoder forward, stash it
        in `self.rollouts`, and reset the Q/K cache so the next forward starts
        fresh. Call this from a forward hook on `embed_image`.
        """
        rollout = self.attention_rollout(add_residual=add_residual)
        self.rollouts.append(rollout)
        for cache in self._layers:
            cache.q = None
            cache.k = None
        return rollout

    def attention_rollout(self, add_residual: bool = True) -> torch.Tensor:
        """Attention rollout (Abnar & Zuidema 2020) across all captured layers.

        Returns a (B, seq, seq) tensor where entry [b, i, j] approximates how
        much token j contributed to token i's final representation.
        """
        attns = self.compute_attentions()
        if not attns:
            raise RuntimeError("No attentions captured — did you run a forward inside the context?")

        # Average over heads, optionally fold in the residual stream.
        per_layer = []
        for a in attns:
            a = a.mean(dim=1)  # (B, seq, seq)
            if add_residual:
                # 0.5 * A + 0.5 * I — residual connection contributes identity each layer.
                eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
                a = a + eye
                a = a / a.sum(dim=-1, keepdim=True)
            per_layer.append(a)

        # Multiply attention matrices layer-by-layer to propagate influence.
        rollout = per_layer[0]
        for a in per_layer[1:]:
            rollout = torch.matmul(a, rollout)
        return rollout
