"""SmolVLA attention capture — VLM vision-encoder backbone wrapper.

Wraps a `SmolVLAPolicy` so a runner / playback loop only has to do:

    viz = SmolVLAAttention(policy)
    with viz:
        ... eval loop ...
        viz.log_overlay(obs)

`SmolVLAAttention` finds the SigLIP vision encoder inside the VLM, installs
the q_proj / k_proj hooks, monkey-patches `embed_image` to snapshot a fresh
attention rollout per camera, and on exit unwinds both. `log_overlay(obs)`
drains the captured rollouts, upsamples each to its camera's resolution,
and writes three rerun streams per camera (image / attention / overlay).

# Design

`torch.nn.functional.scaled_dot_product_attention`, which the HF vision
encoder uses under the hood, never materializes the `softmax(QK^T/√d)`
tensor we'd want to read off the model. Instead we:

1. Install forward hooks on every attention layer's `q_proj` / `k_proj`
   inside the SigLIP vision encoder. Each hook caches the projected Q/K
   tensor.
2. After each `embed_image` call (one per camera per chunk), recompute
   `softmax(Q @ K^T / √(d_head))` manually from the cached tensors, run
   attention rollout (Abnar & Zuidema 2020) across all ViT layers with the
   residual trick, and store the (seq, seq) rollout.
3. Reduce to per-patch importance (column-mean), reshape to the patch
   grid, bilinear-upsample to camera resolution, and overlay via rerun.

Cost: one extra matmul+softmax per ViT layer per image per chunk. On a
4090 laptop this is noise compared to the VLM forward.

# Scope

What this tells you. Where inside each image the vision encoder is
concentrating its own internal attention — i.e. which image regions the
encoder thinks are salient to represent.

What this does NOT tell you. Which image regions actually drive the
*action* output. Vision-encoder attention is a proxy — a cleaner answer
would inspect the expert→prefix cross-attention at the final joint
SmolVLM+expert layer, or run a gradient-based attribution
(IntegratedGradients, Grad-CAM) from the action logits back to the input
pixels. Both are straightforward follow-ups that reuse the same rerun
logging path.

π0 uses the same SigLIP-style encoder and would be a near-identical port
of `SmolVLAAttention`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from ..visualizer.overlay import (
    log_attention_overlay,
    patch_heatmap_to_image,
    rollout_to_patch_heatmap,
)

if TYPE_CHECKING:
    # Heavy import — guarded so users without the [smolvla] extra can still
    # `import lerobot_attention_visualizer` (e.g. an ACT-only setup).
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


@dataclass
class _LayerCache:
    """Captured Q/K for one attention module for one forward pass."""

    q: torch.Tensor | None = None  # (B, seq, num_heads * head_dim)
    k: torch.Tensor | None = None
    num_heads: int | None = None
    head_dim: int | None = None


class VisionAttentionCapture:
    """Low-level: hook every attention layer in a HuggingFace vision encoder.

    Most users want `SmolVLAAttention` instead — this is the underlying
    primitive if you want to do something with attention rollouts that isn't
    "log to rerun".

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
        # One per-layer slot for live Q/K caches (overwritten each forward).
        self._layers = [_LayerCache() for _ in self._attn_modules]
        # Frozen snapshots — one list-of-LayerCaches per embed_image call.
        # Rollout compute is deferred to drain_rollouts() so it doesn't run
        # inside predict_action_chunk and inflate real_latency for the RTC queue.
        self._pending: list[list[_LayerCache]] = []

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
            # after each image (see `SmolVLAAttention` below).
            if which == "q":
                self._layers[layer_idx].q = out.detach()
            else:
                self._layers[layer_idx].k = out.detach()

        return _hook

    def compute_attentions(self) -> list[torch.Tensor]:
        """Return per-layer attention probs shaped (B, heads, seq, seq).

        Recomputed manually from the cached Q / K. Same formula as eager
        attention — softmax(Q @ K^T / sqrt(d_head)), averaged nowhere. Kept
        as an escape hatch for users who want raw per-layer attention; the
        rollout path streams instead and never holds all layers at once.
        """
        out: list[torch.Tensor] = []
        for cache in self._layers:
            probs = self._layer_attention(cache)
            if probs is None:
                continue
            out.append(probs)
        return out

    def _layer_attention(self, cache: _LayerCache) -> torch.Tensor | None:
        """softmax(Q @ K^T / sqrt(d)) for one cached layer, or None if it didn't fire.

        Stays in q.dtype throughout — under the encoder's autocast (bf16 on
        modern GPUs) this is ~2× cheaper than upcasting to fp32, and the
        rollout's head-mean + matmul chain is robust to bf16 noise.
        """
        if cache.q is None or cache.k is None:
            return None
        q = cache.q
        k = cache.k
        b, s, _ = q.shape

        # (B, seq, heads, head_dim) -> (B, heads, seq, head_dim)
        q = q.view(b, s, cache.num_heads, cache.head_dim).transpose(1, 2)
        k = k.view(b, s, cache.num_heads, cache.head_dim).transpose(1, 2)

        scale = cache.head_dim**-0.5
        logits = torch.matmul(q, k.transpose(-1, -2)) * scale
        # SiglipAttention does softmax in float32; match that for numerical
        # consistency across the 729-token sequence (bf16 precision is marginal).
        return F.softmax(logits.float(), dim=-1)

    def snapshot(self) -> None:
        """Freeze the current Q/K tensors into _pending and reset the live caches.

        Intentionally cheap — no GPU compute. Called inside _patched_embed_image
        (inside predict_action_chunk) so it does not inflate the real_latency that
        the RTC queue uses to align action chunks. Rollout compute is deferred to
        drain_rollouts(), which log_overlay() calls after merge() has already fired.
        """
        self._pending.append([
            _LayerCache(q=c.q, k=c.k, num_heads=c.num_heads, head_dim=c.head_dim)
            for c in self._layers
        ])
        for c in self._layers:
            c.q = None
            c.k = None

    def drain_rollouts(
        self,
        last_layer_only: bool = False,
        add_residual: bool = True,
    ) -> list[torch.Tensor]:
        """Compute rollouts from all pending snapshots, clear them, and return the list.

        One tensor per embed_image call (i.e. per camera). Call this from
        log_overlay() — after merge() — so the matmul+softmax compute does not
        contribute to real_latency.
        """
        pending, self._pending = self._pending, []
        results: list[torch.Tensor] = []
        for caches in pending:
            if last_layer_only:
                results.append(self._last_layer_from_caches(caches))
            else:
                results.append(self._rollout_from_caches(caches, add_residual))
        return results

    def _rollout_from_caches(
        self, caches: list[_LayerCache], add_residual: bool
    ) -> torch.Tensor:
        """Attention rollout (Abnar & Zuidema 2020) over an explicit list of caches.

        Streamed: O(S²) peak memory. Returns (B, seq, seq).
        """
        rollout: torch.Tensor | None = None
        eye: torch.Tensor | None = None
        for cache in caches:
            probs = self._layer_attention(cache)
            if probs is None:
                continue
            a = probs.mean(dim=1)  # (B, seq, seq), head-averaged
            if add_residual:
                if eye is None or eye.shape[-1] != a.shape[-1]:
                    eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
                a = a + eye
                a = a / a.sum(dim=-1, keepdim=True)
            rollout = a if rollout is None else torch.matmul(a, rollout)
            del probs, a
        if rollout is None:
            raise RuntimeError("No attentions captured — did you run a forward inside the context?")
        return rollout

    def _last_layer_from_caches(self, caches: list[_LayerCache]) -> torch.Tensor:
        """Head-averaged attention from the last non-empty layer of an explicit cache list."""
        for cache in reversed(caches):
            probs = self._layer_attention(cache)
            if probs is not None:
                return probs.mean(dim=1)  # (B, seq, seq)
        raise RuntimeError("No attentions captured — did you run a forward inside the context?")

    def attention_rollout(self, add_residual: bool = True) -> torch.Tensor:
        """Attention rollout across the LIVE layer caches (for direct use of the low-level API).

        Users who call VisionAttentionCapture directly (not through SmolVLAAttention)
        can call this after a forward pass to get the rollout from self._layers.
        """
        return self._rollout_from_caches(self._layers, add_residual)


class SmolVLAAttention:
    """High-level: SmolVLAPolicy wrapper that streams per-camera attention to rerun.

    Owns the policy-specific glue: locating the SigLIP vision encoder inside the
    VLM, installing q_proj / k_proj hooks, and monkey-patching `embed_image` so
    a fresh rollout is snapshotted after each per-camera vision-encoder forward.
    Runners (and future playback) call `log_overlay(obs)` once per chunk; if no
    forward has happened since the last call (e.g. RTC queue still buffered) it
    is a no-op.

    Usage:
        viz = SmolVLAAttention(policy)
        with viz:
            ...
            actions = policy.predict_action_chunk(batch, ...)
            viz.log_overlay(obs)
    """

    def __init__(self, policy: "SmolVLAPolicy", *, last_layer_only: bool = False):
        self.policy = policy
        self._last_layer_only = last_layer_only
        # Reach the SigLIP encoder buried inside the VLM.
        vision_model = policy.model.vlm_with_expert.get_vlm_model().vision_model
        self._capture = VisionAttentionCapture(vision_model)
        # Holds the original embed_image bound method so we can restore it.
        self._orig_embed_image = None

    def __enter__(self) -> SmolVLAAttention:
        # Install q_proj / k_proj forward hooks on every SigLIP attention layer.
        self._capture.__enter__()

        # `embed_image` is a method on SmolVLMWithExpertModel, not an nn.Module
        # submodule, so register_forward_hook doesn't apply — monkey-patch the
        # bound method. After each call (one per camera per chunk) we snapshot
        # the rollout and reset the Q/K cache.
        self._orig_embed_image = self.policy.model.vlm_with_expert.embed_image

        def _patched_embed_image(image):
            out = self._orig_embed_image(image)
            self._capture.snapshot()  # cheap freeze only — no GPU compute here
            return out

        self.policy.model.vlm_with_expert.embed_image = _patched_embed_image
        return self

    def __exit__(self, *exc) -> None:
        if self._orig_embed_image is not None:
            self.policy.model.vlm_with_expert.embed_image = self._orig_embed_image
            self._orig_embed_image = None
        self._capture.__exit__(*exc)

    def camera_keys(self) -> list[str]:
        """Raw camera keys (e.g. "top", "wrist.top") in the order the policy
        feeds them to the vision encoder."""
        prefix = "observation.images."
        return [
            k[len(prefix):]
            for k in self.policy.config.image_features
            if k.startswith(prefix)
        ]

    def log_overlay(self, obs: dict, *, prefix: str = "attention", clip_percentile: float = 95.0) -> None:
        """Compute rollouts from pending snapshots, then stream image / heatmap / overlay per camera.

        No-op if no forward happened since the last call. Rollout compute (matmul
        + softmax) happens HERE — after merge() has already fired — so it does not
        inflate real_latency and disturb the RTC queue alignment.

        If the rollout count drifts from the camera count (a missed hook), we drop
        the frame rather than misalign overlays.
        """
        # Compute rollouts now, outside the timed inference window.
        rollouts = self._capture.drain_rollouts(last_layer_only=self._last_layer_only)
        if not rollouts:
            return

        camera_keys = self.camera_keys()
        if len(rollouts) != len(camera_keys):
            return

        for cam_key, rollout in zip(camera_keys, rollouts, strict=True):
            # Raw obs dict uses the bare camera name ("top"), not the feature key.
            image = obs.get(cam_key)
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
                continue
            patch_heat = rollout_to_patch_heatmap(rollout)
            heat = patch_heatmap_to_image(patch_heat, target_hw=image.shape[:2], clip_percentile=clip_percentile)
            log_attention_overlay(f"{prefix}/{cam_key}", image, heat)
