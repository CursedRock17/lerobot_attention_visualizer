"""π0 / π0.5 / π0-fast attention capture — PaliGemma vision-tower wrapper.

Wraps a `PI0Policy`, `PI05Policy`, or `PI0FastPolicy` so a runner / playback
loop only has to do:

    viz = Pi0Attention(policy)
    with viz:
        ... eval loop ...
        viz.log_overlay(obs)

# Why one class for three policies

All three lerobot v0.5.x π-family policies share an identical attribute path
to the vision encoder:

    policy.model.paligemma_with_expert.paligemma.model.vision_tower
    policy.model.paligemma_with_expert.embed_image

The vision tower is HuggingFace SigLIP (the same encoder PaliGemma uses), so
its attention layers expose `q_proj` / `k_proj` and `VisionAttentionCapture`
works as-is. The wrapper only differs from `SmolVLAAttention` in (a) the
walk to the vision encoder and (b) the parent object that owns
`embed_image`. We therefore reuse the SmolVLA primitive and re-bind the
two attribute paths.

`Pi05Attention` and `Pi0FastAttention` are aliases for `Pi0Attention` —
they exist so users searching for the model they're loading find a name
that matches.

# Scope

Same caveats as SmolVLA: this shows where the SigLIP encoder concentrates
its own internal attention, not which image regions actually drive the
action expert's output. See `policies/smolvla.py` for the longer
discussion and the standard "next steps" (cross-attention from the
expert, Grad-CAM from the action logits) — they apply identically here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

from ..visualizer.overlay import (
    log_attention_overlay,
    patch_heatmap_to_image,
    rollout_to_patch_heatmap,
)
from .smolvla import VisionAttentionCapture

if TYPE_CHECKING:
    # Heavy imports — guarded so users without the [pi] extra can still
    # `import lerobot_attention_visualizer`.
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

    AnyPi0Policy = Union[PI0Policy, PI05Policy, PI0FastPolicy]


class Pi0Attention:
    """High-level: PI0Policy / PI05Policy / PI0FastPolicy wrapper that streams
    per-camera attention to rerun.

    Owns the policy-specific glue: locating the SigLIP vision tower inside
    `paligemma_with_expert`, installing q_proj / k_proj hooks, and
    monkey-patching `embed_image` so a fresh rollout is snapshotted after
    each per-camera vision-encoder forward. Runners (and future playback)
    call `log_overlay(obs)` once per chunk; if no forward has happened
    since the last call (e.g. RTC queue still buffered) it is a no-op.

    Usage:
        viz = Pi0Attention(policy)
        with viz:
            ...
            actions = policy.predict_action_chunk(batch, ...)
            viz.log_overlay(obs)
    """

    def __init__(self, policy: "AnyPi0Policy", *, last_layer_only: bool = False):
        self.policy = policy
        self._last_layer_only = last_layer_only
        # Reach the SigLIP vision tower buried inside the PaliGemma+expert wrapper.
        vision_model = policy.model.paligemma_with_expert.paligemma.model.vision_tower
        self._capture = VisionAttentionCapture(vision_model)
        # Holds the original embed_image bound method so we can restore it.
        self._orig_embed_image = None

    def __enter__(self) -> Pi0Attention:
        # Install q_proj / k_proj forward hooks on every SigLIP attention layer.
        self._capture.__enter__()

        # `embed_image` is a method on PaliGemmaWithExpertModel (or
        # PI0FastPaliGemma), not an nn.Module submodule, so
        # register_forward_hook doesn't apply — monkey-patch the bound
        # method. After each call (one per camera per chunk) we snapshot
        # the rollout and reset the Q/K cache.
        self._orig_embed_image = self.policy.model.paligemma_with_expert.embed_image

        def _patched_embed_image(image):
            out = self._orig_embed_image(image)
            self._capture.snapshot()  # cheap freeze only — no GPU compute here
            return out

        self.policy.model.paligemma_with_expert.embed_image = _patched_embed_image
        return self

    def __exit__(self, *exc) -> None:
        if self._orig_embed_image is not None:
            self.policy.model.paligemma_with_expert.embed_image = self._orig_embed_image
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

    def log_overlay(
        self,
        obs: dict,
        *,
        prefix: str = "attention",
        clip_percentile: float = 95.0,
        suppress_outliers: bool = False,
    ) -> None:
        """Compute rollouts from pending snapshots, then stream image / heatmap / overlay per camera.

        Rollout compute happens here — after merge() — so it doesn't inflate
        real_latency. No-op if no forward happened since the last call.

        `suppress_outliers` winsorizes SigLIP attention-sink spikes — see
        `patch_heatmap_to_image`.
        """
        rollouts = self._capture.drain_rollouts(last_layer_only=self._last_layer_only)
        if not rollouts:
            return

        camera_keys = self.camera_keys()
        if len(rollouts) != len(camera_keys):
            return

        for cam_key, rollout in zip(camera_keys, rollouts, strict=True):
            image = obs.get(cam_key)
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
                continue
            patch_heat = rollout_to_patch_heatmap(rollout)
            heat = patch_heatmap_to_image(
                patch_heat,
                target_hw=image.shape[:2],
                clip_percentile=clip_percentile,
                suppress_outliers=suppress_outliers,
            )
            log_attention_overlay(f"{prefix}/{cam_key}", image, heat)


# Aliases — the structure is identical for all three π-family policies.
# Users loading PI05Policy or PI0FastPolicy can import the matching name.
Pi05Attention = Pi0Attention
Pi0FastAttention = Pi0Attention
