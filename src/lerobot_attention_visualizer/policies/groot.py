"""Groot N1.6 attention capture — Eagle-2 VLM backbone wrapper.

Wraps a `GrootPolicy` so a runner only has to do:

    viz = GR00TAttention(policy)
    with viz:
        action = policy.select_action(batch)
        viz.log_overlay(obs)

# Architecture

Groot N1.6 uses Eagle-2.5 as its VLM backbone. Eagle-2.5 embeds a
SiglipVisionModel as its visual encoder — the identical architecture to
SmolVLA's SigLIP tower — so `VisionAttentionCapture` is reused unchanged.

Attribute path to the vision encoder:

    policy._groot_model.backbone.eagle_model.vision_model  (SiglipVisionModel)
        └── vision_model.encoder.layers[i].self_attn       (q_proj, k_proj)

# Key difference from SmolVLA

SmolVLA calls `embed_image` once per camera; we snapshot between calls.
Groot calls `backbone.forward_eagle` once with all cameras batched into a
single `pixel_values` tensor of shape `(N_cameras, C, H, W)`. After that
single forward the Q/K tensors in each layer have batch dim = N_cameras.

We patch `forward_eagle` and call `snapshot_split(n_cameras)` afterwards,
which slices the batch dim into one _LayerCache entry per camera —
`drain_rollouts()` then processes them identically to the SmolVLA path.

# Dynamic tiling

Eagle-2.5 supports dynamic image tiling for general VLM chat, but Groot's
processor hardcodes `max_dynamic_tiles=1` (processor_groot.py line 517),
so each camera image always produces exactly one tile. The batch dim of
`pixel_values` equals N_cameras, not N_cameras × N_tiles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..visualizer.overlay import (
    log_attention_overlay,
    patch_heatmap_to_image,
    rollout_to_patch_heatmap,
)
from .smolvla import VisionAttentionCapture

if TYPE_CHECKING:
    from lerobot.policies.groot.modeling_groot import GrootPolicy


class GR00TAttention:
    """High-level: GrootPolicy wrapper that streams per-camera attention to rerun.

    Locates the SigLIP vision encoder inside Eagle-2, installs q_proj / k_proj
    hooks, and patches `forward_eagle` to snapshot all-camera Q/K tensors after
    each batched vision-encoder forward. `log_overlay(obs)` drains the captured
    rollouts and writes three rerun streams per camera (image / attention / overlay).

    Usage:
        viz = GR00TAttention(policy)
        with viz:
            action = policy.select_action(batch)
            viz.log_overlay(obs)
    """

    def __init__(self, policy: "GrootPolicy", *, last_layer_only: bool = False):
        self.policy = policy
        self._last_layer_only = last_layer_only

        # Navigate to the SigLIP encoder inside Eagle-2.
        vision_model = policy._groot_model.backbone.eagle_model.vision_model
        self._capture = VisionAttentionCapture(vision_model)
        self._orig_forward_eagle = None

    def camera_keys(self) -> list[str]:
        """Bare camera names in the order the policy feeds them to the vision encoder."""
        prefix = "observation.images."
        return [
            k[len(prefix):]
            for k in self.policy.config.image_features
            if k.startswith(prefix)
        ]

    def __enter__(self) -> GR00TAttention:
        self._capture.__enter__()

        # Patch forward_eagle so we can snapshot immediately after the single
        # batched vision-model forward that processes all cameras at once.
        orig = self.policy._groot_model.backbone.forward_eagle
        self._orig_forward_eagle = orig
        capture = self._capture
        n_cameras = len(self.camera_keys())

        def _patched_forward_eagle(vl_input):
            result = orig(vl_input)
            # Q/K shape at this point: (N_cameras, n_patches, embed_dim).
            # snapshot_split slices dim 0 into one _LayerCache per camera.
            capture.snapshot_split(n_cameras)
            return result

        self.policy._groot_model.backbone.forward_eagle = _patched_forward_eagle
        return self

    def __exit__(self, *exc) -> None:
        if self._orig_forward_eagle is not None:
            self.policy._groot_model.backbone.forward_eagle = self._orig_forward_eagle
            self._orig_forward_eagle = None
        self._capture.__exit__(*exc)

    def log_overlay(
        self,
        obs: dict,
        *,
        prefix: str = "attention",
        clip_percentile: float = 95.0,
    ) -> None:
        """Compute rollouts from pending snapshots and stream to rerun.

        No-op if no forward happened since the last call. Drops the frame
        silently if the rollout count doesn't match the camera count.
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
            )
            log_attention_overlay(f"{prefix}/{cam_key}", image, heat)
