"""Groot attention capture — Eagle-2 VLM backbone wrapper.

Two adapters are provided:

GR00TAttention
    Wraps lerobot's `GrootPolicy` (Groot N1.5). Attribute path:
        policy._groot_model.backbone.eagle_model.vision_model

GR00TN1d6Attention
    Wraps Isaac-GR00T's native `Gr00tPolicy` (Groot N1.6 / Gr00tN1d6).
    The native loader registers `Gr00tN1d6` with HuggingFace AutoModel;
    lerobot's `GrootPolicy.from_pretrained` cannot load N1.6 checkpoints.
    Attribute path:
        policy.model.backbone.eagle_model.vision_model

Both share the same Eagle-2 / SigLIP vision encoder architecture and the
same `VisionAttentionCapture` + `snapshot_split` capture strategy.

# Why two adapters?

    N1.5 checkpoint → lerobot GrootPolicy → GR00TAttention
    N1.6 checkpoint → Isaac-GR00T Gr00tPolicy → GR00TN1d6Attention
    N1.7 checkpoint → Isaac-GR00T, Qwen3-VL backbone → not supported (no SigLIP)

# Capture strategy (shared)

Eagle-2 processes all camera images in a single batched forward
(`forward_eagle`). Q/K tensors emerge shaped `(N_cameras, n_patches, dim)`.
We patch `forward_eagle` and call `snapshot_split(n_cameras)` to slice
along dim 0, producing one _LayerCache per camera. `drain_rollouts()` then
computes per-camera rollouts identically to the SmolVLA path.

Groot's processor hardcodes `max_dynamic_tiles=1`, so the batch dim always
equals N_cameras (no dynamic image tiling).
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


class GR00TN1d6Attention:
    """High-level: Isaac-GR00T Gr00tPolicy (N1.6) wrapper for rerun attention streaming.

    Use this for checkpoints with `"model_type": "Gr00tN1d6"` loaded via:

        from gr00t.policy.gr00t_policy import Gr00tPolicy
        policy = Gr00tPolicy(
            model_path="nvidia/GR00T-N1.6-3B",
            embodiment_tag="new_embodiment",
            device="cuda",
        )
        viz = GR00TN1d6Attention(policy, camera_keys=["ego", "external"])
        with viz:
            action = policy.get_action(obs)
            viz.log_overlay(obs)

    camera_keys must match the bare names used in your obs dict
    (e.g. "ego", "external" — without the "observation.images." prefix).

    The Isaac-GR00T `Gr00tPolicy` stores the loaded model at `policy.model`
    (not `policy._groot_model` as in lerobot's wrapper). Everything else —
    Eagle backbone, forward_eagle, snapshot_split — is identical to N1.5.
    """

    def __init__(
        self,
        policy,
        *,
        camera_keys: list[str],
        last_layer_only: bool = False,
    ):
        self.policy = policy
        self._camera_keys = camera_keys
        self._last_layer_only = last_layer_only

        # Isaac-GR00T: policy.model is the Gr00tN1d6 PreTrainedModel instance.
        vision_model = policy.model.backbone.eagle_model.vision_model
        self._capture = VisionAttentionCapture(vision_model)
        self._orig_forward_eagle = None

    def camera_keys(self) -> list[str]:
        return self._camera_keys

    def __enter__(self) -> GR00TN1d6Attention:
        self._capture.__enter__()

        orig = self.policy.model.backbone.forward_eagle
        self._orig_forward_eagle = orig
        capture = self._capture
        n_cameras = len(self._camera_keys)

        def _patched_forward_eagle(vl_input):
            result = orig(vl_input)
            capture.snapshot_split(n_cameras)
            return result

        self.policy.model.backbone.forward_eagle = _patched_forward_eagle
        return self

    def __exit__(self, *exc) -> None:
        if self._orig_forward_eagle is not None:
            self.policy.model.backbone.forward_eagle = self._orig_forward_eagle
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

        No-op if no forward happened since the last call.
        """
        rollouts = self._capture.drain_rollouts(last_layer_only=self._last_layer_only)
        if not rollouts:
            return

        if len(rollouts) != len(self._camera_keys):
            return

        for cam_key, rollout in zip(self._camera_keys, rollouts, strict=True):
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
