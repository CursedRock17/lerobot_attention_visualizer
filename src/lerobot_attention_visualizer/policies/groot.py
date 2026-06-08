"""Groot attention capture — Eagle VLM backbone wrappers (N1.5 and N1.6).

Two adapters, sharing the `_EagleCrossAttention` base:

GR00TAttention — Groot N1.5 (lerobot `GrootPolicy`). Eagle-2 = SmolLM2 + SigLIP.
    policy._groot_model.backbone.eagle_model.vision_model
    policy._groot_model.action_head.model            (the cross-attn DiT)
    backbone runs the VLM via `forward_eagle`; vision tokens recovered from
    `eagle_input_ids == image_token_index`.

GR00TN1d6Attention — Groot N1.6 (Isaac-GR00T native `Gr00tPolicy`). Eagle3-VL =
    SigLIP2 + Qwen3 ("AlternateVLDiT" action head). Paths verified against the
    Isaac-GR00T `n1.6-release` source and a real `Gr00tN1d6` checkpoint:
    policy.model.backbone.model.vision_model           (NOT backbone.eagle_model)
    policy.model.action_head.model.transformer_blocks  (diffusers Attention)
    backbone runs the VLM via plain `forward`, which returns `image_mask`
    directly; token index at `backbone.model.config.image_token_index`.

The base only differs between the two in three small hooks (`_forward_attr`,
`_image_token_index`, `_extract_vision_mask`); everything else is shared.

# Two complementary signals

Each adapter streams **two** overlays per camera:

`attention/<cam>/encoder/*`
    Vision-encoder (SigLIP/SigLIP2) self-attention rollout — "which patches the
    (usually frozen) image encoder finds salient." Captured via
    `VisionAttentionCapture` + `snapshot_split`. Best-effort on N1.6: SigLIP2
    tiles at native resolution, so if the per-image patch grid isn't square the
    encoder overlay is skipped (the action overlay below is the key signal).

`attention/<cam>/action/*`
    Action-head cross-attention — "which vision tokens the flow-matching
    denoiser actually looks at while producing the action." Captured via
    `CrossAttentionCapture` on the DiT's cross-attending blocks. This is the
    signal that changes when you fine-tune the action head — the one to watch
    for grounding bugs. See `cross_attention.py`.

The action head attends to the post-projection vision tokens, not the raw
SigLIP patches. We recover their columns from the image-token mask and reshape
per camera (square by default; pass `patch_grid_hw` for N1.6's native-resolution
rectangular grids).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..visualizer.overlay import (
    log_attention_overlay,
    patch_heatmap_to_image,
    rollout_to_patch_heatmap,
)
from .cross_attention import (
    CrossAttentionCapture,
    find_cross_attention_blocks,
    vision_importance_to_grids,
)
from .smolvla import VisionAttentionCapture

if TYPE_CHECKING:
    from lerobot.policies.groot.modeling_groot import GrootPolicy


class _EagleCrossAttention:
    """Shared base for the N1.5 and N1.6 Groot adapters.

    Subclasses resolve the two model-root-dependent attribute paths (the Eagle
    vision model and the action head live under `_groot_model` for N1.5 and under
    `model` for N1.6) and supply `camera_keys()`. Everything else — installing
    the encoder + cross-attention captures, patching `forward_eagle`, and logging
    both overlays — is shared here.
    """

    # Name of the backbone method that runs the Eagle VLM forward. N1.5 exposes
    # `forward_eagle`; N1.6 runs it as the backbone's plain `forward`.
    _forward_attr: str = "forward_eagle"

    def __init__(
        self,
        policy,
        *,
        vision_model,
        model_root,
        last_layer_only: bool,
        cross_attention: bool,
        patch_grid_hw: tuple[int, int] | None = None,
    ):
        self.policy = policy
        self._model_root = model_root
        self._last_layer_only = last_layer_only
        self._cross_enabled = cross_attention
        # Per-camera (h, w) for the action-overlay vision-token grid. None = infer
        # a square grid. Needed when the vision tower tiles at native resolution
        # (N1.6) and the per-camera token count isn't a perfect square.
        self._patch_grid_hw = patch_grid_hw

        self._capture = VisionAttentionCapture(vision_model)
        self._cross: CrossAttentionCapture | None = None
        # (attr_name, original_callable) of the patched backbone forward.
        self._orig_forward: tuple[str, object] | None = None
        self._img_tok: int | None = None
        # Vision-token column mask over the VL key sequence, captured per forward.
        self._vision_mask = None

    def camera_keys(self) -> list[str]:
        raise NotImplementedError

    def _image_token_index(self) -> int:
        """The token id the VLM replaces with vision embeddings (input_ids == this).

        N1.5 default: read off the Eagle-2 model. N1.6 overrides this.
        """
        eagle_model = self._model_root.backbone.eagle_model
        idx = getattr(eagle_model, "image_token_index", None)
        if idx is None:
            idx = eagle_model.config.image_token_index
        return idx

    def _extract_vision_mask(self, vl_input, result):
        """Boolean (seq,) mask of vision-token columns in the VL key sequence.

        N1.5 default: rebuild from the eagle input ids. N1.6 overrides to read the
        `image_mask` the backbone returns directly.
        """
        input_ids = vl_input["eagle_input_ids"]
        return (input_ids[0] == self._img_tok).detach().cpu()

    def __enter__(self) -> "_EagleCrossAttention":
        self._capture.__enter__()
        n_cameras = len(self.camera_keys())
        backbone = self._model_root.backbone
        fwd_attr = self._forward_attr
        orig = getattr(backbone, fwd_attr)
        self._orig_forward = (fwd_attr, orig)
        capture = self._capture

        if self._cross_enabled:
            cross_blocks = find_cross_attention_blocks(
                self._model_root.action_head.model.transformer_blocks
            )
            self._cross = CrossAttentionCapture(cross_blocks)
            self._cross.__enter__()
            self._img_tok = self._image_token_index()

        def _patched_forward(vl_input, *args, **kwargs):
            result = orig(vl_input, *args, **kwargs)
            # Encoder path: Q/K are (N_cameras, n_patches, dim) for a batched
            # vision forward; snapshot_split slices dim 0 per camera.
            capture.snapshot_split(n_cameras)
            if self._cross is not None:
                self._vision_mask = self._extract_vision_mask(vl_input, result)
            return result

        setattr(backbone, fwd_attr, _patched_forward)
        return self

    def __exit__(self, *exc) -> None:
        if self._orig_forward is not None:
            attr, orig = self._orig_forward
            setattr(self._model_root.backbone, attr, orig)
            self._orig_forward = None
        self._capture.__exit__(*exc)
        if self._cross is not None:
            self._cross.__exit__(*exc)
            self._cross = None

    def log_overlay(
        self,
        obs: dict,
        *,
        prefix: str = "attention",
        clip_percentile: float = 95.0,
        suppress_outliers: bool = False,
        gamma: float = 1.0,
        colormap: str = "hot",
    ) -> None:
        """Stream the encoder overlay and the action cross-attention overlay.

        No-op if no forward happened since the last call. Each signal is dropped
        independently if its token count doesn't match the camera count, rather
        than misaligning overlays.

        `suppress_outliers` winsorizes SigLIP attention-sink spikes (mainly helps
        the encoder view) and `gamma` sets the display contrast (>1 = punchier,
        background suppressed) — see `patch_heatmap_to_image`. `colormap` picks the
        palette ("hot", "blue-green", "viridis"). All apply to both overlays.
        """
        camera_keys = self.camera_keys()
        n = len(camera_keys)

        # --- Encoder self-attention (always on) ---
        rollouts = self._capture.drain_rollouts(last_layer_only=self._last_layer_only)
        if rollouts and len(rollouts) == n:
            for cam_key, rollout in zip(camera_keys, rollouts, strict=True):
                image = obs.get(cam_key)
                if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
                    continue
                try:
                    patch_heat = rollout_to_patch_heatmap(rollout)
                except ValueError:
                    # Non-square encoder patch grid (e.g. N1.6 native-resolution
                    # SigLIP2). The encoder view is best-effort; skip it rather
                    # than crash — the action overlay below is the key signal.
                    continue
                heat = patch_heatmap_to_image(
                    patch_heat,
                    target_hw=image.shape[:2],
                    clip_percentile=clip_percentile,
                    suppress_outliers=suppress_outliers,
                    gamma=gamma,
                )
                log_attention_overlay(f"{prefix}/{cam_key}/encoder", image, heat, colormap=colormap)

        # --- Action-head cross-attention (the action-driving signal) ---
        importance = self._cross.drain() if self._cross is not None else None
        if importance is not None and self._vision_mask is not None:
            grids = vision_importance_to_grids(
                importance, self._vision_mask, n, grid_hw=self._patch_grid_hw
            )
            if len(grids) == n:
                for cam_key, grid in zip(camera_keys, grids, strict=True):
                    image = obs.get(cam_key)
                    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
                        continue
                    heat = patch_heatmap_to_image(
                        grid,
                        target_hw=image.shape[:2],
                        clip_percentile=clip_percentile,
                        suppress_outliers=suppress_outliers,
                        gamma=gamma,
                    )
                    log_attention_overlay(f"{prefix}/{cam_key}/action", image, heat, colormap=colormap)


class GR00TAttention(_EagleCrossAttention):
    """High-level: lerobot GrootPolicy (N1.5) wrapper streaming both overlays to rerun.

    Usage:
        viz = GR00TAttention(policy)
        with viz:
            action = policy.select_action(batch)
            viz.log_overlay(obs)

    Pass `cross_attention=False` to stream only the encoder overlay.
    """

    def __init__(
        self,
        policy: "GrootPolicy",
        *,
        last_layer_only: bool = False,
        cross_attention: bool = True,
    ):
        super().__init__(
            policy,
            vision_model=policy._groot_model.backbone.eagle_model.vision_model,
            model_root=policy._groot_model,
            last_layer_only=last_layer_only,
            cross_attention=cross_attention,
        )

    def camera_keys(self) -> list[str]:
        """Bare camera names in the order the policy feeds them to the encoder."""
        prefix = "observation.images."
        return [
            k[len(prefix):]
            for k in self.policy.config.image_features
            if k.startswith(prefix)
        ]


class GR00TN1d6Attention(_EagleCrossAttention):
    """High-level: Isaac-GR00T Gr00tPolicy (N1.6 / Gr00tN1d6) wrapper.

    Use this for checkpoints with `"model_type": "Gr00tN1d6"` (Eagle3-VL =
    SigLIP2 + Qwen3), loaded via Isaac-GR00T:

        from gr00t.policy.gr00t_policy import Gr00tPolicy
        policy = Gr00tPolicy(model_path=..., embodiment_tag="new_embodiment", device="cuda")
        viz = GR00TN1d6Attention(policy, camera_keys=["external_D455", "ego"])
        with viz:
            action = policy.get_action(obs)
            viz.log_overlay(obs)

    `camera_keys` are the bare obs-dict names for your embodiment (e.g.
    `["external_D455", "ego"]`), in feed order.

    Paths verified against the Isaac-GR00T `n1.6-release` source and a real
    `Gr00tN1d6` checkpoint (`model.safetensors.index.json`):
      - vision encoder: `policy.model.backbone.model.vision_model`
      - action DiT:      `policy.model.action_head.model.transformer_blocks`
      - N1.6's `EagleBackbone.forward` returns `image_mask` directly, so we read
        that instead of rebuilding from input ids; the token index lives at
        `backbone.model.config.image_token_index`.

    `patch_grid_hw`: N1.6's SigLIP2 tiles at native resolution, so the per-camera
    vision-token count may not be a perfect square. Pass the per-camera grid
    `(h, w)` (read it once from a real forward — see `examples/visualize_groot_n16.py`)
    to get a correctly-shaped action overlay; leave it None to infer a square grid.
    """

    _forward_attr = "forward"  # N1.6 backbone runs the VLM in plain forward()

    def __init__(
        self,
        policy,
        *,
        camera_keys: list[str],
        last_layer_only: bool = False,
        cross_attention: bool = True,
        patch_grid_hw: tuple[int, int] | None = None,
    ):
        super().__init__(
            policy,
            vision_model=policy.model.backbone.model.vision_model,
            model_root=policy.model,
            last_layer_only=last_layer_only,
            cross_attention=cross_attention,
            patch_grid_hw=patch_grid_hw,
        )
        self._camera_keys = camera_keys

    def camera_keys(self) -> list[str]:
        return self._camera_keys

    def _image_token_index(self) -> int:
        # Eagle3-VL exposes it on the model config.
        cfg = self._model_root.backbone.model.config
        idx = getattr(cfg, "image_token_index", None)
        if idx is None:
            idx = self._model_root.backbone.model.image_token_index
        return idx

    def _extract_vision_mask(self, vl_input, result):
        # N1.6's EagleBackbone.forward returns a BatchFeature with `image_mask`
        # (input_ids == image_token_index). Prefer it; fall back to input_ids.
        mask = None
        try:
            mask = result["image_mask"]
        except (KeyError, TypeError):
            mask = getattr(result, "image_mask", None)
        if mask is not None:
            return mask[0].detach().cpu()
        return (vl_input["input_ids"][0] == self._img_tok).detach().cpu()
