"""ACT activation capture — ResNet18 backbone wrapper.

Wraps an `ACTPolicy` so a runner / playback loop only has to do:

    viz = ACTAttention(policy)
    with viz:
        ... eval loop ...
        viz.log_overlay(obs)

`ACTAttention` installs a forward hook on `policy.model.backbone`, collects
per-camera feature maps each time ACT's internal action queue refills, and
on exit removes the hook. `log_overlay(obs)` reduces the feature maps to
per-spatial-cell activation magnitude, upsamples to camera resolution, and
writes three rerun streams per camera (image / attention / overlay).

# Design

ACT's vision path is a ResNet18 (via `torchvision.models.resnet18`) wrapped
in `IntermediateLayerGetter` to return the `layer4` feature map. There is
no self-attention inside the backbone, so "what is ACT looking at" is best
approximated by the per-spatial-cell activation magnitude of the final
conv stage — `features.norm(dim=1)` gives a (B, H', W') map that
highlights where the CNN has the strongest response before the transformer
mixes them together.

The backbone is called once per camera inside `ACT.forward` (see the
`for img in batch[OBS_IMAGES]` loop in lerobot's `modeling_act.py`), so a
single forward hook on `policy.model.backbone` catches every camera per
chunk.

# Scope

The overlay is an *activation heatmap*, not an attention map. Channel-wise
L2 norm is the standard "where did the CNN fire" proxy when you don't want
to run GradCAM. Caveats:

- Bright doesn't mean *important for the action* — the transformer still
  reweights these tokens via cross-attention. A cleaner "what drives the
  action" signal would be the decoder's cross-attention weights onto the
  image tokens, which is a natural follow-up on the same hook scaffold.
- The overlay only refreshes when ACT's internal action queue refills
  (every `n_action_steps` control steps). Between refills the last overlay
  stays on screen in rerun.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ..visualizer.overlay import log_attention_overlay, patch_heatmap_to_image

if TYPE_CHECKING:
    from lerobot.policies.act.modeling_act import ACTPolicy


class ACTBackboneCapture:
    """Low-level: forward-hook the ACT ResNet backbone and collect per-camera feature maps.

    Most users want `ACTAttention` instead — this is the underlying primitive
    if you want to do something with feature maps that isn't "log to rerun".

    Usage:
        capture = ACTBackboneCapture(policy.model)
        with capture:
            _ = policy.select_action(batch)  # triggers 0 or 1 forwards depending on queue
        heatmaps = capture.activation_heatmaps()  # list[(B, H', W')] — one per camera
        capture.clear()                           # wipe between chunks

    On every call to `policy.model.backbone(img)` the hook appends the returned
    feature map to `self.feature_maps`. Order matches the order the policy feeds
    images (i.e. `policy.config.image_features` insertion order).
    """

    def __init__(self, model: torch.nn.Module):
        self.backbone = model.backbone
        self._handle: torch.utils.hooks.RemovableHandle | None = None
        # One entry per (camera × forward). Caller is responsible for clearing.
        self.feature_maps: list[torch.Tensor] = []

    def __enter__(self) -> ACTBackboneCapture:
        self.feature_maps.clear()

        def _hook(_mod, _inp, out):
            # IntermediateLayerGetter returns {"feature_map": tensor}; be defensive.
            fmap = out["feature_map"] if isinstance(out, dict) else out
            self.feature_maps.append(fmap.detach())

        self._handle = self.backbone.register_forward_hook(_hook)
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def activation_heatmaps(self) -> list[torch.Tensor]:
        """Return per-camera (B, H', W') activation magnitude heatmaps.

        Channel-wise L2 norm is the standard "where did the CNN fire" proxy when
        you don't want to run GradCAM. High values = features in this spatial
        location carry large magnitude into the transformer.
        """
        return [fmap.norm(dim=1) for fmap in self.feature_maps]

    def clear(self) -> None:
        """Drop all captured feature maps. Call this after logging each chunk."""
        self.feature_maps.clear()


class ACTAttention:
    """High-level: ACTPolicy wrapper that streams per-camera activation to rerun.

    Owns the policy-specific glue: hooking `policy.model.backbone`, knowing the
    camera-key order, and turning per-camera feature maps into upsampled rerun
    streams. Runners (and future playback) call `log_overlay(obs)` once per
    control step; on steps where ACT's queue didn't refill (no fresh forward),
    it is a no-op and the previous overlay stays on screen in rerun.

    Usage:
        viz = ACTAttention(policy)
        with viz:
            ...
            action = policy.select_action(batch)
            viz.log_overlay(obs)
    """

    def __init__(self, policy: "ACTPolicy"):
        self.policy = policy
        self._capture = ACTBackboneCapture(policy.model)

    def __enter__(self) -> ACTAttention:
        self._capture.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._capture.__exit__(*exc)

    def camera_keys(self) -> list[str]:
        """Raw camera keys (e.g. "top", "wrist.top") in the order the policy
        feeds them to the backbone."""
        prefix = "observation.images."
        return [
            k[len(prefix):]
            for k in self.policy.config.image_features
            if k.startswith(prefix)
        ]

    def log_overlay(self, obs: dict, *, prefix: str = "attention") -> None:
        """Drain captured feature maps and stream image / heatmap / overlay per camera.

        No-op when ACT's queue didn't refill (no fresh forward this step).
        Drift between heatmap count and camera count drops the frame rather
        than misalign overlays.
        """
        if not self._capture.feature_maps:
            return

        camera_keys = self.camera_keys()
        heatmaps = self._capture.activation_heatmaps()
        self._capture.clear()

        if len(heatmaps) != len(camera_keys):
            return

        for cam_key, heat in zip(camera_keys, heatmaps, strict=True):
            # Raw obs dict uses the bare camera name ("top"), not the feature key.
            image = obs.get(cam_key)
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
                continue
            # Strip batch dim — we only ever run batch size 1 here.
            heat_2d = heat[0]
            # Upsample (H', W') ResNet grid to full camera resolution.
            heat_img = patch_heatmap_to_image(heat_2d, target_hw=image.shape[:2])
            log_attention_overlay(f"{prefix}/{cam_key}", image, heat_img)
