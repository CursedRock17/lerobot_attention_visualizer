"""Capture CNN feature-activation heatmaps from ACT's ResNet backbone.

ACT's vision path is a ResNet18 (via `torchvision.models.resnet18`) wrapped in
`IntermediateLayerGetter` to return the `layer4` feature map. There is no
self-attention inside the backbone, so "what is ACT looking at" is best
approximated by the per-spatial-cell activation magnitude of the final conv
stage — i.e. `features.norm(dim=1)` gives a (B, H', W') map that highlights
the locations where the CNN has the strongest response before the transformer
mixes them together.

The backbone is called once per camera inside `ACT.forward` (see the
`for img in batch[OBS_IMAGES]` loop in `modeling_act.py`), so a single forward
hook on `policy.model.backbone` catches every camera per chunk.
"""

from __future__ import annotations

import torch


class ACTBackboneCapture:
    """Forward-hook the ACT ResNet backbone and collect per-camera feature maps.

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
