"""Heatmap math + rerun logging for attention-rollout overlays."""

from __future__ import annotations

import math

import numpy as np
import rerun as rr
import torch
import torch.nn.functional as F


def rollout_to_patch_heatmap(
    rollout: torch.Tensor, patch_grid_hw: tuple[int, int] | None = None
) -> torch.Tensor:
    """Reduce a (seq, seq) rollout into a (patch_grid_h, patch_grid_w) heatmap.

    SigLIP / Idefics3 vision tokens are a flat sequence of patches with no CLS
    token. We therefore summarize per-patch importance as the attention each
    patch *receives*, averaged across all queries — i.e. the column-mean of the
    rollout (`mean(dim=0)`, since `rollout[i, j]` is query i attending to key j).
    This is the standard trick when the encoder has no class token to query
    against.
    """
    if rollout.ndim == 3:
        # Strip batch dim — we only ever pass batch size 1 through the policy.
        rollout = rollout[0]

    # Column-mean = average attention received by each patch across all queries.
    importance = rollout.mean(dim=0)  # (seq,)
    n = importance.numel()
    if patch_grid_hw is not None:
        h, w = patch_grid_hw
        if h * w != n:
            raise ValueError(
                f"patch_grid_hw {patch_grid_hw} (={h * w}) doesn't match token count {n}."
            )
        return importance.view(h, w)
    # SigLIP produces a square patch grid; infer the side length.
    side = int(math.isqrt(n))
    if side * side != n:
        raise ValueError(
            f"Patch count {n} is not a perfect square — vision encoder produced a "
            "non-square token grid. Pass patch_grid_hw explicitly if you hit this."
        )
    return importance.view(side, side)


def patch_heatmap_to_image(
    heatmap: torch.Tensor,
    target_hw: tuple[int, int],
    clip_percentile: float = 95.0,
    suppress_outliers: bool = False,
    gamma: float = 1.0,
) -> np.ndarray:
    """Upsample a (h_p, w_p) patch heatmap to `target_hw` and normalize to [0, 1].

    `clip_percentile` clips the top (100 - clip_percentile)% of values before
    normalizing. SigLIP's learned positional embeddings create consistent hot
    spots at edge patches; capping at the 95th percentile suppresses those
    structural artifacts so semantic content (e.g. the object being grasped)
    fills the color scale instead.

    `suppress_outliers` (opt-in) additionally winsorizes the *patch grid* — before
    upsampling — to a robust upper bound `median + 6·MAD`. SigLIP attention-sink /
    register patches show up as a handful of isolated spikes that read as
    splotches; pulling them down to the top of the real-signal range lets them
    blend in instead of dominating. `clip_percentile` only caps the *upsampled*
    distribution (a percentile keeps the spikes at full brightness); this removes
    them at the source. Left off by default so the map stays faithful unless asked.

    `gamma` applies a power curve to the final [0, 1] map (`h ** gamma`). The
    default 1.0 is linear (unchanged). `gamma > 1` makes the map more "extreme" —
    it crushes low/mid background toward 0 while leaving the bright peak near 1,
    so the hot region stands out and ambient noise fades (try 2–4). `gamma < 1`
    flattens / brightens the low end. It's a display contrast dial only; the
    underlying attention isn't changed.
    """
    h = heatmap.float()
    if suppress_outliers:
        flat = h.reshape(-1)
        med = flat.median()
        mad = (flat - med).abs().median()
        upper = med + 6.0 * mad
        h = torch.minimum(h, upper)
    # Add batch + channel dims for torch interpolate.
    h = h[None, None]  # (1, 1, H, W)
    # Bilinear upsample from patch grid to full image resolution.
    h = F.interpolate(h, size=target_hw, mode="bilinear", align_corners=False)[0, 0]
    # Percentile-clip then normalize so positional artifacts don't dominate.
    lo = h.min()
    hi = torch.quantile(h.reshape(-1), clip_percentile / 100.0)
    h = (h - lo) / (hi - lo + 1e-8)
    h = h.clamp(0.0, 1.0)
    # Contrast curve: push background down, keep the peak, for a punchier overlay.
    if gamma != 1.0:
        h = h.pow(gamma)
    return h.cpu().numpy()


# Named colormaps as (position, (r, g, b)) control points in [0, 1], linearly
# interpolated per channel. Matplotlib-free to keep the dep list small; add a
# new entry here and it's instantly selectable via the `colormap=` argument.
_COLORMAPS: dict[str, list[tuple[float, tuple[float, float, float]]]] = {
    # black → red → yellow → white. The original ramp — these control points
    # reproduce the old piecewise `clip(3h-k)` output exactly.
    "hot": [(0.0, (0, 0, 0)), (1 / 3, (1, 0, 0)), (2 / 3, (1, 1, 0)), (1.0, (1, 1, 1))],
    # black → deep blue → teal → pale green. A cool "blue-green" scale.
    "blue-green": [
        (0.0, (0, 0, 0)),
        (0.4, (0.0, 0.2, 0.7)),
        (0.7, (0.0, 0.7, 0.6)),
        (1.0, (0.7, 1.0, 0.7)),
    ],
    # perceptual purple → blue → teal → green → yellow (a viridis approximation).
    "viridis": [
        (0.0, (0.267, 0.005, 0.329)),
        (0.25, (0.275, 0.314, 0.545)),
        (0.5, (0.128, 0.566, 0.551)),
        (0.75, (0.369, 0.789, 0.383)),
        (1.0, (0.993, 0.906, 0.144)),
    ],
}


def _apply_colormap(heatmap: np.ndarray, colormap: str) -> np.ndarray:
    """Map a HW float heatmap in [0, 1] to an HWC uint8 RGB image via a named LUT."""
    try:
        pts = _COLORMAPS[colormap]
    except KeyError:
        raise ValueError(
            f"Unknown colormap {colormap!r}; choose from {sorted(_COLORMAPS)}"
        ) from None
    xs = np.array([p[0] for p in pts])
    channels = [np.interp(heatmap, xs, [p[1][c] for p in pts]) for c in range(3)]
    return (np.stack(channels, axis=-1) * 255).astype(np.uint8)


def log_attention_overlay(
    rerun_path: str,
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "hot",
) -> None:
    """Log raw image + attention heatmap + blended overlay under `rerun_path`.

    `image`    HWC uint8 in [0, 255]
    `heatmap`  HW    float in [0, 1]
    `colormap` one of `_COLORMAPS` ("hot", "blue-green", "viridis").
    """
    assert image.dtype == np.uint8
    assert heatmap.shape == image.shape[:2]

    heat_rgb = _apply_colormap(heatmap, colormap)

    # 50/50 blend of image and heatmap so both are visible at once.
    blended = ((1 - alpha) * image + alpha * heat_rgb).astype(np.uint8)

    # Three rerun streams: raw, heatmap, blended overlay.
    rr.log(f"{rerun_path}/image", rr.Image(image))
    rr.log(f"{rerun_path}/attention", rr.Image(heat_rgb))
    rr.log(f"{rerun_path}/overlay", rr.Image(blended))
