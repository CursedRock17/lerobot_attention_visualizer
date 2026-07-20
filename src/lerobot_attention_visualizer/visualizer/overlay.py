"""Heatmap math + rerun logging for attention-rollout overlays."""

from __future__ import annotations

import math

import numpy as np
import rerun as rr
import torch
import torch.nn.functional as F


def rollout_to_patch_heatmap(rollout: torch.Tensor) -> torch.Tensor:
    """Reduce a (seq, seq) rollout into a (patch_grid_h, patch_grid_w) heatmap.

    SigLIP / Idefics3 vision tokens are a flat sequence of patches with no CLS
    token. The choice of reduction must match what the model actually *consumes*:

    - A classifier ViT reads the pooled/CLS token, so its faithful map is the
      CLS *row* of the attention matrix (what the class query attends to). This
      is what timm-style visualizers use.
    - A VLA is different: `embed_image` feeds **every** patch token onward
      (vision_model.last_hidden_state → connector → LLM); the pooled/probe
      output is discarded. So the faithful per-patch importance is how much each
      input patch contributes across *all* output tokens — the **column-mean**
      of the rollout (`mean(dim=0)`, since `rollout[i, j]` is query i attending
      to key j). That is the reduction used here.
    """
    if rollout.ndim == 3:
        # Strip batch dim — we only ever pass batch size 1 through the policy.
        rollout = rollout[0]

    # Column-mean = average attention received by each patch across all queries.
    importance = rollout.mean(dim=0)  # (seq,)
    n = importance.numel()
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
    clip_percentile: float = 100.0,
) -> np.ndarray:
    """Upsample a (h_p, w_p) patch heatmap to `target_hw` and normalize to [0, 1].

    The default is a **faithful min-max** normalization (`clip_percentile=100`):
    the grid's true min maps to 0 and its true max to 1, so the overlay is an
    accurate depiction of the per-patch attention — no values invented or hidden.

    Lowering `clip_percentile` is an OPTIONAL display aid, not the accurate map:
    it clips the top (100 - clip_percentile)% of values before normalizing, which
    stops SigLIP edge/attention-sink artifacts from dominating the color scale at
    the cost of faithfulness. Reach for it only to make a map more readable.
    """
    # Add batch + channel dims for torch interpolate.
    h = heatmap.float()[None, None]  # (1, 1, H, W)
    # Bilinear upsample from patch grid to full image resolution.
    h = F.interpolate(h, size=target_hw, mode="bilinear", align_corners=False)[0, 0]
    # Min-max normalize (clip_percentile=100). A lower percentile winsorizes the
    # top tail first — a readability trade-off, off by default.
    lo = h.min()
    hi = torch.quantile(h.reshape(-1), clip_percentile / 100.0)
    h = (h - lo) / (hi - lo + 1e-8)
    h = h.clamp(0.0, 1.0)
    return h.cpu().numpy()


def log_attention_overlay(
    rerun_path: str,
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> None:
    """Log raw image + attention heatmap + blended overlay under `rerun_path`.

    `image`   HWC uint8 in [0, 255]
    `heatmap` HW    float in [0, 1]
    """
    assert image.dtype == np.uint8
    assert heatmap.shape == image.shape[:2]

    # Colormap: a simple "hot" ramp (black → red → yellow → white). Avoid
    # matplotlib to keep the dep list small.
    r = np.clip(heatmap * 3.0, 0, 1)
    g = np.clip(heatmap * 3.0 - 1.0, 0, 1)
    b = np.clip(heatmap * 3.0 - 2.0, 0, 1)
    heat_rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    # 50/50 blend of image and heatmap so both are visible at once.
    blended = ((1 - alpha) * image + alpha * heat_rgb).astype(np.uint8)

    # Three rerun streams: raw, heatmap, blended overlay.
    rr.log(f"{rerun_path}/image", rr.Image(image))
    rr.log(f"{rerun_path}/attention", rr.Image(heat_rgb))
    rr.log(f"{rerun_path}/overlay", rr.Image(blended))
