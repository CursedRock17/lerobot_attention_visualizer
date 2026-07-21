"""Self-contained tests for visualizer/overlay.py — heatmap math only.

No lerobot, no real model, no rerun viewer required (rerun-sdk imports cleanly
without a viewer connected).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lerobot_attention_visualizer import patch_heatmap_to_image, rollout_to_patch_heatmap
from lerobot_attention_visualizer.visualizer.overlay import _apply_colormap


class TestRolloutToPatchHeatmap:
    def test_strips_batch_dim(self):
        # (1, 9, 9) → (3, 3): batch dim dropped, sqrt(9)=3 inferred grid.
        rollout = torch.rand(1, 9, 9)
        heat = rollout_to_patch_heatmap(rollout)
        assert heat.shape == (3, 3)

    def test_no_batch_dim(self):
        rollout = torch.rand(16, 16)  # 4x4 grid
        heat = rollout_to_patch_heatmap(rollout)
        assert heat.shape == (4, 4)

    def test_column_mean_reduction(self):
        # Build a rollout where column j has constant value j → column-mean = j.
        seq = 4  # 2x2 grid
        rollout = torch.arange(seq, dtype=torch.float32).repeat(seq, 1)
        # rollout[i, j] = j for all i, so column-mean over rows = j.
        heat = rollout_to_patch_heatmap(rollout)
        assert heat.shape == (2, 2)
        # Reshape (seq,) [0, 1, 2, 3] → (2, 2) [[0, 1], [2, 3]]
        torch.testing.assert_close(heat, torch.tensor([[0.0, 1.0], [2.0, 3.0]]))

    def test_non_square_raises(self):
        # seq=7 isn't a perfect square; the function can't infer a patch grid.
        rollout = torch.rand(7, 7)
        with pytest.raises(ValueError, match="not a perfect square"):
            rollout_to_patch_heatmap(rollout)


class TestPatchHeatmapToImage:
    def test_upsample_shape(self):
        heat = torch.rand(3, 3)
        out = patch_heatmap_to_image(heat, target_hw=(48, 64))
        assert isinstance(out, np.ndarray)
        assert out.shape == (48, 64)

    def test_normalized_to_unit_range(self):
        # min should land at 0, max at 1 after normalization.
        heat = torch.tensor([[0.5, 1.0], [2.0, 3.0]])
        out = patch_heatmap_to_image(heat, target_hw=(8, 8))
        assert out.min() == pytest.approx(0.0, abs=1e-6)
        assert out.max() == pytest.approx(1.0, abs=1e-6)

    def test_all_zeros_stays_zero(self):
        # Denominator guard: max() == 0 means we skip the divide.
        heat = torch.zeros(2, 2)
        out = patch_heatmap_to_image(heat, target_hw=(8, 8))
        assert (out == 0).all()
        assert out.shape == (8, 8)

    def test_dtype_is_float(self):
        # Even when given an int tensor, output should be float (interpolate path).
        heat = torch.zeros(2, 2, dtype=torch.int64)
        out = patch_heatmap_to_image(heat, target_hw=(4, 4))
        assert np.issubdtype(out.dtype, np.floating)


class TestFaithfulDefaults:
    """The default render must be the accurate 0-1 grid — no hidden shaping."""

    def test_default_is_plain_min_max(self):
        # A lone spike must survive at 1.0 by default (no clipping/winsorizing).
        heat = torch.full((4, 4), 0.1)
        heat[0, 0] = 100.0
        out = patch_heatmap_to_image(heat, target_hw=(8, 8))
        assert out.max() == pytest.approx(1.0, abs=1e-6)
        assert out.min() == pytest.approx(0.0, abs=1e-6)

    def test_default_matches_explicit_faithful_args(self):
        heat = torch.linspace(0.0, 1.0, 16).reshape(4, 4)
        a = patch_heatmap_to_image(heat, target_hw=(4, 4))
        b = patch_heatmap_to_image(
            heat, target_hw=(4, 4), clip_percentile=100.0, suppress_outliers=False, gamma=1.0
        )
        np.testing.assert_allclose(a, b, atol=1e-6)


class TestOptInDisplayAids:
    def test_suppress_outliers_winsorizes_spike(self):
        heat = torch.full((4, 4), 0.1)
        heat[0, 0] = 100.0
        out = patch_heatmap_to_image(heat, target_hw=(8, 8), suppress_outliers=True)
        assert out.max() < 0.5  # spike no longer dominates

    def test_suppress_outliers_preserves_clean_ramp(self):
        heat = torch.linspace(0.0, 1.0, 16).reshape(4, 4)
        plain = patch_heatmap_to_image(heat, target_hw=(4, 4))
        supp = patch_heatmap_to_image(heat, target_hw=(4, 4), suppress_outliers=True)
        np.testing.assert_allclose(plain, supp, atol=1e-6)  # only touches outliers

    def test_gamma_gt1_suppresses_midtones(self):
        heat = torch.tensor([[0.0, 0.5], [0.5, 1.0]])
        out = patch_heatmap_to_image(heat, target_hw=(2, 2), gamma=3.0)
        assert out.max() == pytest.approx(1.0, abs=1e-6)
        assert out.min() == pytest.approx(0.0, abs=1e-6)
        assert out[0, 1] == pytest.approx(0.125, abs=1e-4)  # 0.5**3


class TestColormap:
    def test_hot_reproduces_legacy_ramp(self):
        # "hot" must be byte-identical to the old hardcoded clip(3h-k) ramp.
        h = np.linspace(0.0, 1.0, 50).reshape(5, 10)
        out = _apply_colormap(h, "hot")
        r = np.clip(h * 3.0, 0, 1)
        g = np.clip(h * 3.0 - 1.0, 0, 1)
        b = np.clip(h * 3.0 - 2.0, 0, 1)
        np.testing.assert_array_equal(out, (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8))

    def test_shape_and_dtype(self):
        out = _apply_colormap(np.linspace(0, 1, 12).reshape(3, 4), "viridis")
        assert out.shape == (3, 4, 3) and out.dtype == np.uint8

    def test_blue_green_is_green_dominant_at_peak(self):
        out = _apply_colormap(np.ones((1, 1), dtype=np.float32), "blue-green")[0, 0]
        assert out[1] > out[0]

    def test_unknown_colormap_raises(self):
        with pytest.raises(ValueError, match="Unknown colormap"):
            _apply_colormap(np.zeros((2, 2)), "rainbow")
