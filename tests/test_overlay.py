"""Self-contained tests for visualizer/overlay.py — heatmap math only.

No lerobot, no real model, no rerun viewer required (rerun-sdk imports cleanly
without a viewer connected).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lerobot_attention_visualizer import patch_heatmap_to_image, rollout_to_patch_heatmap


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
