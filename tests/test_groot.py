"""Smoke tests for GR00TAttention against a mock Eagle-2 / SigLIP architecture.

Covers the Groot-specific gaps:

1. Batched multi-camera forward — all cameras packed into one pixel_values
   tensor; snapshot_split must slice dim 0 correctly.

2. VisionAttentionCapture.snapshot_split — splits (N_cams, seq, dim) Q/K
   into N_cams independent _LayerCache entries.

3. GR00TAttention attribute path — policy._groot_model.backbone.eagle_model
   .vision_model must be found and hooked; forward_eagle must be patched.

4. log_overlay drift guard — rollout count != camera count → silent drop.

5. __exit__ cleanup — forward_eagle restored, hooks removed.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lerobot_attention_visualizer import GR00TAttention
from lerobot_attention_visualizer.policies.smolvla import VisionAttentionCapture
from lerobot_attention_visualizer.visualizer.overlay import rollout_to_patch_heatmap


# ---------------------------------------------------------------------------
# Mock architecture — mirrors the Groot / Eagle-2 nesting
# ---------------------------------------------------------------------------

class _SigLIPAttn(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class _SigLIPLayer(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 8):
        super().__init__()
        self.self_attn = _SigLIPAttn(embed_dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn(x)


class _SigLIPEncoder(nn.Module):
    def __init__(self, num_layers: int = 4, embed_dim: int = 64, num_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList(
            [_SigLIPLayer(embed_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _SigLIPVisionTransformer(nn.Module):
    def __init__(self, num_layers: int = 4, embed_dim: int = 64, num_heads: int = 8):
        super().__init__()
        self.encoder = _SigLIPEncoder(num_layers, embed_dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class _SigLIPVisionModel(nn.Module):
    """Matches SiglipVisionModel — has .vision_model = SiglipVisionTransformer."""

    def __init__(self, num_layers: int = 4, embed_dim: int = 64, num_heads: int = 8):
        super().__init__()
        self.vision_model = _SigLIPVisionTransformer(num_layers, embed_dim, num_heads)

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        # pixel_values: (N_cameras, n_patches, embed_dim) — batched
        hidden = self.vision_model(pixel_values)
        return SimpleNamespace(last_hidden_state=hidden)


def _make_mock_groot_policy(
    num_cameras: int = 2,
    num_patches: int = 16,
    embed_dim: int = 64,
    num_heads: int = 8,
    num_layers: int = 4,
):
    """Build a mock matching Groot's attribute paths.

    GR00TAttention navigates:
        policy._groot_model.backbone.eagle_model.vision_model

    GR00TAttention patches:
        policy._groot_model.backbone.forward_eagle

    camera_keys reads:
        policy.config.image_features
    """
    vision_model = _SigLIPVisionModel(num_layers, embed_dim, num_heads)
    forward_eagle_calls: list = []

    def forward_eagle(vl_input):
        # Simulate batched multi-camera forward: (N_cameras, n_patches, embed_dim)
        x = torch.randn(num_cameras, num_patches, embed_dim)
        vision_model(x)  # fires Q/K hooks for all cameras at once
        forward_eagle_calls.append(vl_input)
        return SimpleNamespace(backbone_features=torch.randn(1, num_patches, embed_dim))

    eagle_model = SimpleNamespace(vision_model=vision_model)
    backbone = SimpleNamespace(
        eagle_model=eagle_model,
        forward_eagle=forward_eagle,
    )
    config = SimpleNamespace(
        image_features={
            f"observation.images.cam{i}": object() for i in range(num_cameras)
        }
    )
    _groot_model = SimpleNamespace(backbone=backbone)
    policy = SimpleNamespace(
        _groot_model=_groot_model,
        config=config,
    )
    return policy, forward_eagle_calls


# ---------------------------------------------------------------------------
# VisionAttentionCapture.snapshot_split
# ---------------------------------------------------------------------------

class TestSnapshotSplit:
    def test_split_produces_n_camera_entries(self):
        policy, _ = _make_mock_groot_policy(num_cameras=3, num_patches=16)
        vm = policy._groot_model.backbone.eagle_model.vision_model
        capture = VisionAttentionCapture(vm)
        with capture:
            # Batched forward: (3, 16, 64)
            vm(torch.randn(3, 16, 64))
            capture.snapshot_split(n_cameras=3)
            assert len(capture._pending) == 3

    def test_split_per_camera_q_shape(self):
        n_cams, n_patches, embed_dim = 2, 9, 64
        policy, _ = _make_mock_groot_policy(
            num_cameras=n_cams, num_patches=n_patches, embed_dim=embed_dim
        )
        vm = policy._groot_model.backbone.eagle_model.vision_model
        capture = VisionAttentionCapture(vm)
        with capture:
            vm(torch.randn(n_cams, n_patches, embed_dim))
            capture.snapshot_split(n_cameras=n_cams)
            for cam_caches in capture._pending:
                for lc in cam_caches:
                    assert lc.q is not None
                    assert lc.q.shape == (1, n_patches, embed_dim)
                    assert lc.k.shape == (1, n_patches, embed_dim)

    def test_split_clears_live_caches(self):
        policy, _ = _make_mock_groot_policy(num_cameras=2, num_patches=9)
        vm = policy._groot_model.backbone.eagle_model.vision_model
        capture = VisionAttentionCapture(vm)
        with capture:
            vm(torch.randn(2, 9, 64))
            capture.snapshot_split(n_cameras=2)
            assert all(lc.q is None for lc in capture._layers)

    def test_split_drain_produces_correct_rollout_shape(self):
        n_cams, n_patches = 2, 16
        policy, _ = _make_mock_groot_policy(num_cameras=n_cams, num_patches=n_patches)
        vm = policy._groot_model.backbone.eagle_model.vision_model
        capture = VisionAttentionCapture(vm)
        with capture:
            vm(torch.randn(n_cams, n_patches, 64))
            capture.snapshot_split(n_cameras=n_cams)
            rollouts = capture.drain_rollouts()
        assert len(rollouts) == n_cams
        for r in rollouts:
            assert r.shape == (1, n_patches, n_patches)

    def test_split_cameras_are_independent(self):
        # Verify cam0 and cam1 Q tensors are different slices, not the same.
        n_cams, n_patches = 2, 4
        policy, _ = _make_mock_groot_policy(num_cameras=n_cams, num_patches=n_patches)
        vm = policy._groot_model.backbone.eagle_model.vision_model
        capture = VisionAttentionCapture(vm)
        with capture:
            vm(torch.randn(n_cams, n_patches, 64))
            capture.snapshot_split(n_cameras=n_cams)
        cam0_q = capture._pending[0][0].q
        cam1_q = capture._pending[1][0].q
        # Different slices of the same (2, 4, 64) tensor — must differ.
        assert not torch.equal(cam0_q, cam1_q)


# ---------------------------------------------------------------------------
# GR00TAttention lifecycle
# ---------------------------------------------------------------------------

class TestGR00TAttentionLifecycle:
    def test_enter_installs_hooks_and_patches_forward_eagle(self):
        policy, _ = _make_mock_groot_policy(num_cameras=2, num_layers=4)
        original_fe = policy._groot_model.backbone.forward_eagle

        viz = GR00TAttention(policy)
        with viz:
            # 4 layers × 2 hooks (q + k) = 8 handles
            assert len(viz._capture._handles) == 8
            assert policy._groot_model.backbone.forward_eagle is not original_fe

        assert viz._capture._handles == []
        assert policy._groot_model.backbone.forward_eagle is original_fe

    def test_forward_eagle_call_snapshots_n_cameras(self):
        num_cameras = 3
        policy, fe_calls = _make_mock_groot_policy(num_cameras=num_cameras)
        viz = GR00TAttention(policy)
        with viz:
            policy._groot_model.backbone.forward_eagle(object())
            assert len(viz._capture._pending) == num_cameras
            assert len(fe_calls) == 1

    def test_camera_keys_strips_prefix(self):
        policy, _ = _make_mock_groot_policy(num_cameras=2)
        viz = GR00TAttention(policy)
        assert viz.camera_keys() == ["cam0", "cam1"]

    def test_log_overlay_no_op_without_forward(self):
        policy, _ = _make_mock_groot_policy()
        viz = GR00TAttention(policy)
        with viz:
            viz.log_overlay({"cam0": MagicMock(), "cam1": MagicMock()})

    def test_log_overlay_drops_on_count_drift(self):
        # 1 forward_eagle call → 2 rollouts, but log_overlay gets obs with 3 cams
        policy, _ = _make_mock_groot_policy(num_cameras=2)
        viz = GR00TAttention(policy)
        with viz:
            policy._groot_model.backbone.forward_eagle(object())
            assert len(viz._capture._pending) == 2
            # Manually corrupt camera count to trigger drift guard
            viz._capture._pending.append(viz._capture._pending[0])  # add phantom entry
            viz.log_overlay({"cam0": MagicMock(), "cam1": MagicMock()})
            # _pending now has 3 entries vs 2 camera keys → drop

    def test_pending_cleared_after_drain(self):
        policy, _ = _make_mock_groot_policy(num_cameras=2, num_patches=16)
        viz = GR00TAttention(policy)
        with viz:
            policy._groot_model.backbone.forward_eagle(object())
            assert len(viz._capture._pending) == 2
            rollouts = viz._capture.drain_rollouts()
            assert len(viz._capture._pending) == 0
        assert len(rollouts) == 2

    def test_last_layer_only_shape(self):
        n_patches = 16
        policy, _ = _make_mock_groot_policy(num_cameras=1, num_patches=n_patches)
        viz = GR00TAttention(policy, last_layer_only=True)
        with viz:
            policy._groot_model.backbone.forward_eagle(object())
            rollouts = viz._capture.drain_rollouts(last_layer_only=True)
        assert rollouts[0].shape == (1, n_patches, n_patches)


# ---------------------------------------------------------------------------
# 729-patch (27×27) grid — actual SigLIP-so400m token count
# ---------------------------------------------------------------------------

class TestGR00TPatchGrid:
    def test_729_patch_reshape_round_trips(self):
        n_patches = 729
        policy, _ = _make_mock_groot_policy(
            num_cameras=1, num_patches=n_patches, num_layers=2
        )
        vm = policy._groot_model.backbone.eagle_model.vision_model
        capture = VisionAttentionCapture(vm)
        with capture:
            vm(torch.randn(1, n_patches, 64))
            capture.snapshot_split(n_cameras=1)
            rollouts = capture.drain_rollouts()
        heatmap = rollout_to_patch_heatmap(rollouts[0])
        assert heatmap.shape == (27, 27)
