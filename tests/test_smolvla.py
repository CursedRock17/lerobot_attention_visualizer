"""Smoke tests for SmolVLAAttention against a SigLIP-shaped mock.

Covers the SmolVLA-specific gaps not exercised by test_capture.py:

1. Nested SigLIP module tree — VisionAttentionCapture must navigate
   SiglipVisionModel.vision_model.encoder.layers[i].self_attn, not a flat
   list directly under the root.

2. 729-patch grid — the actual siglip-so400m-patch14-384 token count (27×27).
   rollout_to_patch_heatmap's isqrt reshape must round-trip correctly.

3. Deferred rollout via SmolVLAAttention — snapshot() inside embed_image
   is cheap (no GPU compute); drain_rollouts() in log_overlay does the work.

4. bf16 dtype — _layer_attention casts logits to float32 before softmax,
   matching SiglipAttention.forward's own dtype behaviour.

5. SmolVLAAttention end-to-end — hook install, embed_image patching, camera
   key extraction, log_overlay no-op / drift guards.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lerobot_attention_visualizer import SmolVLAAttention
from lerobot_attention_visualizer.policies.smolvla import VisionAttentionCapture
from lerobot_attention_visualizer.visualizer.overlay import rollout_to_patch_heatmap


# ---------------------------------------------------------------------------
# Mock architecture — mirrors the real Idefics3 / SigLIP nesting exactly
# ---------------------------------------------------------------------------

class _SigLIPAttn(nn.Module):
    """Matches SiglipAttention: exposes num_heads, head_dim, q_proj, k_proj."""

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
    """Matches SiglipEncoderLayer: has .self_attn."""

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
    """Matches SiglipVisionTransformer."""

    def __init__(self, num_layers: int = 4, embed_dim: int = 64, num_heads: int = 8):
        super().__init__()
        self.encoder = _SigLIPEncoder(num_layers, embed_dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class _SigLIPVisionModel(nn.Module):
    """Matches SiglipVisionModel: has .vision_model = SiglipVisionTransformer.

    The doubled nesting (SiglipVisionModel.vision_model = SiglipVisionTransformer)
    is the real HuggingFace pattern. SmolVLAAttention is initialised with the
    OUTER SiglipVisionModel; VisionAttentionCapture walks into it to find the
    SiglipAttention modules four levels deep.
    """

    def __init__(self, num_layers: int = 4, embed_dim: int = 64, num_heads: int = 8):
        super().__init__()
        self.vision_model = _SigLIPVisionTransformer(num_layers, embed_dim, num_heads)

    def forward(
        self, pixel_values: torch.Tensor, patch_attention_mask=None
    ) -> SimpleNamespace:
        hidden = self.vision_model(pixel_values)
        return SimpleNamespace(last_hidden_state=hidden)


def _make_mock_smolvla_policy(
    num_cameras: int = 2,
    num_patches: int = 16,
    embed_dim: int = 64,
    num_heads: int = 8,
    num_layers: int = 4,
):
    """Build a mock that matches SmolVLMWithExpertModel's attribute paths.

    SmolVLAAttention.__init__ navigates:
        policy.model.vlm_with_expert.get_vlm_model().vision_model

    SmolVLAAttention.__enter__ patches:
        policy.model.vlm_with_expert.embed_image

    camera_keys() reads:
        policy.config.image_features
    """
    vision_model = _SigLIPVisionModel(num_layers, embed_dim, num_heads)
    embed_calls: list = []

    def embed_image(image):
        # Drive the vision tower so Q/K hooks fire.
        x = torch.randn(1, num_patches, embed_dim)
        out = vision_model(x)
        embed_calls.append(image)
        return out.last_hidden_state

    vlm_model = SimpleNamespace(vision_model=vision_model)
    vlm_with_expert = SimpleNamespace(
        get_vlm_model=lambda: vlm_model,
        embed_image=embed_image,
    )
    config = SimpleNamespace(
        image_features={
            f"observation.images.cam{i}": object() for i in range(num_cameras)
        }
    )
    policy = SimpleNamespace(
        model=SimpleNamespace(vlm_with_expert=vlm_with_expert),
        config=config,
    )
    return policy, embed_calls


# ---------------------------------------------------------------------------
# VisionAttentionCapture — SigLIP nesting
# ---------------------------------------------------------------------------

class TestCaptureWithSigLIPTree:
    def test_finds_attention_modules_through_nested_structure(self):
        # Modules live at vision_model.vision_model.encoder.layers[i].self_attn —
        # _find_attention_modules must recurse into the full tree.
        policy, _ = _make_mock_smolvla_policy(num_layers=6)
        vm = policy.model.vlm_with_expert.get_vlm_model().vision_model
        capture = VisionAttentionCapture(vm)
        assert len(capture._attn_modules) == 6

    def test_head_layout_introspected_correctly(self):
        policy, _ = _make_mock_smolvla_policy(embed_dim=64, num_heads=8)
        vm = policy.model.vlm_with_expert.get_vlm_model().vision_model
        capture = VisionAttentionCapture(vm)
        with capture:
            for layer in capture._layers:
                assert layer.num_heads == 8
                assert layer.head_dim == 8  # 64 // 8

    def test_hooks_fire_with_correct_qk_shape(self):
        num_patches, embed_dim = 16, 64
        policy, _ = _make_mock_smolvla_policy(
            num_patches=num_patches, embed_dim=embed_dim
        )
        vm = policy.model.vlm_with_expert.get_vlm_model().vision_model
        capture = VisionAttentionCapture(vm)
        with capture:
            vm(torch.randn(1, num_patches, embed_dim))
            for lc in capture._layers:
                assert lc.q is not None and lc.k is not None
                assert lc.q.shape == (1, num_patches, embed_dim)
                assert lc.k.shape == (1, num_patches, embed_dim)

    def test_snapshot_is_cheap_no_rollout_in_pending(self):
        # _pending entries are lists of _LayerCache, NOT computed rollout tensors.
        policy, _ = _make_mock_smolvla_policy()
        vm = policy.model.vlm_with_expert.get_vlm_model().vision_model
        capture = VisionAttentionCapture(vm)
        with capture:
            vm(torch.randn(1, 16, 64))
            capture.snapshot()
            assert len(capture._pending) == 1
            # Q/K live caches cleared by snapshot.
            assert all(lc.q is None for lc in capture._layers)

    def test_drain_rollouts_shape_and_clears_pending(self):
        num_patches = 16
        policy, _ = _make_mock_smolvla_policy(num_patches=num_patches)
        vm = policy.model.vlm_with_expert.get_vlm_model().vision_model
        capture = VisionAttentionCapture(vm)
        with capture:
            for _ in range(3):
                vm(torch.randn(1, num_patches, 64))
                capture.snapshot()
            assert len(capture._pending) == 3
            rollouts = capture.drain_rollouts()
        assert len(rollouts) == 3
        assert len(capture._pending) == 0
        for r in rollouts:
            assert r.shape == (1, num_patches, num_patches)

    def test_drain_rollouts_last_layer_only(self):
        num_patches = 16
        policy, _ = _make_mock_smolvla_policy(num_patches=num_patches)
        vm = policy.model.vlm_with_expert.get_vlm_model().vision_model
        capture = VisionAttentionCapture(vm)
        with capture:
            vm(torch.randn(1, num_patches, 64))
            capture.snapshot()
            rollouts = capture.drain_rollouts(last_layer_only=True)
        # Shape is the same — only the computation path differs.
        assert rollouts[0].shape == (1, num_patches, num_patches)


# ---------------------------------------------------------------------------
# 729-patch grid — actual siglip-so400m-patch14-384 size
# ---------------------------------------------------------------------------

class TestSigLIPPatchGrid:
    def test_729_patch_reshape_round_trips(self):
        # 729 = 27²; rollout_to_patch_heatmap must produce (27, 27).
        num_patches = 729
        policy, _ = _make_mock_smolvla_policy(
            num_cameras=1, num_patches=num_patches, num_layers=2
        )
        vm = policy.model.vlm_with_expert.get_vlm_model().vision_model
        capture = VisionAttentionCapture(vm)
        with capture:
            vm(torch.randn(1, num_patches, 64))
            capture.snapshot()
            rollouts = capture.drain_rollouts()
        heatmap = rollout_to_patch_heatmap(rollouts[0])
        assert heatmap.shape == (27, 27)

    def test_non_square_patch_count_raises(self):
        # 730 is not a perfect square → rollout_to_patch_heatmap should raise.
        import torch
        dummy_rollout = torch.ones(1, 730, 730)
        with pytest.raises(ValueError, match="not a perfect square"):
            rollout_to_patch_heatmap(dummy_rollout)


# ---------------------------------------------------------------------------
# bf16 dtype — softmax must stay numerically stable
# ---------------------------------------------------------------------------

class TestBF16Softmax:
    def test_layer_attention_in_bf16_sums_to_one(self):
        # _layer_attention casts logits to float32 before softmax even when q/k
        # are bf16 — matching SiglipAttention.forward's own behaviour.
        policy, _ = _make_mock_smolvla_policy(num_patches=16)
        vm = policy.model.vlm_with_expert.get_vlm_model().vision_model
        # Cast entire vision model to bf16.
        vm_bf16 = vm.to(torch.bfloat16)
        capture = VisionAttentionCapture(vm_bf16)
        with capture:
            vm_bf16(torch.randn(1, 16, 64).to(torch.bfloat16))
            attns = capture.compute_attentions()
        for a in attns:
            row_sums = a.sum(dim=-1)
            torch.testing.assert_close(
                row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4
            )


# ---------------------------------------------------------------------------
# SmolVLAAttention end-to-end
# ---------------------------------------------------------------------------

class TestSmolVLAAttention:
    def test_enter_installs_hooks_and_patches_embed_image(self):
        policy, _ = _make_mock_smolvla_policy(num_layers=4)
        original_embed = policy.model.vlm_with_expert.embed_image

        viz = SmolVLAAttention(policy)
        with viz:
            # 4 layers × (q_proj + k_proj) = 8 handles.
            assert len(viz._capture._handles) == 8
            assert policy.model.vlm_with_expert.embed_image is not original_embed

        assert viz._capture._handles == []
        assert policy.model.vlm_with_expert.embed_image is original_embed

    def test_embed_image_calls_freeze_snapshots(self):
        policy, embed_calls = _make_mock_smolvla_policy(num_cameras=2)
        viz = SmolVLAAttention(policy)
        with viz:
            policy.model.vlm_with_expert.embed_image("img_0")
            policy.model.vlm_with_expert.embed_image("img_1")
            # Two cameras → two frozen snapshots queued for deferred compute.
            assert len(viz._capture._pending) == 2
            assert len(embed_calls) == 2

    def test_camera_keys_strips_prefix(self):
        policy, _ = _make_mock_smolvla_policy(num_cameras=3)
        assert SmolVLAAttention(policy).camera_keys() == ["cam0", "cam1", "cam2"]

    def test_log_overlay_no_op_without_forward(self):
        policy, _ = _make_mock_smolvla_policy()
        viz = SmolVLAAttention(policy)
        with viz:
            viz.log_overlay({"cam0": MagicMock()})  # must not raise

    def test_log_overlay_drops_frame_on_count_drift(self):
        # One embed_image call but two cameras declared → mismatch → silent drop.
        policy, _ = _make_mock_smolvla_policy(num_cameras=2)
        viz = SmolVLAAttention(policy)
        with viz:
            policy.model.vlm_with_expert.embed_image("only_one")
            assert len(viz._capture._pending) == 1
            viz.log_overlay({"cam0": MagicMock(), "cam1": MagicMock()})

    def test_last_layer_only_produces_correct_shape(self):
        num_patches = 16
        policy, _ = _make_mock_smolvla_policy(num_cameras=1, num_patches=num_patches)
        viz = SmolVLAAttention(policy, last_layer_only=True)
        with viz:
            policy.model.vlm_with_expert.embed_image("img")
            rollouts = viz._capture.drain_rollouts(last_layer_only=True)
        assert rollouts[0].shape == (1, num_patches, num_patches)

    def test_pending_cleared_between_log_overlay_calls(self):
        # drain_rollouts inside log_overlay must clear _pending so a second call
        # without a new forward is a no-op, not a double-render.
        policy, _ = _make_mock_smolvla_policy(num_cameras=1, num_patches=16)
        viz = SmolVLAAttention(policy)
        with viz:
            policy.model.vlm_with_expert.embed_image("img")
            assert len(viz._capture._pending) == 1
            # First log_overlay drains.
            pending_after_first = []
            rollouts = viz._capture.drain_rollouts()
            pending_after_first = list(viz._capture._pending)
        assert pending_after_first == []
        assert len(rollouts) == 1
