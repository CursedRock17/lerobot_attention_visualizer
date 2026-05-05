"""Smoke tests for Pi0Attention — uses a mock policy with the expected shape.

We mock the deep `policy.model.paligemma_with_expert.paligemma.model.vision_tower`
chain since the real PaliGemma is heavy to instantiate and lives behind the
[pi] extra. The structural test (the wrapper installs hooks, monkey-patches
embed_image, restores both on exit) is what we actually want to validate.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lerobot_attention_visualizer import Pi05Attention, Pi0Attention, Pi0FastAttention


class _MiniAttn(nn.Module):
    """Same shape as a SigLIP attention layer — q_proj / k_proj submodules."""

    def __init__(self, dim: int = 8, num_heads: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x) + self.k_proj(x)


class _MiniVisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_MiniAttn() for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _make_mock_pi0_policy(num_cameras: int = 2):
    """Build the shape Pi0Attention reaches into without needing real lerobot."""
    vision_tower = _MiniVisionTower()
    embed_calls = []

    def embed_image(image):
        # Drive the vision tower so the q_proj / k_proj hooks fire.
        x = torch.randn(1, 4, 8)
        _ = vision_tower(x)
        embed_calls.append(image)
        return torch.zeros(1, 4, 8)

    paligemma_with_expert = SimpleNamespace(
        paligemma=SimpleNamespace(model=SimpleNamespace(vision_tower=vision_tower)),
        embed_image=embed_image,
    )
    config = SimpleNamespace(
        image_features={
            f"observation.images.cam{i}": object() for i in range(num_cameras)
        }
    )
    policy = SimpleNamespace(
        model=SimpleNamespace(paligemma_with_expert=paligemma_with_expert),
        config=config,
    )
    return policy, embed_calls


class TestPi0Attention:
    def test_aliases_point_to_same_class(self):
        # PI05Policy / PI0FastPolicy share the structure exactly, so we expose
        # the same class under three names for discoverability.
        assert Pi05Attention is Pi0Attention
        assert Pi0FastAttention is Pi0Attention

    def test_enter_installs_hooks_and_patches_embed_image(self):
        policy, _ = _make_mock_pi0_policy()
        original_embed = policy.model.paligemma_with_expert.embed_image

        viz = Pi0Attention(policy)
        with viz:
            # Hooks: 2 attn layers × (q_proj + k_proj) = 4 handles.
            assert len(viz._capture._handles) == 4
            # embed_image is replaced by the patched closure.
            assert policy.model.paligemma_with_expert.embed_image is not original_embed

        # On exit: hooks removed, embed_image restored.
        assert viz._capture._handles == []
        assert policy.model.paligemma_with_expert.embed_image is original_embed

    def test_embed_image_calls_snapshot_per_camera(self):
        policy, embed_calls = _make_mock_pi0_policy(num_cameras=3)
        viz = Pi0Attention(policy)

        with viz:
            for i in range(3):
                policy.model.paligemma_with_expert.embed_image(f"image_{i}")

            # snapshot() freezes Q/K into _pending (no GPU compute); rollout
            # compute is deferred to drain_rollouts() called by log_overlay.
            assert len(viz._capture._pending) == 3
            assert len(embed_calls) == 3
            rollouts = viz._capture.drain_rollouts()
            assert len(rollouts) == 3
            assert len(viz._capture._pending) == 0

    def test_camera_keys_strips_prefix(self):
        policy, _ = _make_mock_pi0_policy(num_cameras=2)
        viz = Pi0Attention(policy)
        assert viz.camera_keys() == ["cam0", "cam1"]

    def test_log_overlay_no_op_without_forward(self):
        policy, _ = _make_mock_pi0_policy()
        viz = Pi0Attention(policy)
        # No forwards happened → no rollouts → log_overlay is a silent no-op.
        with viz:
            viz.log_overlay({"cam0": MagicMock()})  # would crash if it tried to render

    def test_log_overlay_drops_frame_on_count_drift(self):
        # If the camera count and rollout count disagree (e.g. one hook missed),
        # log_overlay should drop the frame rather than misalign.
        policy, _ = _make_mock_pi0_policy(num_cameras=2)
        viz = Pi0Attention(policy)
        with viz:
            # Only one camera embedded — but the policy declares two.
            policy.model.paligemma_with_expert.embed_image("only_one")
            assert len(viz._capture._pending) == 1
            # Should NOT raise; should NOT log.
            viz.log_overlay({"cam0": MagicMock(), "cam1": MagicMock()})
