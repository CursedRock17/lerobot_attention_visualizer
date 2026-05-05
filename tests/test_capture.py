"""Self-contained tests for VisionAttentionCapture.

Builds a tiny dummy encoder with the HF q_proj / k_proj attention layout and
runs the full capture / rollout path. No lerobot, no SmolVLA, no real ViT —
just nn.Modules wired the way SigLIP-style encoders are wired.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lerobot_attention_visualizer import VisionAttentionCapture


class _DummyAttn(nn.Module):
    """Mimics a HF attention layer: q_proj / k_proj submodules + head metadata."""

    def __init__(self, dim: int = 8, num_heads: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Real attention math is irrelevant — the hooks only need q_proj / k_proj
        # to fire so their outputs land in the cache.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.out_proj(q + k + v)


class _NoProjBlock(nn.Module):
    """A non-attention block — should NOT be matched by _find_attention_modules."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _DummyEncoder(nn.Module):
    """Stack of dummy attention layers, with a non-attention block sprinkled in."""

    def __init__(self, num_layers: int = 3, dim: int = 8, num_heads: int = 2):
        super().__init__()
        self.attn_layers = nn.ModuleList(
            [_DummyAttn(dim, num_heads) for _ in range(num_layers)]
        )
        self.distractor = _NoProjBlock(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.attn_layers:
            x = layer(x) + x
        x = self.distractor(x)
        return x


class TestFindAttentionModules:
    def test_finds_only_q_k_modules(self):
        encoder = _DummyEncoder(num_layers=3)
        # _find_attention_modules is static — call directly.
        hits = VisionAttentionCapture._find_attention_modules(encoder)
        # Exactly the 3 _DummyAttn layers; the _NoProjBlock distractor must be skipped.
        assert len(hits) == 3
        assert all(hasattr(m, "q_proj") and hasattr(m, "k_proj") for m in hits)

    def test_empty_when_no_attention_layers(self):
        # An encoder made only of non-attention blocks should yield no hits.
        encoder = nn.Sequential(_NoProjBlock(), _NoProjBlock())
        hits = VisionAttentionCapture._find_attention_modules(encoder)
        assert hits == []


class TestCaptureLifecycle:
    def test_hooks_install_and_remove(self):
        encoder = _DummyEncoder(num_layers=2)
        capture = VisionAttentionCapture(encoder)

        # Before entering: no handles.
        assert capture._handles == []

        with capture:
            # Inside: 2 layers × 2 hooks (q_proj + k_proj) = 4 handles.
            assert len(capture._handles) == 4

        # After exit: all handles removed.
        assert capture._handles == []

    def test_q_k_cached_after_forward(self):
        torch.manual_seed(0)
        encoder = _DummyEncoder(num_layers=2)
        capture = VisionAttentionCapture(encoder)

        with capture:
            x = torch.randn(1, 9, 8)  # (B=1, S=9, dim=8) — 9 = 3x3 patches
            _ = encoder(x)
            # Q/K should be populated for every layer.
            for layer_cache in capture._layers:
                assert layer_cache.q is not None
                assert layer_cache.k is not None
                assert layer_cache.q.shape == (1, 9, 8)
                assert layer_cache.k.shape == (1, 9, 8)


class TestRollout:
    def test_rollout_shape_matches_seq(self):
        torch.manual_seed(0)
        encoder = _DummyEncoder(num_layers=3)
        capture = VisionAttentionCapture(encoder)

        with capture:
            x = torch.randn(1, 9, 8)
            _ = encoder(x)
            rollout = capture.attention_rollout()

        # Streamed rollout returns (B, S, S) regardless of layer count.
        assert rollout.shape == (1, 9, 9)

    def test_rollout_without_residual(self):
        torch.manual_seed(0)
        encoder = _DummyEncoder(num_layers=2)
        capture = VisionAttentionCapture(encoder)

        with capture:
            x = torch.randn(1, 4, 8)
            _ = encoder(x)
            rollout = capture.attention_rollout(add_residual=False)

        assert rollout.shape == (1, 4, 4)

    def test_rollout_raises_without_forward(self):
        encoder = _DummyEncoder(num_layers=2)
        capture = VisionAttentionCapture(encoder)
        with capture:
            # No forward → no Q/K cached → rollout has nothing to chain.
            with pytest.raises(RuntimeError, match="No attentions captured"):
                capture.attention_rollout()


class TestComputeAttentions:
    def test_per_layer_shape(self):
        torch.manual_seed(0)
        encoder = _DummyEncoder(num_layers=2)
        capture = VisionAttentionCapture(encoder)

        with capture:
            x = torch.randn(1, 4, 8)
            _ = encoder(x)
            attns = capture.compute_attentions()

        # One (B, H, S, S) tensor per attention layer; head count from the dummy.
        assert len(attns) == 2
        for a in attns:
            assert a.shape == (1, 2, 4, 4)

    def test_softmax_sums_to_one(self):
        torch.manual_seed(0)
        encoder = _DummyEncoder(num_layers=2)
        capture = VisionAttentionCapture(encoder)

        with capture:
            x = torch.randn(1, 4, 8)
            _ = encoder(x)
            attns = capture.compute_attentions()

        # softmax along the key dim → each row should sum to 1.
        for a in attns:
            row_sums = a.sum(dim=-1)
            torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)


class TestSnapshot:
    def test_snapshot_appends_and_clears(self):
        torch.manual_seed(0)
        encoder = _DummyEncoder(num_layers=2)
        capture = VisionAttentionCapture(encoder)

        with capture:
            # Two forwards, snapshot after each — simulates two cameras.
            for _ in range(2):
                _ = encoder(torch.randn(1, 4, 8))
                capture.snapshot()
                # snapshot() resets Q/K so subsequent hooks see a fresh slate.
                for layer_cache in capture._layers:
                    assert layer_cache.q is None
                    assert layer_cache.k is None
                # ... but the previous Q/K need to be repopulated by the next forward
                # for the next snapshot. So we run a fresh forward each iteration.

            # Frozen snapshots live in _pending; rollout compute is deferred.
            assert len(capture._pending) == 2

            # drain_rollouts() computes and clears.
            rollouts = capture.drain_rollouts()
            assert len(rollouts) == 2
            assert len(capture._pending) == 0
            for r in rollouts:
                assert r.shape == (1, 4, 4)
