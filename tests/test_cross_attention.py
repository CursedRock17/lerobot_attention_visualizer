"""Unit tests for the action-head cross-attention primitive.

Covers, in isolation from any policy:

1. find_cross_attention_blocks — identifies cross-attending diffusers Attention
   modules by `to_q.in_features != to_k.in_features` (cross_attention_dim ≠
   query_dim), skipping self-attention blocks.

2. CrossAttentionCapture — hooks to_q/to_k, recomputes softmax(QK^T/√d), reduces
   over heads + query rows, and accumulates a mean over every fire (the
   denoiser's steps × cross-attn layers). Includes the key-length drift guard.

3. vision_importance_to_grids — masks the per-VL-token importance to the vision
   columns, splits per camera, reshapes to a square patch grid, and drops
   (returns []) on non-divisible / non-square counts.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lerobot_attention_visualizer.policies.cross_attention import (
    CrossAttentionCapture,
    find_cross_attention_blocks,
    vision_importance_to_grids,
)


# ---------------------------------------------------------------------------
# Mock diffusers-style Attention / DiT block
# ---------------------------------------------------------------------------

class _DiffusersAttn(nn.Module):
    """Mimics diffusers Attention: to_q/to_k/to_v projections + `heads`.

    A cross-attending module consumes a different feature dim for keys
    (`kv_dim`) than for queries (`query_dim`); a self-attending module uses the
    same dim for both.
    """

    def __init__(self, query_dim: int, kv_dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(kv_dim, query_dim, bias=False)
        self.to_v = nn.Linear(kv_dim, query_dim, bias=False)

    def forward(self, hidden_states, encoder_hidden_states=None):
        kv = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        # Fire to_q then to_k then to_v, matching diffusers' call order.
        self.to_q(hidden_states)
        self.to_k(kv)
        self.to_v(kv)
        return hidden_states


class _DiTBlock(nn.Module):
    def __init__(self, query_dim: int, kv_dim: int, cross: bool, heads: int = 4):
        super().__init__()
        self.attn1 = _DiffusersAttn(query_dim, kv_dim if cross else query_dim, heads)
        self._cross = cross

    def forward(self, hidden, vl):
        self.attn1(hidden, encoder_hidden_states=vl if self._cross else None)


def _make_dit(query_dim=64, kv_dim=96, heads=4, num_layers=4):
    """Interleaved DiT: even blocks cross-attend, odd blocks self-attend."""
    return nn.ModuleList(
        [_DiTBlock(query_dim, kv_dim, cross=(i % 2 == 0), heads=heads) for i in range(num_layers)]
    )


# ---------------------------------------------------------------------------
# find_cross_attention_blocks
# ---------------------------------------------------------------------------

class TestFindCrossAttentionBlocks:
    def test_picks_only_cross_blocks(self):
        blocks = _make_dit(num_layers=4)  # 2 cross, 2 self
        cross = find_cross_attention_blocks(blocks)
        assert len(cross) == 2
        for attn in cross:
            assert attn.to_q.in_features != attn.to_k.in_features

    def test_empty_when_all_self_attention(self):
        # query_dim == kv_dim everywhere → no cross blocks.
        blocks = nn.ModuleList(
            [_DiTBlock(64, 64, cross=False) for _ in range(3)]
        )
        assert find_cross_attention_blocks(blocks) == []


# ---------------------------------------------------------------------------
# CrossAttentionCapture
# ---------------------------------------------------------------------------

class TestCrossAttentionCapture:
    def test_drain_none_without_forward(self):
        blocks = _make_dit()
        cap = CrossAttentionCapture(find_cross_attention_blocks(blocks))
        with cap:
            pass
        assert cap.drain() is None

    def test_accumulates_importance_vector_shape(self):
        query_dim, kv_dim, heads = 64, 96, 4
        q_len, k_len = 10, 12
        blocks = _make_dit(query_dim, kv_dim, heads, num_layers=4)
        cross = find_cross_attention_blocks(blocks)
        cap = CrossAttentionCapture(cross)
        hidden = torch.randn(1, q_len, query_dim)
        vl = torch.randn(1, k_len, kv_dim)
        with cap:
            # Simulate the denoiser: 3 steps, every block fires once per step.
            for _ in range(3):
                for b in blocks:
                    b(hidden, vl)
            importance = cap.drain()
        assert importance is not None
        assert importance.shape == (k_len,)
        # Each row of a softmax over k_len sums to 1; mean over heads+rows keeps that.
        assert torch.allclose(importance.sum(), torch.tensor(1.0), atol=1e-4)

    def test_mean_over_fires_is_stable(self):
        # Same (q, k) fired N times → mean equals a single fire (deterministic).
        query_dim, kv_dim = 32, 48
        blocks = _make_dit(query_dim, kv_dim, heads=4, num_layers=2)  # 1 cross block
        cross = find_cross_attention_blocks(blocks)
        hidden = torch.randn(1, 5, query_dim)
        vl = torch.randn(1, 7, kv_dim)

        cap1 = CrossAttentionCapture(cross)
        with cap1:
            cross[0](hidden, encoder_hidden_states=vl)
            one = cap1.drain()

        cap_n = CrossAttentionCapture(cross)
        with cap_n:
            for _ in range(5):
                cross[0](hidden, encoder_hidden_states=vl)
            many = cap_n.drain()

        torch.testing.assert_close(one, many, atol=1e-5, rtol=1e-5)

    def test_drain_resets(self):
        blocks = _make_dit(num_layers=2)
        cross = find_cross_attention_blocks(blocks)
        cap = CrossAttentionCapture(cross)
        with cap:
            cross[0](torch.randn(1, 4, 64), encoder_hidden_states=torch.randn(1, 6, 96))
            assert cap.drain() is not None
            # Second drain with no new forward → None.
            assert cap.drain() is None

    def test_key_length_drift_skipped(self):
        # If a fire arrives with a different key length, it must not corrupt the
        # running mean — it is skipped.
        query_dim, kv_dim, heads = 64, 96, 4
        attn = _DiffusersAttn(query_dim, kv_dim, heads)
        cap = CrossAttentionCapture([attn])
        with cap:
            attn(torch.randn(1, 5, query_dim), encoder_hidden_states=torch.randn(1, 12, kv_dim))
            # Different key length — skipped by the shape guard.
            attn(torch.randn(1, 5, query_dim), encoder_hidden_states=torch.randn(1, 8, kv_dim))
            importance = cap.drain()
        assert importance.shape == (12,)  # the first (kept) fire's key length


# ---------------------------------------------------------------------------
# vision_importance_to_grids
# ---------------------------------------------------------------------------

class TestVisionImportanceToGrids:
    def test_masks_splits_and_reshapes(self):
        # 12 VL tokens: 8 vision (4 per camera, 2×2 grid), 4 text.
        importance = torch.arange(12, dtype=torch.float32)
        vision_mask = torch.tensor(
            [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0], dtype=torch.bool
        )  # 8 True
        grids = vision_importance_to_grids(importance, vision_mask, n_cameras=2)
        assert len(grids) == 2
        for g in grids:
            assert g.shape == (2, 2)

    def test_preserves_camera_order(self):
        # 2 vision tokens, 2 cameras → one 1×1 grid each, in camera order.
        importance = torch.tensor([10.0, 20.0])
        vision_mask = torch.tensor([1, 1], dtype=torch.bool)
        grids = vision_importance_to_grids(importance, vision_mask, n_cameras=2)
        assert grids[0].flatten().tolist() == [10.0]
        assert grids[1].flatten().tolist() == [20.0]

    def test_non_divisible_returns_empty(self):
        # 6 vision tokens can't split evenly into 4 cameras.
        importance = torch.arange(6, dtype=torch.float32)
        vision_mask = torch.ones(6, dtype=torch.bool)
        assert vision_importance_to_grids(importance, vision_mask, n_cameras=4) == []

    def test_non_square_per_camera_returns_empty(self):
        # 6 vision tokens / 1 camera = 6, not a perfect square.
        importance = torch.arange(6, dtype=torch.float32)
        vision_mask = torch.ones(6, dtype=torch.bool)
        assert vision_importance_to_grids(importance, vision_mask, n_cameras=1) == []

    def test_no_vision_tokens_returns_empty(self):
        importance = torch.arange(5, dtype=torch.float32)
        vision_mask = torch.zeros(5, dtype=torch.bool)
        assert vision_importance_to_grids(importance, vision_mask, n_cameras=1) == []
