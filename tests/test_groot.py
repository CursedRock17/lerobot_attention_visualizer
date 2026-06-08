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

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn

from lerobot_attention_visualizer import GR00TAttention, GR00TN1d6Attention
from lerobot_attention_visualizer.policies.smolvla import VisionAttentionCapture
from lerobot_attention_visualizer.visualizer import overlay as overlay_mod
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


IMAGE_TOKEN_INDEX = 99  # sentinel id Eagle replaces with vision embeddings


class _DiffusersAttn(nn.Module):
    """Mimics diffusers Attention: to_q/to_k/to_v + `heads`. Cross iff kv_dim != query_dim."""

    def __init__(self, query_dim: int, kv_dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(kv_dim, query_dim, bias=False)
        self.to_v = nn.Linear(kv_dim, query_dim, bias=False)

    def forward(self, hidden_states, encoder_hidden_states=None):
        kv = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
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


class _MockDiT(nn.Module):
    """Interleaved DiT: even blocks cross-attend to vl, odd blocks self-attend."""

    def __init__(self, query_dim: int, kv_dim: int, heads: int = 4, num_layers: int = 4):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [_DiTBlock(query_dim, kv_dim, cross=(i % 2 == 0), heads=heads) for i in range(num_layers)]
        )

    def run_denoiser(self, hidden, vl, steps: int = 2):
        for _ in range(steps):
            for b in self.transformer_blocks:
                b(hidden, vl)


def _vl_input(num_cameras: int, vision_per_cam: int, n_text: int = 2):
    """Build a mock eagle vl_input dict with image-token placeholders.

    Layout per camera: `n_text` text tokens then `vision_per_cam` image tokens,
    so the vision columns appear in camera order (mirrors Eagle's prompt).
    """
    ids: list[int] = []
    for _ in range(num_cameras):
        ids.extend([1] * n_text)
        ids.extend([IMAGE_TOKEN_INDEX] * vision_per_cam)
    return {"eagle_input_ids": torch.tensor([ids])}  # (1, seq_len)


def _make_mock_groot_policy(
    num_cameras: int = 2,
    num_patches: int = 16,
    embed_dim: int = 64,
    num_heads: int = 8,
    num_layers: int = 4,
    *,
    dit_query_dim: int = 32,
    dit_kv_dim: int = 48,
    dit_heads: int = 4,
    dit_layers: int = 4,
):
    """Build a mock matching Groot's attribute paths.

    GR00TAttention navigates:
        policy._groot_model.backbone.eagle_model.vision_model
        policy._groot_model.action_head.model.transformer_blocks   (cross-attn DiT)

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

    eagle_model = SimpleNamespace(vision_model=vision_model, image_token_index=IMAGE_TOKEN_INDEX)
    dit = _MockDiT(dit_query_dim, dit_kv_dim, dit_heads, dit_layers)
    backbone = SimpleNamespace(
        eagle_model=eagle_model,
        forward_eagle=forward_eagle,
    )
    action_head = SimpleNamespace(model=dit)
    config = SimpleNamespace(
        image_features={
            f"observation.images.cam{i}": object() for i in range(num_cameras)
        }
    )
    _groot_model = SimpleNamespace(backbone=backbone, action_head=action_head)
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
            policy._groot_model.backbone.forward_eagle(_vl_input(num_cameras, 4))
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
            policy._groot_model.backbone.forward_eagle(_vl_input(2, 4))
            assert len(viz._capture._pending) == 2
            # Manually corrupt camera count to trigger drift guard
            viz._capture._pending.append(viz._capture._pending[0])  # add phantom entry
            viz.log_overlay({"cam0": MagicMock(), "cam1": MagicMock()})
            # _pending now has 3 entries vs 2 camera keys → drop

    def test_pending_cleared_after_drain(self):
        policy, _ = _make_mock_groot_policy(num_cameras=2, num_patches=16)
        viz = GR00TAttention(policy)
        with viz:
            policy._groot_model.backbone.forward_eagle(_vl_input(2, 4))
            assert len(viz._capture._pending) == 2
            rollouts = viz._capture.drain_rollouts()
            assert len(viz._capture._pending) == 0
        assert len(rollouts) == 2

    def test_last_layer_only_shape(self):
        n_patches = 16
        policy, _ = _make_mock_groot_policy(num_cameras=1, num_patches=n_patches)
        viz = GR00TAttention(policy, last_layer_only=True)
        with viz:
            policy._groot_model.backbone.forward_eagle(_vl_input(1, 4))
            rollouts = viz._capture.drain_rollouts(last_layer_only=True)
        assert rollouts[0].shape == (1, n_patches, n_patches)


# ---------------------------------------------------------------------------
# GR00TAttention action-head cross-attention
# ---------------------------------------------------------------------------

class TestGR00TCrossAttention:
    def test_enter_hooks_only_cross_blocks(self):
        # dit_layers=4 → 2 cross blocks; each cross block adds 2 hooks (q + k).
        policy, _ = _make_mock_groot_policy(num_cameras=2, dit_layers=4)
        viz = GR00TAttention(policy)
        with viz:
            assert viz._cross is not None
            assert len(viz._cross._modules) == 2
            assert len(viz._cross._handles) == 4

    def test_exit_removes_cross_hooks(self):
        policy, _ = _make_mock_groot_policy(num_cameras=2)
        viz = GR00TAttention(policy)
        with viz:
            pass
        assert viz._cross is None

    def test_cross_attention_disabled(self):
        policy, _ = _make_mock_groot_policy(num_cameras=2)
        viz = GR00TAttention(policy, cross_attention=False)
        with viz:
            assert viz._cross is None
            policy._groot_model.backbone.forward_eagle(_vl_input(2, 4))
            # No cross capture → vision_mask never populated.
            assert viz._vision_mask is None

    def test_forward_eagle_populates_vision_mask(self):
        vision_per_cam = 4
        policy, _ = _make_mock_groot_policy(num_cameras=2)
        viz = GR00TAttention(policy)
        with viz:
            policy._groot_model.backbone.forward_eagle(_vl_input(2, vision_per_cam))
            assert viz._vision_mask is not None
            # 2 cameras × 4 vision tokens = 8 True entries.
            assert int(viz._vision_mask.sum()) == 2 * vision_per_cam

    def test_log_overlay_streams_action_overlay(self, monkeypatch):
        # End-to-end: forward_eagle + a simulated denoiser run, then log_overlay
        # should emit action/* streams for each camera.
        logged: list[str] = []
        monkeypatch.setattr(overlay_mod.rr, "log", lambda path, *a, **k: logged.append(path))

        num_cameras, vision_per_cam = 2, 4  # 2×2 grid per camera
        policy, _ = _make_mock_groot_policy(
            num_cameras=num_cameras, dit_query_dim=32, dit_kv_dim=48, dit_layers=4
        )
        viz = GR00TAttention(policy)
        dit = policy._groot_model.action_head.model

        with viz:
            vl_in = _vl_input(num_cameras, vision_per_cam)
            policy._groot_model.backbone.forward_eagle(vl_in)
            # Simulate the flow-matching denoiser: vl key length == eagle seq len.
            vl_len = vl_in["eagle_input_ids"].shape[1]
            hidden = torch.randn(1, 6, 32)
            vl = torch.randn(1, vl_len, 48)
            dit.run_denoiser(hidden, vl, steps=3)

            obs = {f"cam{i}": np.zeros((48, 48, 3), dtype=np.uint8) for i in range(num_cameras)}
            viz.log_overlay(obs)

        action_paths = [p for p in logged if "/action/" in p]
        # 3 streams (image/attention/overlay) × 2 cameras.
        assert len(action_paths) == 6
        assert any("cam0/action/overlay" in p for p in action_paths)
        assert any("cam1/action/overlay" in p for p in action_paths)


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


# ---------------------------------------------------------------------------
# GR00TN1d6Attention — N1.6 (Eagle3-VL: SigLIP2 + Qwen3) attribute paths
# ---------------------------------------------------------------------------

def _vl_input_n16(num_cameras: int, vision_per_cam: int, n_text: int = 2):
    """N1.6 backbone input: uses `input_ids` (not `eagle_input_ids`)."""
    ids: list[int] = []
    for _ in range(num_cameras):
        ids.extend([1] * n_text)
        ids.extend([IMAGE_TOKEN_INDEX] * vision_per_cam)
    return {"input_ids": torch.tensor([ids])}


def _make_mock_groot_n16_policy(
    num_cameras: int = 2,
    num_patches: int = 16,
    embed_dim: int = 64,
    num_heads: int = 8,
    num_layers: int = 4,
    *,
    vision_per_cam: int = 4,
    n_text: int = 2,
    dit_query_dim: int = 32,
    dit_kv_dim: int = 48,
    dit_heads: int = 4,
    dit_layers: int = 4,
):
    """Mock matching the VERIFIED N1.6 paths:

        policy.model.backbone.model.vision_model            (NOT backbone.eagle_model)
        policy.model.backbone.forward  -> returns image_mask (NOT forward_eagle)
        policy.model.backbone.model.config.image_token_index
        policy.model.action_head.model.transformer_blocks   (diffusers Attention)
    """
    vision_model = _SigLIPVisionModel(num_layers, embed_dim, num_heads)
    forward_calls: list = []
    seq_len = num_cameras * (n_text + vision_per_cam)

    def backbone_forward(vl_input):
        # Fire the SigLIP2 q/k hooks for all cameras (batched), like the real fwd.
        vision_model(torch.randn(num_cameras, num_patches, embed_dim))
        forward_calls.append(vl_input)
        input_ids = vl_input["input_ids"]
        return {
            "backbone_features": torch.randn(1, seq_len, dit_kv_dim),
            "backbone_attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
            "image_mask": input_ids == IMAGE_TOKEN_INDEX,  # (1, seq_len)
        }

    eagle3 = SimpleNamespace(
        vision_model=vision_model,
        config=SimpleNamespace(image_token_index=IMAGE_TOKEN_INDEX),
    )
    backbone = SimpleNamespace(model=eagle3, forward=backbone_forward)
    action_head = SimpleNamespace(model=_MockDiT(dit_query_dim, dit_kv_dim, dit_heads, dit_layers))
    model = SimpleNamespace(backbone=backbone, action_head=action_head)
    policy = SimpleNamespace(model=model)
    return policy, forward_calls


class TestGR00TN1d6Attention:
    def test_init_resolves_n16_vision_path(self):
        # Must reach backbone.model.vision_model (not backbone.eagle_model) and
        # hook every SigLIP2 attention layer.
        policy, _ = _make_mock_groot_n16_policy(num_cameras=2, num_layers=4)
        viz = GR00TN1d6Attention(policy, camera_keys=["external_D455", "ego"])
        assert len(viz._capture._attn_modules) == 4
        assert viz._forward_attr == "forward"

    def test_enter_patches_backbone_forward(self):
        policy, _ = _make_mock_groot_n16_policy(num_cameras=2)
        original = policy.model.backbone.forward
        viz = GR00TN1d6Attention(policy, camera_keys=["external_D455", "ego"])
        with viz:
            assert policy.model.backbone.forward is not original
            # 2 cross blocks (of 4 interleaved) × (q + k) = 4 hooks.
            assert len(viz._cross._handles) == 4
        assert policy.model.backbone.forward is original

    def test_forward_reads_returned_image_mask(self):
        vision_per_cam = 4
        policy, calls = _make_mock_groot_n16_policy(num_cameras=2, vision_per_cam=vision_per_cam)
        viz = GR00TN1d6Attention(policy, camera_keys=["external_D455", "ego"])
        with viz:
            policy.model.backbone.forward(_vl_input_n16(2, vision_per_cam))
            assert len(calls) == 1
            assert viz._vision_mask is not None
            assert int(viz._vision_mask.sum()) == 2 * vision_per_cam

    def test_image_token_index_from_model_config(self):
        policy, _ = _make_mock_groot_n16_policy()
        viz = GR00TN1d6Attention(policy, camera_keys=["external_D455", "ego"])
        assert viz._image_token_index() == IMAGE_TOKEN_INDEX

    def test_action_overlay_with_rectangular_grid(self, monkeypatch):
        # 6 vision tokens/camera → non-square; patch_grid_hw=(2,3) must reshape it.
        logged: list[str] = []
        monkeypatch.setattr(overlay_mod.rr, "log", lambda path, *a, **k: logged.append(path))

        num_cameras, vision_per_cam = 2, 6
        policy, _ = _make_mock_groot_n16_policy(
            num_cameras=num_cameras, vision_per_cam=vision_per_cam,
            dit_query_dim=32, dit_kv_dim=48, dit_layers=4,
        )
        viz = GR00TN1d6Attention(
            policy, camera_keys=["external_D455", "ego"], patch_grid_hw=(2, 3)
        )
        dit = policy.model.action_head.model
        with viz:
            vl_in = _vl_input_n16(num_cameras, vision_per_cam)
            policy.model.backbone.forward(vl_in)
            seq_len = vl_in["input_ids"].shape[1]
            dit.run_denoiser(torch.randn(1, 6, 32), torch.randn(1, seq_len, 48), steps=3)
            obs = {c: np.zeros((48, 48, 3), dtype=np.uint8) for c in ["external_D455", "ego"]}
            viz.log_overlay(obs)

        action_paths = [p for p in logged if "/action/" in p]
        assert len(action_paths) == 6  # 3 streams × 2 cameras
        assert any("external_D455/action/overlay" in p for p in action_paths)

    def test_rectangular_grid_dropped_without_patch_grid_hw(self, monkeypatch):
        # Same non-square case but no patch_grid_hw → square inference fails → no
        # action overlay (frame dropped), rather than a crash.
        logged: list[str] = []
        monkeypatch.setattr(overlay_mod.rr, "log", lambda path, *a, **k: logged.append(path))

        policy, _ = _make_mock_groot_n16_policy(num_cameras=2, vision_per_cam=6)
        viz = GR00TN1d6Attention(policy, camera_keys=["external_D455", "ego"])  # no grid
        dit = policy.model.action_head.model
        with viz:
            vl_in = _vl_input_n16(2, 6)
            policy.model.backbone.forward(vl_in)
            seq_len = vl_in["input_ids"].shape[1]
            dit.run_denoiser(torch.randn(1, 6, 32), torch.randn(1, seq_len, 48), steps=2)
            obs = {c: np.zeros((48, 48, 3), dtype=np.uint8) for c in ["external_D455", "ego"]}
            viz.log_overlay(obs)
        assert [p for p in logged if "/action/" in p] == []
