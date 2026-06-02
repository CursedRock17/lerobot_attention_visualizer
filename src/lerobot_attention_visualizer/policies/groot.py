"""Groot attention capture — Eagle-2 VLM backbone wrapper.

Two adapters are provided:

GR00TAttention
    Wraps lerobot's `GrootPolicy` (Groot N1.5). Attribute path:
        policy._groot_model.backbone.eagle_model.vision_model
        policy._groot_model.action_head.model            (the cross-attn DiT)

GR00TN1d6Attention
    Wraps Isaac-GR00T's native `Gr00tPolicy` (Groot N1.6 / Gr00tN1d6).
    The native loader registers `Gr00tN1d6` with HuggingFace AutoModel;
    lerobot's `GrootPolicy.from_pretrained` cannot load N1.6 checkpoints.
    Assumed attribute path (INHERITED FROM N1.5 — SEE WARNING BELOW):
        policy.model.backbone.eagle_model.vision_model
        policy.model.action_head.model

    !!! N1.6 ARCHITECTURE IS NOT THE SAME AS N1.5 — PATHS UNVERIFIED !!!
    N1.5 uses the Eagle-2 VLM (SmolLM2 LLM + SigLIP). N1.6 swapped to a
    **SigLIP2 vision encoder + Qwen3 language model** with a reworked VL→DiT
    connector; its action head is the "AlternateVLDiT" (still a cross-attention
    flow-matching DiT that interleaves cross-attn every 2 blocks, so the
    `CrossAttentionCapture` *approach* still applies). BUT the module paths
    above and the image-token mask (`eagle_input_ids == image_token_index`,
    Eagle/InternVL-style) are copied from N1.5 and have NOT been verified
    against a real N1.6 checkpoint. Run a `policy.model` module-tree probe and
    fix the paths before trusting N1.6 maps. Tracked on branch `groot_n16`.

The N1.5 `GR00TAttention` below is verified; only N1.6 is provisional. They
currently share the `_EagleCrossAttention` base on the (unconfirmed) assumption
that the vision-encoder + cross-attn-DiT layout carries over.

# Two complementary signals

This adapter streams **two** overlays per camera:

`attention/<cam>/encoder/*`
    SigLIP vision-encoder self-attention rollout — "which patches the (usually
    frozen) image encoder finds salient to represent." Captured via
    `VisionAttentionCapture` + `snapshot_split` (Eagle batches all cameras into
    one `forward_eagle`, so Q/K emerge as `(N_cameras, n_patches, dim)` and we
    slice along dim 0).

`attention/<cam>/action/*`
    Action-head cross-attention — "which vision tokens the flow-matching
    denoiser actually looks at while producing the action." Captured via
    `CrossAttentionCapture` on the DiT's cross-attending blocks. This is the
    signal that changes when you fine-tune the action head, so it is the one to
    watch for grounding bugs. See `cross_attention.py` for the mechanism.

The vision tokens the action head attends to are NOT the 729 raw SigLIP patches:
Eagle pixel-shuffles them down (downsample 0.5 → 256 tokens, a 16×16 grid per
camera) before they enter the LLM sequence. We recover their columns from
`eagle_input_ids == image_token_index`, captured inside the patched
`forward_eagle`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..visualizer.overlay import (
    log_attention_overlay,
    patch_heatmap_to_image,
    rollout_to_patch_heatmap,
)
from .cross_attention import (
    CrossAttentionCapture,
    find_cross_attention_blocks,
    vision_importance_to_grids,
)
from .smolvla import VisionAttentionCapture

if TYPE_CHECKING:
    from lerobot.policies.groot.modeling_groot import GrootPolicy


class _EagleCrossAttention:
    """Shared base for the N1.5 and N1.6 Groot adapters.

    Subclasses resolve the two model-root-dependent attribute paths (the Eagle
    vision model and the action head live under `_groot_model` for N1.5 and under
    `model` for N1.6) and supply `camera_keys()`. Everything else — installing
    the encoder + cross-attention captures, patching `forward_eagle`, and logging
    both overlays — is shared here.
    """

    def __init__(
        self,
        policy,
        *,
        vision_model,
        model_root,
        last_layer_only: bool,
        cross_attention: bool,
    ):
        self.policy = policy
        self._model_root = model_root
        self._last_layer_only = last_layer_only
        self._cross_enabled = cross_attention

        self._capture = VisionAttentionCapture(vision_model)
        self._cross: CrossAttentionCapture | None = None
        self._orig_forward_eagle = None
        # Vision-token column mask over the VL key sequence, captured per forward.
        self._vision_mask = None

    def camera_keys(self) -> list[str]:
        raise NotImplementedError

    def _image_token_index(self) -> int:
        """The token id Eagle replaces with vision embeddings (input_ids == this)."""
        eagle_model = self._model_root.backbone.eagle_model
        idx = getattr(eagle_model, "image_token_index", None)
        if idx is None:
            idx = eagle_model.config.image_token_index
        return idx

    def __enter__(self) -> "_EagleCrossAttention":
        self._capture.__enter__()
        n_cameras = len(self.camera_keys())
        backbone = self._model_root.backbone
        orig = backbone.forward_eagle
        self._orig_forward_eagle = orig
        capture = self._capture

        img_tok = None
        if self._cross_enabled:
            cross_blocks = find_cross_attention_blocks(
                self._model_root.action_head.model.transformer_blocks
            )
            self._cross = CrossAttentionCapture(cross_blocks)
            self._cross.__enter__()
            img_tok = self._image_token_index()

        def _patched_forward_eagle(vl_input):
            result = orig(vl_input)
            # Q/K shape here: (N_cameras, n_patches, embed_dim). snapshot_split
            # slices dim 0 into one _LayerCache per camera for the encoder path.
            capture.snapshot_split(n_cameras)
            if self._cross is not None:
                # Vision tokens are the columns of the VL key sequence where the
                # eagle input id is the image placeholder. Order is camera-major.
                input_ids = vl_input["eagle_input_ids"]
                self._vision_mask = (input_ids[0] == img_tok).detach().cpu()
            return result

        backbone.forward_eagle = _patched_forward_eagle
        return self

    def __exit__(self, *exc) -> None:
        if self._orig_forward_eagle is not None:
            self._model_root.backbone.forward_eagle = self._orig_forward_eagle
            self._orig_forward_eagle = None
        self._capture.__exit__(*exc)
        if self._cross is not None:
            self._cross.__exit__(*exc)
            self._cross = None

    def log_overlay(
        self,
        obs: dict,
        *,
        prefix: str = "attention",
        clip_percentile: float = 95.0,
        suppress_outliers: bool = False,
        gamma: float = 1.0,
    ) -> None:
        """Stream the encoder overlay and the action cross-attention overlay.

        No-op if no forward happened since the last call. Each signal is dropped
        independently if its token count doesn't match the camera count, rather
        than misaligning overlays.

        `suppress_outliers` winsorizes SigLIP attention-sink spikes (mainly helps
        the encoder view) and `gamma` sets the display contrast (>1 = punchier,
        background suppressed) — see `patch_heatmap_to_image`. Both apply to the
        encoder and action overlays alike.
        """
        camera_keys = self.camera_keys()
        n = len(camera_keys)

        # --- Encoder self-attention (always on) ---
        rollouts = self._capture.drain_rollouts(last_layer_only=self._last_layer_only)
        if rollouts and len(rollouts) == n:
            for cam_key, rollout in zip(camera_keys, rollouts, strict=True):
                image = obs.get(cam_key)
                if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
                    continue
                patch_heat = rollout_to_patch_heatmap(rollout)
                heat = patch_heatmap_to_image(
                    patch_heat,
                    target_hw=image.shape[:2],
                    clip_percentile=clip_percentile,
                    suppress_outliers=suppress_outliers,
                    gamma=gamma,
                )
                log_attention_overlay(f"{prefix}/{cam_key}/encoder", image, heat)

        # --- Action-head cross-attention (the action-driving signal) ---
        importance = self._cross.drain() if self._cross is not None else None
        if importance is not None and self._vision_mask is not None:
            grids = vision_importance_to_grids(importance, self._vision_mask, n)
            if len(grids) == n:
                for cam_key, grid in zip(camera_keys, grids, strict=True):
                    image = obs.get(cam_key)
                    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
                        continue
                    heat = patch_heatmap_to_image(
                        grid,
                        target_hw=image.shape[:2],
                        clip_percentile=clip_percentile,
                        suppress_outliers=suppress_outliers,
                        gamma=gamma,
                    )
                    log_attention_overlay(f"{prefix}/{cam_key}/action", image, heat)


class GR00TAttention(_EagleCrossAttention):
    """High-level: lerobot GrootPolicy (N1.5) wrapper streaming both overlays to rerun.

    Usage:
        viz = GR00TAttention(policy)
        with viz:
            action = policy.select_action(batch)
            viz.log_overlay(obs)

    Pass `cross_attention=False` to stream only the encoder overlay.
    """

    def __init__(
        self,
        policy: "GrootPolicy",
        *,
        last_layer_only: bool = False,
        cross_attention: bool = True,
    ):
        super().__init__(
            policy,
            vision_model=policy._groot_model.backbone.eagle_model.vision_model,
            model_root=policy._groot_model,
            last_layer_only=last_layer_only,
            cross_attention=cross_attention,
        )

    def camera_keys(self) -> list[str]:
        """Bare camera names in the order the policy feeds them to the encoder."""
        prefix = "observation.images."
        return [
            k[len(prefix):]
            for k in self.policy.config.image_features
            if k.startswith(prefix)
        ]


class GR00TN1d6Attention(_EagleCrossAttention):
    """High-level: Isaac-GR00T Gr00tPolicy (N1.6) wrapper streaming both overlays.

    Use this for checkpoints with `"model_type": "Gr00tN1d6"` loaded via:

        from gr00t.policy.gr00t_policy import Gr00tPolicy
        policy = Gr00tPolicy(
            model_path="nvidia/GR00T-N1.6-3B",
            embodiment_tag="new_embodiment",
            device="cuda",
        )
        viz = GR00TN1d6Attention(policy, camera_keys=["ego", "external"])
        with viz:
            action = policy.get_action(obs)
            viz.log_overlay(obs)

    camera_keys must match the bare names used in your obs dict (e.g. "ego",
    "external" — without the "observation.images." prefix).

    The Isaac-GR00T `Gr00tPolicy` stores the loaded model at `policy.model`
    (not `policy._groot_model`).

    PROVISIONAL: the `policy.model.backbone.eagle_model.vision_model` and
    `action_head.model.transformer_blocks` paths below are inherited from N1.5
    and are NOT confirmed for N1.6 (SigLIP2 + Qwen3, not Eagle-2). They will
    raise `AttributeError` if N1.6 renamed/restructured the backbone. Verify
    against a real N1.6 module tree and adjust before relying on the maps — see
    the module docstring and branch `groot_n16`.
    """

    def __init__(
        self,
        policy,
        *,
        camera_keys: list[str],
        last_layer_only: bool = False,
        cross_attention: bool = True,
    ):
        super().__init__(
            policy,
            vision_model=policy.model.backbone.eagle_model.vision_model,
            model_root=policy.model,
            last_layer_only=last_layer_only,
            cross_attention=cross_attention,
        )
        self._camera_keys = camera_keys

    def camera_keys(self) -> list[str]:
        return self._camera_keys
