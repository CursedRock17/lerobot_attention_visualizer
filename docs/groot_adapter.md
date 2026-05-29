# Groot N1.6 Adapter — Design Notes

Design document for `policies/groot.py` — the attention visualizer adapter for NVIDIA's
Groot N1.6 policy, which uses the Eagle-2 VLM backbone.

---

## Architecture overview

```
GrootPolicy
└── model: GR00TN15
    ├── backbone: EagleBackbone
    │   └── eagle_model: Eagle25VLForConditionalGeneration
    │       ├── vision_model: SiglipVisionModel       ← hook target
    │       │   └── vision_model.encoder.layers[i]
    │       │       └── self_attn  (q_proj, k_proj)   ← Q/K capture points
    │       ├── mlp1                                   ← pixel-shuffle + projection
    │       └── language_model: Qwen2                 ← not captured
    └── action_head: ...
```

Eagle-2 uses **SiglipVisionModel** as its vision encoder — the identical architecture
to SmolVLA's SigLIP tower. The `self_attn.q_proj` / `self_attn.k_proj` layout is
the same, so `VisionAttentionCapture` can be reused without modification.

---

## Key difference from SmolVLA

| | SmolVLA | Groot N1.6 |
|---|---|---|
| Per-image entry point | `vlm_with_expert.embed_image(img)` — called **once per camera** | No per-camera call — all cameras batched into one `pixel_values` forward |
| Q/K shape after hooks | `(1, n_patches, embed_dim)` — single image | `(N_cameras, n_patches, embed_dim)` — all cameras in batch dim |
| Snapshot strategy | Patch `embed_image`; call `snapshot()` on each call | Patch `extract_feature` (or hook post-vision-model); split batch dim after |
| Token count | 729 patches (27×27 at 384 px, patch 14) | ~256 tokens after pixel-shuffle downsampling (config-dependent) |

---

## Capture strategy

Because all cameras are processed in a single `vision_model` forward, we cannot use
the per-call snapshot pattern. Instead:

1. **Install Q/K hooks** on `eagle_model.vision_model` via `VisionAttentionCapture`
   as usual — the hooks will fire once per forward with `q/k` shaped
   `(N_cameras, n_patches, embed_dim)`.

2. **Patch `EagleBackbone.forward_eagle`** instead of `embed_image`. After the
   patched method returns, call a new method `snapshot_split(N_cameras)` that:
   - Reads the current `_layers[i].q` / `.k` tensors (shape `(N_cams, S, D)`)
   - Slices them on dim 0 to produce one `_LayerCache` per camera
   - Appends `N_cameras` entries to `_pending` in camera order
   - Clears `_layers[i].q` / `.k`

3. **`drain_rollouts`** stays unchanged — it already processes each entry in
   `_pending` independently.

This means `log_overlay` sees exactly `N_cameras` rollouts after each policy forward,
matching the camera keys from `policy.config.image_features`, just like SmolVLA.

---

## Token / patch count

Eagle-2 applies **pixel shuffle** (`downsample_ratio=0.5` by default) after the SigLIP
encoder, reducing the 729-patch (27×27) output to ~182 tokens (roughly 13×14,
non-square due to the reshape). The attention capture happens **before** pixel shuffle
(we hook the SigLIP `self_attn` outputs directly), so `rollout_to_patch_heatmap`
still receives a square 27×27 grid.

Verify at runtime: `capture._layers[0].q.shape[1]` should be 729 for a 384 px input.

---

## Hook path

```python
# Navigate to the SigLIP vision model inside Eagle-2
vision_model = policy.model.backbone.eagle_model.vision_model

capture = VisionAttentionCapture(vision_model)
```

Camera keys are read from `policy.config.image_features` (same as SmolVLA).

---

## Patching entry point

```python
_orig_forward_eagle = policy.model.backbone.forward_eagle

def _patched_forward_eagle(vl_input):
    result = _orig_forward_eagle(vl_input)
    # After vision model has fired, Q/K shape = (N_cameras, n_patches, embed_dim)
    n_cameras = len(camera_keys)
    capture.snapshot_split(n_cameras)
    return result

policy.model.backbone.forward_eagle = _patched_forward_eagle
```

---

## `snapshot_split` — new method on `VisionAttentionCapture`

```python
def snapshot_split(self, n_cameras: int) -> None:
    """Freeze one _LayerCache snapshot per camera from a batched forward.

    When the vision model processes all cameras in a single forward pass
    (batch dim = N_cameras), each layer's Q/K is shaped (N_cams, S, D).
    This splits on dim 0 and appends N_cameras entries to _pending.
    """
    for cam_idx in range(n_cameras):
        self._pending.append([
            _LayerCache(
                q=c.q[cam_idx : cam_idx + 1] if c.q is not None else None,
                k=c.k[cam_idx : cam_idx + 1] if c.k is not None else None,
                num_heads=c.num_heads,
                head_dim=c.head_dim,
            )
            for c in self._layers
        ])
    for c in self._layers:
        c.q = None
        c.k = None
```

---

## `GR00TAttention` class outline

```python
class GR00TAttention:
    def __init__(self, policy, last_layer_only: bool = False):
        vision_model = policy.model.backbone.eagle_model.vision_model
        self._capture = VisionAttentionCapture(vision_model)
        self._policy = policy
        self._last_layer_only = last_layer_only
        self._camera_keys = [
            k[len("observation.images."):]
            for k in policy.config.image_features
            if k.startswith("observation.images.")
        ]
        self._orig_forward_eagle = None

    def camera_keys(self) -> list[str]:
        return self._camera_keys

    def __enter__(self):
        self._capture.__enter__()
        orig = self._policy.model.backbone.forward_eagle
        self._orig_forward_eagle = orig
        capture = self._capture
        n_cams = len(self._camera_keys)

        def _patched(vl_input):
            result = orig(vl_input)
            capture.snapshot_split(n_cams)
            return result

        self._policy.model.backbone.forward_eagle = _patched
        return self

    def __exit__(self, *exc):
        self._policy.model.backbone.forward_eagle = self._orig_forward_eagle
        self._orig_forward_eagle = None
        self._capture.__exit__(*exc)

    def log_overlay(self, obs: dict, *, prefix: str = "attention",
                    clip_percentile: float = 95.0) -> None:
        rollouts = self._capture.drain_rollouts(
            last_layer_only=self._last_layer_only
        )
        if not rollouts or len(rollouts) != len(self._camera_keys):
            return
        for cam_key, rollout in zip(self._camera_keys, rollouts):
            image = obs.get(cam_key)
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
                continue
            patch_heat = rollout_to_patch_heatmap(rollout)
            heat = patch_heatmap_to_image(
                patch_heat, target_hw=image.shape[:2],
                clip_percentile=clip_percentile
            )
            log_attention_overlay(f"{prefix}/{cam_key}", image, heat)
```

---

## Open questions — resolved

1. **Does `forward_eagle` fire exactly once per `get_action` call?**
   Confirm by printing inside a patched version before writing hooks.

2. ~~Is `pixel_values` always `(N_cameras, C, H, W)` or can it be a flat
   concatenation of tiles (dynamic resolution)?~~
   **Resolved.** `processor_groot.py` hardcodes `max_dynamic_tiles=1` —
   every camera image produces exactly one tile. `pixel_values` is always
   `(N_cameras, C, H, W)` in robot inference. `snapshot_split` is safe.
   Eagle-2.5's dynamic tiling (used in general VLM chat) is disabled here.

3. **Flash attention compatibility.**
   Eagle-2 forces `flash_attention_2` on the SigLIP tower. Flash attention
   does not return attention weights, so `output_attentions=True` won't work.
   Our Q/K hook approach (hooking `q_proj` / `k_proj` outputs, computing
   attention ourselves) bypasses this entirely — no change needed.

4. **`select_layer` setting.**
   `EagleBackbone` uses `self.select_layer` to pick which hidden state to
   pass downstream. This does not affect which layers we hook — we hook
   all SigLIP encoder layers regardless.

---

## Testing plan

- Mock matching the Groot attribute path:
  `policy.model.backbone.eagle_model.vision_model` → `_SigLIPVisionModel`
- Verify `snapshot_split(n_cameras=2)` produces 2 entries in `_pending`
  with `q.shape == (1, n_patches, embed_dim)` per entry
- Verify `drain_rollouts()` produces 2 rollouts of shape `(1, n_patches, n_patches)`
- Verify `__exit__` restores `forward_eagle` and removes all hooks
- 729-patch and dynamic-tile edge cases
