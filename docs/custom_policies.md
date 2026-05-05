# Visualizing attention for a custom policy

This walks through what you need to do to get attention overlays for a
policy that **isn't** stock SmolVLA / π0 / ACT — for example, your own
fine-tune that subclasses one of those, a research policy you registered
through lerobot's custom-policy mechanism, or an entirely new
architecture that happens to use a HuggingFace-style vision encoder.

The TL;DR: the library contracts on three things, and as long as your
policy satisfies them you can wire it up in well under 50 lines.

## The contract

`SmolVLAAttention` is a thin glue layer over `VisionAttentionCapture`. It
makes three assumptions about the policy you hand it:

1. **There is a vision encoder.** A submodule, anywhere inside your
   policy, that takes a `(B, C, H, W)` image tensor and runs a stack of
   transformer layers over flattened patch tokens. SigLIP, CLIP-ViT,
   DINO-v2, the SmolVLM image tower — all qualify.
2. **Each attention layer in that encoder exposes `q_proj` and `k_proj`
   submodules.** This is the standard HuggingFace ViT / SigLIP / Idefics3
   layout. The library walks the encoder's module tree and hooks every
   module that has both projections; it also reads `num_heads` (or
   `num_attention_heads`) and `head_dim` off the same module.
3. **There is a per-image entry point you can wrap.** A method, anywhere
   in your policy, that runs the encoder once per camera. We monkey-patch
   it so we can snapshot one rollout per camera per chunk.

Stock `SmolVLAAttention` hardcodes (1) as
`policy.model.vlm_with_expert.get_vlm_model().vision_model` and (3) as
`policy.model.vlm_with_expert.embed_image`. Different policies will need
different paths to the same pair of objects.

If your encoder doesn't satisfy (2) — e.g. it uses a fused `qkv_proj`, or
a custom flash-attention module — you can still get rollouts by
subclassing `VisionAttentionCapture` (Path C below).

## Path A — your custom policy reuses SmolVLA's structure

The easy case. If your policy subclasses `SmolVLAPolicy`, or otherwise
keeps the `vlm_with_expert.get_vlm_model().vision_model` +
`vlm_with_expert.embed_image` layout intact (e.g. a LoRA fine-tune, an
extra head bolted onto the action expert, or extra preprocessing), just
use `SmolVLAAttention` directly:

```python
from lerobot_attention_visualizer import SmolVLAAttention

policy = MyCustomSmolVLAPolicy.from_pretrained("user/my-finetune")
viz = SmolVLAAttention(policy)

with viz:
    actions = policy.predict_action_chunk(obs_frame, ...)
    viz.log_overlay(obs)
```

If the rollout looks blank the first time you run it, the most common
cause is that you replaced or wrapped `embed_image`; add a `print` in
your override and confirm it still gets called once per camera.

## Path B — custom encoder path, HF-style attention layers

Your policy has its own vision encoder (still ViT-shaped, still
`q_proj` / `k_proj`-shaped) and its own per-image entry point. Reuse the
low-level `VisionAttentionCapture` and write your own wrapper:

```python
from __future__ import annotations
import contextlib

from lerobot_attention_visualizer import VisionAttentionCapture
from lerobot_attention_visualizer.visualizer.overlay import (
    log_attention_overlay,
    patch_heatmap_to_image,
    rollout_to_patch_heatmap,
)


class MyPolicyAttention:
    def __init__(self, policy):
        self.policy = policy
        # (1) point at the vision encoder.
        self._capture = VisionAttentionCapture(policy.vision_tower)
        self._orig_encode = None

    def __enter__(self):
        self._capture.__enter__()
        # (3) monkey-patch the per-image entry point.
        self._orig_encode = self.policy.encode_image

        def _patched(image):
            out = self._orig_encode(image)
            self._capture.snapshot()  # rollout for this camera, then reset Q/K cache
            return out

        self.policy.encode_image = _patched
        return self

    def __exit__(self, *exc):
        if self._orig_encode is not None:
            self.policy.encode_image = self._orig_encode
            self._orig_encode = None
        self._capture.__exit__(*exc)

    def camera_keys(self) -> list[str]:
        # Order the policy feeds them to the encoder.
        prefix = "observation.images."
        return [
            k[len(prefix):]
            for k in self.policy.config.image_features
            if k.startswith(prefix)
        ]

    def log_overlay(self, obs: dict, *, prefix: str = "attention") -> None:
        rollouts = list(self._capture.rollouts)
        self._capture.rollouts.clear()
        if not rollouts:
            return  # no fresh forward this step
        cams = self.camera_keys()
        if len(rollouts) != len(cams):
            return  # drift — drop frame rather than misalign
        for cam, rollout in zip(cams, rollouts, strict=True):
            image = obs.get(cam)
            if image is None:
                continue
            patch_heat = rollout_to_patch_heatmap(rollout)
            heat = patch_heatmap_to_image(patch_heat, target_hw=image.shape[:2])
            log_attention_overlay(f"{prefix}/{cam}", image, heat)
```

That's the entire wrapper. It mirrors `SmolVLAAttention` line-for-line —
the only policy-specific bits are `policy.vision_tower` and
`policy.encode_image`. Substitute the right attribute names for your
architecture.

### Where to find your encoder and entry point

If you didn't write the policy yourself, the fastest way to locate them
is to print the module tree and grep:

```python
for name, _ in policy.named_modules():
    if "vision" in name.lower() or "vit" in name.lower() or "siglip" in name.lower():
        print(name)
```

For the entry point, look for a method on the policy (or a submodule)
that accepts an image and returns embeddings. SmolVLA calls it
`embed_image`; other policies use names like `encode_image`,
`forward_vision`, `vision_forward`, or just inline the call inside
`forward`. If it's inlined, you can't monkey-patch it cleanly — see
Path C.

## Path C — encoder without q_proj / k_proj

Some encoders use a single fused `qkv_proj` (linear projection that
returns concatenated Q/K/V), or a custom attention module that doesn't
expose the projections as submodules. In that case
`_find_attention_modules` won't match anything and the rollout will be
empty.

Two options:

### Option 1 — subclass and override the matcher

```python
import torch
from lerobot_attention_visualizer import VisionAttentionCapture


class FusedQKVCapture(VisionAttentionCapture):
    @staticmethod
    def _find_attention_modules(root: torch.nn.Module) -> list[torch.nn.Module]:
        return [m for m in root.modules() if hasattr(m, "qkv_proj")]

    def __enter__(self):
        # ... set up _layers / num_heads / head_dim like the base class ...
        # then hook qkv_proj instead of (q_proj, k_proj). Inside the hook,
        # split the projection into (Q, K, V) on the last dim before storing.
        ...
```

This is a real refactor and won't be a copy-paste job — you have to
adapt the slicing to your encoder's exact `qkv_proj` output layout
(`(B, S, 3*H*d)` packed differently in different libraries).

### Option 2 — fall back to activation magnitude

If you only need a "where is the encoder firing" picture and don't care
about cross-token rollout, the ACT path is cheaper and works on any
encoder. Hook the final encoder block's output and take its channel-wise
L2 norm. See `src/lerobot_attention_visualizer/policies/act.py` for the
full pattern; the core is `feature_map.norm(dim=1)`.

## Working with lerobot's custom policy registry

LeRobot has a policy-registration system that lets you load custom
architectures via the same `make_policy` / `from_pretrained` flow as
stock policies. Once your policy is registered there, **none of the code
above changes** — you load it the lerobot way and then wrap the
resulting `policy` object with whichever adapter you wrote.

```python
from lerobot.policies.factory import make_policy   # exact import varies by lerobot version

policy = make_policy("my_custom_policy_id", ...)   # however lerobot loads it
viz = MyPolicyAttention(policy)
with viz:
    ...
```

LeRobot's policy registry API has moved across minor versions, so check
the lerobot v0.5.0 docs for the exact `register_policy` /
`PreTrainedPolicy` boilerplate. The relevant point for this library is
that we never reach into lerobot's registry — we only reach into the
`policy` object's module tree, and as long as the contract above holds,
how you obtained the policy doesn't matter.

## Observation dict shape

`viz.log_overlay(obs)` expects `obs` to be a dict mapping **bare camera
names** (e.g. `"top"`, `"wrist.top"`) to HWC `uint8` ndarrays — the
shape returned by `lerobot.robots.so_follower.SO101Follower.get_observation`
and friends. If your obs pipeline returns:

- chw / float / normalized tensors → convert to HWC uint8 first
- keys like `"observation.images.top"` → strip the prefix before passing
  to `log_overlay`, or override `camera_keys()` in your wrapper

If shapes drift, `log_overlay` skips the frame rather than logging
mis-sized overlays — set a breakpoint there if heatmaps stop appearing.

## Sanity checks

After wiring up your wrapper, work through this list before assuming
the visualizer is broken:

1. **Run one forward, check `viz._capture.rollouts`.** If empty, your
   per-image entry point isn't being called from inside the policy
   forward. Add a print to confirm.
2. **Check `len(viz._capture._attn_modules)`.** Should match the number
   of attention layers in your encoder (e.g. 26 for SigLIP-base). If 0,
   the encoder doesn't have `q_proj` / `k_proj` — see Path C.
3. **Check rollout shape.** `(B, S, S)` where `S` is the patch count.
   For 384×384 SigLIP with patch 14, `S = 729 = 27²`. If `S` isn't a
   perfect square, `rollout_to_patch_heatmap` will raise — pass an
   explicit patch grid.
4. **Check `obs` keys against `camera_keys()`.** A silent name mismatch
   is the most common reason heatmaps don't render.

If all four pass and the heatmaps still look wrong (uniform, all on one
patch, flickering), the issue is interpretive rather than mechanical —
see the "Scope" section of `policies/smolvla.py` for the caveats around
what attention rollout actually tells you.
