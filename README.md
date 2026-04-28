# lerobot_attention_visualizer

Watch where a lerobot policy's vision path is focused while it drives the arm
or replays a recorded episode. Per-camera heatmaps stream to **rerun** next
to the raw image so you can eyeball whether the model is locking onto the
block, the gripper, a cable in the background, etc.

Targets **lerobot v0.5.0** and **LeRobotDataset v3.0**. Hardware-agnostic:
works with any robot lerobot supports (SO-100, SO-101, Aloha, …). CUDA is
preferred for SmolVLA / π0; ACT runs comfortably on CPU.

## Policies wired up

- **SmolVLA** (and **π0**, near-identical port) — attention rollout
  (Abnar & Zuidema 2020) across the SigLIP ViT layers. Computed manually
  because HF's vision encoder uses
  `torch.nn.functional.scaled_dot_product_attention` under the hood, which
  never materializes the `softmax(QK^T/√d)` tensor.
- **ACT** — per-spatial-cell activation magnitude of the ResNet18 backbone's
  final conv stage (`layer4`). ACT has no self-attention in the backbone, so
  this is the CNN analogue: "where does the encoder fire hardest before the
  transformer mixes tokens together?"

## What you get

Three rerun streams per camera per chunk:

```
attention/<cam>/image       # raw RGB
attention/<cam>/attention   # heatmap (red = high attention)
attention/<cam>/overlay     # blended 50/50
```

Updated once per RTC chunk (~every 10–20 control steps) for SmolVLA, and once
per ACT-queue refill (every `n_action_steps`) for ACT — enough to read the
story without burning compute.

## Layout

```
lerobot_attention_visualizer/
├── visualizer/         # shared heatmap math + rerun streams
├── policies/           # per-policy capture wrappers (smolvla, act)
└── running/            # live eval loops (one per policy)
```

A peer `playback/` section will land next to `running/` for replaying a
recorded `LeRobotDataset` and visualizing attention frame-by-frame, without
any robot hardware.

## Install

```bash
pip install -e .
# Or, if developing against a local lerobot checkout:
# pip install -e /path/to/lerobot
# pip install -e .
```

## Run

The live loops connect to a real robot (e.g. SO-101) and stream overlays to
rerun. Edit the constants at the top of each script (`HF_USER`, follower port,
camera serials, task description) to match your setup, then:

```bash
python -m lerobot_attention_visualizer.running.smolvla   # SmolVLA + RTC + rollout
python -m lerobot_attention_visualizer.running.act       # ACT + ResNet activation
```

Toggle `ATTENTION_ENABLED = False` at the top of either script to run the
same control loop without the capture — useful for A/B-comparing the policy's
behavior with the instrumentation removed.

## Design

`torch.nn.functional.scaled_dot_product_attention` (which the HF vision
encoder uses under the hood) never materializes `softmax(QK^T/√d)`, so we
can't just read attention weights off the model. Instead we:

1. Install forward hooks on every attention layer's `q_proj` / `k_proj`
   inside the SigLIP vision encoder. Each hook caches the projected Q/K
   tensor.
2. After each `embed_image` call (one per camera per chunk), recompute
   `softmax(Q @ K^T / √(d_head))` manually from the cached tensors, run
   attention rollout across all ViT layers with the residual trick, and
   store the (seq, seq) rollout.
3. Reduce the rollout to a per-patch importance vector (column-mean),
   reshape to the patch grid, bilinear-upsample to the camera image size,
   and overlay via rerun.

Cost: one extra matmul+softmax per ViT layer per image per chunk. On a
4090 laptop this is noise compared to the VLM forward pass.

## Scope

**What this tells you.** Where inside each image the vision encoder is
concentrating its own internal attention — i.e. which image regions the
encoder thinks are salient to represent.

**What this does NOT tell you.** Which image regions actually drive the
*action* output. Vision-encoder attention is a proxy — a cleaner answer
would inspect the expert→prefix cross-attention at the final joint
SmolVLM+expert layer, or run a gradient-based attribution
(IntegratedGradients, Grad-CAM) from the action logits back to the input
pixels. Both are straightforward follow-ups that reuse the same rerun
logging path.

## ACT-specific notes

ACT's ResNet is a pure CNN, so the overlay you see is an *activation
heatmap*, not an attention map. Channel-wise L2 norm on the `layer4`
feature map highlights where the CNN has the largest pre-transformer
response. Caveats:

- Bright doesn't mean *important for the action* — the transformer still
  reweights these tokens via cross-attention. A cleaner "what drives the
  action" signal would be the decoder's cross-attention weights onto the
  image tokens, which is a natural follow-up on the same hook scaffold.
- The overlay only refreshes when ACT's internal action queue refills
  (every `n_action_steps` control steps). Between refills the last overlay
  stays on screen in rerun.
