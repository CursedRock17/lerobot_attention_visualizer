"""Stream GR00T-N1.6 attention overlays to rerun (Isaac-GR00T `Gr00tPolicy`).

This is an INTEGRATION TEMPLATE, not a turnkey replay script. N1.6 loads and runs
through Isaac-GR00T's native API (`Gr00tPolicy`, Eagle3-VL = SigLIP2 + Qwen3), not
lerobot — and the obs format + data loading are environment-specific. So this
shows how to wrap *your existing* N1.6 policy + eval loop with
`GR00TN1d6Attention`, plus a one-shot probe to read the per-camera patch grid.

Two overlays per camera stream to rerun (see `policies/groot.py`):
    attention/<cam>/encoder/*   SigLIP2 self-attention (best-effort on N1.6)
    attention/<cam>/action/*    action-head cross-attention  ← the signal to watch

Prerequisites (the verified GB10 stack — see README "Groot N1.6"):
    CUDA 13 / torch nightly cu130, flash-attn, transformers==4.51.3,
    diffusers<0.36, numpy==1.26.0, Isaac-GR00T @ ead5283, and `rerun-sdk`
    (NOT pulled by --no-deps installs — add it explicitly).

Usage: adapt the three TODO blocks (policy load, camera keys, obs/loop) to your rig,
set PROBE=True for the first run to discover PATCH_GRID_HW, then set it and rerun.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from lerobot.utils.visualization_utils import init_rerun

from lerobot_attention_visualizer import GR00TN1d6Attention

# rr.spawn() needs the viewer binary on PATH even inside conda.
_local_bin = str(Path.home() / ".local" / "bin")
if _local_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _local_bin + os.pathsep + os.environ.get("PATH", "")

# ---------------------------------------------------------------------------
# Edit these to match your setup
# ---------------------------------------------------------------------------
MODEL_PATH      = "CursedRock17/so101_teleop_vials_sim_and_real_finetune"  # or a local checkpoint dir
EMBODIMENT_TAG  = "new_embodiment"
# Bare obs-dict camera names for your embodiment, in feed order. For this
# checkpoint's `new_embodiment`, the processor lists these two:
CAMERA_KEYS     = ["external_D455", "ego"]
# Per-camera vision-token patch grid (h, w) for the ACTION overlay. N1.6's
# SigLIP2 tiles at native resolution, so the grid is usually NOT square — leaving
# this None infers a square grid and silently drops the action overlay when the
# count isn't a perfect square. Run with PROBE=True (below) to discover it.
PATCH_GRID_HW   = None          # e.g. (16, 22)
PROBE           = True          # first run: print the per-camera token count, then exit
LAST_LAYER_ONLY = False
CLIP_PERCENTILE = 99.0
SUPPRESS_OUTLIERS = True
GAMMA           = 2.5
COLORMAP        = "hot"         # "hot" | "blue-green" | "viridis"
# ---------------------------------------------------------------------------

# --- TODO 1: load YOUR N1.6 policy (must load with flash_attention_2 + bf16) ---
# from gr00t.policy.gr00t_policy import Gr00tPolicy   # adjust import to your Isaac-GR00T
# policy = Gr00tPolicy(model_path=MODEL_PATH, embodiment_tag=EMBODIMENT_TAG, device="cuda")
#
# Tip: don't construct a throwaway policy just to probe — reuse the one your demo
# already loads, to avoid the "Qwen3 must use flash_attention_2 but got None"
# assertion (that surfaces when attn_implementation isn't propagated at load).
raise SystemExit(
    "Edit TODO 1–3 in this template: load your Gr00tPolicy, build the obs dict, "
    "and drive the loop. See the module docstring."
)

init_rerun(session_name="groot_n16_attention")

viz = GR00TN1d6Attention(
    policy,                       # noqa: F821 — provided by TODO 1
    camera_keys=CAMERA_KEYS,
    last_layer_only=LAST_LAYER_ONLY,
    patch_grid_hw=PATCH_GRID_HW,
)

with viz:
    # --- TODO 2: produce an obs dict: bare cam name -> HWC uint8 ndarray ---
    # obs = {cam: frame_hwc_uint8 for cam in CAMERA_KEYS}

    # --- TODO 3: your control / replay loop ---
    # for step in range(num_steps):
    #     obs = get_observation()                 # your source
    #     action = policy.get_action(build_groot_inputs(obs))   # your usual call
    #
    #     if PROBE:
    #         # One forward has now run; viz captured the vision-token mask.
    #         m = viz._vision_mask
    #         if m is not None:
    #             total = int(m.sum())
    #             per_cam = total // len(CAMERA_KEYS)
    #             print(f"[probe] vision tokens: total={total}, per_camera={per_cam}")
    #             print(f"[probe] set PATCH_GRID_HW to (h, w) with h*w == {per_cam} "
    #                   f"(from the processed image patch grid / spatial_shapes).")
    #         raise SystemExit("Probe done — set PATCH_GRID_HW and rerun with PROBE=False.")
    #
    #     viz.log_overlay(
    #         obs,
    #         clip_percentile=CLIP_PERCENTILE,
    #         suppress_outliers=SUPPRESS_OUTLIERS,
    #         gamma=GAMMA,
    #         colormap=COLORMAP,
    #     )
    pass  # ← replace with TODO 2–3 above

print("Done. Check the rerun viewer under attention/<cam>/encoder/ and /action/.")
