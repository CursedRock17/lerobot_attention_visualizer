"""Replay a recorded LeRobotDataset and stream SmolVLA attention overlays to rerun.

No robot hardware required — loads camera frames directly from the dataset and
feeds them through the policy exactly as the live eval loop would, then logs
image / heatmap / overlay per camera to rerun.

Useful for:
  - Verifying the attention visualizer without a connected arm
  - Offline debugging ("was the model looking at the block during failure ep 3?")
  - Generating attention visualizations for a paper / slide deck

Usage
-----
Edit the four constants at the top, then:
    conda activate lav
    python examples/visualize_smolvla_dataset.py

Rerun must be running or will be launched automatically. Open the rerun viewer
and navigate to the `attention/` tree to see per-camera overlays.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.visualization_utils import init_rerun

from lerobot_attention_visualizer import SmolVLAAttention

# Ensure ~/.local/bin is in PATH so rr.spawn() can find the rerun viewer binary
# even when running inside a conda environment that doesn't inherit it.
_local_bin = str(Path.home() / ".local" / "bin")
if _local_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _local_bin + os.pathsep + os.environ.get("PATH", "")

# ---------------------------------------------------------------------------
# Edit these to match your setup
# ---------------------------------------------------------------------------
POLICY_PATH      = "Sa74ll/smolvla_so101_pickandplace"
DATASET_REPO_ID  = "lerobot/svla_so101_pickplace"
TASK_DESCRIPTION = "pink lego brick into the transparent box"

EPISODE_IDX = 45         # which episode to replay
# False = full attention rollout (smoother, recommended). True = raw last-layer
# self-attention, which is dominated by SigLIP attention-sink / register patches
# and reads as splotchy noise — only useful for inspecting a single layer.
LAST_LAYER_ONLY = False
PLAYBACK_FPS = 30       # pacing; set to None to run as fast as possible
# Clip the top (100 - CLIP_PERCENTILE)% before normalizing the heatmap.
# Suppresses SigLIP's structural edge hot spots so the object being manipulated
# fills the color scale. Lower = more clipping; 100 = plain min-max.
CLIP_PERCENTILE = 95.0
# Opt-in artifact suppression: winsorize SigLIP attention-sink / register spikes
# (median + 6·MAD) so they don't read as splotches. Flip to True and re-run to
# A/B against the default in the rerun viewer.
SUPPRESS_OUTLIERS = False
# Display contrast. 1.0 = linear. >1 makes the map more "extreme" — crushes
# background noise toward black and isolates the bright hot spot (try 2–4).
GAMMA = 1.0
# Heatmap palette: "hot" (red→yellow→white), "blue-green", or "viridis".
COLORMAP = "hot"
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load one episode so the download is small.
# revision="main" bypasses the Hub version-tag check — some community datasets
# are not tagged with a codebase version even though their info.json is valid v3.0.
dataset = LeRobotDataset(DATASET_REPO_ID, episodes=[EPISODE_IDX], revision="main")

# Derive the episode length.
# dataset.meta.episodes is a HF datasets.Dataset; column-indexed with bare Python ints.
# Because we loaded with episodes=[EPISODE_IDX], frame indices are 0-based local indices.
episode_len = len(dataset)
print(f"Episode {EPISODE_IDX}: {episode_len} frames at {dataset.fps} Hz")

# Camera keys as bare names ("top", "wrist.top") — matches log_overlay's obs dict.
cam_prefix = "observation.images."
camera_keys = [
    k[len(cam_prefix):]
    for k in dataset.features
    if k.startswith(cam_prefix)
]
print(f"Cameras: {camera_keys}")

policy = SmolVLAPolicy.from_pretrained(POLICY_PATH)
# Disable RTC for playback — predict_action_chunk without RTC always runs
# the full forward (i.e. always calls embed_image, which fires the hooks).
policy.config.rtc_config = None

# Remap image_features to match the actual dataset camera names.
# smolvla_base ships with placeholder keys (camera1/camera2/camera3); real datasets
# use names like "up"/"side". prepare_images() does a dict lookup, so keys must match.
_vis_feature = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
# image_features is a read-only property derived from input_features; patch the source.
# Keep all non-VISUAL entries (state, etc.) and replace VISUAL placeholders with
# the actual camera names from the dataset.
_non_visual = {
    k: v for k, v in policy.config.input_features.items()
    if v.type is not FeatureType.VISUAL
}
policy.config.input_features = {
    **_non_visual,
    **{f"{cam_prefix}{cam}": _vis_feature for cam in camera_keys},
}

policy.to(device).eval()

preprocessor, _ = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=POLICY_PATH,
    dataset_stats=None,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

init_rerun(session_name="smolvla_dataset_attention")

viz = SmolVLAAttention(policy, last_layer_only=LAST_LAYER_ONLY)
interval = (1.0 / PLAYBACK_FPS) if PLAYBACK_FPS else 0.0

with viz:
    for frame_idx in range(episode_len):
        t0 = time.perf_counter()

        frame = dataset[frame_idx]

        # Build obs dict for log_overlay: bare cam name → HWC uint8 numpy.
        # dataset images are (C, H, W) float32 in [0, 1] after hf_transform_to_torch.
        obs: dict[str, np.ndarray] = {}
        for cam in camera_keys:
            key = f"{cam_prefix}{cam}"
            if key in frame:
                img_t = frame[key]  # (C, H, W) float32 [0, 1]
                obs[cam] = (img_t * 255).byte().permute(1, 2, 0).cpu().numpy()

        # Build the policy batch: tensors need a batch dim + device + task string.
        batch: dict = {}
        for k, v in frame.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0).to(device)
        batch["task"] = [TASK_DESCRIPTION]

        # Preprocessor normalises observations to the policy's expected range.
        batch = preprocessor(batch)

        # Forward pass — embed_image fires per camera, snapshot() freezes Q/K.
        with torch.inference_mode():
            policy.predict_action_chunk(batch)

        # drain_rollouts() + rr.log() happen here, outside the timed window.
        viz.log_overlay(
            obs,
            clip_percentile=CLIP_PERCENTILE,
            suppress_outliers=SUPPRESS_OUTLIERS,
            gamma=GAMMA,
            colormap=COLORMAP,
        )

        step = frame_idx + 1
        if step % 10 == 0:
            print(f"  frame {step}/{episode_len}")

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, interval - elapsed))

print("Done. Check the rerun viewer under attention/<cam>/.")
