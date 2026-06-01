"""Replay a recorded LeRobotDataset and stream Groot N1.6 attention overlays to rerun.

No robot hardware required — loads camera frames directly from the dataset and
feeds them through the policy exactly as the live eval loop would, then logs
image / heatmap / overlay per camera to rerun.

Groot uses Eagle-2's SigLIP vision encoder. The attention visualizer hooks the
same q_proj / k_proj layers as SmolVLA, but all cameras are batched into a
single vision-model forward so `GR00TAttention` splits the resulting Q/K
tensors by camera index rather than snapshotting between per-camera calls.

Usage
-----
Prerequisites — install Groot's dependencies FIRST (see Install section in
README.md), then edit the constants below and run:

    conda activate lav
    python examples/visualize_groot_dataset.py

Rerun must be running or will be launched automatically. Open the rerun viewer
and navigate to the `attention/` tree to see per-camera overlays.

Finding a dataset
-----------------
Use any LeRobotDataset whose camera names match what your Groot checkpoint was
trained on. NVIDIA's GR00T-N1.5-3B was trained on SO101 data; community
fine-tunes are available at:
    https://huggingface.co/models?other=base_model:finetune:nvidia/GR00T-N1.5-3B

The `input_features` remapping below handles any camera-name mismatch
automatically, so DATASET_REPO_ID and POLICY_PATH can come from different repos
as long as the number of cameras matches.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
from lerobot.utils.visualization_utils import init_rerun

from lerobot_attention_visualizer import GR00TAttention

# Ensure ~/.local/bin is in PATH so rr.spawn() can find the rerun viewer binary
# even when running inside a conda environment that doesn't inherit it.
_local_bin = str(Path.home() / ".local" / "bin")
if _local_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _local_bin + os.pathsep + os.environ.get("PATH", "")

# ---------------------------------------------------------------------------
# Edit these to match your setup
# ---------------------------------------------------------------------------
POLICY_PATH      = "nvidia/GR00T-N1.5-3B"  # Hub repo or local path
DATASET_REPO_ID  = "aravindhs-NV/so100-orig-groot-vials-rack-left-cosmos-70"                       # e.g. "your-username/your-groot-dataset"
TASK_DESCRIPTION = "pick up the vial and place in the yellow rack"                       # must match the task used during training

EPISODE_IDX      = 0       # which episode to replay
LAST_LAYER_ONLY  = True    # True = crisper maps; False = full attention rollout
PLAYBACK_FPS     = 10      # pacing; set to None to run as fast as possible
CLIP_PERCENTILE  = 95.0    # suppress SigLIP edge artifacts (lower = more clipping)
# ---------------------------------------------------------------------------

if not DATASET_REPO_ID:
    raise ValueError(
        "Set DATASET_REPO_ID to a LeRobotDataset hub repo trained with Groot. "
        "See the module docstring for guidance on finding a compatible dataset."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load one episode so the download is small.
dataset = LeRobotDataset(DATASET_REPO_ID, episodes=[EPISODE_IDX])

episode_len = len(dataset)
print(f"Episode {EPISODE_IDX}: {episode_len} frames at {dataset.fps} Hz")

# Camera keys as bare names ("front", "wrist") — matches log_overlay's obs dict.
cam_prefix = "observation.images."
camera_keys = [
    k[len(cam_prefix):]
    for k in dataset.features
    if k.startswith(cam_prefix)
]
print(f"Cameras: {camera_keys}")

policy = GrootPolicy.from_pretrained(POLICY_PATH)
policy.to(device).eval()

# Remap image_features to match the actual dataset camera names.
# Groot checkpoints ship with placeholder keys; prepare_images does a dict
# lookup so keys must match the actual batch. We patch input_features
# (the source that image_features derives from) instead of the read-only property.
_vis_feature = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
_non_visual = {
    k: v for k, v in policy.config.input_features.items()
    if v.type is not FeatureType.VISUAL
}
policy.config.input_features = {
    **_non_visual,
    **{f"{cam_prefix}{cam}": _vis_feature for cam in camera_keys},
}

preprocessor, _ = make_groot_pre_post_processors(
    config=policy.config,
    dataset_stats=None,
)

init_rerun(session_name="groot_dataset_attention")

viz = GR00TAttention(policy, last_layer_only=LAST_LAYER_ONLY)
interval = (1.0 / PLAYBACK_FPS) if PLAYBACK_FPS else 0.0

with viz:
    for frame_idx in range(episode_len):
        t0 = time.perf_counter()

        frame = dataset[frame_idx]

        # Build obs dict for log_overlay: bare cam name → HWC uint8 numpy.
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

        # Groot preprocessor: normalise → Eagle-encode → eagle_* keys.
        batch = preprocessor(batch)

        # Forward pass — forward_eagle fires once for all cameras, snapshot_split
        # slices the (N_cameras, n_patches, dim) Q/K batch into per-camera entries.
        with torch.inference_mode():
            policy.predict_action_chunk(batch)

        # drain_rollouts() + rr.log() happen here, outside the timed window.
        viz.log_overlay(obs, clip_percentile=CLIP_PERCENTILE)

        step = frame_idx + 1
        if step % 10 == 0:
            print(f"  frame {step}/{episode_len}")

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, interval - elapsed))

print("Done. Check the rerun viewer under attention/<cam>/.")
