"""Replay a recorded LeRobotDataset and stream Groot N1.6 attention overlays to rerun.

No robot hardware required — loads camera frames directly from the dataset and
feeds them through the policy exactly as the live eval loop would, then logs
image / heatmap / overlay per camera to rerun.

Two overlays are streamed per camera (navigate to each in the rerun viewer):

  attention/<cam>/encoder/*   SigLIP vision-encoder self-attention rollout —
                              what the (frozen) image encoder finds salient.
  attention/<cam>/action/*    action-head cross-attention — which vision tokens
                              the flow-matching denoiser actually attends to
                              while producing the action. THIS is the signal
                              that moves when you fine-tune; watch it for
                              grounding bugs. (Set cross_attention=False on
                              GR00TAttention to stream only the encoder view.)

Groot uses Eagle-2's SigLIP vision encoder. The encoder overlay hooks the same
q_proj / k_proj layers as SmolVLA, but all cameras are batched into a single
vision-model forward so `GR00TAttention` splits the resulting Q/K tensors by
camera index. The action overlay hooks the cross-attending DiT blocks in the
action head — see docs/groot_adapter.md.

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
from lerobot.policies.groot.configuration_groot import GrootConfig
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
# Base model weights to load. This is NVIDIA's pretrained checkpoint — NOT a
# lerobot GrootPolicy checkpoint. GrootPolicy is constructed fresh from a
# GrootConfig built from the dataset's features; it then downloads these weights.
BASE_MODEL_PATH  = "nvidia/GR00T-N1.5-3B"
DATASET_REPO_ID  = ""       # e.g. "sreetz-nv/so101_teleop_vials_rack_left_sim_and_left"
TASK_DESCRIPTION = ""       # must match the task description used during recording

EMBODIMENT_TAG   = "new_embodiment"  # matches GR00T's SO-100/101 embodiment
EPISODE_IDX      = 0       # which episode to replay
# False = full attention rollout (smoother, recommended). True = raw last-layer
# self-attention, which is dominated by SigLIP attention-sink / register patches
# and reads as splotchy noise — only useful for inspecting a single layer.
LAST_LAYER_ONLY  = False
PLAYBACK_FPS     = 10      # pacing; set to None to run as fast as possible
CLIP_PERCENTILE  = 95.0    # suppress SigLIP edge artifacts (lower = more clipping)
# Opt-in artifact suppression: winsorize SigLIP attention-sink / register spikes
# (median + 6·MAD) so they don't read as splotches. Flip to True and re-run to
# A/B against the default in the rerun viewer. Mainly affects the encoder view.
SUPPRESS_OUTLIERS = False
# Display contrast. 1.0 = linear. >1 makes the map more "extreme" — crushes
# background noise toward black and isolates the bright hot spot (try 2–4).
GAMMA = 1.0
# Heatmap palette: "hot" (red→yellow→white), "blue-green", or "viridis".
COLORMAP = "hot"
# ---------------------------------------------------------------------------

if not DATASET_REPO_ID:
    raise ValueError(
        "Set DATASET_REPO_ID to a LeRobotDataset hub repo. "
        "See the module docstring for guidance on finding a compatible dataset."
    )
if not TASK_DESCRIPTION:
    raise ValueError("Set TASK_DESCRIPTION to the task string used when recording the dataset.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load one episode so the download is small.
# revision="main" bypasses the Hub version-tag check — some community datasets
# are not tagged with a codebase version even though their info.json is valid v3.0.
dataset = LeRobotDataset(DATASET_REPO_ID, episodes=[EPISODE_IDX], revision="main")

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

# Build GrootConfig from the dataset's actual features.
# nvidia/GR00T-N1.5-3B is the base weights, not a lerobot GrootPolicy checkpoint —
# GrootPolicy.from_pretrained() would fail trying to parse NVIDIA's native config
# through draccus. Instead we construct GrootConfig explicitly and GrootPolicy()
# calls GR00TN15.from_pretrained(BASE_MODEL_PATH) internally.
_state_feat = dataset.features.get("observation.state")
_state_dim = _state_feat["shape"][0] if _state_feat else 6
_action_feat = dataset.features.get("action")
_action_dim = _action_feat["shape"][0] if _action_feat else 6

_groot_config = GrootConfig(
    input_features={
        **{f"{cam_prefix}{cam}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
           for cam in camera_keys},
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(_state_dim,)),
    },
    output_features={
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(_action_dim,)),
    },
    base_model_path=BASE_MODEL_PATH,
    embodiment_tag=EMBODIMENT_TAG,
)
policy = GrootPolicy(_groot_config)
policy.to(device).eval()

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

print("Done. Check the rerun viewer under attention/<cam>/encoder/ and /action/.")
