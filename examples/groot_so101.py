"""Evaluate Groot N1.5 on a live SO-101 AND stream Eagle-2 attention overlays to rerun.

Mirrors NVIDIA's Groot eval loop but wraps the policy with `GR00TAttention`,
so the SigLIP vision encoder's per-camera attention rollout streams to rerun
alongside the raw image while the policy drives the arm.

Unlike SmolVLA (which uses RTC), Groot manages its own internal action queue
via `select_action`. The queue refills automatically when empty by calling
`predict_action_chunk` under the hood — that forward triggers our Eagle
vision-encoder hooks. Between refills the last overlay stays visible in rerun
and `log_overlay` is a no-op.

Toggle `ATTENTION_ENABLED = False` to run a plain eval loop identical to
NVIDIA's reference script, with zero overhead from the capture.

Usage
-----
Edit the constants below to match your setup, then:
    conda activate lav
    python examples/groot_so101.py
"""

from __future__ import annotations

import contextlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.configs import ColorMode
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

from lerobot_attention_visualizer import GR00TAttention

# ---------------------------------------------------------------------------
# Edit these to match your setup
# ---------------------------------------------------------------------------
# Base model weights. This is NVIDIA's pretrained checkpoint, not a lerobot
# GrootPolicy checkpoint — the policy is constructed fresh from GrootConfig.
BASE_MODEL_PATH  = "nvidia/GR00T-N1.5-3B"
device           = torch.device("cuda")

# Task
TASK_DESCRIPTION = "Pick up the object and place it in the target location"
NUM_EPISODES     = 5
EPISODE_TIME_SEC = 30
ROBOT_TYPE       = "so101_follower"

# Attention visualization — set False to run plain eval with zero capture overhead.
ATTENTION_ENABLED = True
# False = full attention rollout (smoother, recommended). True = raw last-layer
# self-attention, dominated by SigLIP attention-sink / register patches (splotchy).
LAST_LAYER_ONLY   = False
CLIP_PERCENTILE   = 95.0   # suppress SigLIP edge artifacts

# Control rate — Groot refills its action queue every n_action_steps control steps,
# which is when the Eagle forward fires and the overlay updates.
CONTROL_FPS  = 10
CAMERA_FPS   = 30
FRAME_W, FRAME_H = 640, 480

# torch.compile — compiles Groot's flow-matching action head for faster inference
# on the second+ call (first call is a ~30s warmup). Safe to enable alongside
# attention capture; we hook q_proj/k_proj before compile sees the graph.
# Set False if you hit shape-related compile errors.
TORCH_COMPILE  = False
EMBODIMENT_TAG = "new_embodiment"   # matches GR00T's SO-100/101 embodiment
# ---------------------------------------------------------------------------

follower_config = SO101FollowerConfig(
    port="/dev/ttyACM1",
    id="SO101",
    cameras={
        # Adjust camera names, indices, and serials to match your rig.
        "ego": OpenCVCameraConfig(
            index_or_path=0,
            width=FRAME_W,
            height=FRAME_H,
            fps=CAMERA_FPS,
            fourcc="MJPG",
        ),
        "external": RealSenseCameraConfig(
            serial_number_or_name="",   # fill in your RealSense serial
            width=FRAME_W,
            height=FRAME_H,
            fps=CAMERA_FPS,
            use_depth=False,
            color_mode=ColorMode.RGB,
        ),
    },
)
follower = SO101Follower(follower_config)

# Build GrootConfig from the robot's feature schema and load base weights.
# nvidia/GR00T-N1.5-3B is NVIDIA's pretrained checkpoint — not a lerobot
# GrootPolicy checkpoint. Calling from_pretrained on it would fail because
# draccus can't parse NVIDIA's native config as GrootConfig.
action_features  = hw_to_dataset_features(follower.action_features, "action")
obs_features     = hw_to_dataset_features(follower.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

_action_dim = next(v["shape"][0] for k, v in action_features.items())
_state_dim  = next(
    v["shape"][0] for k, v in obs_features.items()
    if "state" in k and "image" not in k
)
_cam_keys   = [k.removeprefix("observation.images.") for k in obs_features if "observation.images." in k]

_groot_config = GrootConfig(
    input_features={
        **{"observation.images." + cam: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
           for cam in _cam_keys},
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

if TORCH_COMPILE:
    # Compile only the flow-matching action head — the vision backbone uses
    # dynamic shapes (variable token counts) that would break fullgraph compile.
    # Our q_proj/k_proj hooks on the vision encoder are unaffected.
    policy._groot_model.action_head = torch.compile(
        policy._groot_model.action_head, mode="reduce-overhead"
    )

preprocessor, postprocessor = make_groot_pre_post_processors(
    config=policy.config,
    dataset_stats=None,
)

_, events = init_keyboard_listener()
init_rerun(session_name="groot_so101_attention")

shutdown_event = threading.Event()

# GR00TAttention hooks Eagle-2's SigLIP q_proj/k_proj. On each queue refill
# (predict_action_chunk fires forward_eagle once for all cameras), snapshot_split
# slices the batched Q/K into per-camera entries. log_overlay drains them after
# the action is dispatched, so the capture adds zero latency to the control loop.
viz = GR00TAttention(policy, last_layer_only=LAST_LAYER_ONLY) \
    if ATTENTION_ENABLED else contextlib.nullcontext()

# Background thread so rerun serialization doesn't stall the control loop.
_log_executor = ThreadPoolExecutor(max_workers=1) if ATTENTION_ENABLED else None


def control_loop():
    """Single loop: observe → infer → act → log attention, at CONTROL_FPS."""
    interval = 1.0 / CONTROL_FPS
    while not shutdown_event.is_set():
        t0 = time.perf_counter()

        # Read cameras + joint state from the robot.
        obs = follower.get_observation()

        # Package into the tensor dict Groot's preprocessor expects.
        obs_frame = build_inference_frame(
            observation=obs,
            ds_features=dataset_features,
            device=device,
            task=TASK_DESCRIPTION,
            robot_type=ROBOT_TYPE,
        )
        # Eagle VLM tokenization, normalization, device move.
        batch = preprocessor(obs_frame)

        # select_action pops from Groot's internal queue. When the queue is
        # empty it calls predict_action_chunk → forward_eagle fires, hooks
        # capture Q/K. Otherwise this is a cheap dequeue with no forward pass.
        with torch.inference_mode():
            action = policy.select_action(batch)

        # Unnormalize from [-1, 1] to real joint-angle space.
        action = postprocessor(action.unsqueeze(0)).squeeze(0)

        # Send to arm.
        action_dict = make_robot_action(action.unsqueeze(0), dataset_features)
        follower.send_action(action_dict)

        if ATTENTION_ENABLED:
            # Log overlay off the control thread so rerun serialization
            # doesn't eat into the next control step budget.
            _log_executor.submit(
                viz.log_overlay, dict(obs), clip_percentile=CLIP_PERCENTILE
            )

        dt = time.perf_counter() - t0
        time.sleep(max(0.0, interval - dt))


follower.connect()
log_say(f"Robot connected. attention={'on' if ATTENTION_ENABLED else 'off'}")

try:
    with viz:
        control_thread = threading.Thread(target=control_loop, daemon=True)
        control_thread.start()

        for episode_idx in range(NUM_EPISODES):
            if events["stop_recording"]:
                break
            log_say(f"Eval episode {episode_idx + 1}/{NUM_EPISODES}")

            episode_end = time.perf_counter() + EPISODE_TIME_SEC
            while time.perf_counter() < episode_end:
                if events["stop_recording"]:
                    break
                time.sleep(0.05)

except KeyboardInterrupt:
    log_say("Interrupted by user")

finally:
    shutdown_event.set()
    try:
        control_thread.join(timeout=5.0)
    except NameError:
        pass
    if _log_executor is not None:
        _log_executor.shutdown(wait=True)
    follower.disconnect()
    log_say("Evaluation finished")
