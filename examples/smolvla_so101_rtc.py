"""Evaluate SmolVLA with Real-Time Chunking AND attention overlays.

Mirrors `smolvla/evaluate.py` but wraps the policy with `SmolVLAAttention`,
so the SigLIP vision encoder's per-camera attention rollout streams to rerun
alongside the raw image while the policy drives the arm.

This is a read-only extension — toggling the capture off (set
ATTENTION_ENABLED = False) gives you the same control loop as smolvla/evaluate.py.
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.configs import ColorMode
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

from lerobot_attention_visualizer import SmolVLAAttention

# Configuration
HF_USER = "CursedRock17"
DATASET_NAME = "so101_block_grab"
POLICY_PATH = f"{HF_USER}/{DATASET_NAME}_smolvla"
device = torch.device("cuda")

# Evaluation
TASK_DESCRIPTION = "Grab a block, place it in the bin"
NUM_EPISODES = 5
EPISODE_TIME_SEC = 20
ROBOT_TYPE = "so101_follower"

# RTC
RTC_ENABLED = True
EXECUTION_HORIZON = 10
QUEUE_THRESHOLD = 15
MAX_GUIDANCE_WEIGHT = 10.0

# Attention visualization — flip off to run a plain RTC loop.
ATTENTION_ENABLED = True

# Rates: cameras stream at CAMERA_FPS, the arm is driven at CONTROL_FPS.
# Capture adds a matmul+softmax per ViT layer per chunk on top of the VLM
# forward, so lower control rate keeps the laptop GPU comfortable.
CAMERA_FPS = 30
CONTROL_FPS = 10
FRAME_W, FRAME_H = 640, 480

follower_config = SO101FollowerConfig(
    port="/dev/ttyACM1",
    id="Raven",
    cameras={
        "top": OpenCVCameraConfig(
            index_or_path="/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_4AF3193F-video-index0",
            width=FRAME_W,
            height=FRAME_H,
            fps=CAMERA_FPS,
            fourcc="MJPG",
        ),
        "wrist.top": RealSenseCameraConfig(
            serial_number_or_name="353322271691",
            width=FRAME_W,
            height=FRAME_H,
            fps=CAMERA_FPS,
            use_depth=False,
            color_mode=ColorMode.RGB,
        ),
    },
)
follower = SO101Follower(follower_config)

policy = SmolVLAPolicy.from_pretrained(POLICY_PATH)
policy.config.rtc_config = RTCConfig(
    enabled=RTC_ENABLED,
    execution_horizon=EXECUTION_HORIZON,
    max_guidance_weight=MAX_GUIDANCE_WEIGHT,
)
policy.init_rtc_processor()
policy.to(device)
policy.eval()

action_features = hw_to_dataset_features(follower.action_features, "action")
obs_features = hw_to_dataset_features(follower.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=POLICY_PATH,
    dataset_stats=None,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

_, events = init_keyboard_listener()
init_rerun(session_name="smolvla_eval_rtc_viz")

# Queue holds pending actions for the actor thread to consume.
action_queue = ActionQueue(policy.config.rtc_config)
# Rolling window of recent inference latencies — RTC uses max() to pick inference_delay.
latency_tracker = LatencyTracker(maxlen=20)
# Seconds per action at control rate — used to translate latency into a step count.
time_per_chunk = 1.0 / CONTROL_FPS
shutdown_event = threading.Event()

# Wraps the policy: hooks SigLIP Q/K and freezes a snapshot per embed_image call
# inside predict_action_chunk (zero GPU compute on the hot path). log_overlay
# computes rollouts + streams to rerun AFTER merge() has fired.
viz = SmolVLAAttention(policy) if ATTENTION_ENABLED else contextlib.nullcontext()

# Single-worker pool so rerun serialization runs off the inference thread.
# max_workers=1 keeps frames in order and bounds memory (one frame queued at a time).
_log_executor = ThreadPoolExecutor(max_workers=1) if ATTENTION_ENABLED else None


def get_actions_loop():
    """Producer: refill the action queue via RTC, then log per-camera attention."""
    while not shutdown_event.is_set():
        # With RTC off, we only refill when the queue is fully empty.
        threshold = QUEUE_THRESHOLD if RTC_ENABLED else 0
        # Skip if queue still has enough buffered actions.
        if action_queue.qsize() > threshold:
            time.sleep(0.01)
            continue

        # Snapshot queue state before inference so RTC can blend correctly.
        idx_before = action_queue.get_action_index()
        prev_actions = action_queue.get_left_over()
        # Predict how many control steps inference will take, based on recent runs.
        inference_delay = math.ceil((latency_tracker.max() or 0.0) / time_per_chunk)

        t0 = time.perf_counter()
        # Grab latest camera frames + joint state from the follower.
        obs = follower.get_observation()
        # Assemble the dict SmolVLA expects (tokenized task, normalized tensors).
        obs_frame = build_inference_frame(
            observation=obs,
            ds_features=dataset_features,
            device=device,
            task=TASK_DESCRIPTION,
            robot_type=ROBOT_TYPE,
        )
        obs_frame = preprocessor(obs_frame)

        # Each embed_image call inside this forward triggers viz to snapshot a rollout.
        actions = policy.predict_action_chunk(
            obs_frame,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_actions,
        )
        # Keep an unpost-processed copy for the RTC prefix next iteration.
        original_actions = actions.squeeze(0).clone()
        post_actions = postprocessor(actions).squeeze(0)

        # Record real latency and translate into steps for the next inference_delay.
        real_latency = time.perf_counter() - t0
        latency_tracker.add(real_latency)
        real_delay = math.ceil(real_latency / time_per_chunk)
        # RTC blends the new chunk into the queue at the correct offset.
        action_queue.merge(original_actions, post_actions, real_delay, idx_before)

        if ATTENTION_ENABLED:
            # drain_rollouts() + rr.log() happen in the background thread so
            # rerun serialization (~100ms for two 640×480 cameras) doesn't stall
            # the next inference iteration. Pass a copy of obs so the inference
            # thread can immediately overwrite it on the next get_observation().
            _log_executor.submit(viz.log_overlay, dict(obs))


def actor_loop():
    """Consumer: pop one action at a time at CONTROL_FPS and send it to the robot."""
    # Target wall-clock budget per action (0.1s at 10Hz).
    interval = 1.0 / CONTROL_FPS
    while not shutdown_event.is_set():
        t0 = time.perf_counter()
        # Pop one action from the queue; may be None if producer is still warming up.
        action = action_queue.get()
        if action is not None:
            # Package the 6-DoF action into the robot's expected dict schema.
            action_dict = make_robot_action(action.unsqueeze(0), dataset_features)
            follower.send_action(action_dict)
        # Sleep the remainder of the control period so we send at a steady rate.
        dt = time.perf_counter() - t0
        time.sleep(max(0.0, interval - dt))


follower.connect()
log_say(
    f"Robot connected. RTC={'on' if RTC_ENABLED else 'off'} "
    f"attention={'on' if ATTENTION_ENABLED else 'off'}"
)

try:
    with viz:
        inference_thread = threading.Thread(target=get_actions_loop, daemon=True)
        control_thread = threading.Thread(target=actor_loop, daemon=True)
        inference_thread.start()
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
        inference_thread.join(timeout=5.0)
        control_thread.join(timeout=5.0)
    except NameError:
        pass
    if _log_executor is not None:
        _log_executor.shutdown(wait=True)
    follower.disconnect()
    log_say("Evaluation finished")
