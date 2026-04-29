"""Evaluate SmolVLA with Real-Time Chunking AND attention overlays.

Mirrors `smolvla/evaluate.py` but wraps the policy with a
`VisionAttentionCapture` on the SigLIP vision encoder. After every chunk
request we log an attention-rollout heatmap per camera to rerun, so you can
watch where the vision encoder is focused while the policy drives the arm.

This is a read-only extension — toggling the capture off (set
ATTENTION_ENABLED = False) gives you the same control loop as smolvla/evaluate.py.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import rerun as rr
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

from lerobot_attention_visualizer import (
    VisionAttentionCapture,
    log_attention_overlay,
    patch_heatmap_to_image,
    rollout_to_patch_heatmap,
)

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

# SigLIP encoder inside the VLM — this is where we hook Q/K projections.
vision_model = policy.model.vlm_with_expert.get_vlm_model().vision_model
capture = VisionAttentionCapture(vision_model)
# Holds the original embed_image bound method so we can restore it on shutdown.
_orig_embed_image = None


def _camera_keys_in_policy_order() -> list[str]:
    """Return the raw camera keys (e.g. "top", "wrist.top") in the same order
    `prepare_images` feeds them into the vision encoder."""
    prefix = "observation.images."
    keys = []
    for k in policy.config.image_features:
        if k.startswith(prefix):
            keys.append(k[len(prefix) :])
    return keys


def _log_attention_for_obs(obs: dict) -> None:
    """Consume capture.rollouts (one entry per camera) and overlay on the raw
    camera images. Must be called exactly once per predict_action_chunk."""
    # Camera order here matches the order embed_image was called during forward.
    camera_keys = _camera_keys_in_policy_order()
    if len(capture.rollouts) != len(camera_keys):
        # Fewer rollouts than cameras — usually means a hook was missed. Skip
        # this frame rather than misalign the overlay.
        capture.rollouts.clear()
        return

    for cam_key, rollout in zip(camera_keys, capture.rollouts, strict=True):
        # Raw obs dict uses the bare camera name ("top"), not the feature key.
        image = obs.get(cam_key)
        if image is None:
            continue
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            continue

        # (seq, seq) rollout → (patch_h, patch_w) per-patch importance.
        patch_heat = rollout_to_patch_heatmap(rollout)
        # Upsample the patch grid to the full camera resolution for overlay.
        heat = patch_heatmap_to_image(patch_heat, target_hw=image.shape[:2])
        log_attention_overlay(f"attention/{cam_key}", image, heat)

    capture.rollouts.clear()


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

        # Each embed_image call inside this forward triggers capture.snapshot().
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

        # Overlay the captured rollouts on the raw camera frames via rerun.
        if ATTENTION_ENABLED:
            _log_attention_for_obs(obs)


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
    if ATTENTION_ENABLED:
        # Install q_proj / k_proj forward hooks on every SigLIP attention layer.
        capture.__enter__()

        # `embed_image` is a method on SmolVLMWithExpertModel, not an nn.Module
        # submodule, so forward_hook doesn't apply — monkey-patch the bound
        # method. After each call (one per camera per chunk) we snapshot the
        # rollout and reset the Q/K cache.
        _orig_embed_image = policy.model.vlm_with_expert.embed_image

        def _patched_embed_image(image):
            # Run the original encoder forward, then freeze its attention rollout.
            out = _orig_embed_image(image)
            capture.snapshot()
            return out

        policy.model.vlm_with_expert.embed_image = _patched_embed_image

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
    if ATTENTION_ENABLED and _orig_embed_image is not None:
        policy.model.vlm_with_expert.embed_image = _orig_embed_image
        capture.__exit__(None, None, None)
    follower.disconnect()
    log_say("Evaluation finished")
