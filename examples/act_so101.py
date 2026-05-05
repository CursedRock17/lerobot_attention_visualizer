"""Evaluate ACT with ResNet activation-heatmap overlays logged to rerun.

Mirrors `act/evaluate.py` but drops `record_loop` for a hand-rolled control
loop so we can wrap the policy with `ACTAttention` and stream overlays
without fighting record_loop's own rerun display. Every time ACT's internal
action queue refills, the backbone runs once per camera; `viz.log_overlay`
drains those feature maps, reduces to per-spatial-cell activation magnitude,
and logs alongside the raw camera frames.

Toggle `ATTENTION_ENABLED = False` to run the same loop without the capture
for an A/B comparison.
"""

from __future__ import annotations

import contextlib
import time

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.configs import ColorMode
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

from lerobot_attention_visualizer import ACTAttention

# Configuration
HF_USER = "CursedRock17"
DATASET_NAME = "so101_block_grab_act"
POLICY_PATH = f"{HF_USER}/{DATASET_NAME}_act_0"
device = torch.device("cuda")

# Evaluation
TASK_DESCRIPTION = "Grab a block, place it in the bin"
NUM_EPISODES = 5
EPISODE_TIME_SEC = 20
ROBOT_TYPE = "so101_follower"

# Flip off to run the same control loop without the backbone capture.
ATTENTION_ENABLED = True

# ACT refills its internal queue every `n_action_steps`, so the overlay refreshes
# at that cadence, not every control step. 10Hz is plenty for a laptop GPU.
FPS = 10
FRAME_W, FRAME_H = 640, 480

follower_config = SO101FollowerConfig(
    port="/dev/ttyACM1",
    id="Raven",
    cameras={
        "top": OpenCVCameraConfig(
            index_or_path="/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_4AF3193F-video-index0",
            width=FRAME_W,
            height=FRAME_H,
            fps=FPS,
            fourcc="MJPG",
        ),
        "wrist.top": RealSenseCameraConfig(
            serial_number_or_name="353322271691",
            width=FRAME_W,
            height=FRAME_H,
            fps=FPS,
            use_depth=False,
            color_mode=ColorMode.RGB,
        ),
    },
)
follower = SO101Follower(follower_config)

policy = ACTPolicy.from_pretrained(POLICY_PATH)
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
init_rerun(session_name="act_eval_viz")

# Wraps the policy: hooks `policy.model.backbone`. log_overlay only fires on
# steps where ACT's internal queue refilled (every `n_action_steps`).
viz = ACTAttention(policy) if ATTENTION_ENABLED else contextlib.nullcontext()

follower.connect()
log_say(f"Robot connected. attention={'on' if ATTENTION_ENABLED else 'off'}")

# Seconds per control step — sleep the remainder so we send at a steady rate.
interval = 1.0 / FPS

try:
    with viz:
        for episode_idx in range(NUM_EPISODES):
            if events["stop_recording"]:
                break
            log_say(f"Eval episode {episode_idx + 1}/{NUM_EPISODES}")
            # Clear ACT's internal action queue so the next step triggers a fresh forward.
            policy.reset()

            episode_end = time.perf_counter() + EPISODE_TIME_SEC
            while time.perf_counter() < episode_end:
                if events["stop_recording"]:
                    break
                t0 = time.perf_counter()

                # Grab latest camera frames + joint state from the follower.
                obs = follower.get_observation()
                # Assemble the dict ACT expects (normalized tensors keyed by feature name).
                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_features,
                    device=device,
                    task=TASK_DESCRIPTION,
                    robot_type=ROBOT_TYPE,
                )
                obs_frame = preprocessor(obs_frame)

                # select_action only runs the model when its internal queue is empty —
                # on those steps the backbone hook fires and viz.log_overlay sees fresh maps.
                action = policy.select_action(obs_frame)
                action_post = postprocessor(action).squeeze(0)
                # Package the 6-DoF action into the robot's expected dict schema.
                action_dict = make_robot_action(action_post.unsqueeze(0), dataset_features)
                follower.send_action(action_dict)

                if ATTENTION_ENABLED:
                    viz.log_overlay(obs)

                # Sleep the remainder of the control period so we hold a steady rate.
                dt = time.perf_counter() - t0
                time.sleep(max(0.0, interval - dt))

except KeyboardInterrupt:
    log_say("Interrupted by user")

finally:
    follower.disconnect()
    log_say("Evaluation finished")
