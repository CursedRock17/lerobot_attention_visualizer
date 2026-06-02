import torch
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot_attention_visualizer import GR00TAttention
from lerobot_attention_visualizer.visualizer.overlay import rollout_to_patch_heatmap

DATASET_REPO_ID = "sreetz-nv/so101_teleop_vials_rack_left_sim_and_real"
EPISODE_IDX = 79

dataset = LeRobotDataset(DATASET_REPO_ID, episodes=[EPISODE_IDX], revision="main")
cam_prefix = "observation.images."
camera_keys = [k[len(cam_prefix):] for k in dataset.features if k.startswith(cam_prefix)]
print("cameras:", camera_keys)

_state_dim = dataset.features["observation.state"]["shape"][0]
_action_dim = dataset.features["action"]["shape"][0]

config = GrootConfig(
    input_features={
        **{f"{cam_prefix}{cam}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
           for cam in camera_keys},
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(_state_dim,)),
    },
    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(_action_dim,))},
    base_model_path="nvidia/GR00T-N1.5-3B",
    embodiment_tag="new_embodiment",
)
policy = GrootPolicy(config).cuda().eval()

preprocessor, _ = make_groot_pre_post_processors(config=config, dataset_stats=None)
frame = dataset[0]
batch = {k: v.unsqueeze(0).cuda() for k, v in frame.items() if isinstance(v, torch.Tensor)}
batch["task"] = ["Pick up the vial and place it in the rack"]
batch = preprocessor(batch)

# --- run with GR00TAttention, compare last_layer_only vs full rollout ---
for mode, last_only in [("last_layer_only", True), ("full_rollout", False)]:
    viz = GR00TAttention(policy, last_layer_only=last_only)
    with viz:
        with torch.inference_mode():
            policy.predict_action_chunk(batch)
        rollouts = viz._capture.drain_rollouts(last_layer_only=last_only)

    print(f"\n=== {mode} ===")
    for i, (cam, rollout) in enumerate(zip(camera_keys, rollouts)):
        # Column mean — what the heatmap shows
        r = rollout[0]  # (256, 256)
        col_mean = r.mean(dim=0).cpu().numpy()  # (256,) importance per patch

        pcts = np.percentile(col_mean, [0, 10, 25, 50, 75, 90, 95, 99, 100])
        print(f"  cam={cam}")
        print(f"    col_mean: min={pcts[0]:.5f}  p50={pcts[3]:.5f}  p90={pcts[5]:.5f}  p95={pcts[6]:.5f}  max={pcts[8]:.5f}")
        print(f"    col_mean std={col_mean.std():.5f}  range={pcts[8]-pcts[0]:.5f}")
        print(f"    top-3 patch indices: {col_mean.argsort()[-3:][::-1].tolist()}")

        # What the heatmap grid looks like (16x16)
        grid = col_mean.reshape(16, 16)
        print(f"    heatmap grid (16x16) min={grid.min():.5f} max={grid.max():.5f}")
        # Print as ASCII for quick visual check
        lo, hi = grid.min(), grid.max()
        chars = " .:-=+*#@"
        print("    grid:")
        for row in grid:
            print("      " + "".join(chars[int((v - lo) / (hi - lo + 1e-8) * (len(chars)-1))] for v in row))

        # Per-head attention check (does any head show useful signal?)
        with GR00TAttention(policy, last_layer_only=last_only) as viz2:
            with torch.inference_mode():
                policy.predict_action_chunk(batch)
            caches = viz2._capture._pending[i] if len(viz2._capture._pending) > i else None
        if caches:
            last = caches[-1]
            if last.q is not None and last.k is not None:
                q = last.q.float()
                k = last.k.float()
                b, s, d = q.shape
                nh, hd = last.num_heads, last.head_dim
                q = q.view(b, s, nh, hd).transpose(1, 2)
                k = k.view(b, s, nh, hd).transpose(1, 2)
                attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * hd**-0.5, dim=-1)
                head_entropy = -(attn * (attn + 1e-8).log()).sum(-1).mean(-1)[0]
                print(f"    per-head entropy (last layer, cam={cam}): min={head_entropy.min():.3f} max={head_entropy.max():.3f}")
                print(f"      most focused head: {head_entropy.argmin().item()} (entropy={head_entropy.min():.3f})")
