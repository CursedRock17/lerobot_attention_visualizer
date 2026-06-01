from importlib.metadata import PackageNotFoundError, version

from .policies.act import ACTAttention, ACTBackboneCapture
from .policies.groot import GR00TAttention, GR00TN1d6Attention
from .policies.pi0 import Pi05Attention, Pi0Attention, Pi0FastAttention
from .policies.smolvla import SmolVLAAttention, VisionAttentionCapture
from .visualizer.overlay import (
    log_attention_overlay,
    patch_heatmap_to_image,
    rollout_to_patch_heatmap,
)

try:
    __version__ = version("lerobot-attention-visualizer")
except PackageNotFoundError:
    # Running from a source checkout without an installed dist (e.g. CI tests).
    __version__ = "0.0.0+unknown"

__all__ = [
    "ACTAttention",
    "ACTBackboneCapture",
    "GR00TAttention",
    "GR00TN1d6Attention",
    "Pi05Attention",
    "Pi0Attention",
    "Pi0FastAttention",
    "SmolVLAAttention",
    "VisionAttentionCapture",
    "__version__",
    "log_attention_overlay",
    "patch_heatmap_to_image",
    "rollout_to_patch_heatmap",
]
