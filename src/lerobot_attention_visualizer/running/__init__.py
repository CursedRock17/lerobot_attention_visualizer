"""Live eval loops with attention overlays — one module per policy.

Each module connects to a real robot at import time and runs an eval loop, so
they're meant to be invoked as scripts (e.g. `python -m
lerobot_attention_visualizer.running.smolvla`), not imported from library code.
"""
