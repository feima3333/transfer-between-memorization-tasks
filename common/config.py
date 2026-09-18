"""Single source of truth for paths and shared constants.

Every path in this repo derives from here, so a fresh clone runs with no path edits.
Each of the three roots defaults to a folder next to this package and can be overridden
by an environment variable of the same name (e.g. ``DATA_DIR=/data/cifar``).
"""
import os
from pathlib import Path

# Repo root = the directory that holds this ``common/`` package.
REPO_ROOT = Path(__file__).resolve().parent.parent


def _dir(env_var: str, default: str) -> Path:
    override = os.environ.get(env_var)
    return Path(override) if override else REPO_ROOT / default


# Training writes model weights (.pth) here -- large, local only, not committed.
CHECKPOINT_DIR = _dir("CHECKPOINT_DIR", "checkpoints")
# metrics.json + cache CSVs the figures read -- committed, so figures reproduce without retraining.
FIGURE_DATA_DIR = _dir("FIGURE_DATA_DIR", "figure_data")
# Figure notebooks save PNGs here.
OUTPUT_DIR = _dir("OUTPUT_DIR", "outputs")
# CIFAR-10 is auto-downloaded here by torchvision.
DATA_DIR = _dir("DATA_DIR", "data")

# ---- shared model / data constants (the paper's config; see README) ----
NUM_CLASSES = 10
WIDTHS = (128, 256, 512)  # the three conv-block widths of WideCNN_BN

# CIFAR-10 channel statistics used for normalization everywhere.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
