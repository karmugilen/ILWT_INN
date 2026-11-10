"""Shared helpers for the user-facing CLI scripts in this folder."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow scripts to be run from any cwd: scripts/ is a sibling of core/ and shell/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from core.config import ModelConfig
from core.models import create_model
from shell.io import load_model_weights


def select_device(prefer: str = "auto") -> torch.device:
    """Pick a torch device. 'auto' uses CUDA when available."""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" or (prefer == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def load_pipeline_model(
    model_path: Path | str,
    num_blocks: int = 8,
    hidden_channels: int = 128,
    size: int = 256,
    device: torch.device | None = None,
):
    """Load the trained StegoWaveFlow model from a .pth file."""
    device = device or select_device()
    cfg = ModelConfig(
        path=str(model_path),
        num_blocks=num_blocks,
        hidden_channels=hidden_channels,
        size=size,
    )
    model = create_model(cfg)
    load_model_weights(model, Path(model_path), device)
    model = model.to(device).eval()
    return model, device, cfg


def default_model_path() -> Path:
    """The model file shipped at the repo root."""
    return REPO_ROOT / "ilwt_steganography_model_research.pth"
