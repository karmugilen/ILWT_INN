#!/usr/bin/env python3
"""
Recover the hidden secret from a stego image and write a PNG.

Usage:
    python scripts/extract_image.py --stego stego.png --output recovered.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _common import default_model_path, load_pipeline_model, select_device
from shell.io import load_8bit_image, save_8bit_png


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stego", required=True, help="Stego image with hidden data")
    p.add_argument("--output", required=True, help="Output recovered secret path (.png)")
    p.add_argument("--model", default=str(default_model_path()), help="Trained model .pth")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--num_blocks", type=int, default=8)
    p.add_argument("--hidden_channels", type=int, default=128)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model, device, _ = load_pipeline_model(
        args.model, args.num_blocks, args.hidden_channels, args.size, device
    )
    print(f"Loaded model from {args.model} on {device}")

    stego = load_8bit_image(Path(args.stego), args.size).unsqueeze(0).to(device).float()
    stego_like = torch.cat([stego, torch.zeros_like(stego)], dim=1)

    with torch.no_grad():
        reconstructed = model.inverse(stego_like)
        recovered_secret = reconstructed[:, 3:]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_8bit_png(recovered_secret[0], out_path)
    print(f"Wrote recovered secret: {out_path}")


if __name__ == "__main__":
    main()
