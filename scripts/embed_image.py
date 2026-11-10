#!/usr/bin/env python3
"""
Embed one secret image into one cover image and write a stego PNG.

Usage:
    python scripts/embed_image.py --cover cover.png --secret secret.png --output stego.png

The stego.png is an 8-bit RGB PNG that can be shared or stored in any normal
image pipeline. Run scripts/extract_image.py on it to recover the secret.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _common import default_model_path, load_pipeline_model, select_device
from shell.io import load_image, save_8bit_png


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cover", required=True, help="Cover image (any common format)")
    p.add_argument("--secret", required=True, help="Secret image to hide")
    p.add_argument("--output", required=True, help="Output stego image path (.png)")
    p.add_argument("--model", default=str(default_model_path()), help="Trained model .pth")
    p.add_argument("--size", type=int, default=256, help="Square size for processing (default 256)")
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

    cover = load_image(Path(args.cover), args.size).unsqueeze(0).to(device)
    secret = load_image(Path(args.secret), args.size).unsqueeze(0).to(device)
    inputs = torch.cat([cover, secret], dim=1)

    with torch.no_grad():
        stego_output, _ = model(inputs)
        stego_host = stego_output[:, :3]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_8bit_png(stego_host[0], out_path)
    print(f"Wrote stego image: {out_path}")


if __name__ == "__main__":
    main()
