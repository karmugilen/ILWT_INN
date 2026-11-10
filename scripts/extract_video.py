#!/usr/bin/env python3
"""
Recover hidden secrets from a stego video. Writes one PNG per frame.

Usage:
    python scripts/extract_video.py --stego stego.mkv --output_dir recovered/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from _common import default_model_path, load_pipeline_model, select_device
from shell.io import extract_frames, load_8bit_image, save_8bit_png


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stego", required=True, help="Stego video (any format ffmpeg can read)")
    p.add_argument("--output_dir", required=True, help="Directory to write recovered PNGs")
    p.add_argument("--workdir", default=None,
                   help="Scratch directory for extracted frames (default: alongside output_dir)")
    p.add_argument("--model", default=str(default_model_path()), help="Trained model .pth")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--num_blocks", type=int, default=8)
    p.add_argument("--hidden_channels", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--keep_frames", action="store_true", help="Don't delete intermediate frame dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model, device, _ = load_pipeline_model(
        args.model, args.num_blocks, args.hidden_channels, args.size, device
    )
    print(f"Loaded model from {args.model} on {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir) if args.workdir else out_dir.parent / f".{out_dir.name}_work"
    extracted_dir = workdir / "extracted_frames"

    stego_path = Path(args.stego)
    print(f"Extracting frames from {stego_path} -> {extracted_dir}")
    stego_frames = extract_frames(stego_path, extracted_dir, args.size)
    print(f"Found {len(stego_frames)} stego frames")

    bs = args.batch_size
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(stego_frames), bs), desc="Recovering"):
            batch = stego_frames[batch_start : batch_start + bs]
            stego = torch.stack([load_8bit_image(p, args.size) for p in batch]).to(device).float()
            stego_like = torch.cat([stego, torch.zeros_like(stego)], dim=1)
            reconstructed = model.inverse(stego_like)
            recovered_secrets = reconstructed[:, 3:]
            for i, frame in enumerate(recovered_secrets):
                save_8bit_png(frame, out_dir / f"recovered_{batch_start + i + 1:05d}.png")

    print(f"Done. {len(stego_frames)} recovered images in {out_dir}")
    if not args.keep_frames:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
        print(f"Cleaned up workdir: {workdir}")


if __name__ == "__main__":
    main()
