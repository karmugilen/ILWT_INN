#!/usr/bin/env python3
"""
Embed secret images into a cover video and write a stego video (FFV1/MKV).

Usage:
    python scripts/embed_video.py --video cover.mp4 --hide_dir secrets/ --output stego.mkv

If --hide_dir contains fewer images than the video has frames, the images are
cycled. The output is a real 8-bit FFV1/MKV file (lossless, plays in standard
tools). Recover the secrets with scripts/extract_video.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from _common import default_model_path, load_pipeline_model, select_device
from shell.io import (
    extract_frames,
    frames_to_video,
    get_video_fps,
    list_images_in_dir,
    load_image,
    save_8bit_png,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, help="Cover video (any format ffmpeg can read)")
    p.add_argument("--hide_dir", required=True, help="Directory of secret images to embed")
    p.add_argument("--output", required=True, help="Output stego video path (.mkv)")
    p.add_argument("--workdir", default=None,
                   help="Scratch directory for intermediate frames (default: alongside --output)")
    p.add_argument("--model", default=str(default_model_path()), help="Trained model .pth")
    p.add_argument("--size", type=int, default=256, help="Frame size (square)")
    p.add_argument("--num_blocks", type=int, default=8)
    p.add_argument("--hidden_channels", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8, help="Frames per GPU batch")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--keep_frames", action="store_true", help="Don't delete intermediate frame dirs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model, device, _ = load_pipeline_model(
        args.model, args.num_blocks, args.hidden_channels, args.size, device
    )
    print(f"Loaded model from {args.model} on {device}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir) if args.workdir else out_path.parent / f".{out_path.stem}_work"
    cover_frames_dir = workdir / "cover_frames"
    stego_frames_dir = workdir / "stego_frames"

    video_path = Path(args.video)
    hide_dir = Path(args.hide_dir)
    print(f"Extracting frames from {video_path} -> {cover_frames_dir}")
    cover_frames = extract_frames(video_path, cover_frames_dir, args.size)
    fps = get_video_fps(video_path)

    hide_files = list_images_in_dir(hide_dir, "*.png") or list_images_in_dir(hide_dir, "*.jpg")
    if not hide_files:
        raise SystemExit(f"No images found in {hide_dir}")
    print(f"Found {len(cover_frames)} cover frames, {len(hide_files)} secret images, fps={fps:.2f}")

    stego_frames_dir.mkdir(parents=True, exist_ok=True)
    bs = args.batch_size
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(cover_frames), bs), desc="Embedding"):
            batch = cover_frames[batch_start : batch_start + bs]
            covers = torch.stack([load_image(p, args.size) for p in batch]).to(device)
            secrets = torch.stack(
                [load_image(hide_files[(batch_start + i) % len(hide_files)], args.size)
                 for i in range(len(batch))]
            ).to(device)
            stego_output, _ = model(torch.cat([covers, secrets], dim=1))
            stego_host = stego_output[:, :3]
            for i, frame in enumerate(stego_host):
                save_8bit_png(frame, stego_frames_dir / f"frame_{batch_start + i + 1:05d}.png")

    print(f"Encoding stego video -> {out_path}")
    frames_to_video(stego_frames_dir, out_path, fps)
    print(f"Done. Stego video: {out_path}")

    if not args.keep_frames:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
        print(f"Cleaned up workdir: {workdir}")


if __name__ == "__main__":
    main()
