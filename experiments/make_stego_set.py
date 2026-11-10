#!/usr/bin/env python
"""
Build the 24-BPP cover/stego frame set from the 20 Xiph sequences with the trained
ILWT-INN model, and measure per-frame recovery from the stored 8-bit stego.

Outputs (under --out):
  cover/<seq>/frame_00001.png     8-bit cover frame (256x256, same ffmpeg resize as the standard pipeline)
  stego/<seq>/frame_00001.png     8-bit stego frame (what is stored on disk)
  per_frame.csv                   seq, frame, secret, H-PSNR, R-PSNR, BER(all bits), BER bit0..bit7
  bitplane_summary.csv            mean BER per bit plane, mean BER, BSC capacity per plane
Used by: steganalysis, bit-plane BER, ECC payload.

Secrets (default): the "test" list of splits/imagenet_split_seed42.json, assigned
deterministically (frame k of sequence j gets test[(j*1000 + k) % N]).
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(os.path.expanduser("~/ilwt_workspace"))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))
from _common import load_pipeline_model  # noqa: E402
from shell.io import load_image  # noqa: E402

SEQUENCES = {
    "Akiyo": "Television/akiyo_cif.y4m", "Bus": "MotorVehicles/bus_cif.y4m",
    "Carphone": "Foreman/carphone_qcif.y4m", "City": "city_cif.y4m",
    "Coastguard": "MotorVehicles/coastguard_cif.y4m", "Container": "MotorVehicles/container_cif.y4m",
    "Crew": "crew_cif.y4m", "Foreman": "Foreman/foreman_cif.y4m",
    "Hall Monitor": "hall_monitor_cif.y4m", "Hall Objects": "hall_objects_qcif.y4m",
    "Harbour": "MotorVehicles/harbour_cif.y4m", "Ice": "ice_cif.y4m",
    "News": "Television/news_cif.y4m", "Pamphlet": "pamphlet_cif.y4m",
    "Salesman": "salesman_qcif.y4m", "Silent": "silent_cif.y4m",
    "Soccer": "playing/soccer_cif.y4m", "Suzie": "suzie_qcif.y4m",
    "Trevor": "trevor_qcif.y4m", "Vtc1Nw": "vtc1nw_422_cif.y4m",
}


def h2(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def to_u8(t):
    """[-1,1] CHW tensor -> uint8 HWC RGB, same rounding as shell.io.save_8bit_png."""
    x = ((t.clamp(-1, 1) + 1) / 2).clamp(0, 1)
    return (x.detach().cpu().numpy() * 255.0).round().astype(np.uint8).transpose(1, 2, 0)


def from_u8(a, device):
    return torch.from_numpy(a.transpose(2, 0, 1).copy()).float().div(255).mul(2).sub(1).to(device)


def psnr_u8(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(ROOT / "data/sd_videos"))
    ap.add_argument("--images", default=str(ROOT / "data/imagenet256"))
    ap.add_argument("--split", default=str(ROOT / "splits/imagenet_split_seed42.json"))
    ap.add_argument("--model", default=str(ROOT / "code/ilwt_steganography_model_research.pth"))
    ap.add_argument("--out", default=str(ROOT / "runs/stego_set_24bpp"))
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames per sequence")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--secret-dir", default="",
                    help="standard protocol: sorted *.png in this dir, frame i of every sequence gets secret[i % N]")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, device, _ = load_pipeline_model(args.model, 8, 128, 256)
    if args.secret_dir:
        secret_root = Path(args.secret_dir)
        secrets = [p.name for p in sorted(secret_root.glob("*.png"))]
        pick = lambda j, k: secrets[k % len(secrets)]                      # noqa: E731  (standard pipeline)
    else:
        secret_root = Path(args.images)
        secrets = json.load(open(args.split))["test"]
        pick = lambda j, k: secrets[(j * 1000 + k) % len(secrets)]         # noqa: E731
    print(f"model on {device}; {len(secrets)} secrets from {secret_root}")

    plane_err, plane_n = np.zeros(8), 0
    csv_f = open(out / "per_frame.csv", "w", newline="")
    w = csv.writer(csv_f)
    w.writerow(["seq", "frame", "secret", "H_PSNR", "R_PSNR", "BER_all"] + [f"BER_bit{b}" for b in range(8)])

    for j, (seq, rel) in enumerate(SEQUENCES.items()):
        cdir, sdir = out / "cover" / seq, out / "stego" / seq
        cdir.mkdir(parents=True, exist_ok=True); sdir.mkdir(parents=True, exist_ok=True)
        # same frame extraction as shell.io.extract_frames (ffmpeg lanczos scale to 256x256, rgb24)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(Path(args.videos) / rel),
                        "-vf", "scale=256:256:flags=lanczos", "-pix_fmt", "rgb24", str(cdir / "frame_%05d.png")],
                       check=True)
        frames = sorted(cdir.glob("frame_*.png"))
        if args.max_frames:
            for f in frames[args.max_frames:]:
                f.unlink()
            frames = frames[:args.max_frames]
        seq_h, seq_r, seq_b = [], [], []
        for b0 in range(0, len(frames), args.batch):
            batch = frames[b0:b0 + args.batch]
            covers_u8 = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in batch]
            names = [pick(j, b0 + i) for i in range(len(batch))]
            covers = torch.stack([from_u8(c, device) for c in covers_u8])
            secs = torch.stack([load_image(secret_root / n, 256) for n in names]).to(device)
            with torch.no_grad():
                stego_out, _ = model(torch.cat([covers, secs], 1))
                stego_u8 = [to_u8(s) for s in stego_out[:, :3]]
                stego_t = torch.stack([from_u8(s, device) for s in stego_u8])      # 8-bit round trip
                rec = model.inverse(torch.cat([stego_t, torch.zeros_like(stego_t)], 1))[:, 3:]
            for i, p in enumerate(batch):
                cv2.imwrite(str(sdir / p.name), cv2.cvtColor(stego_u8[i], cv2.COLOR_RGB2BGR))
                sec_u8, rec_u8 = to_u8(secs[i]), to_u8(rec[i])
                diff = np.bitwise_xor(sec_u8, rec_u8)
                per_plane = np.array([((diff >> b) & 1).mean() for b in range(8)])
                plane_err += per_plane; plane_n += 1
                hp, rp, ber = psnr_u8(covers_u8[i], stego_u8[i]), psnr_u8(sec_u8, rec_u8), per_plane.mean()
                seq_h.append(hp); seq_r.append(rp); seq_b.append(ber)
                w.writerow([seq, p.stem, names[i], f"{hp:.4f}", f"{rp:.4f}", f"{ber:.6f}"] + [f"{x:.6f}" for x in per_plane])
        csv_f.flush()
        print(f"{seq:13s} frames {len(frames):4d}  H-PSNR {np.mean(seq_h):.2f}  R-PSNR {np.mean(seq_r):.2f}  "
              f"BER {np.mean(seq_b):.4f}", flush=True)
    csv_f.close()

    plane = plane_err / plane_n
    with open(out / "bitplane_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bit_plane", "mean_BER", "BSC_capacity_bits_per_bit"])
        for b in range(8):
            w.writerow([b, f"{plane[b]:.6f}", f"{1 - h2(plane[b]):.6f}"])
        cap = sum(3 * (1 - h2(p)) for p in plane)   # 3 colour channels x 8 planes
        w.writerow(["all", f"{plane.mean():.6f}", f"total BSC capacity {cap:.3f} BPP of 24"])
    print("bit-plane BER (bit0=LSB):", " ".join(f"{p:.4f}" for p in plane))
    print(f"mean BER {plane.mean():.4f}; BSC capacity {cap:.2f} of 24 BPP; frames {plane_n}")


if __name__ == "__main__":
    main()
