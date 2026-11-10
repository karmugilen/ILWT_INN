#!/usr/bin/env python
"""
native-resolution test.

The same deployed model (fully convolutional, trained at 256x256) is run on every frame of the
20 Xiph sequences in two conditions, with the same frames and the same secrets (frame i -> hide
secret[i % 300], standard protocol):
  256     cover: ffmpeg lanczos 256x256 (the standard pipeline); secret: shell.io.load_image at 256x256
  native  cover: the decoded frame at its native size, no resize (CIF 352x288, QCIF 176x144);
          secret: the same torchvision Resize, to the cover's H x W
The stego is rounded to 8 bits and the secret is recovered blindly from it.
BPP stays 24 (one RGB secret pixel per cover pixel); bits per frame scale with H x W.

Outputs (--out): per_frame.csv, summary.csv (per condition and per resolution class:
mean, SD, 95% CI across sequences), per_seq.csv.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as E  # noqa: E402
from make_stego_set import SEQUENCES  # noqa: E402

QCIF = {s for s, rel in SEQUENCES.items() if "qcif" in rel}


def decode(path, scale):
    w, h = map(int, subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=width,height", "-of", "csv=p=0", str(path)],
                                   check=True, capture_output=True, text=True).stdout.strip().split(","))
    vf = ["-vf", "scale=256:256:flags=lanczos"] if scale else []
    if scale:
        w = h = 256
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), *vf, "-pix_fmt", "rgb24",
                          "-f", "rawvideo", "-"], check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def secret_u8(path, h, w):
    t = T.Compose([T.Resize((h, w)), T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])(Image.open(path).convert("RGB"))
    return E.t_to_u8(t)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--secret-dir", required=True, help="hide folder (sorted; first 300 used)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=1, help="use every k-th frame")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model, device = E.load_model()
    secrets = sorted(Path(args.secret_dir).glob("*.png"))[:300]
    rows = []
    for seq, rel in SEQUENCES.items():
        for cond in ("256", "native"):
            covers = decode(Path(args.videos) / rel, scale=(cond == "256"))[::args.every]
            idx = np.arange(0, len(covers) * args.every, args.every)
            h, w = covers.shape[1:3]
            secs = np.stack([secret_u8(secrets[i % len(secrets)], h, w) for i in idx])
            stego = []
            with torch.no_grad():
                for b in range(0, len(covers), args.batch):
                    x = torch.cat([E.u8_to_t(covers[b:b + args.batch], device), E.u8_to_t(secs[b:b + args.batch], device)], 1)
                    stego.append(E.t_to_u8(model(x)[0][:, :3]))
            stego = np.concatenate(stego)
            recs = E.recover(model, stego, device, bs=args.batch)
            for i, c, st, s, r in zip(idx, covers, stego, secs, recs):
                m = E.frame_metrics(s, r, device)
                rows.append({"cond": cond, "res": f"{w}x{h}", "seq": seq, "frame": int(i) + 1,
                             "H_PSNR": round(E.psnr_u8(c, st), 4), "H_SSIM": round(E.ssim_u8(c, st, device), 6),
                             **{k: round(v, 6) for k, v in m.items()}})
            cur = [x for x in rows if x["cond"] == cond and x["seq"] == seq]
            print(f"{seq:13s} {cond:6s} {w}x{h} {len(cur):4d} fr  H {np.mean([x['H_PSNR'] for x in cur]):.2f}  "
                  f"R {np.mean([x['R_PSNR'] for x in cur]):.2f}  NC {np.mean([x['NC'] for x in cur]):.4f}  "
                  f"BER {np.mean([x['BER'] for x in cur]):.4f}", flush=True)
        E.write_csv(out / "per_frame.csv", rows)
    met = ("H_PSNR", "H_SSIM", "R_PSNR", "SSIM", "NC", "BER", "E_XAI")
    summ = E.summarize([{**r, "group": r["cond"]} for r in rows], ["group"], met) + \
        E.summarize([{**r, "group": f"{r['cond']} ({'CIF' if r['seq'] not in QCIF else 'QCIF'} seq)"} for r in rows], ["group"], met)
    E.write_csv(out / "summary.csv", summ)
    seqs = list(SEQUENCES)
    E.write_csv(out / "per_seq.csv", [{"seq": s, "cond": c, **{k: round(np.mean([x[k] for x in rows if x["seq"] == s and x["cond"] == c]), 4) for k in met}}
                                      for s in seqs for c in ("256", "native")])
    print("\ngroup             n_seq  H-PSNR  H-SSIM  R-PSNR (95% CI)              R-SSIM  NC      BER")
    for r in summ:
        print(f"{r['group']:16s} {r['n_seq']:5d}  {r['H_PSNR']:6.2f}  {r['H_SSIM']:.4f}  {r['R_PSNR']:6.2f} {r['R_PSNR_CI95']:20s}  "
              f"{r['SSIM']:.4f}  {r['NC']:.4f}  {r['BER']:.4f}")


if __name__ == "__main__":
    main()
