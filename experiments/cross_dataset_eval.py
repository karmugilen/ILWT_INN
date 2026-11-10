#!/usr/bin/env python
"""
Cross-dataset secrets.

Scores stego frame sets written by make_stego_set.py --secret-dir <dataset>. Each set holds
all frames of the 20 Xiph sequences, frame i carrying secret[i % N] of that dataset. Use the standard
hide images (in-domain, the training distribution) as the reference, and DIV2K validation
(never seen in training) as the cross-dataset test.

Per frame: H_PSNR, H_SSIM (cover vs 8-bit stego), then R_PSNR, SSIM, NC, BER, E_XAI (secret vs blind
recovery from the 8-bit stego). Summary: mean, SD and 95% CI across the 20 sequences, per set.

Usage: cross_dataset_eval.py --sets name=set_dir=secret_dir [name=set_dir=secret_dir ...] --out DIR
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out", default=str(E.ROOT / "runs/cross_dataset"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model, device = E.load_model()
    rows = []
    for spec in args.sets:
        name, set_dir, secret_dir = spec.split("=")
        set_dir, secret_dir = Path(set_dir), Path(secret_dir)
        for seq, frames in E.read_frame_table(set_dir).items():
            covers = np.stack([E.read_rgb(set_dir / "cover" / seq / f"{f}.png") for f, _ in frames])
            stegos = np.stack([E.read_rgb(set_dir / "stego" / seq / f"{f}.png") for f, _ in frames])
            recs = E.recover(model, stegos, device)
            for (f, s), c, st, r in zip(frames, covers, stegos, recs):
                sec = E.load_secret_u8(secret_dir / s)
                m = E.frame_metrics(sec, r, device)
                rows.append({"secrets": name, "seq": seq, "frame": f, "secret": s,
                             "H_PSNR": round(E.psnr_u8(c, st), 4), "H_SSIM": round(E.ssim_u8(c, st, device), 6),
                             **{k: round(v, 6) for k, v in m.items()}, "band": E.band(m["NC"])})
            cur = [x for x in rows if x["secrets"] == name and x["seq"] == seq]
            print(f"{name:8s} {seq:13s} {len(cur):4d} fr  H {np.mean([x['H_PSNR'] for x in cur]):.2f}  "
                  f"R {np.mean([x['R_PSNR'] for x in cur]):.2f}  BER {np.mean([x['BER'] for x in cur]):.4f}", flush=True)
            E.write_csv(out / "per_frame.csv", rows)

    summ = E.summarize(rows, ["secrets"], ("H_PSNR", "H_SSIM", "R_PSNR", "SSIM", "NC", "BER", "E_XAI"))
    E.write_csv(out / "summary.csv", summ)
    print("\nsecrets   H-PSNR  H-SSIM   R-PSNR (95% CI)              SSIM    NC      BER")
    for r in summ:
        print(f"{r['secrets']:8s} {r['H_PSNR']:7.2f}  {r['H_SSIM']:.4f}  {r['R_PSNR']:6.2f} {r['R_PSNR_CI95']:20s}  "
              f"{r['SSIM']:.4f}  {r['NC']:.4f}  {r['BER']:.4f}")


if __name__ == "__main__":
    main()
