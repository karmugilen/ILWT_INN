#!/usr/bin/env python
"""
Attack robustness on all 20 Xiph sequences.

The original two-cover atlases (Akiyo, Silent) used 12 attacks at one strength each.
Here the same 12 attacks run at 3 strengths (the original value included) on --frames-per-seq
evenly spaced frames of every sequence. Stego frames come from the standard-protocol set
written by make_stego_set.py (8-bit stego PNGs; standard secrets), so the "none" row reproduces
the clean recovery.

Outputs (--out):
  per_frame.csv   seq, frame, attack, level, attack_PSNR (attacked vs clean stego), R_PSNR, SSIM, NC, BER, E_XAI, band
  summary.csv     per attack/level: mean, SD and 95% CI across the 20 sequences, band fractions
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as E  # noqa: E402

sys.path.insert(0, str(E.ROOT / "code" / "Dec-18-E-and-E-V3"))
import steg_attacks as A  # noqa: E402  (original attack implementations; BGR uint8 in/out)


def sharpen(img, s):
    """3x3 Laplacian sharpening with strength s (s = 1 is the original steg_attacks.laplacian_sharpen)."""
    lap = cv2.filter2D(img.astype(np.float32), -1, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32))
    return np.clip(img.astype(np.float32) - s * lap, 0, 255).astype(np.uint8)


def watermark(img, alpha):
    """Visible text watermark; alpha = 0.3 is the original steg_attacks.watermark."""
    out = img.copy()
    cv2.putText(out, "COPYRIGHT", (img.shape[1] // 4, img.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5 * img.shape[1] / 352, (255, 255, 255), 3, cv2.LINE_AA)
    return cv2.addWeighted(img, 1 - alpha, out, alpha, 0)


# name -> (function(img_bgr, level), levels). Original strengths: Akiyo atlas (uniform +-5, sharpen 3x3,
# crop 15%, translate 10 px, watermark, saturation 1.2) and Silent atlas (Gaussian sigma 2, motion blur 5,
# contrast 1.1, rotation 0.5 deg, brightness +30, histogram equalisation).
ATTACKS = {
    "none":           (lambda im, v: im, [0]),
    "gaussian_noise": (lambda im, v: A.add_noise(im, v / 255.0), [1, 2, 4]),
    "uniform_noise":  (A.uniform_noise, [2, 5, 10]),
    "sharpen":        (sharpen, [0.25, 0.5, 1.0]),
    "motion_blur":    (A.motion_blur, [3, 5, 7]),
    "contrast":       (A.contrast, [1.05, 1.1, 1.2]),
    "brightness":     (A.brightness, [10, 30, 50]),
    "saturation":     (A.saturation, [1.1, 1.2, 1.5]),
    "hist_eq":        (lambda im, v: A.hist_eq(im), [1]),
    "rotation_deg":   (A.rotate, [0.25, 0.5, 1.0]),
    "translate_px":   (A.translate, [5, 10, 20]),
    "crop_pct":       (A.crop, [0.05, 0.15, 0.25]),
    "watermark":      (watermark, [0.15, 0.3, 0.5]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=str(E.ROOT / "runs/stego_hide_current_may28"))
    ap.add_argument("--secret-dir", default=str(E.ROOT / "data/hide300"))
    ap.add_argument("--frames-per-seq", type=int, default=50)
    ap.add_argument("--out", default=str(E.ROOT / "runs/attacks_20seq"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model, device = E.load_model()
    table = E.read_frame_table(args.set)
    rows = []
    for seq, frames in table.items():
        pick = np.unique(np.linspace(0, len(frames) - 1, args.frames_per_seq).round().astype(int))
        chosen = [frames[i] for i in pick]
        stego_bgr = [cv2.imread(str(Path(args.set) / "stego" / seq / f"{f}.png")) for f, _ in chosen]
        secrets = [E.load_secret_u8(Path(args.secret_dir) / s) for _, s in chosen]
        for name, (fn, levels) in ATTACKS.items():
            for lv in levels:
                np.random.seed(args.seed)
                attacked_bgr = [fn(im, lv) for im in stego_bgr]
                attacked_rgb = np.stack([cv2.cvtColor(a, cv2.COLOR_BGR2RGB) for a in attacked_bgr])
                recs = E.recover(model, attacked_rgb, device)
                for (f, _), a, s, r, clean in zip(chosen, attacked_bgr, secrets, recs, stego_bgr):
                    m = E.frame_metrics(s, r, device)
                    rows.append({"seq": seq, "frame": f, "attack": name, "level": lv,
                                 "attack_PSNR": round(E.psnr_u8(a, clean), 3),
                                 **{k: round(v, 6) for k, v in m.items()}, "band": E.band(m["NC"])})
        done = [r for r in rows if r["seq"] == seq and r["attack"] == "none"]
        print(f"{seq:13s} {len(chosen)} frames  clean R-PSNR {np.mean([r['R_PSNR'] for r in done]):.2f}  "
              f"NC {np.mean([r['NC'] for r in done]):.4f}", flush=True)
        E.write_csv(out / "per_frame.csv", rows)

    summ = E.summarize(rows, ["attack", "level"], ("R_PSNR", "SSIM", "NC", "BER", "E_XAI", "attack_PSNR"))
    E.write_csv(out / "summary.csv", summ)
    print("\nattack          level   R-PSNR   NC (95% CI)                 BER     robust stress critical")
    for r in summ:
        print(f"{r['attack']:15s} {str(r['level']):6s} {r['R_PSNR']:7.2f}  {r['NC']:.4f} {r['NC_CI95']:22s} "
              f"{r['BER']:.4f}  {r['frac_robust']:.2f}   {r['frac_stress']:.2f}   {r['frac_critical']:.2f}")


if __name__ == "__main__":
    main()
