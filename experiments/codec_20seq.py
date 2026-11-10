#!/usr/bin/env python
"""
Lossy video re-encoding of the stego videos.

Every 20-sequence stego video (8-bit stego frames, standard protocol) is re-encoded with each codec
setting below, decoded back to 8-bit RGB, and the secrets are extracted blindly.

Settings:
  ffv1            FFV1 lossless, gbrp (the standard storage format)        - reference
  x264rgb_crf0    H.264 lossless in RGB (libx264rgb -crf 0)
  x264_444_crf0   H.264 "lossless" after RGB->YUV 4:4:4 conversion (rounding only)
  x264_crf18/23/28  H.264 yuv420p, preset medium (typical distribution settings; 23 = x264 default)
  x265_crf18/23/28  H.265 yuv420p, preset medium (28 = x265 default)

Outputs (--out):
  per_frame.csv   seq, frame, codec, stego_PSNR_after_codec, R_PSNR, SSIM, NC, BER, E_XAI, band
  per_video.csv   seq, codec, file size (bytes), bitrate (kbit/s), frames
  summary.csv     per codec: mean, SD, 95% CI across sequences, band fractions,
                  failure rate = fraction of frames with NC < 0.85 (critical band)
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as E  # noqa: E402

FPS = 30
CODECS = {
    "ffv1":          ["-c:v", "ffv1", "-level", "3", "-coder", "1", "-context", "1", "-g", "1",
                      "-slices", "24", "-slicecrc", "1", "-pix_fmt", "gbrp"],
    "x264rgb_crf0":  ["-c:v", "libx264rgb", "-preset", "medium", "-crf", "0", "-pix_fmt", "rgb24"],
    "x264_444_crf0": ["-c:v", "libx264", "-preset", "medium", "-crf", "0", "-pix_fmt", "yuv444p"],
    "x264_crf18":    ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"],
    "x264_crf23":    ["-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p"],
    "x264_crf28":    ["-c:v", "libx264", "-preset", "medium", "-crf", "28", "-pix_fmt", "yuv420p"],
    "x265_crf18":    ["-c:v", "libx265", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-x265-params", "log-level=error"],
    "x265_crf23":    ["-c:v", "libx265", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-x265-params", "log-level=error"],
    "x265_crf28":    ["-c:v", "libx265", "-preset", "medium", "-crf", "28", "-pix_fmt", "yuv420p", "-x265-params", "log-level=error"],
}


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=str(E.ROOT / "runs/stego_hide_current_may28"))
    ap.add_argument("--secret-dir", default=str(E.ROOT / "data/hide300"))
    ap.add_argument("--out", default=str(E.ROOT / "runs/codec_20seq"))
    ap.add_argument("--codecs", default=",".join(CODECS))
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames (for quick tests)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model, device = E.load_model()
    table = E.read_frame_table(args.set)
    rows, vids = [], []
    for seq, frames in table.items():
        if args.max_frames:
            frames = frames[:args.max_frames]
        secrets = [E.load_secret_u8(Path(args.secret_dir) / s) for _, s in frames]
        clean = [E.read_rgb(Path(args.set) / "stego" / seq / f"{f}.png") for f, _ in frames]
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            tmp = Path(tmp)
            src = tmp / "src"; src.mkdir()
            for k, (f, _) in enumerate(frames):        # contiguous numbering for ffmpeg
                os.symlink(Path(args.set) / "stego" / seq / f"{f}.png", src / f"f_{k + 1:05d}.png")
            for codec in args.codecs.split(","):
                vid = tmp / f"{codec}.mkv"
                run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(src / "f_%05d.png"),
                     *CODECS[codec], str(vid)])
                size = vid.stat().st_size
                dec = tmp / f"dec_{codec}"; dec.mkdir()
                run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(vid), "-pix_fmt", "rgb24", str(dec / "d_%05d.png")])
                decoded = np.stack([E.read_rgb(p) for p in sorted(dec.glob("d_*.png"))])
                assert len(decoded) == len(frames), f"{seq}/{codec}: decoded {len(decoded)} of {len(frames)} frames"
                recs = E.recover(model, decoded, device)
                for (f, _), s, r, d, c in zip(frames, secrets, recs, decoded, clean):
                    m = E.frame_metrics(s, r, device)
                    rows.append({"seq": seq, "frame": f, "codec": codec,
                                 "stego_PSNR_after_codec": round(E.psnr_u8(d, c), 3),
                                 **{k: round(v, 6) for k, v in m.items()}, "band": E.band(m["NC"])})
                vids.append({"seq": seq, "codec": codec, "bytes": size, "frames": len(frames),
                             "kbps": round(size * 8 / 1000 / (len(frames) / FPS), 1)})
                shutil.rmtree(dec)
                cur = [r for r in rows if r["seq"] == seq and r["codec"] == codec]
                print(f"{seq:13s} {codec:14s} {size/1e6:7.2f} MB  R-PSNR {np.mean([r['R_PSNR'] for r in cur]):6.2f}  "
                      f"NC {np.mean([r['NC'] for r in cur]):.4f}  BER {np.mean([r['BER'] for r in cur]):.4f}", flush=True)
        E.write_csv(out / "per_frame.csv", rows)
        E.write_csv(out / "per_video.csv", vids)

    summ = E.summarize(rows, ["codec"], ("R_PSNR", "SSIM", "NC", "BER", "E_XAI", "stego_PSNR_after_codec"))
    kb = {c: np.mean([v["kbps"] for v in vids if v["codec"] == c]) for c in CODECS}
    for r in summ:
        r["mean_kbps"] = round(kb[r["codec"]], 1)
        r["failure_rate_NC_lt_0.85"] = r["frac_critical"]
    E.write_csv(out / "summary.csv", summ)
    print("\ncodec           kbps     R-PSNR  SSIM    NC      BER     failure(NC<0.85)")
    for r in summ:
        print(f"{r['codec']:14s} {r['mean_kbps']:8.1f}  {r['R_PSNR']:6.2f}  {r['SSIM']:.4f}  {r['NC']:.4f}  "
              f"{r['BER']:.4f}  {r['frac_critical']:.3f}")


if __name__ == "__main__":
    main()
