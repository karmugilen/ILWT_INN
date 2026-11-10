"""Shared helpers for the evaluation scripts (attacks, codecs, cross-dataset).

All metrics are computed on 8-bit images, the same way as the standard pipeline:
  R-PSNR  PSNR(secret, recovered) on the 0..255 scale
  SSIM    same maths as train.calculate_ssim (11x11 Gaussian, sigma 1.5, C1=0.01^2, C2=0.03^2, [0,1]),
          but the window is built with numpy: CPU torch matmul segfaults in a container that routes BLAS through NVBLAS
  NC      normalised cross-correlation on [0,1] pixels (core/metrics.py definition)
  BER     all-bit-plane bit error rate between the 8-bit secret and the 8-bit recovery
  E_XAI   mean |S - S_hat| / 255  (Eq. for the XAI damage energy)
Robustness bands: NC >= 0.95 robust, 0.85 <= NC < 0.95 stress, NC < 0.85 critical.
"""
import csv
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(os.path.expanduser("~/ilwt_workspace"))
CODE = Path(os.environ.get("ILWT_CODE", ROOT / "code"))   # or set ILWT_CODE=<path to this repository>
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "scripts"))
from _common import load_pipeline_model  # noqa: E402
from shell.io import load_image  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MODEL = CODE / "ilwt_steganography_model_research.pth"


def load_model(path=MODEL):
    model, device, _ = load_pipeline_model(str(path), 8, 128, 256)
    return model, device


def u8_to_t(rgb_u8, device):
    """uint8 HWC RGB (or NHWC) -> float [-1,1] (N)CHW tensor."""
    a = np.asarray(rgb_u8)
    if a.ndim == 3:
        a = a[None]
    t = torch.from_numpy(np.ascontiguousarray(a.transpose(0, 3, 1, 2))).float().div(255).mul(2).sub(1)
    return t.to(device)


def t_to_u8(t):
    """[-1,1] (N)CHW tensor -> uint8 NHWC RGB, same rounding as shell.io.save_8bit_png."""
    x = ((t.clamp(-1, 1) + 1) / 2).clamp(0, 1)
    if x.ndim == 3:
        x = x[None]
    return (x.detach().cpu().numpy() * 255.0).round().astype(np.uint8).transpose(0, 2, 3, 1)


@torch.no_grad()
def recover(model, stego_rgb_u8, device, bs=32):
    """Blind extraction from 8-bit stego frames (N,H,W,3 RGB uint8) -> recovered uint8 RGB."""
    out = []
    for i in range(0, len(stego_rgb_u8), bs):
        s = u8_to_t(stego_rgb_u8[i:i + bs], device)
        rec = model.inverse(torch.cat([s, torch.zeros_like(s)], 1))[:, 3:]
        out.append(t_to_u8(rec))
    return np.concatenate(out)


def read_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def load_secret_u8(path):
    """Secret exactly as the model sees it (shell.io.load_image: resize to 256x256), as uint8 RGB."""
    return t_to_u8(load_image(Path(path), 256))[0]


def psnr_u8(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def nc_u8(a, b):
    x = a.astype(np.float64).ravel() / 255.0
    y = b.astype(np.float64).ravel() / 255.0
    x -= x.mean(); y -= y.mean()
    return float((x * y).sum() / (math.sqrt((x * x).sum() * (y * y).sum()) + 1e-8))


def ber_u8(a, b):
    return float(np.unpackbits(np.bitwise_xor(a, b)).mean())


_WIN = {}


def _window(device, ch=3, size=11, sigma=1.5):
    if (device, ch) not in _WIN:
        g = np.exp(-((np.arange(size) - size // 2) ** 2) / (2 * sigma ** 2)); g /= g.sum()
        w = torch.from_numpy(np.outer(g, g).astype(np.float32))[None, None].expand(ch, 1, size, size).contiguous()
        _WIN[(device, ch)] = w.to(device)
    return _WIN[(device, ch)]


def ssim_u8(a, b, device):
    x = u8_to_t(a, device).add(1).div(2).clamp(0, 1)
    y = u8_to_t(b, device).add(1).div(2).clamp(0, 1)
    w, p, c = _window(device), 5, 3
    mu1, mu2 = F.conv2d(x, w, padding=p, groups=c), F.conv2d(y, w, padding=p, groups=c)
    s1 = F.conv2d(x * x, w, padding=p, groups=c) - mu1 ** 2
    s2 = F.conv2d(y * y, w, padding=p, groups=c) - mu2 ** 2
    s12 = F.conv2d(x * y, w, padding=p, groups=c) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
    return float(m.mean())


def frame_metrics(secret_u8, rec_u8, device):
    return {
        "R_PSNR": psnr_u8(secret_u8, rec_u8),
        "SSIM": ssim_u8(secret_u8, rec_u8, device),
        "NC": nc_u8(secret_u8, rec_u8),
        "BER": ber_u8(secret_u8, rec_u8),
        "E_XAI": float(np.abs(secret_u8.astype(np.float64) - rec_u8.astype(np.float64)).mean() / 255.0),
    }


def band(nc):
    return "robust" if nc >= 0.95 else ("stress" if nc >= 0.85 else "critical")


def read_frame_table(set_dir):
    """per_frame.csv written by make_stego_set.py -> {seq: [(frame_stem, secret_name), ...]}."""
    table = {}
    for r in csv.DictReader(open(Path(set_dir) / "per_frame.csv")):
        table.setdefault(r["seq"], []).append((r["frame"], r["secret"]))
    return table


def summarize(rows, keys, metrics=("R_PSNR", "SSIM", "NC", "BER", "E_XAI")):
    """Group frame rows by `keys`; report per-sequence means, then mean, SD and 95% CI across sequences,
    plus band fractions over all frames."""
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for gk, rs in groups.items():
        row = dict(zip(keys, gk))
        seqs = sorted({r["seq"] for r in rs})
        row["n_seq"], row["n_frames"] = len(seqs), len(rs)
        for m in metrics:
            per_seq = np.array([np.mean([r[m] for r in rs if r["seq"] == s]) for s in seqs])
            mean, sd = per_seq.mean(), (per_seq.std(ddof=1) if len(per_seq) > 1 else 0.0)
            half = 1.96 * sd / math.sqrt(len(per_seq)) if len(per_seq) > 1 else 0.0
            row[m] = round(mean, 4); row[m + "_sd"] = round(sd, 4)
            row[m + "_CI95"] = f"[{mean - half:.4f}, {mean + half:.4f}]"
        bands = [band(r["NC"]) for r in rs]
        for b in ("robust", "stress", "critical"):
            row["frac_" + b] = round(bands.count(b) / len(bands), 4)
        out.append(row)
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
