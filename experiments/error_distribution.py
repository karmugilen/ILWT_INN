#!/usr/bin/env python
"""
recovery-error distribution and the BER it implies.

Same protocol as E12's 256 condition (native_res.py): deployed model, every frame of the 20 Xiph sequences
(ffmpeg lanczos 256x256), standard secrets (sorted hide folder, first 300, frame i -> secret[i % 300]),
8-bit stego, blind recovery.

For every recovered pixel value (3 channels) the signed error e = recovered - secret (integer, -255..255)
is histogrammed, and the per-plane BER is measured directly (XOR of the 8-bit values).

Derivation checked here. Bit k of v differs from bit k of v + e exactly when an odd number of the
boundaries b = j 2^k, 0 < b < 256 (there are 2^(8-k) - 1 of them), lie in (v, v + e]. If the secret value v is
uniform over the values for which v + e stays in 0..255 and independent of e, the flip probability P_k(e) is
the fraction of those v; it is computed exactly by counting. For |e| < 2^k at most one boundary can be crossed,
which gives the closed form
    P_k(e) = |e| (2^(8-k) - 1) / (256 - |e|),
so for the upper planes BER_k is set by the mean of |e| / (256 - |e|), i.e. by the mean absolute error, while
the lower planes approach 1/2. BER_k = sum_e h(e) P_k(e) depends on the whole error histogram h, not only on
its second moment (MSE). Since each boundary is crossed by at most |e| values of v, the closed form is also
an upper bound for every e:  BER_k <= (2^(8-k) - 1) E[|e| / (256 - |e|)].
For comparison the same formula is evaluated for discretised Gaussian and Laplacian errors with the
measured MSE, to show that equal PSNR does not mean equal BER.

Outputs (--out): hist.csv (e, count), planes.csv (plane, measured BER, predicted BER, bound and the
Gaussian / Laplacian predictions), summary.csv.
"""
import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as E  # noqa: E402
from make_stego_set import SEQUENCES  # noqa: E402
from native_res import decode, secret_u8  # noqa: E402

E_RANGE = np.arange(-255, 256)


def _flip_table():
    """P[k, e + 255]: share of v in 0..255 with v + e in 0..255 whose bit k differs from that of v + e."""
    v = np.arange(256)
    tab = np.zeros((8, 511))
    for j, e in enumerate(E_RANGE):
        w = v[(v + e >= 0) & (v + e <= 255)]
        x = np.bitwise_xor(w, w + e)
        tab[:, j] = [((x >> k) & 1).mean() for k in range(8)]
    return tab


FLIP = _flip_table()


def flip_prob(e, k):
    return FLIP[k, np.asarray(e) + 255]


def predicted(h):
    p = h / h.sum()
    return np.array([(p * flip_prob(E_RANGE, k)).sum() for k in range(8)])


def discretised(kind, mse):
    if kind == "gaussian":
        sd = math.sqrt(mse)
        w = np.exp(-0.5 * (E_RANGE / sd) ** 2)
    else:
        b = math.sqrt(mse / 2)
        w = np.exp(-np.abs(E_RANGE) / b)
    # rescale so that the discrete distribution has exactly the target MSE
    lo, hi = 0.05, 20.0
    for _ in range(100):
        s = (lo + hi) / 2
        q = w ** (1 / s ** 2) if kind == "gaussian" else w ** (1 / s)
        if (q / q.sum() * E_RANGE ** 2).sum() < mse:
            lo = s
        else:
            hi = s
    return q / q.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--secret-dir", required=True, help="hide folder (sorted; first 300 used)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=0, help="per sequence, for quick tests")
    args = ap.parse_args()
    out = Path(args.out)
    assert not (out / "planes.csv").exists(), f"{out} already has results; choose a new --out"
    out.mkdir(parents=True, exist_ok=True)
    model, device = E.load_model()
    secrets = sorted(Path(args.secret_dir).glob("*.png"))[:300]
    hist = np.zeros(511, np.int64)
    flips = np.zeros(8, np.int64)
    n_val, n_frames, frame_psnr = 0, 0, []
    for seq, rel in SEQUENCES.items():
        covers = decode(Path(args.videos) / rel, scale=True)
        if args.max_frames:
            covers = covers[:args.max_frames]
        secs = np.stack([secret_u8(secrets[i % len(secrets)], 256, 256) for i in range(len(covers))])
        stego = []
        with torch.no_grad():
            for b in range(0, len(covers), args.batch):
                x = torch.cat([E.u8_to_t(covers[b:b + args.batch], device), E.u8_to_t(secs[b:b + args.batch], device)], 1)
                stego.append(E.t_to_u8(model(x)[0][:, :3]))
        recs = E.recover(model, np.concatenate(stego), device, bs=args.batch)
        e = recs.astype(np.int16) - secs.astype(np.int16)
        hist += np.bincount((e + 255).ravel(), minlength=511)
        x = np.bitwise_xor(recs, secs)
        flips += np.array([int(((x >> k) & 1).sum()) for k in range(8)], np.int64)
        n_val += x.size; n_frames += len(covers)
        frame_psnr += [E.psnr_u8(s, r) for s, r in zip(secs, recs)]
        print(f"{seq:13s} {len(covers):4d} fr  BER {np.mean([((x >> k) & 1).mean() for k in range(8)]):.4f}  "
              f"mean|e| {np.abs(e).mean():.3f}", flush=True)

    p = hist / hist.sum()
    mse = float((p * E_RANGE ** 2).sum()); mae = float((p * np.abs(E_RANGE)).sum())
    meas = flips / n_val
    r = float((p * np.abs(E_RANGE) / (256 - np.abs(E_RANGE))).sum())
    bound = [min(1.0, (2 ** (8 - k) - 1) * r) for k in range(8)]
    pred = predicted(hist)
    g, l = discretised("gaussian", mse), discretised("laplacian", mse)
    pg = np.array([(g * flip_prob(E_RANGE, k)).sum() for k in range(8)])
    pl = np.array([(l * flip_prob(E_RANGE, k)).sum() for k in range(8)])
    with open(out / "hist.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["e", "count"])
        w.writerows([[int(v), int(c)] for v, c in zip(E_RANGE, hist) if c])
    with open(out / "planes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bit_plane", "BER_measured", "BER_predicted_uniform_v", "bound_closed_form",
                    "BER_gaussian_same_MSE", "BER_laplacian_same_MSE"])
        for k in range(8):
            w.writerow([k, f"{meas[k]:.6f}", f"{pred[k]:.6f}", f"{bound[k]:.6f}", f"{pg[k]:.6f}", f"{pl[k]:.6f}"])
        w.writerow(["all", f"{meas.mean():.6f}", f"{pred.mean():.6f}", "", f"{pg.mean():.6f}", f"{pl.mean():.6f}"])
    a = np.abs(E_RANGE)
    share = {"abs_e_0": p[a == 0].sum(), "abs_e_1": p[a == 1].sum(), "abs_e_2_3": p[(a >= 2) & (a <= 3)].sum(),
             "abs_e_4_7": p[(a >= 4) & (a <= 7)].sum(), "abs_e_8_15": p[(a >= 8) & (a <= 15)].sum(),
             "abs_e_ge16": p[a >= 16].sum()}
    summ = {"frames": n_frames, "values": n_val, "MSE": round(mse, 4), "pooled_PSNR": round(10 * math.log10(255 ** 2 / mse), 4),
            "mean_frame_PSNR": round(float(np.mean(frame_psnr)), 4), "mean_abs_e": round(mae, 4),
            "BER_measured": round(float(meas.mean()), 6), "BER_predicted": round(float(pred.mean()), 6),
            "BER_gaussian_same_MSE": round(float(pg.mean()), 6), "BER_laplacian_same_MSE": round(float(pl.mean()), 6),
            **{k: round(float(v), 6) for k, v in share.items()}}
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ)); w.writeheader(); w.writerow(summ)
    print(summ)
    print("plane  measured  predicted  bound  gaussian  laplacian")
    for k in range(8):
        print(f"{k:5d}  {meas[k]:.4f}    {pred[k]:.4f}     {bound[k]:.4f} {pg[k]:.4f}    {pl[k]:.4f}")


if __name__ == "__main__":
    main()
