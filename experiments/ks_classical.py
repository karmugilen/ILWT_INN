#!/usr/bin/env python
"""
classical statistics as hypothesis tests and as detectors.

The original table (Chi-Square, histogram correlation, entropy difference, ...) gives one score per video and no
test. Here every statistic is computed per frame on the cover AND on its stego, on the E04 frame set
(make_stego_set.py layout: cover/<seq>/*.png, stego/<seq>/*.png; all 5,811 frames of the 20 sequences), and asked
two questions:
  1. are cover and stego distributions different?  two-sample Kolmogorov-Smirnov over frames, and a paired
     Wilcoxon signed-rank test on the 20 per-sequence means (sequence = independent unit)
  2. how well does the statistic alone detect stego?  AUC (direction-free, max(AUC, 1-AUC); the direction is fixed
     on all frames, then a 95% cluster-bootstrap CI over sequences, see cluster_ci), and the accuracy of a single threshold learned on 4/5 of the sequences and
     applied to the other 1/5 (5-fold, grouped by sequence)

Statistics (standard definitions; the original Dih / Gefr scripts are not available):
  chi2_p       Westfeld-Pfitzmann chi-square attack, P(embedding) = chi2.sf(stat, df) over the pairs of values
               (2k, 2k+1) of the R, G, B histograms (pairs with no pixels skipped)
  entropy_Y / entropy_Cb / entropy_Cr   Shannon entropy (bits) of the 256-bin histogram
  hp_Y / hp_C  mean |KV high-pass residual| on Y and on Cb, Cr (the SRM 5x5 KV kernel)
  temporal_Y   mean |Y_t - Y_(t-1)| between consecutive frames of a sequence (frame 1 has none)

Outputs (--out): per_frame.csv, summary.csv (one row per statistic).
"""
import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import chi2, ks_2samp, wilcoxon
from sklearn.metrics import roc_auc_score

KV = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], np.float64) / 12.0
STATS = ("chi2_p", "entropy_Y", "entropy_Cb", "entropy_Cr", "hp_Y", "hp_C", "temporal_Y")


def entropy(ch):
    p = np.bincount(ch.ravel(), minlength=256).astype(np.float64)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def chi2_p(rgb):
    stat, df = 0.0, 0
    for k in range(3):
        h = np.bincount(rgb[..., k].ravel(), minlength=256).astype(np.float64)
        a, b = h[0::2], h[1::2]
        e = (a + b) / 2
        m = e > 0
        stat += float((((a[m] - e[m]) ** 2) / e[m]).sum()); df += int(m.sum())
    return float(chi2.sf(stat, max(df - 1, 1)))


def features(path):
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    hp = lambda ch: float(np.abs(cv2.filter2D(ch.astype(np.float64), -1, KV, borderType=cv2.BORDER_REFLECT)).mean())  # noqa: E731
    return y.astype(np.float64), {"chi2_p": chi2_p(rgb), "entropy_Y": entropy(y), "entropy_Cb": entropy(cb),
                                  "entropy_Cr": entropy(cr), "hp_Y": hp(y), "hp_C": (hp(cb) + hp(cr)) / 2}


def seq_rows(job):
    root, seq = job
    rows = []
    for label, kind in ((0, "cover"), (1, "stego")):
        prev = None
        for p in sorted((root / kind / seq).glob("*.png")):
            if label == 1 and not (root / "cover" / seq / p.name).exists():
                continue
            if label == 0 and not (root / "stego" / seq / p.name).exists():
                continue
            y, f = features(p)
            f["temporal_Y"] = float(np.abs(y - prev).mean()) if prev is not None else float("nan")
            prev = y
            rows.append({"seq": seq, "frame": p.stem, "label": label, **f})
    return rows


def cluster_ci(seq, y, s, draws=2000, seed=0):
    """95% cluster-bootstrap CI of the pooled AUC, resampling sequences. AUC is a sum over (cover, stego) pairs;
    W[a, b] holds the wins of sequence b's stegos over sequence a's covers. In a draw, pairs between two copies of
    the SAME sequence are dropped (only same-slot pairs kept), so a draw has the same mix of within- and
    between-sequence pairs as the data; plain resampling over-weights within-sequence pairs and biases the AUC."""
    us = np.unique(seq)
    cov = [np.sort(s[(seq == u) & (y == 0)]) for u in us]
    stg = [np.sort(s[(seq == u) & (y == 1)]) for u in us]
    nc, ns = np.array([len(x) for x in cov], float), np.array([len(x) for x in stg], float)
    W = np.zeros((len(us), len(us)))
    for a, c in enumerate(cov):
        for b, g in enumerate(stg):
            right, left = np.searchsorted(g, c, "right"), np.searchsorted(g, c, "left")
            W[a, b] = (len(g) - right).sum() + 0.5 * (right - left).sum()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        k = rng.integers(len(us), size=len(us))
        mask = (k[:, None] != k[None, :]) | np.eye(len(us), dtype=bool)
        vals.append((W[np.ix_(k, k)] * mask).sum() / ((nc[k][:, None] * ns[k][None, :]) * mask).sum())
    return np.percentile(vals, [2.5, 97.5])


def cv_accuracy(seq, y, s, folds=5, seed=42):
    us = np.random.default_rng(seed).permutation(np.unique(seq))
    correct = 0
    for f in range(folds):
        te = np.isin(seq, us[f::folds]); tr = ~te
        best = (0, 0.0, 1)
        for t in np.unique(np.percentile(s[tr], np.linspace(0, 100, 401))):
            for d in (1, -1):
                acc = np.mean(((d * (s[tr] - t)) > 0) == (y[tr] == 1))
                if acc > best[0]:
                    best = (acc, t, d)
        correct += np.sum(((best[2] * (s[te] - best[1])) > 0) == (y[te] == 1))
    return correct / len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="frame set with cover/ and stego/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seqs = sorted(p.name for p in (root / "cover").iterdir() if p.is_dir())
    with Pool(args.workers) as pool:
        rows = [r for rs in pool.map(seq_rows, [(root, s) for s in seqs]) for r in rs]
    with open(out / "per_frame.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{args.data}: {sum(r['label'] == 0 for r in rows)} covers, {sum(r['label'] == 1 for r in rows)} stegos, "
          f"{len(seqs)} sequences")

    summ = []
    for st in STATS:
        r = [x for x in rows if np.isfinite(x[st])]
        seq = np.array([x["seq"] for x in r]); y = np.array([x["label"] for x in r]); s = np.array([x[st] for x in r])
        c, g = s[y == 0], s[y == 1]
        ks = ks_2samp(c, g)
        mc = np.array([s[(seq == q) & (y == 0)].mean() for q in seqs])
        mg = np.array([s[(seq == q) & (y == 1)].mean() for q in seqs])
        d = mg - mc
        wp = float("nan") if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
        raw = roc_auc_score(y, s); sign = 1 if raw >= 0.5 else -1      # direction fixed on all frames
        auc = max(raw, 1 - raw)
        lo, hi = cluster_ci(seq, y, sign * s)
        acc = cv_accuracy(seq, y, s)
        summ.append({"statistic": st, "n_cover": len(c), "n_stego": len(g),
                     "cover_mean": f"{c.mean():.6g}", "stego_mean": f"{g.mean():.6g}",
                     "cover_median": f"{np.median(c):.6g}", "stego_median": f"{np.median(g):.6g}",
                     "KS_D": f"{ks.statistic:.4f}", "KS_p": f"{ks.pvalue:.3g}",
                     "seq_mean_diff_median": f"{np.median(d):.6g}", "wilcoxon_p_20seq": f"{wp:.3g}",
                     "AUC": f"{auc:.4f}", "AUC_CI95": f"[{lo:.4f}, {hi:.4f}]", "CV_accuracy": f"{acc:.4f}"})
        print(f"{st:11s} cover {c.mean():10.5g}  stego {g.mean():10.5g}  KS D {ks.statistic:.3f} p {ks.pvalue:.2g}  "
              f"Wilcoxon p {wp:.2g}  AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]  CV acc {acc:.3f}", flush=True)
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)


if __name__ == "__main__":
    main()
