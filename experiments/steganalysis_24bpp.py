#!/usr/bin/env python
"""
Deep steganalysis at the headline 24-BPP operating point.

Detectors are retrained from scratch on ILWT-INN cover/stego frames:
  xunet, yenet, srnet                 grayscale (luminance Y) input, as in their papers
  xunet_color, yenet_color, srnet_color   YCbCr 3-channel input (colour-aware; the method embeds in Cb/Cr)

Protocol
  * 5-fold cross-validation grouped by VIDEO SEQUENCE (frames of one video never appear in
    both train and test). Inside each training fold 2 sequences are held out for early stopping.
  * Paired training (a cover and its own stego in the same batch), random flips / 90-degree
    rotations applied identically to both.
  * Test metrics on every out-of-fold frame (about 5.5k covers + 5.5k stegos in total):
    AUC, P_E = min_t (P_FA + P_MD)/2, accuracy and P_FA / P_MD at threshold 0.5.
  * 95% confidence intervals by bootstrap over the 20 sequences (cluster bootstrap, 2000 draws),
    plus mean +- SD across folds.

Input: the frame set written by make_stego_set.py (cover/<seq>/*.png, stego/<seq>/*.png).
Outputs under --out:
  scores_<det>.csv      seq, frame, label, fold, score  (out-of-fold stego probability)
  folds_<det>.csv       per-fold metrics
  summary.csv           one row per detector with CIs
  roc_<det>.csv         pooled ROC curve (fpr, tpr)
Resumable: finished (detector, fold) pairs are skipped.
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(os.path.expanduser("~/ilwt_workspace"))


# ----------------------------------------------------------------------------
# Fixed high-pass filters
# ----------------------------------------------------------------------------
KV = np.array([[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0


def srm_30():
    """The 30 basic SRM residual filters (5x5), built from their definitions."""
    f = []

    def put(k3):
        k = np.zeros((5, 5), np.float32); k[1:4, 1:4] = k3; return k
    # 1st order: 8 directions
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
        k = np.zeros((5, 5), np.float32); k[2, 2] = -1; k[2 + dy, 2 + dx] = 1; f.append(k)
    # 2nd order: 4 directions, /2
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        k = np.zeros((5, 5), np.float32); k[2, 2] = -2; k[2 + dy, 2 + dx] = 1; k[2 - dy, 2 - dx] = 1; f.append(k / 2)
    # 3rd order: 4 directions x 2 shifts, /3
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        for s in (1, -1):
            k = np.zeros((5, 5), np.float32)
            for t, c in zip((-1, 0, 1, 2), (1, -3, 3, -1)):
                k[2 + s * t * dy, 2 + s * t * dx] = c
            f.append(k / 3)
    sq3 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], np.float32)
    f.append(put(sq3) / 4)
    e3 = np.array([[-1, 2, -1], [2, -4, 2], [0, 0, 0]], np.float32)
    for r in range(4):
        f.append(put(np.rot90(e3, r)) / 4)
    f.append(KV.copy())
    e5 = KV * 12.0; e5[3:, :] = 0
    for r in range(4):
        f.append(np.rot90(e5, r).copy() / 12)
    assert len(f) == 30
    return np.stack(f)


class FixedHPF(nn.Module):
    """Applies the same filter bank to every input channel."""
    def __init__(self, bank, in_ch):
        super().__init__()
        w = torch.from_numpy(bank)[:, None]                      # (K,1,5,5)
        self.register_buffer("w", w.repeat(in_ch, 1, 1, 1))     # (in_ch*K,1,5,5)
        self.in_ch = in_ch

    def forward(self, x):
        return F.conv2d(x, self.w, padding=2, groups=self.in_ch)


# ----------------------------------------------------------------------------
# Detectors
# ----------------------------------------------------------------------------
class XuNet(nn.Module):
    """Xu, Wu, Shi, IEEE SPL 2016."""
    def __init__(self, in_ch=1):
        super().__init__()
        self.hpf = FixedHPF(KV[None], in_ch)
        c = in_ch

        def grp(i, o, k, act, absval=False):
            return nn.ModuleDict(dict(conv=nn.Conv2d(i, o, k, padding=k // 2, bias=False), bn=nn.BatchNorm2d(o))), act, absval
        self.groups = [grp(c, 8, 5, torch.tanh, True), grp(8, 16, 5, torch.tanh),
                       grp(16, 32, 1, F.relu), grp(32, 64, 1, F.relu), grp(64, 128, 1, F.relu)]
        self.mods = nn.ModuleList([g[0] for g in self.groups])
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.hpf(x)
        for i, (m, act, absval) in enumerate(self.groups):
            x = m["conv"](x)
            if absval:
                x = x.abs()
            x = act(m["bn"](x))
            x = F.avg_pool2d(x, 5, 2, 2) if i < 4 else x.mean((2, 3))
        return self.fc(x)


class YeNet(nn.Module):
    """Ye, Ni, Yi, IEEE TIFS 2017 (SRM-initialised first layer, TLU), with BatchNorm added for stable training."""
    def __init__(self, in_ch=1, tlu=3.0):
        super().__init__()
        bank = torch.from_numpy(srm_30())[:, None].repeat(1, in_ch, 1, 1) / in_ch
        self.srm = nn.Conv2d(in_ch, 30, 5, padding=2, bias=False)
        self.srm.weight.data.copy_(bank)
        self.tlu = tlu

        def cbr(i, o, k, s=1):
            return nn.Sequential(nn.Conv2d(i, o, k, s, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.body = nn.Sequential(
            cbr(30, 30, 3), cbr(30, 30, 3), cbr(30, 30, 3), nn.AvgPool2d(2, 2),
            cbr(30, 32, 5), nn.AvgPool2d(3, 2, 1),
            cbr(32, 32, 5), nn.AvgPool2d(3, 2, 1),
            cbr(32, 32, 5), nn.AvgPool2d(3, 2, 1),
            cbr(32, 16, 3), cbr(16, 16, 3, 3))
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = torch.clamp(self.srm(x), -self.tlu, self.tlu)
        return self.fc(self.body(x).mean((2, 3)))


class SRNet(nn.Module):
    """Boroumand, Chen, Fridrich, IEEE TIFS 2019."""
    def __init__(self, in_ch=1):
        super().__init__()

        def cb(i, o, s=1):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o))
        self.t1 = nn.Sequential(cb(in_ch, 64), nn.ReLU(inplace=True), cb(64, 16), nn.ReLU(inplace=True))
        self.t2 = nn.ModuleList([nn.Sequential(cb(16, 16), nn.ReLU(inplace=True), cb(16, 16)) for _ in range(5)])
        self.t3 = nn.ModuleList()
        for i, o in [(16, 16), (16, 64), (64, 128), (128, 256)]:
            self.t3.append(nn.ModuleDict(dict(
                main=nn.Sequential(cb(i, o), nn.ReLU(inplace=True), cb(o, o), nn.AvgPool2d(3, 2, 1)),
                short=nn.Sequential(nn.Conv2d(i, o, 1, 2, bias=False), nn.BatchNorm2d(o)))))
        self.t4 = nn.Sequential(cb(256, 512), nn.ReLU(inplace=True), cb(512, 512))
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.t1(x)
        for b in self.t2:
            x = x + b(x)
        for b in self.t3:
            x = b["main"](x) + b["short"](x)
        return self.fc(self.t4(x).mean((2, 3)))


DETECTORS = {
    "xunet": (XuNet, 1), "yenet": (YeNet, 1), "srnet": (SRNet, 1),
    "xunet_color": (XuNet, 3), "yenet_color": (YeNet, 3), "srnet_color": (SRNet, 3),
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def to_input(rgb_u8, in_ch):
    """uint8 (N,H,W,3) RGB -> float (N,C,H,W) on 0..255 scale: Y for 1 channel, YCbCr for 3."""
    x = rgb_u8.astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    if in_ch == 1:
        return y[:, None]
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], 1)


def load_set(root):
    seqs = sorted(p.name for p in (root / "cover").iterdir() if p.is_dir())
    covers, stegos, seq_of, names = [], [], [], []
    for s in seqs:
        for p in sorted((root / "cover" / s).glob("*.png")):
            q = root / "stego" / s / p.name
            if not q.exists():
                continue
            covers.append(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
            stegos.append(cv2.cvtColor(cv2.imread(str(q)), cv2.COLOR_BGR2RGB))
            seq_of.append(s); names.append(p.stem)
    return seqs, np.stack(covers), np.stack(stegos), np.array(seq_of), np.array(names)


def augment(c, s, rng):
    k = rng.integers(4)
    c, s = np.rot90(c, k, (1, 2)), np.rot90(s, k, (1, 2))
    if rng.random() < 0.5:
        c, s = c[:, :, ::-1], s[:, :, ::-1]
    return np.ascontiguousarray(c), np.ascontiguousarray(s)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def metrics(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    pe = float(np.min(0.5 * (fpr + 1 - tpr)))
    pred = p >= 0.5
    p_fa = float(pred[y == 0].mean()); p_md = float((~pred[y == 1]).mean())
    return dict(AUC=float(roc_auc_score(y, p)), P_E=pe, ACC=float((pred == y).mean()), P_FA=p_fa, P_MD=p_md)


def cluster_bootstrap(seq, y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(seq)
    idx_of = {s: np.where(seq == s)[0] for s in uniq}
    out = {k: [] for k in ("AUC", "P_E", "ACC", "P_FA", "P_MD")}
    for _ in range(n):
        idx = np.concatenate([idx_of[s] for s in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        for k, v in metrics(y[idx], p[idx]).items():
            out[k].append(v)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


# ----------------------------------------------------------------------------
# Train / test one fold
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict(net, x_u8, in_ch, device, bs=64):
    net.eval()
    out = []
    for i in range(0, len(x_u8), bs):
        x = torch.from_numpy(to_input(x_u8[i:i + bs], in_ch)).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.softmax(net(x).float(), 1)[:, 1].cpu().numpy())
    net.train()
    return np.concatenate(out)


def run_fold(det, train_idx, val_idx, covers, stegos, args, device, log):
    cls, in_ch = DETECTORS[det]
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    net = cls(in_ch).to(device)
    opt = torch.optim.Adamax(net.parameters(), lr=args.lr, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    half = args.batch // 2
    best, best_state, bad = -1.0, None, 0
    val_y = np.r_[np.zeros(len(val_idx)), np.ones(len(val_idx))]
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        perm = rng.permutation(train_idx)
        losses = []
        for i in range(0, len(perm) - half + 1, half):
            b = perm[i:i + half]
            c, s = augment(covers[b], stegos[b], rng)
            x = torch.from_numpy(to_input(np.concatenate([c, s]), in_ch)).to(device)
            y = torch.cat([torch.zeros(len(b)), torch.ones(len(b))]).long().to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(net(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(loss.item())
        sched.step()
        pv = np.r_[predict(net, covers[val_idx], in_ch, device), predict(net, stegos[val_idx], in_ch, device)]
        m = metrics(val_y, pv)
        log(f"    ep {ep:3d} loss {np.mean(losses):.4f}  val AUC {m['AUC']:.4f} P_E {m['P_E']:.4f}  ({time.time()-t0:.0f}s)")
        score = m["AUC"] - m["P_E"]
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if bad >= args.patience or (m["AUC"] > 0.9999 and m["P_E"] < 1e-4 and ep >= 3):
            log(f"    early stop at epoch {ep}")
            break
    net.load_state_dict(best_state)
    return net, in_ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "runs/stego_set_24bpp"))
    ap.add_argument("--out", default=str(ROOT / "runs/steganalysis_24bpp"))
    ap.add_argument("--detectors", default="xunet,yenet,srnet,srnet_color,xunet_color,yenet_color")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32, help="covers+stegos per batch (pairs = batch/2)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-frames-per-seq", type=int, default=0, help="for quick tests")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    device = torch.device("cuda")
    seqs, covers, stegos, seq_of, names = load_set(Path(args.data))
    if args.max_frames_per_seq:
        keep = np.concatenate([np.where(seq_of == s)[0][:args.max_frames_per_seq] for s in seqs])
        covers, stegos, seq_of, names = covers[keep], stegos[keep], seq_of[keep], names[keep]
    log(f"loaded {len(covers)} cover/stego pairs from {len(seqs)} sequences")

    order = [str(s) for s in np.random.default_rng(args.seed).permutation(seqs)]
    fold_seqs = [list(order[i::args.folds]) for i in range(args.folds)]
    json.dump({"folds": fold_seqs, "args": vars(args)}, open(out / "folds.json", "w"), indent=1)

    summary_rows = []
    for det in args.detectors.split(","):
        score_path = out / f"scores_{det}.csv"
        done = {}
        if score_path.exists():
            for r in csv.DictReader(open(score_path)):
                done.setdefault(int(r["fold"]), []).append(r)
        for f, test_s in enumerate(fold_seqs):
            if f in done:
                log(f"[{det}] fold {f} already done, skipping"); continue
            rest = [s for s in order if s not in test_s]
            val_s = rest[:2]; train_s = rest[2:]
            tr = np.where(np.isin(seq_of, train_s))[0]
            va = np.where(np.isin(seq_of, val_s))[0]
            te = np.where(np.isin(seq_of, test_s))[0]
            log(f"[{det}] fold {f}: test {test_s}  val {val_s}  train {len(tr)} pairs")
            t0 = time.time()
            net, in_ch = run_fold(det, tr, va, covers, stegos, args, device, log)
            pc, ps = predict(net, covers[te], in_ch, device), predict(net, stegos[te], in_ch, device)
            y = np.r_[np.zeros(len(te)), np.ones(len(te))]
            m = metrics(y, np.r_[pc, ps])
            log(f"[{det}] fold {f} TEST  AUC {m['AUC']:.4f}  P_E {m['P_E']:.4f}  ACC {m['ACC']:.4f}  "
                f"P_FA {m['P_FA']:.4f}  P_MD {m['P_MD']:.4f}  ({(time.time()-t0)/60:.1f} min)")
            new = not score_path.exists()
            with open(score_path, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["seq", "frame", "label", "fold", "score"])
                for lab, pp in ((0, pc), (1, ps)):
                    for i, sc in zip(te, pp):
                        w.writerow([seq_of[i], names[i], lab, f, f"{sc:.6f}"])

        rows = list(csv.DictReader(open(score_path)))
        seq = np.array([r["seq"] for r in rows]); y = np.array([int(r["label"]) for r in rows])
        p = np.array([float(r["score"]) for r in rows]); fold = np.array([int(r["fold"]) for r in rows])
        per_fold = [metrics(y[fold == f], p[fold == f]) for f in range(args.folds)]
        with open(out / f"folds_{det}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["fold", "AUC", "P_E", "ACC", "P_FA", "P_MD"])
            w.writeheader()
            for f, m in enumerate(per_fold):
                w.writerow({"fold": f, **{k: f"{v:.6f}" for k, v in m.items()}})
        pooled = metrics(y, p); ci = cluster_bootstrap(seq, y, p)
        fpr, tpr, _ = roc_curve(y, p)
        np.savetxt(out / f"roc_{det}.csv", np.c_[fpr, tpr], delimiter=",", header="fpr,tpr", comments="")
        row = {"detector": det, "n_cover": int((y == 0).sum()), "n_stego": int((y == 1).sum())}
        for k in ("AUC", "P_E", "ACC", "P_FA", "P_MD"):
            vals = [m[k] for m in per_fold]
            row[k] = f"{pooled[k]:.4f}"
            row[f"{k}_CI95"] = f"[{ci[k][0]:.4f}, {ci[k][1]:.4f}]"
            row[f"{k}_fold_mean_sd"] = f"{np.mean(vals):.4f} +- {np.std(vals, ddof=1):.4f}"
        summary_rows.append(row)
        log(f"[{det}] POOLED  AUC {row['AUC']} {row['AUC_CI95']}  P_E {row['P_E']} {row['P_E_CI95']}  "
            f"P_FA {row['P_FA']}  P_MD {row['P_MD']}")
        with open(out / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)
    log("done")


if __name__ == "__main__":
    main()
