#!/usr/bin/env python
"""
DeepMIH quantization-aware fine-tune.

Same recipe as the HiNet fine-tune (hinet_ste_finetune.py, E06), applied to DeepMIH's released TPAMI-2022
weights and its own training objective (train_old_version.py, the version the released weights come from):
  * full cascade net1 + importance map net3 + net2, DeepMIH's native 2-secret training (cover, secret 1,
    secret 2 = three different ImageNet training images)
  * the transmitted stego (steg 2) is rounded to the 8-bit grid with a straight-through estimator before the
    reverse pass, exactly as ILWT-INN and the HiNet fine-tune; the reverse pass then runs net2 (Gaussian z,
    24 ch) -> steg 1 (first 12 ch) and secret 2 (MIDDLE 12 ch, old-version layout) -> net1 (Gaussian z) ->
    secret 1 (last 12 ch)
  * DeepMIH loss (sum-reduced MSE): 2 r1 + 2 r2 + lambda_guide (g1 + g2) + l1 + l2, with g2 on the 8-bit steg 2
  * same data (ImageNet 256 PNGs, seed-42 split, 256x256 crops), optimiser settings, warm-up, spike skipping,
    gradient clipping, gradient checkpointing and time cap as the HiNet run
Validation each epoch on --val-images ImageNet validation images, from the stored 8-bit steg 2:
  1-secret mode (secret 2 = 0.5 grey; the 24-BPP comparison): H-PSNR, R-PSNR float and 8-bit for secret 1
  2-secret mode (48 BPP): R-PSNR 8-bit for secrets 1 and 2
best.pt = best 1-secret 8-bit R-PSNR. export/best_{1,2,3}.pt hold {"net": state_dict} in the released layout,
so deepmih_eval.py --weights <out>/export/best_ evaluates them unchanged.

Resumes automatically from <out>/last.pt. Stops after --epochs or --max-hours.
"""
import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.expanduser("~/ilwt_workspace")
DEEPMIH_DIR = os.path.join(ROOT, "baselines/DeepMIH")
sys.path.insert(0, DEEPMIH_DIR)
from model import Model_1, Model_2  # noqa: E402  (DeepMIH repo)
from imp_subnet import ImpMapBlock  # noqa: E402
import modules.Unet_common as common  # noqa: E402


class ImageList(Dataset):
    """uint8 CHW numpy arrays (same as hinet_ste_finetune.py: avoids the container's small /dev/shm)."""
    def __init__(self, image_dir, names, size=256, train=True):
        self.paths = [os.path.join(image_dir, n) for n in names]
        self.tf = T.Compose([T.RandomCrop(size, pad_if_needed=True, padding_mode="reflect"),
                             T.RandomHorizontalFlip(), T.RandomVerticalFlip()]) if train else \
            T.Compose([T.Resize(size), T.CenterCrop(size)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.tf(Image.open(self.paths[i]).convert("RGB"))
        return np.ascontiguousarray(np.asarray(img, dtype=np.uint8).transpose(2, 0, 1))


def numpy_collate(batch):
    return np.stack(batch)


def to_device(x, device):
    return torch.from_numpy(x).to(device, non_blocking=True).float().div_(255.0)


dwt, iwt = common.DWT(), common.IWT()


def quantize_ste(x):
    q = torch.round(x.clamp(0, 1) * 255.0) / 255.0
    return x + (q - x).detach()


def forward(nets, cover, s1, s2):
    net1, net2, net3 = nets
    cd = dwt(cover)
    st1d = net1(torch.cat((cd, dwt(s1)), 1)).narrow(1, 0, 12)
    steg1 = iwt(st1d)
    imp = net3(cover, s1, steg1)
    st2d = net2(torch.cat((st1d, dwt(imp), dwt(s2)), 1)).narrow(1, 0, 12)
    return cd, st1d, steg1, st2d, iwt(st2d)


def reverse(nets, steg2_stored):
    net1, net2, _ = nets
    d = dwt(steg2_stored)
    r2 = net2(torch.cat((d, torch.randn(d.shape[0], 24, *d.shape[2:], device=d.device)), 1), rev=True)
    r1 = net1(torch.cat((r2.narrow(1, 0, 12), torch.randn_like(d)), 1), rev=True)
    return iwt(r1.narrow(1, 12, 12)), iwt(r2.narrow(1, 12, 12))       # secret 1, secret 2 (old-version split)


def psnr(a, b):
    mse = F.mse_loss(a.clamp(0, 1) * 255, b.clamp(0, 1) * 255, reduction="none").mean(dim=(1, 2, 3))
    return (10 * torch.log10(255.0 ** 2 / mse.clamp_min(1e-10))).mean().item()


@torch.no_grad()
def evaluate(nets, loader, device):
    for n in nets:
        n.eval()
    m = {k: [] for k in ("H_8bit", "R1_float", "R1_8bit", "R1_8bit_2sec", "R2_8bit_2sec")}
    for x in loader:
        x = to_device(x, device)
        k = x.shape[0] // 3
        cover, s1, s2 = x[:k], x[k:2 * k], x[2 * k:3 * k]
        *_, steg2 = forward(nets, cover, s1, torch.full_like(cover, 0.5))         # 1-secret mode
        q = torch.round(steg2.clamp(0, 1) * 255) / 255
        m["H_8bit"].append(psnr(q, cover))
        m["R1_float"].append(psnr(reverse(nets, steg2.clamp(0, 1))[0], s1))
        m["R1_8bit"].append(psnr(reverse(nets, q)[0], s1))
        *_, steg2 = forward(nets, cover, s1, s2)                                  # 2-secret mode
        q = torch.round(steg2.clamp(0, 1) * 255) / 255
        r1, r2 = reverse(nets, q)
        m["R1_8bit_2sec"].append(psnr(r1, s1)); m["R2_8bit_2sec"].append(psnr(r2, s2))
    for n in nets:
        n.train()
    return {k: float(np.mean(v)) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(ROOT, "data/imagenet256"))
    ap.add_argument("--split", default=os.path.join(ROOT, "splits/imagenet_split_seed42.json"))
    ap.add_argument("--init", default=os.path.join(ROOT, "baselines/DeepMIH_weights/model_checkpoint_03000_"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs/deepmih_ste"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--max-hours", type=float, default=84.0)
    ap.add_argument("--triples", type=int, default=2, help="(cover, secret 1, secret 2) triples per batch")
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--lambda-guide", type=float, default=2.0)
    ap.add_argument("--warmup-iters", type=int, default=1000)
    ap.add_argument("--spike-factor", type=float, default=5.0)
    ap.add_argument("--lr-halve-every", type=int, default=30)
    ap.add_argument("--val-images", type=int, default=999)
    ap.add_argument("--max-iters-per-epoch", type=int, default=0, help="0 = full epoch (for quick tests)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every-epoch", action="store_true")
    args = ap.parse_args()
    assert os.path.realpath(args.out) != os.path.realpath(os.path.dirname(args.init)), "never write into the released weights"

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda")
    os.makedirs(os.path.join(args.out, "export"), exist_ok=True)
    split = json.load(open(args.split))
    bs = 3 * args.triples
    train_dl = DataLoader(ImageList(args.images, split["train"], train=True), batch_size=bs, shuffle=True,
                          num_workers=8, drop_last=True, persistent_workers=True, collate_fn=numpy_collate)
    val_dl = DataLoader(ImageList(args.images, split["val"][:args.val_images], train=False), batch_size=9,
                        shuffle=False, num_workers=4, drop_last=True, collate_fn=numpy_collate)
    print(f"train {len(split['train'])} images -> {len(train_dl)} batches of {args.triples} triples; val {args.val_images}")

    nets = [torch.nn.DataParallel(M().to(device), device_ids=[0]) for M in (Model_1, Model_2, ImpMapBlock)]
    for n in nets[:2]:          # gradient checkpointing on the 16 invertible blocks of each stage (memory only)
        for name, blk in n.module.model.named_children():
            if name.startswith("inv"):
                blk.forward = (lambda f: lambda x, rev=False: checkpoint(f, x, rev, use_reentrant=False)
                               if torch.is_grad_enabled() else f(x, rev))(blk.forward)
    params = [p for n in nets for p in n.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=args.lr, betas=(args.beta1, 0.999), eps=1e-6, weight_decay=1e-5)

    last = os.path.join(args.out, "last.pt")
    start_epoch, elapsed_before, best = 0, 0.0, -1.0
    mpath = os.path.join(args.out, "metrics.csv")
    cols = ["epoch", "hours", "lr", "train_loss", "H_8bit", "R1_float", "R1_8bit", "R1_8bit_2sec", "R2_8bit_2sec"]
    if os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        for n, k in zip(nets, ("net1", "net2", "net3")):
            n.load_state_dict(ck[k])
        optim.load_state_dict(ck["opt"])
        start_epoch, elapsed_before, best = ck["epoch"], ck["elapsed_s"], ck["best"]
        for g in optim.param_groups:
            g["lr"] = ck["lr"]
        print(f"resumed from epoch {start_epoch} ({elapsed_before / 3600:.1f} h already used)")
    else:
        for i, n in enumerate(nets, 1):
            sd = torch.load(f"{args.init}{i}.pt", map_location=device, weights_only=True)["net"]   # downloaded file
            n.load_state_dict({k: v for k, v in sd.items() if "tmp_var" not in k})
        print(f"initialised from released DeepMIH weights {args.init}{{1,2,3}}.pt")
        m = evaluate(nets, val_dl, device)
        print("[released, before fine-tune] " + "  ".join(f"{k} {v:.2f}" for k, v in m.items()), flush=True)
        with open(mpath, "w", newline="") as f:
            csv.writer(f).writerow(cols)
            csv.writer(f).writerow([0, 0, args.lr, ""] + [f"{m[k]:.4f}" for k in cols[4:]])

    t0 = time.time()
    step = start_epoch * (args.max_iters_per_epoch or len(train_dl))
    base_lr = args.lr if step < args.warmup_iters else optim.param_groups[0]["lr"]   # resume inside warm-up: ramp from the target LR
    for epoch in range(start_epoch + 1, args.epochs + 1):
        losses, bad, skipped, oom, ema = [], 0, 0, 0, None
        for it, x in enumerate(train_dl):
            if args.max_iters_per_epoch and it >= args.max_iters_per_epoch:
                break
            x = to_device(x, device)
            k = args.triples
            cover, s1, s2 = x[:k], x[k:2 * k], x[2 * k:3 * k]
            try:
                cd, st1d, steg1, st2d, steg2 = forward(nets, cover, s1, s2)
                steg2_q = quantize_ste(steg2)
                r1, r2 = reverse(nets, steg2_q)
                mse = lambda a, b: F.mse_loss(a, b, reduction="sum")  # noqa: E731
                loss = (2 * mse(r1, s1) + 2 * mse(r2, s2) + args.lambda_guide * (mse(steg1, cover) + mse(steg2_q, cover))
                        + mse(st1d.narrow(1, 0, 3), cd.narrow(1, 0, 3)) + mse(st2d.narrow(1, 0, 3), cd.narrow(1, 0, 3)))
                lv = loss.item() if torch.isfinite(loss) else float("inf")
                if args.spike_factor and ema is not None and lv > args.spike_factor * ema:
                    skipped += 1; optim.zero_grad(set_to_none=True); continue
                if not np.isfinite(lv):
                    bad += 1; optim.zero_grad(set_to_none=True)
                    if bad > 20:
                        raise RuntimeError("training exploded (NaN/Inf); lower --lr and resume")
                    continue
                ema = lv if ema is None else 0.99 * ema + 0.01 * lv
                step += 1
                if step <= args.warmup_iters:
                    for g in optim.param_groups:
                        g["lr"] = base_lr * step / args.warmup_iters
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                optim.step(); optim.zero_grad(set_to_none=True)
            except torch.OutOfMemoryError:
                optim.zero_grad(set_to_none=True); torch.cuda.empty_cache(); oom += 1
                if oom == 1:
                    print(f"  OOM at it {it}: peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", flush=True)
                continue
            losses.append(lv)
            if it % 200 == 0:
                print(f"epoch {epoch} it {it}/{len(train_dl)} loss {lv:.1f} peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB "
                      f"({(time.time() - t0 + elapsed_before) / 3600:.2f} h)", flush=True)

        hours = (time.time() - t0 + elapsed_before) / 3600
        m = evaluate(nets, val_dl, device)
        lr = optim.param_groups[0]["lr"]
        print(f"[epoch {epoch}] {hours:.2f} h  loss {np.mean(losses):.1f}  skipped {skipped} oom {oom}  "
              + "  ".join(f"{k} {v:.2f}" for k, v in m.items()), flush=True)
        with open(mpath, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{hours:.3f}", lr, f"{np.mean(losses):.3f}"] + [f"{m[k]:.4f}" for k in cols[4:]])
        best = max(best, m["R1_8bit"])
        state = {"net1": nets[0].state_dict(), "net2": nets[1].state_dict(), "net3": nets[2].state_dict(),
                 "opt": optim.state_dict(), "epoch": epoch, "elapsed_s": hours * 3600, "lr": lr, "best": best,
                 "metrics": m, "args": vars(args)}
        torch.save(state, last + ".tmp"); os.replace(last + ".tmp", last)
        if m["R1_8bit"] >= best:
            torch.save(state, os.path.join(args.out, "best.pt"))
            for i, n in enumerate(nets, 1):
                torch.save({"net": n.state_dict(), "epoch": epoch}, os.path.join(args.out, "export", f"best_{i}.pt"))
        if args.save_every_epoch:
            torch.save({"net1": state["net1"], "net2": state["net2"], "net3": state["net3"], "epoch": epoch},
                       os.path.join(args.out, f"epoch_{epoch:03d}.pt"))
        if epoch % args.lr_halve_every == 0:
            for g in optim.param_groups:
                g["lr"] *= 0.5
        if hours >= args.max_hours:
            print(f"time cap reached ({hours:.1f} h >= {args.max_hours} h); stopping after epoch {epoch}")
            break
    print("done")


if __name__ == "__main__":
    main()
