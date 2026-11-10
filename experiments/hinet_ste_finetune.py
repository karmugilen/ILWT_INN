#!/usr/bin/env python
"""
HiNet quantization-aware fine-tune.

Starts from the released HiNet weights and fine-tunes with the same 8-bit
straight-through-estimator (STE) rounding that ILWT-INN uses: the stego image is
rounded to integer 0..255 values before the reverse (recovery) pass, the
gradient passes straight through the rounding.

Same data protocol as ILWT-INN: ImageNet 256 PNGs, 70/15/15 split with seed 42,
256x256 crops. The split is written to splits/imagenet_split_seed42.json so every
other experiment script uses the identical file lists.

Resumes automatically from <out>/last.pt. Stops after --epochs or --max-hours,
whichever comes first.

Example:
  python experiments/hinet_ste_finetune.py --max-hours 120
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torch.utils.checkpoint import checkpoint

ROOT = os.path.expanduser("~/ilwt_workspace")
HINET_DIR = os.path.join(ROOT, "baselines/HiNet")
sys.path.insert(0, HINET_DIR)
from model import Model  # noqa: E402  (HiNet repo)
import modules.Unet_common as common  # noqa: E402

# ----------------------------------------------------------------------------
# Data split shared by all experiment scripts
# ----------------------------------------------------------------------------
def make_split(image_dir, split_path, seed=42, ratios=(0.70, 0.15, 0.15)):
    if os.path.exists(split_path):
        with open(split_path) as f:
            return json.load(f)
    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    idx = np.arange(len(files))
    np.random.RandomState(seed).shuffle(idx)
    n_tr, n_va = int(ratios[0] * len(files)), int(ratios[1] * len(files))
    split = {
        "seed": seed,
        "image_dir": image_dir,
        "train": [files[i] for i in idx[:n_tr]],
        "val": [files[i] for i in idx[n_tr:n_tr + n_va]],
        "test": [files[i] for i in idx[n_tr + n_va:]],
    }
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split, f)
    return split


class ImageList(Dataset):
    """Returns uint8 CHW numpy arrays. Workers send numpy (pickled through a pipe)
    instead of torch tensors, so the small 64 MB /dev/shm of the Docker container
    is never used."""
    def __init__(self, image_dir, names, size=256, train=True):
        self.paths = [os.path.join(image_dir, n) for n in names]
        if train:
            self.tf = T.Compose([
                T.RandomCrop(size, pad_if_needed=True, padding_mode="reflect"),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
            ])
        else:
            self.tf = T.Compose([T.Resize(size), T.CenterCrop(size)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.tf(Image.open(self.paths[i]).convert("RGB"))
        return np.ascontiguousarray(np.asarray(img, dtype=np.uint8).transpose(2, 0, 1))


def numpy_collate(batch):
    return np.stack(batch)


def to_device(x, device):
    return torch.from_numpy(x).to(device, non_blocking=True).float().div_(255.0)


# ----------------------------------------------------------------------------
# HiNet forward / reverse with 8-bit STE
# ----------------------------------------------------------------------------
dwt, iwt = common.DWT(), common.IWT()


def quantize_ste(x):
    """Round to the 8-bit grid in the forward pass, identity gradient in backward."""
    q = torch.round(x.clamp(0, 1) * 255.0) / 255.0
    return x + (q - x).detach()


def hide_and_recover(net, cover, secret, quantize):
    out = net(torch.cat((dwt(cover), dwt(secret)), 1))
    out_steg = out.narrow(1, 0, 12)
    steg = iwt(out_steg)
    steg_used = quantize_ste(steg) if quantize else steg
    z = torch.randn(out.shape[0], out.shape[1] - 12, *out.shape[2:], device=out.device)
    rev = net(torch.cat((dwt(steg_used), z), 1), rev=True)
    secret_rev = iwt(rev.narrow(1, 12, rev.shape[1] - 12))
    return steg, steg_used, out_steg, secret_rev


def psnr(a, b):
    mse = F.mse_loss(a.clamp(0, 1) * 255, b.clamp(0, 1) * 255, reduction="none").mean(dim=(1, 2, 3))
    return (10 * torch.log10(255.0 ** 2 / mse.clamp_min(1e-10))).mean().item()


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    h, r_float, r_8bit = [], [], []
    for x in loader:
        x = to_device(x, device)
        half = x.shape[0] // 2
        secret, cover = x[:half], x[half:2 * half]
        steg, _, _, rec_f = hide_and_recover(net, cover, secret, quantize=False)
        steg_q = torch.round(steg.clamp(0, 1) * 255) / 255           # true 8-bit storage
        h.append(psnr(steg_q, cover))
        r_float.append(psnr(rec_f, secret))
        # recover from the stored 8-bit stego
        steg_q_dwt = dwt(steg_q)
        rev = net(torch.cat((steg_q_dwt, torch.randn_like(steg_q_dwt)), 1), rev=True)
        r_8bit.append(psnr(iwt(rev.narrow(1, 12, 12)), secret))
    net.train()
    return float(np.mean(h)), float(np.mean(r_float)), float(np.mean(r_8bit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(ROOT, "data/imagenet256"))
    ap.add_argument("--split", default=os.path.join(ROOT, "splits/imagenet_split_seed42.json"))
    ap.add_argument("--init", default=os.path.join(HINET_DIR, "model/released/model.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs/hinet_ste"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--max-hours", type=float, default=120.0)
    ap.add_argument("--batch", type=int, default=8, help="images per batch (half cover, half secret)")
    ap.add_argument("--lr", type=float, default=10 ** -4.5)
    ap.add_argument("--val-images", type=int, default=1000)
    ap.add_argument("--max-iters-per-epoch", type=int, default=0, help="0 = full epoch (for quick tests)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup-iters", type=int, default=1000, help="linear LR warm-up from 0")
    ap.add_argument("--load-opt", action="store_true", help="also load the released Adam state")
    ap.add_argument("--beta1", type=float, default=0.5, help="Adam beta1 (HiNet uses 0.5)")
    ap.add_argument("--lambda-guide", type=float, default=1.0, help="weight of the hiding (guide) loss")
    ap.add_argument("--spike-factor", type=float, default=5.0,
                    help="skip a batch whose loss exceeds this multiple of the running loss (0 = off)")
    ap.add_argument("--save-every-epoch", action="store_true")
    ap.add_argument("--lr-halve-every", type=int, default=10, help="halve the LR every N epochs")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)

    split = make_split(args.images, args.split)
    print(f"split: train {len(split['train'])}  val {len(split['val'])}  test {len(split['test'])}")
    train_dl = DataLoader(ImageList(args.images, split["train"], train=True), batch_size=args.batch,
                          shuffle=True, num_workers=8, drop_last=True, persistent_workers=True,
                          collate_fn=numpy_collate)
    val_dl = DataLoader(ImageList(args.images, split["val"][:args.val_images], train=False),
                        batch_size=8, shuffle=False, num_workers=4, drop_last=True,
                        collate_fn=numpy_collate)

    net = torch.nn.DataParallel(Model().to(device), device_ids=[0])
    # Gradient checkpointing on each of HiNet's 16 invertible blocks: batch 8 at 256x256 needs
    # ~15 GB otherwise, which overflows the 16 GB RTX A4000. Same maths, less memory.
    for name, blk in net.module.model.named_children():
        if name.startswith("inv"):
            fwd = blk.forward
            blk.forward = (lambda f: lambda x, rev=False: checkpoint(f, x, rev, use_reentrant=False)
                           if torch.is_grad_enabled() else f(x, rev))(fwd)
    optim = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr,
                             betas=(args.beta1, 0.999), eps=1e-6, weight_decay=1e-5)

    last = os.path.join(args.out, "last.pt")
    start_epoch, elapsed_before = 0, 0.0
    if os.path.exists(last):
        ck = torch.load(last, map_location=device)
        net.load_state_dict(ck["net"]); optim.load_state_dict(ck["opt"])
        start_epoch, elapsed_before = ck["epoch"], ck["elapsed_s"]
        for g in optim.param_groups:
            g["lr"] = ck["lr"]
        print(f"resumed from epoch {start_epoch} ({elapsed_before/3600:.1f} h already used)")
    else:
        ck = torch.load(args.init, map_location=device)
        net.load_state_dict({k: v for k, v in ck["net"].items() if "tmp_var" not in k})
        if args.load_opt and "opt" in ck:
            optim.load_state_dict(ck["opt"])
            for g in optim.param_groups:
                g["lr"] = args.lr
        print(f"initialized from released weights {args.init} (optimizer state: {args.load_opt})")
        h, rf, r8 = evaluate(net, val_dl, device)
        print(f"[released, before fine-tune] H-PSNR(8bit) {h:.2f}  R-PSNR float {rf:.2f}  R-PSNR 8bit {r8:.2f}")
        with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "hours", "lr", "train_loss", "H_PSNR_8bit", "R_PSNR_float", "R_PSNR_8bit"])
            csv.writer(f).writerow([0, 0, args.lr, "", f"{h:.4f}", f"{rf:.4f}", f"{r8:.4f}"])

    t0 = time.time()
    step = start_epoch * (args.max_iters_per_epoch or len(train_dl))
    base_lr = optim.param_groups[0]["lr"]
    best_r8 = ck.get("best_r8", -1.0) if os.path.exists(last) else -1.0
    for epoch in range(start_epoch + 1, args.epochs + 1):
        losses, bad_steps, skipped, oom = [], 0, 0, 0
        ema = None
        for it, x in enumerate(train_dl):
            if args.max_iters_per_epoch and it >= args.max_iters_per_epoch:
                break
            x = to_device(x, device)
            half = x.shape[0] // 2
            secret, cover = x[:half], x[half:]
            try:
                steg, steg_q, out_steg, secret_rev = hide_and_recover(net, cover, secret, quantize=True)
                # HiNet losses (sum-reduced MSE, lambdas 5/1/1); guide loss on the 8-bit stego that is stored
                g_loss = F.mse_loss(steg_q, cover, reduction="sum")
                r_loss = F.mse_loss(secret_rev, secret, reduction="sum")
                l_loss = F.mse_loss(out_steg.narrow(1, 0, 3), dwt(cover).narrow(1, 0, 3), reduction="sum")
                loss = 5 * r_loss + args.lambda_guide * g_loss + l_loss
                lv = loss.item() if torch.isfinite(loss) else float("inf")
                if args.spike_factor and ema is not None and lv > args.spike_factor * ema:
                    skipped += 1
                    optim.zero_grad(set_to_none=True)
                    continue
                if np.isfinite(lv):
                    ema = lv if ema is None else 0.99 * ema + 0.01 * lv
                if not torch.isfinite(loss):
                    bad_steps += 1
                    optim.zero_grad(set_to_none=True)
                    if bad_steps > 20:
                        raise RuntimeError("training exploded (NaN/Inf); lower --lr and resume")
                    continue
                step += 1
                if step <= args.warmup_iters:
                    for g in optim.param_groups:
                        g["lr"] = base_lr * step / args.warmup_iters
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                optim.step(); optim.zero_grad(set_to_none=True)
            except torch.OutOfMemoryError:
                # rare large-memory batch: drop it instead of killing a multi-day run
                optim.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                oom += 1
                if oom == 1:
                    print(f"  OOM at it {it}: peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
                continue
            losses.append(loss.item())
            if it % 200 == 0:
                print(f"epoch {epoch} it {it}/{len(train_dl)} loss {loss.item():.1f} "
                      f"({(time.time()-t0)/3600 + elapsed_before/3600:.2f} h)", flush=True)

        hours = (time.time() - t0 + elapsed_before) / 3600
        h, rf, r8 = evaluate(net, val_dl, device)
        lr = optim.param_groups[0]["lr"]
        print(f"[epoch {epoch}] {hours:.2f} h  loss {np.mean(losses):.1f}  skipped {skipped} oom {oom}  H-PSNR(8bit) {h:.2f}  "
              f"R-PSNR float {rf:.2f}  R-PSNR 8bit {r8:.2f}", flush=True)
        with open(os.path.join(args.out, "metrics.csv"), "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{hours:.3f}", lr, f"{np.mean(losses):.3f}", f"{h:.4f}", f"{rf:.4f}", f"{r8:.4f}"])

        best_r8 = max(best_r8, r8)
        state = {"net": net.state_dict(), "opt": optim.state_dict(), "epoch": epoch,
                 "elapsed_s": hours * 3600, "lr": lr, "best_r8": best_r8, "args": vars(args)}
        torch.save(state, last + ".tmp"); os.replace(last + ".tmp", last)
        if r8 >= best_r8:
            torch.save(state, os.path.join(args.out, "best.pt"))
        if args.save_every_epoch:
            torch.save({"net": state["net"], "epoch": epoch}, os.path.join(args.out, f"epoch_{epoch:03d}.pt"))
        if epoch % args.lr_halve_every == 0:
            for g in optim.param_groups:
                g["lr"] *= 0.5
        if hours >= args.max_hours:
            print(f"time cap reached ({hours:.1f} h >= {args.max_hours} h); stopping after epoch {epoch}")
            break
    print("done")


if __name__ == "__main__":
    main()
