"""
Ablation sweep: identify components in StegoWaveFlow whose compute cost does not
buy meaningful PSNR. Trains baseline + 3 reduced variants on a fixed 300-image
subset for a fixed budget, then prints a side-by-side comparison.
"""
import time
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from train import (
    ImageSteganographyDataset,
    perturb_stego_like_for_inverse,
    steganography_loss,
    calculate_psnr,
    calculate_ssim,
)
from core.models import (
    StegoWaveFlow,
    StegoWaveFlowBlock,
    ILWT53_2D,
    LearnableSubbandWeights,
    LearnableYCbCrScaling,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)


# ----------------------------------------------------------------------------
# Model variants
# ----------------------------------------------------------------------------

class StegoWaveFlowNoCond(StegoWaveFlow):
    """Variant with no cover-conditioning network — coupling layers see only x1."""

    def __init__(self, channels=6, num_blocks=8, hidden_channels=128, transform_type="ilwt53"):
        nn.Module.__init__(self)
        self.ilwt = ILWT53_2D(channels)
        self.inn_channels = channels * 4
        self.cond_net = nn.Identity()
        self.blocks = nn.ModuleList(
            [StegoWaveFlowBlock(self.inn_channels, hidden_channels, cond_channels=0) for _ in range(num_blocks)]
        )
        self.subband_weights = LearnableSubbandWeights()
        self.ycbcr_scaling = LearnableYCbCrScaling()

    def forward(self, x):
        cover = x[:, :3, :, :]
        x_freq = self.ilwt(x)
        log_det_total = 0
        z = x_freq
        for block in self.blocks:
            z, log_det = block(z, cond=None)
            log_det_total += log_det
        residual_spatial = self.ilwt.inverse(z)
        cover_ycc = rgb_to_ycbcr(cover)
        resid_ycc = rgb_to_ycbcr(residual_spatial[:, :3, :, :])
        kY, kC = self.ycbcr_scaling.get_scales()
        scale = torch.stack([kY, kC, kC]).view(1, 3, 1, 1).to(cover.device)
        stego_ycc = cover_ycc + torch.tanh(resid_ycc) * scale
        stego_rgb = ycbcr_to_rgb(stego_ycc)
        stego_spatial = x.clone()
        stego_spatial[:, :3, :, :] = stego_rgb
        stego_spatial[:, 3:, :, :] = residual_spatial[:, 3:, :, :]
        return stego_spatial, log_det_total

    def inverse(self, z):
        z_freq = self.ilwt(z)
        x_freq = z_freq
        for i in range(len(self.blocks) - 1, -1, -1):
            x_freq = self.blocks[i].inverse(x_freq, cond=None)
        residual_spatial = self.ilwt.inverse(x_freq)
        z_ycc = rgb_to_ycbcr(z[:, :3, :, :])
        resid_ycc = rgb_to_ycbcr(residual_spatial[:, :3, :, :])
        kY, kC = self.ycbcr_scaling.get_scales()
        scale = torch.stack([kY, kC, kC]).view(1, 3, 1, 1).to(z.device)
        host_ycc = z_ycc - torch.tanh(resid_ycc) * scale
        host_rgb = ycbcr_to_rgb(host_ycc)
        x_spatial = z.clone()
        x_spatial[:, :3, :, :] = host_rgb
        x_spatial[:, 3:, :, :] = residual_spatial[:, 3:, :, :]
        return x_spatial


CONFIGS = [
    {"name": "baseline",     "num_blocks": 8, "hidden_channels": 128, "use_cond": True},
    {"name": "A_4blocks",    "num_blocks": 4, "hidden_channels": 128, "use_cond": True},
    {"name": "B_64ch",       "num_blocks": 8, "hidden_channels": 64,  "use_cond": True},
    {"name": "C_no_cond",    "num_blocks": 8, "hidden_channels": 128, "use_cond": False},
]


def build_model(cfg):
    if cfg["use_cond"]:
        return StegoWaveFlow(
            channels=6,
            num_blocks=cfg["num_blocks"],
            hidden_channels=cfg["hidden_channels"],
            transform_type="ilwt53",
        )
    return StegoWaveFlowNoCond(
        channels=6,
        num_blocks=cfg["num_blocks"],
        hidden_channels=cfg["hidden_channels"],
        transform_type="ilwt53",
    )


# ----------------------------------------------------------------------------
# Training / eval loop
# ----------------------------------------------------------------------------

def train_one_config(cfg, train_loader, val_loader, device, num_epochs):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    epoch_secs = []
    last_val_hid = last_val_rec = last_val_ssim = 0.0

    for epoch in range(num_epochs):
        model.train()
        t0 = time.perf_counter()
        running_loss = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"  [{cfg['name']}] ep {epoch+1}/{num_epochs}", leave=False, ncols=100)
        for input_tensor, host_tensor, secret_tensor in pbar:
            input_tensor = input_tensor.to(device, non_blocking=True)
            host_tensor = host_tensor.to(device, non_blocking=True)
            secret_tensor = secret_tensor.to(device, non_blocking=True)

            optimizer.zero_grad()
            stego_output, _ = model(input_tensor)
            stego_host = stego_output[:, :3, :, :]
            stego_host_q = perturb_stego_like_for_inverse(stego_host)
            stego_like = torch.cat([stego_host_q, torch.zeros_like(stego_host_q)], dim=1)
            reconstructed = model.inverse(stego_like)
            recovered_secret = reconstructed[:, 3:, :, :]

            loss, _, _, _, _, _ = steganography_loss(
                secret_tensor,
                recovered_secret,
                host_tensor,
                stego_host_q,
                model.ilwt,
                model.subband_weights,
                target_psnr=42.0,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            n_seen += 1
            pbar.set_postfix(loss=f"{running_loss / n_seen:.3f}")
        scheduler.step()
        t1 = time.perf_counter()
        epoch_secs.append(t1 - t0)

        # Val every 5 epochs and on the last
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            hids, recs, ssims = [], [], []
            with torch.no_grad():
                for input_tensor, host_tensor, secret_tensor in tqdm(val_loader, desc=f"  [{cfg['name']}] val ep {epoch+1}", leave=False, ncols=100):
                    input_tensor = input_tensor.to(device, non_blocking=True)
                    host_tensor = host_tensor.to(device, non_blocking=True)
                    secret_tensor = secret_tensor.to(device, non_blocking=True)
                    stego_output, _ = model(input_tensor)
                    stego_host = stego_output[:, :3, :, :]
                    stego_host_q = perturb_stego_like_for_inverse(stego_host)
                    stego_like = torch.cat([stego_host_q, torch.zeros_like(stego_host_q)], dim=1)
                    reconstructed = model.inverse(stego_like)
                    recovered_secret = reconstructed[:, 3:, :, :]
                    h = calculate_psnr(stego_host_q, host_tensor)
                    r = calculate_psnr(recovered_secret, secret_tensor)
                    s = calculate_ssim(recovered_secret, secret_tensor)
                    hids.append(h.item() if torch.is_tensor(h) else float(h))
                    recs.append(r.item() if torch.is_tensor(r) else float(r))
                    ssims.append(s.item() if torch.is_tensor(s) else float(s))
            last_val_hid = float(np.mean(hids))
            last_val_rec = float(np.mean(recs))
            last_val_ssim = float(np.mean(ssims))
            avg_ep_secs = float(np.mean(epoch_secs[-5:]))
            remaining_ep = num_epochs - (epoch + 1)
            cfg_eta_secs = avg_ep_secs * remaining_ep
            print(f"  [{cfg['name']}] epoch {epoch+1}/{num_epochs}  "
                  f"hidPSNR={last_val_hid:.2f}  recPSNR={last_val_rec:.2f}  "
                  f"recSSIM={last_val_ssim:.3f}  s/ep={avg_ep_secs:.1f}  "
                  f"this-cfg ETA: ~{cfg_eta_secs/60:.1f}m")

    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    return {
        "name": cfg["name"],
        "params": n_params,
        "avg_sec_per_epoch": float(np.mean(epoch_secs)),
        "val_hiding_psnr": last_val_hid,
        "val_recovery_psnr": last_val_rec,
        "val_recovery_ssim": last_val_ssim,
        "peak_gpu_mem_mb": float(peak_mem_mb),
    }


def main():
    NUM_IMAGES = 300
    NUM_EPOCHS = 20
    IMG_SIZE = 256
    DATA_DIR = "training_data"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    full = ImageSteganographyDataset(DATA_DIR, img_size=IMG_SIZE)
    print(f"Full dataset: {len(full)} images, using {NUM_IMAGES}")
    rng = np.random.default_rng(42)
    idxs = rng.permutation(len(full))[:NUM_IMAGES].tolist()

    n_train = int(0.7 * NUM_IMAGES)
    n_val = int(0.2 * NUM_IMAGES)
    train_idx, val_idx = idxs[:n_train], idxs[n_train : n_train + n_val]
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    train_loader = DataLoader(Subset(full, train_idx), batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(Subset(full, val_idx),   batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    results = []
    sweep_start = time.perf_counter()
    cfg_durations = []
    print(f"\nStarting sweep: {len(CONFIGS)} configs × {NUM_EPOCHS} epochs × {len(train_idx)} train steps")
    print(f"Rough estimate (~7 it/s GPU): {len(CONFIGS) * NUM_EPOCHS * len(train_idx) / 7 / 60:.0f}m total")

    for i, cfg in enumerate(CONFIGS, 1):
        print(f"\n=== [{i}/{len(CONFIGS)}] {cfg['name']} ({cfg}) ===")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        cfg_start = time.perf_counter()
        r = train_one_config(cfg, train_loader, val_loader, device, NUM_EPOCHS)
        cfg_elapsed = time.perf_counter() - cfg_start
        cfg_durations.append(cfg_elapsed)
        results.append(r)

        elapsed_total = time.perf_counter() - sweep_start
        remaining_configs = len(CONFIGS) - i
        avg_cfg_time = sum(cfg_durations) / len(cfg_durations)
        eta_secs = avg_cfg_time * remaining_configs
        print(f"  -> params={r['params']:,}  sec/epoch={r['avg_sec_per_epoch']:.1f}  "
              f"hidPSNR={r['val_hiding_psnr']:.2f}  recPSNR={r['val_recovery_psnr']:.2f}  "
              f"mem={r['peak_gpu_mem_mb']:.0f}MB")
        print(f"  this config: {cfg_elapsed/60:.1f}m  |  "
              f"elapsed: {elapsed_total/60:.1f}m  |  "
              f"remaining: {remaining_configs} cfg(s), ~{eta_secs/60:.1f}m  |  "
              f"projected total: {(elapsed_total + eta_secs)/60:.1f}m")

    print("\n" + "=" * 90)
    print(f"{'Config':<14} {'Params':>10}  {'s/ep':>6}  {'hidPSNR':>9}  {'recPSNR':>9}  {'recSSIM':>9}  {'GPU MB':>8}")
    print("-" * 90)
    base = results[0]
    for r in results:
        d_rec = r["val_recovery_psnr"] - base["val_recovery_psnr"]
        speedup = base["avg_sec_per_epoch"] / max(r["avg_sec_per_epoch"], 1e-6)
        print(f"{r['name']:<14} {r['params']:>10,}  {r['avg_sec_per_epoch']:>6.1f}  "
              f"{r['val_hiding_psnr']:>9.2f}  {r['val_recovery_psnr']:>9.2f}  "
              f"{r['val_recovery_ssim']:>9.3f}  {r['peak_gpu_mem_mb']:>8.0f}  "
              f"(Δrec={d_rec:+.2f}  {speedup:.2f}x)")
    print("=" * 90)

    Path("ablation_results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved ablation_results.json")


if __name__ == "__main__":
    main()
