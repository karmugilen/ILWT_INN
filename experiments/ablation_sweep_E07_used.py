"""
Full ablation sweep for the ablation table.

Runs 19 unique configurations on a fixed 1000-image split, writes resumable
per-config outputs, and rebuilds grouped CSV/Markdown tables after every run.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import csv
import json
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from shell.io import save_8bit_png
from train import (
    ImageSteganographyDataset,
    quantize_to_8bit_ste,
    steganography_loss,
    calculate_psnr,
    calculate_ssim,
    calculate_ber,
    calculate_ncc,
    calculate_lpips,
)
from core.models import StegoWaveFlow


@dataclass(frozen=True)
class Cfg:
    name: str
    group: str
    label: str
    num_blocks: int = 8
    hidden_channels: int = 128
    transform_type: str = "ilwt53"
    use_conditioning: bool = True
    ycbcr_mode: str = "learnable"
    ycbcr_init_kY: float = 0.02
    ycbcr_init_kC: float = 0.06
    use_subband_weights: bool = True
    use_actnorm: bool = True
    use_inv1x1: bool = True
    use_ste_in_training: bool = True


BASELINE = Cfg(name="baseline", group="-", label="ILWT 5/3 (Proposed)")

CONFIGS = [
    Cfg("depth_2", "Depth", "2 blocks", num_blocks=2),
    Cfg("width_32", "Width", "32", hidden_channels=32),
    Cfg("depth_4", "Depth", "4 blocks", num_blocks=4),
    Cfg("width_64", "Width", "64", hidden_channels=64),
    BASELINE,
    Cfg("no_cond", "Conditioning", "No-Cond", use_conditioning=False),
    Cfg("no_actnorm", "INN Block", "No-ActNorm", use_actnorm=False),
    Cfg("no_inv1x1", "INN Block", "No-Inv1x1", use_inv1x1=False),
    Cfg("ste_off", "Quantization", "STE-Off", use_ste_in_training=False),
    Cfg("subband_uniform", "Weight", "Fixed/Uniform", use_subband_weights=False),
    Cfg("ycbcr_frozen", "Color Space", "Frozen YCbCr (0.02, 0.06)", ycbcr_mode="frozen"),
    Cfg("ycbcr_fixed", "Color Space", "Fixed YCbCr (0.04, 0.10)", ycbcr_mode="fixed", ycbcr_init_kY=0.04, ycbcr_init_kC=0.10),
    Cfg("ycbcr_off", "Color Space", "No YCbCr (full RGB)", ycbcr_mode="off"),
    Cfg("haar_lifting", "Transform", "Haar Lifting", transform_type="haar"),
    Cfg("dwt97_float", "Transform", "Float DWT 9/7", transform_type="dwt97_float"),
    Cfg("depth_12", "Depth", "12 blocks", num_blocks=12),
    Cfg("width_256", "Width", "256", hidden_channels=256),
    Cfg("depth_16", "Depth", "16 blocks", num_blocks=16),
    Cfg("spatial_only", "Transform", "Identity (Spatial)", transform_type="none"),
]

CFG_BY_NAME = {cfg.name: cfg for cfg in CONFIGS}

OUT_ROOT = Path("ablation_runs")
NUM_IMAGES = 200
NUM_EPOCHS = 100
IMG_SIZE = 256
DATA_DIR = "hide"
ACCUM_STEPS = 8
VALIDATE_EVERY = 5

TABLE_ORDER = [
    ("Transform", ["spatial_only", "haar_lifting", "dwt97_float", "baseline"]),
    ("Depth", ["depth_2", "depth_4", "baseline", "depth_12", "depth_16"]),
    ("Width", ["width_32", "width_64", "baseline", "width_256"]),
    ("Weight", ["subband_uniform", "baseline"]),
    ("Color Space", ["ycbcr_off", "ycbcr_fixed", "ycbcr_frozen", "baseline"]),
    ("Conditioning", ["no_cond", "baseline"]),
    ("Quantization", ["ste_off", "baseline"]),
    ("INN Block", ["no_actnorm", "no_inv1x1", "baseline"]),
]


def _fmt_duration(seconds: float) -> str:
    """Format seconds as human-readable duration like '1h 23m 45s'."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def install_log_tee(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", buffering=1)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    print(f"\n\n===== ablation_sweep start {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    return log_file, old_stdout, old_stderr


def build_model(cfg: Cfg):
    return StegoWaveFlow(
        channels=6,
        num_blocks=cfg.num_blocks,
        hidden_channels=cfg.hidden_channels,
        transform_type=cfg.transform_type,
        use_conditioning=cfg.use_conditioning,
        ycbcr_mode=cfg.ycbcr_mode,
        ycbcr_init_kY=cfg.ycbcr_init_kY,
        ycbcr_init_kC=cfg.ycbcr_init_kC,
        use_subband_weights=cfg.use_subband_weights,
        use_actnorm=cfg.use_actnorm,
        use_inv1x1=cfg.use_inv1x1,
        use_ste=cfg.use_ste_in_training,
    )


def shared_splits(full_dataset):
    if len(full_dataset) < NUM_IMAGES:
        raise ValueError(f"{DATA_DIR} has {len(full_dataset)} images; {NUM_IMAGES} required")
    rng = np.random.default_rng(42)
    idxs = rng.permutation(len(full_dataset))[:NUM_IMAGES].tolist()
    n_train = int(0.70 * NUM_IMAGES)
    n_val = int(0.15 * NUM_IMAGES)
    return idxs[:n_train], idxs[n_train:n_train + n_val], idxs[n_train + n_val:]


def stego_roundtrip(model, input_tensor, host_tensor, cfg: Cfg):
    stego_output, _ = model(input_tensor)
    stego_host = stego_output[:, :3]
    stego_eval = quantize_to_8bit_ste(stego_host) if cfg.use_ste_in_training else stego_host
    stego_like = torch.cat([stego_eval, torch.zeros_like(stego_eval)], dim=1)
    reconstructed = model.inverse(stego_like)
    return stego_eval, reconstructed[:, 3:]


def summarize_metrics(values):
    return float(np.mean(values)) if values else 0.0


def evaluate_loader(model, cfg: Cfg, loader, device, with_lpips: bool):
    model.eval()
    metrics = {
        "hiding_psnr": [],
        "recovery_psnr": [],
        "ber": [],
        "ncc": [],
        "rec_ssim": [],
        "lpips": [],
    }
    with torch.no_grad():
        for input_tensor, host_tensor, secret_tensor in tqdm(loader, desc=f"  [{cfg.name}] eval", leave=False, ncols=100):
            input_tensor = input_tensor.to(device, non_blocking=True)
            host_tensor = host_tensor.to(device, non_blocking=True)
            secret_tensor = secret_tensor.to(device, non_blocking=True)
            stego_host, recovered_secret = stego_roundtrip(model, input_tensor, host_tensor, cfg)
            metrics["hiding_psnr"].append(float(calculate_psnr(stego_host, host_tensor).item()))
            metrics["recovery_psnr"].append(float(calculate_psnr(recovered_secret, secret_tensor).item()))
            metrics["ber"].append(float(calculate_ber(recovered_secret, secret_tensor)))
            metrics["ncc"].append(float(calculate_ncc(recovered_secret, secret_tensor)))
            metrics["rec_ssim"].append(float(calculate_ssim(recovered_secret, secret_tensor).item()))
            if with_lpips:
                metrics["lpips"].append(float(calculate_lpips(recovered_secret, secret_tensor)))
    return {key: summarize_metrics(value) for key, value in metrics.items()}


def save_samples(model, cfg: Cfg, test_loader, device, run_dir: Path, count: int = 5):
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_metrics = []
    model.eval()
    saved = 0
    with torch.no_grad():
        for input_tensor, host_tensor, secret_tensor in test_loader:
            input_tensor = input_tensor.to(device, non_blocking=True)
            host_tensor = host_tensor.to(device, non_blocking=True)
            secret_tensor = secret_tensor.to(device, non_blocking=True)
            stego_host, recovered_secret = stego_roundtrip(model, input_tensor, host_tensor, cfg)
            idx = saved + 1
            save_8bit_png(host_tensor[0].cpu(), samples_dir / f"{idx:02d}_cover.png")
            save_8bit_png(stego_host[0].cpu(), samples_dir / f"{idx:02d}_stego.png")
            save_8bit_png(secret_tensor[0].cpu(), samples_dir / f"{idx:02d}_secret.png")
            save_8bit_png(recovered_secret[0].cpu(), samples_dir / f"{idx:02d}_recovered.png")
            sample_metrics.append({
                "sample": idx,
                "hiding_psnr": float(calculate_psnr(stego_host, host_tensor).item()),
                "recovery_psnr": float(calculate_psnr(recovered_secret, secret_tensor).item()),
                "ber_percent": float(calculate_ber(recovered_secret, secret_tensor) * 100.0),
                "ncc": float(calculate_ncc(recovered_secret, secret_tensor)),
                "rec_ssim": float(calculate_ssim(recovered_secret, secret_tensor).item()),
            })
            saved += 1
            if saved >= count:
                break
    (samples_dir / "metrics.json").write_text(json.dumps(sample_metrics, indent=2))


def model_config_dict(cfg: Cfg):
    return {
        "channels": 6,
        "num_blocks": cfg.num_blocks,
        "hidden_channels": cfg.hidden_channels,
        "transform_type": cfg.transform_type,
        "size": IMG_SIZE,
        "use_conditioning": cfg.use_conditioning,
        "ycbcr_mode": cfg.ycbcr_mode,
        "ycbcr_init_kY": cfg.ycbcr_init_kY,
        "ycbcr_init_kC": cfg.ycbcr_init_kC,
        "use_subband_weights": cfg.use_subband_weights,
        "use_actnorm": cfg.use_actnorm,
        "use_inv1x1": cfg.use_inv1x1,
        "use_ste": cfg.use_ste_in_training,
    }


def run_one(cfg: Cfg, train_loader, val_loader, test_loader, device):
    run_dir = OUT_ROOT / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    ckpt_path = run_dir / "checkpoint.pth"
    if ckpt_path.exists():
        print(f"[skip] {cfg.name}: checkpoint already exists")
        return evaluate_on_test(cfg, ckpt_path, test_loader, device, run_dir)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    with (run_dir / "train_log.csv").open("w", newline="") as train_csv, (run_dir / "epoch_log.csv").open("w", newline="") as epoch_csv:
        train_w = csv.writer(train_csv)
        epoch_w = csv.writer(epoch_csv)
        train_w.writerow(["step", "epoch", "loss"])
        epoch_w.writerow(["epoch", "train_loss", "val_hidPSNR", "val_recPSNR", "val_BER", "val_NCC", "val_SSIM", "val_LPIPS", "s_per_epoch", "peak_mem_MB"])

        global_step = 0
        epoch_secs = []
        config_t0 = time.perf_counter()
        for epoch in range(NUM_EPOCHS):
            model.train()
            optimizer.zero_grad()
            t0 = time.perf_counter()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"  [{cfg.name}] ep {epoch + 1}/{NUM_EPOCHS}", leave=False, ncols=100)
            for batch_idx, (input_tensor, host_tensor, secret_tensor) in enumerate(pbar):
                input_tensor = input_tensor.to(device, non_blocking=True)
                host_tensor = host_tensor.to(device, non_blocking=True)
                secret_tensor = secret_tensor.to(device, non_blocking=True)

                stego_host, recovered_secret = stego_roundtrip(model, input_tensor, host_tensor, cfg)
                loss, _, _, _, _, _ = steganography_loss(
                    secret_tensor,
                    recovered_secret,
                    host_tensor,
                    stego_host,
                    model.ilwt,
                    model.subband_weights,
                    target_psnr=42.0,
                )
                (loss / ACCUM_STEPS).backward()
                if ((batch_idx + 1) % ACCUM_STEPS == 0) or ((batch_idx + 1) == len(train_loader)):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                running_loss += loss.item()
                global_step += 1
                train_w.writerow([global_step, epoch + 1, f"{loss.item():.8f}"])
                pbar.set_postfix(loss=f"{running_loss / (batch_idx + 1):.4f}")

            scheduler.step()
            sec = time.perf_counter() - t0
            epoch_secs.append(sec)
            avg_loss = running_loss / max(1, len(train_loader))

            # ETA for this config
            avg_sec = np.mean(epoch_secs)
            remaining_epochs = NUM_EPOCHS - (epoch + 1)
            eta_config_sec = avg_sec * remaining_epochs
            elapsed_config = time.perf_counter() - config_t0

            if ((epoch + 1) % VALIDATE_EVERY == 0) or ((epoch + 1) == NUM_EPOCHS):
                val_metrics = evaluate_loader(model, cfg, val_loader, device, with_lpips=False)
                peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
                epoch_w.writerow([
                    epoch + 1,
                    f"{avg_loss:.8f}",
                    f"{val_metrics['hiding_psnr']:.4f}",
                    f"{val_metrics['recovery_psnr']:.4f}",
                    f"{val_metrics['ber'] * 100.0:.6f}",
                    f"{val_metrics['ncc']:.6f}",
                    f"{val_metrics['rec_ssim']:.6f}",
                    "",
                    f"{sec:.3f}",
                    f"{peak_mem:.1f}",
                ])
                print(
                    f"  [{cfg.name}] epoch {epoch + 1}/{NUM_EPOCHS} "
                    f"hidPSNR={val_metrics['hiding_psnr']:.2f} "
                    f"recPSNR={val_metrics['recovery_psnr']:.2f} "
                    f"BER={val_metrics['ber'] * 100.0:.4f}% "
                    f"s/ep={sec:.1f} "
                    f"elapsed={_fmt_duration(elapsed_config)} "
                    f"ETA={_fmt_duration(eta_config_sec)}"
                )

    save_samples(model, cfg, test_loader, device, run_dir)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config_dict(cfg),
            "ablation_config": asdict(cfg),
            "params": n_params,
            "avg_sec_per_epoch": float(np.mean(epoch_secs)),
        },
        ckpt_path,
    )
    return evaluate_on_test(cfg, ckpt_path, test_loader, device, run_dir)


def evaluate_on_test(cfg: Cfg, ckpt_path: Path, test_loader, device, run_dir: Path):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    params = int(checkpoint.get("params", sum(p.numel() for p in model.parameters())))
    sec_per_epoch = float(checkpoint.get("avg_sec_per_epoch", 0.0))
    metrics = evaluate_loader(model, cfg, test_loader, device, with_lpips=True)
    row = {
        "name": cfg.name,
        "group": cfg.group,
        "label": cfg.label,
        "hiding_psnr": metrics["hiding_psnr"],
        "recovery_psnr": metrics["recovery_psnr"],
        "ber_percent": metrics["ber"] * 100.0,
        "ncc": metrics["ncc"],
        "rec_ssim": metrics["rec_ssim"],
        "lpips": metrics["lpips"],
        "params": params,
        "sec_per_epoch": sec_per_epoch,
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(row, indent=2))
    return row


def format_row(row):
    return [
        row["group"],
        row["label"],
        f"{row['hiding_psnr']:.3f}",
        f"{row['recovery_psnr']:.3f}",
        f"{row['ber_percent']:.6f}",
        f"{row['ncc']:.6f}",
        f"{row['rec_ssim']:.6f}",
        f"{row['lpips']:.6f}",
        str(row["params"]),
        f"{row['sec_per_epoch']:.3f}",
    ]


def table_rows(rows):
    by_name = {row["name"]: row for row in rows}
    out = []
    for group, names in TABLE_ORDER:
        for name in names:
            if name not in by_name:
                continue
            row = dict(by_name[name])
            row["group"] = group
            if name == "baseline":
                proposed_labels = {
                    "Transform": "ILWT 5/3 (Proposed)",
                    "Depth": "8 blocks (Proposed)",
                    "Width": "128 (Proposed)",
                    "Weight": "Learnable (Proposed)",
                    "Color Space": "Learnable YCbCr (Proposed)",
                    "Conditioning": "With-Cond (Proposed)",
                    "Quantization": "STE-On (Proposed)",
                    "INN Block": "Full Block (Proposed)",
                }
                row["label"] = proposed_labels[group]
            out.append(row)
    return out


def write_table(rows):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    headers = ["Group", "Row", "Hiding PSNR", "Recovery PSNR", "BER (%)", "NCC", "Rec-SSIM", "LPIPS", "Params", "sec/epoch"]
    ordered = table_rows(rows)
    with (OUT_ROOT / "ablation_table.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in ordered:
            writer.writerow(format_row(row))

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in ordered:
        lines.append("| " + " | ".join(format_row(row)) + " |")
    (OUT_ROOT / "ablation_table.md").write_text("\n".join(lines) + "\n")


def parse_csv_names(value: str):
    names = []
    for part in value.split(","):
        name = part.strip()
        if name:
            names.append(name)
    return names


def select_configs(args):
    if args.list:
        return []

    if args.all:
        selected = list(CONFIGS)
    elif args.config:
        requested = parse_csv_names(args.config)
        unknown = [name for name in requested if name not in CFG_BY_NAME]
        if unknown:
            valid = ", ".join(cfg.name for cfg in CONFIGS)
            raise ValueError(f"Unknown config(s): {', '.join(unknown)}\nValid configs: {valid}")
        selected = [CFG_BY_NAME[name] for name in requested]
    elif args.from_config or args.to_config:
        names = [cfg.name for cfg in CONFIGS]
        start_name = args.from_config or names[0]
        end_name = args.to_config or names[-1]
        if start_name not in names:
            raise ValueError(f"Unknown --from config: {start_name}")
        if end_name not in names:
            raise ValueError(f"Unknown --to config: {end_name}")
        start = names.index(start_name)
        end = names.index(end_name)
        if start > end:
            raise ValueError("--from must come before --to in the configured run order")
        selected = CONFIGS[start:end + 1]
    else:
        selected = [CONFIGS[0]]

    if args.limit is not None:
        selected = selected[:args.limit]
    return selected


def parse_number_selection(selection: str):
    selected = []
    seen = set()
    max_idx = len(CONFIGS)
    for raw_part in selection.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = [p.strip() for p in part.split("-", 1)]
            if not start_text.isdigit() or not end_text.isdigit():
                raise ValueError(f"Invalid range: {part}")
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid range order: {part}")
            numbers = range(start, end + 1)
        else:
            if not part.isdigit():
                raise ValueError(f"Invalid selection: {part}")
            numbers = [int(part)]

        for number in numbers:
            if number < 1 or number > max_idx:
                raise ValueError(f"Selection {number} is out of range 1-{max_idx}")
            if number not in seen:
                selected.append(CONFIGS[number - 1])
                seen.add(number)
    return selected


def interactive_select_configs():
    print_config_list()
    print("")
    print("Select configs by number, separated by commas.")
    print("Examples: 1   or   1,2,3   or   5-9")
    print("Type 'all' for all configs, 'q' to quit.")
    while True:
        choice = input("Selection: ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            return []
        if choice == "all":
            return list(CONFIGS)
        try:
            selected = parse_number_selection(choice)
        except ValueError as exc:
            print(f"Error: {exc}")
            continue
        if selected:
            return selected
        print("No configs selected.")


def read_existing_rows():
    rows = []
    for cfg in CONFIGS:
        metrics_path = OUT_ROOT / cfg.name / "test_metrics.json"
        if not metrics_path.exists():
            continue
        rows.append(json.loads(metrics_path.read_text()))
    return rows


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run selected ablation-study configs and auto-save logs/tables."
    )
    parser.add_argument("--list", action="store_true", help="List configs in run order and exit.")
    parser.add_argument("--all", action="store_true", help="Run all configs in the planned order.")
    parser.add_argument("--config", help="Comma-separated config names to run, e.g. depth_2,width_32.")
    parser.add_argument("--from", dest="from_config", help="Run from this config name in planned order.")
    parser.add_argument("--to", dest="to_config", help="Run through this config name in planned order.")
    parser.add_argument("--limit", type=int, help="Run at most this many selected configs.")
    parser.add_argument("--log", default=str(OUT_ROOT / "sweep.log"), help="Log file path.")
    parser.add_argument("--rebuild-table", action="store_true", help="Only rebuild tables from existing test_metrics.json files.")
    parser.add_argument("--no-menu", action="store_true", help="Do not show the interactive menu; run the first config if no selection flags are passed.")
    return parser


def print_config_list():
    print("Ablation configs in run order:")
    for i, cfg in enumerate(CONFIGS, 1):
        print(f"{i:02d}. {cfg.name:<16} {cfg.group:<13} {cfg.label}")


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_file, old_stdout, old_stderr = install_log_tee(Path(args.log))
    try:
        if args.list:
            print_config_list()
            return

        has_selection_flags = any([args.all, args.config, args.from_config, args.to_config, args.limit is not None])
        if has_selection_flags or args.no_menu:
            selected = select_configs(args)
        else:
            selected = interactive_select_configs()
        if args.rebuild_table:
            rows = []
            for cfg in CONFIGS:
                metrics_path = OUT_ROOT / cfg.name / "test_metrics.json"
                if metrics_path.exists():
                    rows.append(json.loads(metrics_path.read_text()))
            write_table(rows)
            print(f"Rebuilt table from {len(rows)} existing metrics file(s).")
            return

        if not selected:
            print("No configs selected.")
            return

        print("Selected configs:")
        for cfg in selected:
            print(f"  - {cfg.name}")

        completed_rows = read_existing_rows()
        rows_by_name = {row["name"]: row for row in completed_rows}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        full = ImageSteganographyDataset(DATA_DIR, img_size=IMG_SIZE, augment=True)
        full_noaug = ImageSteganographyDataset(DATA_DIR, img_size=IMG_SIZE, augment=False)
        train_idx, val_idx, test_idx = shared_splits(full)
        print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        train_loader = DataLoader(Subset(full, train_idx), batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(Subset(full_noaug, val_idx), batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(Subset(full_noaug, test_idx), batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        sweep_t0 = time.perf_counter()
        config_durations: list[float] = []
        for i, cfg in enumerate(selected, 1):
            print(f"\n=== [{i}/{len(selected)}] {cfg.name} ===")
            cfg_t0 = time.perf_counter()
            row = run_one(cfg, train_loader, val_loader, test_loader, device)
            cfg_dur = time.perf_counter() - cfg_t0
            config_durations.append(cfg_dur)
            rows_by_name[row["name"]] = row
            write_table([rows_by_name[name] for name in rows_by_name])

            # Sweep-level ETA
            sweep_elapsed = time.perf_counter() - sweep_t0
            avg_cfg_dur = np.mean(config_durations)
            remaining_configs = len(selected) - i
            sweep_eta = avg_cfg_dur * remaining_configs
            print(
                f"  -> test hidPSNR={row['hiding_psnr']:.2f} "
                f"recPSNR={row['recovery_psnr']:.2f} "
                f"BER={row['ber_percent']:.4f}%"
            )
            print(
                f"  ⏱  config took {_fmt_duration(cfg_dur)} | "
                f"sweep elapsed {_fmt_duration(sweep_elapsed)} | "
                f"{remaining_configs} configs left | "
                f"sweep ETA {_fmt_duration(sweep_eta)}"
            )

        total_sweep = time.perf_counter() - sweep_t0
        print(f"\nDone. Total sweep time: {_fmt_duration(total_sweep)}")
        print("See ablation_runs/ablation_table.csv")
    finally:
        print(f"===== ablation_sweep end {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_file.close()


if __name__ == "__main__":
    main()
