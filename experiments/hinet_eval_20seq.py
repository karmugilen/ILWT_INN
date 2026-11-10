#!/usr/bin/env python
"""
HiNet vs ILWT-INN on the 20 Xiph sequences under the same 8-bit storage.

Same frames and secrets as the standard protocol (stego set from make_stego_set.py --secret-dir
hide300: frame i of every sequence carries secret[i % 300]). Covers are re-extracted with the
same ffmpeg lanczos 256x256 resize. For every frame:
  ILWT-INN       its stored 8-bit stego frame, blind recovery from it
  HiNet <name>   embed cover+secret, round the stego to 8 bits, recover from the 8-bit stego with
                 Gaussian z (HiNet's own reverse; torch seed 0 per sequence)
  --float        additionally every model "[float]": recovery from the unquantised float stego (a
                 no-quantisation ceiling; a float frame cannot be stored in a video file)
Metrics (eval_common): H_PSNR, H_SSIM (cover vs stego), R_PSNR, SSIM, NC, BER, E_XAI.

Outputs (--out): per_frame.csv, summary.csv (mean, SD, 95% CI across sequences),
wilcoxon.csv (two-sided Wilcoxon signed-rank on the 20 per-sequence means, ILWT-INN vs each HiNet).

Usage: hinet_eval_20seq.py --hinet name=checkpoint.pt [name=checkpoint.pt ...]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as E  # noqa: E402
from make_stego_set import SEQUENCES  # noqa: E402

S = 256
METRICS = ("H_PSNR", "H_SSIM", "R_PSNR", "SSIM", "NC", "BER", "E_XAI")


def covers_of(videos, rel):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(Path(videos) / rel),
                          "-vf", f"scale={S}:{S}:flags=lanczos", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, S, S, 3)


def to01(u8, device):
    return torch.from_numpy(np.ascontiguousarray(u8.transpose(0, 3, 1, 2))).to(device).float().div(255)


def from01(t):
    return (t.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=str(E.ROOT / "runs/stego_hide_current_may28"))
    ap.add_argument("--secret-dir", default=str(E.ROOT / "data/hide300"))
    ap.add_argument("--videos", default=str(E.ROOT / "data/sd_videos"))
    ap.add_argument("--hinet-dir", default=str(E.ROOT / "baselines/HiNet"))
    ap.add_argument("--hinet", nargs="+", required=True, help="name=checkpoint (HiNet {'net': state_dict})")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--float", action="store_true", help="also score every model on its float (unquantised) stego")
    ap.add_argument("--out", default=str(E.ROOT / "runs/hinet_eval_20seq"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model, device = E.load_model()
    sys.path.insert(0, args.hinet_dir)
    from model import Model
    import modules.Unet_common as common
    dwt, iwt = common.DWT(), common.IWT()
    nets = {}
    for spec in args.hinet:
        name, ck = spec.split("=")
        net = torch.nn.DataParallel(Model().to(device), device_ids=[0])
        sd = torch.load(ck, map_location=device, weights_only=False)["net"]
        net.load_state_dict({k: v for k, v in sd.items() if "tmp_var" not in k}); net.eval()
        nets[f"HiNet {name}"] = net

    table = E.read_frame_table(args.set)
    rows = []
    for seq, frames in table.items():
        allc = covers_of(args.videos, SEQUENCES[seq])
        idx = [int(f.split("_")[-1]) - 1 for f, _ in frames]
        covers = allc[idx]
        secrets = np.stack([E.load_secret_u8(Path(args.secret_dir) / s) for _, s in frames])
        stegos = {"ILWT-INN": np.stack([E.read_rgb(Path(args.set) / "stego" / seq / f"{f}.png") for f, _ in frames])}
        recs = {"ILWT-INN": E.recover(model, stegos["ILWT-INN"], device)}
        with torch.no_grad():
            for name, net in nets.items():
                torch.manual_seed(0)
                st, rc = [], []
                for i in range(0, len(frames), args.batch):
                    c, s = to01(covers[i:i + args.batch], device), to01(secrets[i:i + args.batch], device)
                    steg_q = torch.round(iwt(net(torch.cat((dwt(c), dwt(s)), 1)).narrow(1, 0, 12)).clamp(0, 1) * 255) / 255
                    d = dwt(steg_q)
                    rev = net(torch.cat((d, torch.randn_like(d)), 1), rev=True)
                    st.append(from01(steg_q)); rc.append(from01(iwt(rev.narrow(1, 12, 12))))
                stegos[name], recs[name] = np.concatenate(st), np.concatenate(rc)
            if args.float:
                # no-quantisation ceiling: recover from the float stego (not storable as a video frame)
                st, rc = [], []
                for i in range(0, len(frames), args.batch):
                    x = torch.cat([E.u8_to_t(covers[i:i + args.batch], device), E.u8_to_t(secrets[i:i + args.batch], device)], 1)
                    sf = model(x)[0][:, :3].clamp(-1, 1)
                    st.append(((sf + 1) / 2 * 255).permute(0, 2, 3, 1).cpu().numpy())
                    rc.append(E.t_to_u8(model.inverse(torch.cat([sf, torch.zeros_like(sf)], 1))[:, 3:]))
                stegos["ILWT-INN [float]"], recs["ILWT-INN [float]"] = np.concatenate(st), np.concatenate(rc)
                for name, net in nets.items():
                    torch.manual_seed(0)
                    st, rc = [], []
                    for i in range(0, len(frames), args.batch):
                        c, s = to01(covers[i:i + args.batch], device), to01(secrets[i:i + args.batch], device)
                        steg = iwt(net(torch.cat((dwt(c), dwt(s)), 1)).narrow(1, 0, 12)).clamp(0, 1)
                        d = dwt(steg)
                        rev = net(torch.cat((d, torch.randn_like(d)), 1), rev=True)
                        st.append((steg * 255).permute(0, 2, 3, 1).cpu().numpy()); rc.append(from01(iwt(rev.narrow(1, 12, 12))))
                    stegos[f"{name} [float]"], recs[f"{name} [float]"] = np.concatenate(st), np.concatenate(rc)
        for name in stegos:
            for (f, s), c, st, sec, r in zip(frames, covers, stegos[name], secrets, recs[name]):
                m = E.frame_metrics(sec, r, device)
                rows.append({"model": name, "seq": seq, "frame": f, "secret": s,
                             "H_PSNR": round(E.psnr_u8(c, st), 4), "H_SSIM": round(E.ssim_u8(c, st, device), 6),
                             **{k: round(v, 6) for k, v in m.items()}})
            cur = [x for x in rows if x["model"] == name and x["seq"] == seq]
            print(f"{seq:13s} {name:24s} H {np.mean([x['H_PSNR'] for x in cur]):.2f}  "
                  f"R {np.mean([x['R_PSNR'] for x in cur]):.2f}  BER {np.mean([x['BER'] for x in cur]):.4f}", flush=True)
        E.write_csv(out / "per_frame.csv", rows)

    summ = E.summarize(rows, ["model"], METRICS)
    E.write_csv(out / "summary.csv", [{k: v for k, v in r.items() if not k.startswith("frac_")} for r in summ])
    seqs = list(table)
    per_seq = {(m, s): {k: np.mean([x[k] for x in rows if x["model"] == m and x["seq"] == s]) for k in METRICS}
               for m in {x["model"] for x in rows} for s in seqs}
    wil = []
    pairs = [("ILWT-INN", n) for n in nets] + ([("ILWT-INN [float]", f"{n} [float]") for n in nets] if args.float else [])
    for ref, name in pairs:
        for k in METRICS:
            a = np.array([per_seq[(ref, s)][k] for s in seqs]); b = np.array([per_seq[(name, s)][k] for s in seqs])
            stat, p = wilcoxon(a, b)
            wil.append({"ref": ref, "vs": name, "metric": k, "ILWT_INN_mean": round(a.mean(), 4), "HiNet_mean": round(b.mean(), 4),
                        "median_diff": round(float(np.median(a - b)), 4), "W": stat, "p": f"{p:.3g}", "n_seq": len(seqs)})
    E.write_csv(out / "wilcoxon.csv", wil)
    print("\nmodel             H-PSNR  H-SSIM   R-PSNR  SSIM    NC      BER")
    for r in summ:
        print(f"{r['model']:24s} {r['H_PSNR']:7.2f}  {r['H_SSIM']:.4f}  {r['R_PSNR']:6.2f}  {r['SSIM']:.4f}  "
              f"{r['NC']:.4f}  {r['BER']:.4f}")
    for w in wil:
        print(w)


if __name__ == "__main__":
    main()
