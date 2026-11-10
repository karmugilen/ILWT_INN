#!/usr/bin/env python
"""
stego bit depth 8 / 16 / 32 (float) for every compared model.

Same models, no retraining. Each model embeds once; its stego is then stored at
  8   round to 1/255   (8-bit PNG / FFV1 gbrp: the deployed storage)
  16  round to 1/65535 (16-bit PNG / FFV1 gbrp16le)
  32  float32, no rounding (TIFF/EXR float; the no-quantisation ceiling)
and the secret is recovered blindly from that stego. Covers: every --every-th frame of the 20 Xiph
sequences (ffmpeg lanczos 256x256, standard resize); secrets: standard protocol, frame i -> hide sorted[i % 300].
HiNet recovers with Gaussian z (its own reverse), torch seed 0 per sequence.

Outputs (--out): per_frame.csv, summary.csv (model x depth: mean, SD, 95% CI across sequences).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as E  # noqa: E402
from make_stego_set import SEQUENCES  # noqa: E402

DEPTHS = {8: 255.0, 16: 65535.0, 32: None}
METRICS = ("H_PSNR", "H_SSIM", "R_PSNR", "SSIM", "NC", "BER", "E_XAI")


def covers_of(videos, rel):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(Path(videos) / rel), "-vf",
                          "scale=256:256:flags=lanczos", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 256, 256, 3)


def store(x01, depth):
    """x in [0,1] -> stored value in [0,1] at the given bit depth."""
    x01 = x01.clamp(0, 1)
    return x01 if DEPTHS[depth] is None else torch.round(x01 * DEPTHS[depth]) / DEPTHS[depth]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--secret-dir", required=True)
    ap.add_argument("--hinet-dir", default="")
    ap.add_argument("--hinet", nargs="*", default=[], help="name=checkpoint")
    ap.add_argument("--deepmih-dir", default="")
    ap.add_argument("--deepmih", default="", help="DeepMIH stage-1 checkpoint (model_checkpoint_03000_1.pt); "
                    "single-secret 24 BPP = net1 only. Run in its own process (its model/config clash with HiNet's)")
    ap.add_argument("--skip-ilwt", action="store_true")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model, device = E.load_model()
    hide = sorted(Path(args.secret_dir).glob("*.png"))[:300]

    def ilwt(c_u8, s_u8):
        x = torch.cat([E.u8_to_t(c_u8, device), E.u8_to_t(s_u8, device)], 1)
        s01 = (model(x)[0][:, :3].clamp(-1, 1) + 1) / 2
        res = {}
        for d in DEPTHS:
            st = store(s01, d)
            rec = model.inverse(torch.cat([st * 2 - 1, torch.zeros_like(st)], 1))[:, 3:]
            res[d] = ((st * 255).permute(0, 2, 3, 1).cpu().numpy(), E.t_to_u8(rec))
        return res

    models = {} if args.skip_ilwt else {"ILWT-INN": ilwt}
    if args.hinet or args.deepmih:
        sys.path.insert(0, args.hinet_dir or args.deepmih_dir)
        import modules.Unet_common as common
        dwt, iwt = common.DWT(), common.IWT()

        def make(net):
            def run(c_u8, s_u8):
                c = torch.from_numpy(np.ascontiguousarray(c_u8.transpose(0, 3, 1, 2))).to(device).float() / 255
                s = torch.from_numpy(np.ascontiguousarray(s_u8.transpose(0, 3, 1, 2))).to(device).float() / 255
                s01 = iwt(net(torch.cat((dwt(c), dwt(s)), 1)).narrow(1, 0, 12)).clamp(0, 1)
                res = {}
                for d in DEPTHS:
                    st = store(s01, d)
                    dd = dwt(st)
                    rev = net(torch.cat((dd, torch.randn_like(dd)), 1), rev=True)
                    rec = (iwt(rev.narrow(1, 12, 12)).clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
                    res[d] = ((st * 255).permute(0, 2, 3, 1).cpu().numpy(), rec)
                return res
            return run

        for spec in args.hinet:
            from model import Model
            name, ck = spec.split("=")
            net = torch.nn.DataParallel(Model().to(device), device_ids=[0])
            sd = torch.load(ck, map_location=device, weights_only=False)["net"]
            net.load_state_dict({k: v for k, v in sd.items() if "tmp_var" not in k}); net.eval()
            models[f"HiNet {name}"] = make(net)
        if args.deepmih:
            from model import Model_1
            net = torch.nn.DataParallel(Model_1().to(device), device_ids=[0])
            sd = torch.load(args.deepmih, map_location=device, weights_only=True)["net"]   # downloaded file
            net.load_state_dict({k: v for k, v in sd.items() if "tmp_var" not in k}); net.eval()
            models["DeepMIH (stage 1)"] = make(net)

    rows = []
    for seq, rel in SEQUENCES.items():
        covers = covers_of(args.videos, rel)[::args.every]
        idx = np.arange(len(covers)) * args.every
        secrets = np.stack([E.load_secret_u8(hide[i % len(hide)]) for i in idx])
        for name, fn in models.items():
            torch.manual_seed(0)
            acc = {d: ([], []) for d in DEPTHS}
            with torch.no_grad():
                for b in range(0, len(covers), args.batch):
                    for d, (st, rc) in fn(covers[b:b + args.batch], secrets[b:b + args.batch]).items():
                        acc[d][0].append(st); acc[d][1].append(rc)
            for d, (st, rc) in acc.items():
                st, rc = np.concatenate(st), np.concatenate(rc)
                for i, c, s_, sec, r in zip(idx, covers, st, secrets, rc):
                    m = E.frame_metrics(sec, r, device)
                    rows.append({"model": name, "depth": d, "seq": seq, "frame": int(i) + 1,
                                 "H_PSNR": round(E.psnr_u8(c, s_), 4), "H_SSIM": round(E.ssim_u8(c, s_, device), 6),
                                 **{k: round(v, 6) for k, v in m.items()}})
                cur = [x for x in rows if x["model"] == name and x["depth"] == d and x["seq"] == seq]
                print(f"{seq:13s} {name:16s} {d:2d}-bit  H {np.mean([x['H_PSNR'] for x in cur]):.2f}  "
                      f"R {np.mean([x['R_PSNR'] for x in cur]):.2f}  BER {np.mean([x['BER'] for x in cur]):.4f}", flush=True)
        E.write_csv(out / "per_frame.csv", rows)
    summ = E.summarize(rows, ["model", "depth"], METRICS)
    E.write_csv(out / "summary.csv", [{k: v for k, v in r.items() if not k.startswith("frac_")} for r in summ])
    print("\nmodel            depth  H-PSNR  H-SSIM  R-PSNR (95% CI)              R-SSIM  NC      BER")
    for r in summ:
        print(f"{r['model']:16s} {r['depth']:5d}  {r['H_PSNR']:6.2f}  {r['H_SSIM']:.4f}  {r['R_PSNR']:6.2f} "
              f"{r['R_PSNR_CI95']:20s}  {r['SSIM']:.4f}  {r['NC']:.4f}  {r['BER']:.4f}")


if __name__ == "__main__":
    main()
