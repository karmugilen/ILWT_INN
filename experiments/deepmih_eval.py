#!/usr/bin/env python
"""
DeepMIH (released TPAMI-2022 weights, full cascade net1 + net2 + importance map net3) on the
20 Xiph sequences, stego stored at 8 / 16 / 32-float bits.

Same protocol as E14 (bitdepth_sweep.py): every --every-th frame, ffmpeg lanczos 256x256, frame i ->
hide sorted[i % 300]. DeepMIH's own test procedure: forward net1 (cover, s1) -> steg1; imp = net3(cover,
s1, steg1); forward net2 (steg1, imp, s2) -> steg2 (the transmitted image); reverse net2 on stored steg2
with Gaussian z -> (steg1', s2'); reverse net1 on steg1' with Gaussian z -> s1'. torch seed 0 per sequence.
The released weights were trained with train_old_version.py, so secret 2 is read from the MIDDLE 12
channels of the net2 reverse output (README "Further explanation"; test_oldversion.py), not the last 12.

Modes
  1sec   24 BPP, the payload of ILWT-INN: s1 = standard secrets, s2 = constant mid-grey (0.5); scored on s1
  2sec   48 BPP, DeepMIH's native use: s1 = hide[i % 300], s2 = hide[(i + 150) % 300]; scored on s1 and s2
Hiding metrics are cover vs the stored steg2 in every row.

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
    x01 = x01.clamp(0, 1)
    return x01 if DEPTHS[depth] is None else torch.round(x01 * DEPTHS[depth]) / DEPTHS[depth]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--secret-dir", required=True)
    ap.add_argument("--deepmih-dir", required=True)
    ap.add_argument("--weights", required=True, help="prefix of model_checkpoint_03000_{1,2,3}.pt")
    ap.add_argument("--modes", default="1sec,2sec")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "no CUDA device (container lost the GPU?)"
    device = torch.device("cuda")

    for m in ("model", "config", "modules"):          # make sure DeepMIH's modules are the ones imported
        sys.modules.pop(m, None)
    sys.path.insert(0, args.deepmih_dir)
    import config as c
    from model import Model_1, Model_2
    from imp_subnet import ImpMapBlock
    import modules.Unet_common as common
    assert c.use_imp_map, "released weights use the importance map"
    dwt, iwt = common.DWT(), common.IWT()
    nets = []
    for i, M in ((1, Model_1), (2, Model_2), (3, ImpMapBlock)):
        n = torch.nn.DataParallel(M().to(device), device_ids=[0])
        sd = torch.load(f"{args.weights}{i}.pt", map_location=device, weights_only=True)["net"]   # downloaded file
        n.load_state_dict({k: v for k, v in sd.items() if "tmp_var" not in k}); n.eval(); nets.append(n)
    net1, net2, net3 = nets
    print("DeepMIH params (net1+net2+net3): %.3f M" % (sum(p.numel() for n in nets for p in n.parameters()) / 1e6), flush=True)

    t01 = lambda u8: torch.from_numpy(np.ascontiguousarray(u8.transpose(0, 3, 1, 2))).to(device).float() / 255
    u8 = lambda t: (t.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()

    def run(c_u8, s1_u8, s2_u8):
        cover, s1 = t01(c_u8), t01(s1_u8)
        s2 = torch.full_like(cover, 0.5) if s2_u8 is None else t01(s2_u8)
        st1d = net1(torch.cat((dwt(cover), dwt(s1)), 1)).narrow(1, 0, 12)
        imp = net3(cover, s1, iwt(st1d))
        st2 = iwt(net2(torch.cat((st1d, dwt(imp), dwt(s2)), 1)).narrow(1, 0, 12)).clamp(0, 1)
        res = {}
        for d in DEPTHS:
            st = store(st2, d); sd_ = dwt(st)
            r2 = net2(torch.cat((sd_, torch.randn(sd_.shape[0], 24, *sd_.shape[2:], device=device)), 1), rev=True)
            rs2 = iwt(r2.narrow(1, 12, 12))                      # old-version split (released weights)
            r1 = net1(torch.cat((r2.narrow(1, 0, 12), torch.randn_like(sd_)), 1), rev=True)
            rs1 = iwt(r1.narrow(1, 12, 12))
            res[d] = ((st * 255).permute(0, 2, 3, 1).cpu().numpy(), u8(rs1), u8(rs2))
        return res

    hide = sorted(Path(args.secret_dir).glob("*.png"))[:300]
    modes = args.modes.split(",")
    rows = []
    for seq, rel in SEQUENCES.items():
        covers = covers_of(args.videos, rel)[::args.every]
        idx = np.arange(len(covers)) * args.every
        sec1 = np.stack([E.load_secret_u8(hide[i % 300]) for i in idx])
        sec2 = np.stack([E.load_secret_u8(hide[(i + 150) % 300]) for i in idx])
        for mode in modes:
            torch.manual_seed(0)
            acc = {d: ([], [], []) for d in DEPTHS}
            with torch.no_grad():
                for b in range(0, len(covers), args.batch):
                    s2b = sec2[b:b + args.batch] if mode == "2sec" else None
                    for d, outs in run(covers[b:b + args.batch], sec1[b:b + args.batch], s2b).items():
                        for a, o in zip(acc[d], outs):
                            a.append(o)
            for d, (st, r1, r2) in acc.items():
                st, r1, r2 = np.concatenate(st), np.concatenate(r1), np.concatenate(r2)
                scored = [("DeepMIH 1-secret (24 BPP)", sec1, r1)] if mode == "1sec" else \
                         [("DeepMIH 2-secret (48 BPP) s1", sec1, r1), ("DeepMIH 2-secret (48 BPP) s2", sec2, r2)]
                for name, secs, recs in scored:
                    for i, cv, s_, sec, r in zip(idx, covers, st, secs, recs):
                        m = E.frame_metrics(sec, r, device)
                        rows.append({"model": name, "depth": d, "seq": seq, "frame": int(i) + 1,
                                     "H_PSNR": round(E.psnr_u8(cv, s_), 4), "H_SSIM": round(E.ssim_u8(cv, s_, device), 6),
                                     **{k: round(v, 6) for k, v in m.items()}})
                    cur = [x for x in rows if x["model"] == name and x["depth"] == d and x["seq"] == seq]
                    print(f"{seq:13s} {name:30s} {d:2d}-bit  H {np.mean([x['H_PSNR'] for x in cur]):.2f}  "
                          f"R {np.mean([x['R_PSNR'] for x in cur]):.2f}  BER {np.mean([x['BER'] for x in cur]):.4f}", flush=True)
        E.write_csv(out / "per_frame.csv", rows)
    summ = E.summarize(rows, ["model", "depth"], METRICS)
    E.write_csv(out / "summary.csv", [{k: v for k, v in r.items() if not k.startswith("frac_")} for r in summ])
    print("\nmodel                          depth  H-PSNR  H-SSIM  R-PSNR (95% CI)              R-SSIM  NC      BER")
    for r in summ:
        print(f"{r['model']:30s} {r['depth']:5d}  {r['H_PSNR']:6.2f}  {r['H_SSIM']:.4f}  {r['R_PSNR']:6.2f} "
              f"{r['R_PSNR_CI95']:20s}  {r['SSIM']:.4f}  {r['NC']:.4f}  {r['BER']:.4f}")


if __name__ == "__main__":
    main()
