#!/usr/bin/env python
"""
architectural invertibility vs end-to-end recovery.

Same frames and secrets as the native-resolution and error-distribution scripts (deployed model, every frame
of the 20 Xiph sequences at 256x256, standard secrets). Per batch, in float32:
  lifting   x -> ILWT -> ILWT^-1: max / mean |error| (8-bit units); share of non-integer coefficients
            when the input is 8-bit (the deployed lifting has no rounding step)
  INN       ILWT(x) -> 8 blocks -> 8 inverse blocks, same cover conditioning: max / mean |error|
  recovery R-PSNR of the secret (8-bit), mean over sequences of per-frame values, in five settings:
    exact_inverse    INN output z and cover conditioning kept: ILWT^-1(INN^-1(z | cond(cover))) -> secret.
                     Uses z, which is never transmitted; shows the architecture itself inverts.
    own_output       the full 6-channel forward output (stego + auxiliary channels) fed to model.inverse
    float_zero       float stego + zeros for the auxiliary channels (extraction as trained, no storage)
    q8_zero          8-bit stego + zeros (the deployed extraction)

Outputs (--out): summary.csv, per_seq.csv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as E  # noqa: E402
from make_stego_set import SEQUENCES  # noqa: E402
from native_res import decode, secret_u8  # noqa: E402

MODES = ("exact_inverse", "own_output", "float_zero", "q8_zero")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--secret-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    assert not (out / "summary.csv").exists(), f"{out} already has results; choose a new --out"
    out.mkdir(parents=True, exist_ok=True)
    model, dev = E.load_model()
    m = model.module if hasattr(model, "module") else model
    secrets = sorted(Path(args.secret_dir).glob("*.png"))[:300]
    lift_max = inn_max = 0.0
    lift_sum = inn_sum = 0.0; n_lift = n_inn = 0
    nonint = 0; n_coef = 0
    per_seq = []
    for seq, rel in SEQUENCES.items():
        covers = decode(Path(args.videos) / rel, scale=True)
        if args.max_frames:
            covers = covers[:args.max_frames]
        secs = np.stack([secret_u8(secrets[i % len(secrets)], 256, 256) for i in range(len(covers))])
        ps = {k: [] for k in MODES}
        with torch.no_grad():
            for b in range(0, len(covers), args.batch):
                c, s = covers[b:b + args.batch], secs[b:b + args.batch]
                x = torch.cat([E.u8_to_t(c, dev), E.u8_to_t(s, dev)], 1)
                xf = m.ilwt(x)
                d = (m.ilwt.inverse(xf) - x).abs() * 127.5
                lift_max = max(lift_max, d.max().item()); lift_sum += d.sum().item(); n_lift += d.numel()
                coef = m.ilwt(x) * 127.5                         # coefficients on the 8-bit scale
                nonint += ((coef - coef.round()).abs() > 1e-3).sum().item(); n_coef += coef.numel()
                cond = F.interpolate(m.cond_net(x[:, :3]), size=xf.shape[2:], mode="bilinear", align_corners=False)
                z = xf
                for blk in m.blocks:
                    z, _ = blk(z, cond)
                y = z
                for blk in reversed(m.blocks):
                    y = blk.inverse(y, cond)
                d = (y - xf).abs() * 127.5
                inn_max = max(inn_max, d.max().item()); inn_sum += d.sum().item(); n_inn += d.numel()
                rec = {"exact_inverse": m.ilwt.inverse(y)[:, 3:]}
                st, _ = m(x)
                rec["own_output"] = m.inverse(st)[:, 3:]
                zeros = torch.zeros_like(st[:, :3])
                rec["float_zero"] = m.inverse(torch.cat([st[:, :3], zeros], 1))[:, 3:]
                st8 = E.u8_to_t(E.t_to_u8(st[:, :3]), dev)
                rec["q8_zero"] = m.inverse(torch.cat([st8, zeros], 1))[:, 3:]
                for k, r in rec.items():
                    ps[k] += [E.psnr_u8(a, bb) for a, bb in zip(s, E.t_to_u8(r))]
        row = {"seq": seq, "frames": len(covers), **{k: round(float(np.mean(v)), 4) for k, v in ps.items()}}
        per_seq.append(row)
        print(row, flush=True)
    summ = {"frames": sum(r["frames"] for r in per_seq),
            "lifting_max_abs_err_8bit": f"{lift_max:.3g}", "lifting_mean_abs_err_8bit": f"{lift_sum / n_lift:.3g}",
            "lifting_noninteger_share": round(nonint / n_coef, 4),
            "inn_max_abs_err_8bit": f"{inn_max:.3g}", "inn_mean_abs_err_8bit": f"{inn_sum / n_inn:.3g}",
            **{f"R_PSNR_{k}": round(float(np.mean([r[k] for r in per_seq])), 2) for k in MODES}}
    for name, rows in (("per_seq.csv", per_seq), ("summary.csv", [summ])):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(summ)


if __name__ == "__main__":
    main()
