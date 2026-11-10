#!/usr/bin/env python
"""
E18 input: HiNet / DeepMIH stego frame sets matched to the E04 (ILWT-INN) steganalysis set, so the same six
retrained detectors can be run on the baselines.

Paired with E04 frame by frame: the SAME cover PNGs (cover/ is a symlink to the E04 set) and the SAME secret for
every frame (read from the E04 per_frame.csv, ImageNet test split, loaded with the same Resize + ToTensor as
make_stego_set.py). Each baseline embeds 24 BPP (one RGB secret) and its stego is stored as 8-bit PNG.
  hinet     released HiNet (or any HiNet checkpoint via --hinet-ckpt)
  deepmih   released DeepMIH full cascade, 1-secret mode (secret 2 = constant 0.5), as in E15
Recovery from the stored 8-bit PNG is scored as a sanity check (Gaussian z, torch seed 0 per sequence).

Outputs (--out): cover -> <ref>/cover, stego/<seq>/frame_NNNNN.png, per_frame.csv (seq, frame, secret, H_PSNR, R_PSNR).
One model per process (HiNet's and DeepMIH's model/config modules share names).
"""
import argparse
import csv
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

ROOT = Path(os.path.expanduser("~/ilwt_workspace"))


def psnr_u8(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("hinet", "deepmih"))
    ap.add_argument("--ref", default=str(ROOT / "runs/stego_set_24bpp"))
    ap.add_argument("--images", default=str(ROOT / "data/imagenet256"))
    ap.add_argument("--hinet-dir", default=str(ROOT / "baselines/HiNet"))
    ap.add_argument("--hinet-ckpt", default=str(ROOT / "baselines/HiNet/model/released/model.pt"))
    ap.add_argument("--deepmih-dir", default=str(ROOT / "baselines/DeepMIH"))
    ap.add_argument("--deepmih-weights", default=str(ROOT / "baselines/DeepMIH_weights/model_checkpoint_03000_"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-frames-per-seq", type=int, default=0, help="for quick tests")
    args = ap.parse_args()
    ref, out = Path(args.ref).resolve(), Path(args.out)
    assert out.resolve() != ref, "refusing to write into the reference set"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "cover").exists():
        os.symlink(ref / "cover", out / "cover")
    assert torch.cuda.is_available(), "no CUDA device"
    device = torch.device("cuda")

    for m in ("model", "config", "modules", "hinet", "invblock", "rrdb_denselayer", "denseblock"):
        sys.modules.pop(m, None)
    sys.path.insert(0, args.hinet_dir if args.model == "hinet" else args.deepmih_dir)
    import modules.Unet_common as common
    dwt, iwt = common.DWT(), common.IWT()
    if args.model == "hinet":
        from model import Model
        net = torch.nn.DataParallel(Model().to(device), device_ids=[0])
        sd = torch.load(args.hinet_ckpt, map_location=device, weights_only=False)["net"]
        net.load_state_dict({k: v for k, v in sd.items() if "tmp_var" not in k}); net.eval()

        def embed_recover(c, s):
            st = iwt(net(torch.cat((dwt(c), dwt(s)), 1)).narrow(1, 0, 12)).clamp(0, 1)
            st = torch.round(st * 255) / 255
            d = dwt(st)
            rec = iwt(net(torch.cat((d, torch.randn_like(d)), 1), rev=True).narrow(1, 12, 12))
            return st, rec
    else:
        from model import Model_1, Model_2
        from imp_subnet import ImpMapBlock
        nets = []
        for i, M in ((1, Model_1), (2, Model_2), (3, ImpMapBlock)):
            n = torch.nn.DataParallel(M().to(device), device_ids=[0])
            sd = torch.load(f"{args.deepmih_weights}{i}.pt", map_location=device, weights_only=True)["net"]
            n.load_state_dict({k: v for k, v in sd.items() if "tmp_var" not in k}); n.eval(); nets.append(n)
        net1, net2, net3 = nets

        def embed_recover(c, s):
            s2 = torch.full_like(c, 0.5)
            st1d = net1(torch.cat((dwt(c), dwt(s)), 1)).narrow(1, 0, 12)
            imp = net3(c, s, iwt(st1d))
            st = iwt(net2(torch.cat((st1d, dwt(imp), dwt(s2)), 1)).narrow(1, 0, 12)).clamp(0, 1)
            st = torch.round(st * 255) / 255
            d = dwt(st)
            r2 = net2(torch.cat((d, torch.randn(d.shape[0], 24, *d.shape[2:], device=device)), 1), rev=True)
            r1 = net1(torch.cat((r2.narrow(1, 0, 12), torch.randn_like(d)), 1), rev=True)
            return st, iwt(r1.narrow(1, 12, 12))

    load_secret = T.Compose([T.Resize((256, 256)), T.ToTensor()])           # = make_stego_set.load_image, in [0,1]
    rows = list(csv.DictReader(open(ref / "per_frame.csv")))
    by_seq = {}
    for r in rows:
        by_seq.setdefault(r["seq"], []).append(r)
    res = []
    for seq, rs in by_seq.items():
        if args.max_frames_per_seq:
            rs = rs[:args.max_frames_per_seq]
        (out / "stego" / seq).mkdir(parents=True, exist_ok=True)
        torch.manual_seed(0)
        hs, rps = [], []
        for b in range(0, len(rs), args.batch):
            bt = rs[b:b + args.batch]
            cov = [cv2.cvtColor(cv2.imread(str(ref / "cover" / seq / f"{r['frame']}.png")), cv2.COLOR_BGR2RGB) for r in bt]
            c = torch.from_numpy(np.stack(cov).transpose(0, 3, 1, 2).copy()).float().div(255).to(device)
            s = torch.stack([load_secret(Image.open(Path(args.images) / r["secret"]).convert("RGB")) for r in bt]).to(device)
            with torch.no_grad():
                st, rec = embed_recover(c, s)
            st8 = (st * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            rec8 = (rec.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            sec8 = (s * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            for r, cv, sg, sc, rc in zip(bt, cov, st8, sec8, rec8):
                cv2.imwrite(str(out / "stego" / seq / f"{r['frame']}.png"), cv2.cvtColor(sg, cv2.COLOR_RGB2BGR))
                h, rp = psnr_u8(cv, sg), psnr_u8(sc, rc)
                hs.append(h); rps.append(rp)
                res.append({"seq": seq, "frame": r["frame"], "secret": r["secret"], "H_PSNR": f"{h:.4f}", "R_PSNR": f"{rp:.4f}"})
        print(f"{seq:13s} frames {len(rs):4d}  H-PSNR {np.mean(hs):.2f}  R-PSNR {np.mean(rps):.2f}", flush=True)
    with open(out / "per_frame.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
    sm = {k: np.mean([np.mean([float(x[k]) for x in res if x["seq"] == q]) for q in by_seq]) for k in ("H_PSNR", "R_PSNR")}
    print(f"{args.model}: {len(res)} stego frames, mean over sequences H-PSNR {sm['H_PSNR']:.2f}  R-PSNR {sm['R_PSNR']:.2f}")


if __name__ == "__main__":
    main()
