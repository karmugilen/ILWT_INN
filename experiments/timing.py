#!/usr/bin/env python
"""
Computational cost. Run on an otherwise idle GPU.

1. model.csv     params, FLOPs (torch FlopCounterMode: conv/matmul only; lifting adds are excluded),
                 peak GPU memory, and model-only latency per frame for embed (forward + 8-bit
                 quantisation) and blind extract (inverse), batch 1 and 4, cuda.synchronize() around each
                 call, --warmup then --iters timed calls -> mean, SD, median, p95 ms/frame and FPS.
                 With --hinet the released HiNet (DWT + 16 INV blocks) is timed the same way; with
                 --deepmih the released DeepMIH full cascade (net1 + importance map net3 + net2; embed =
                 both forward stages + 8-bit, extract = net2 reverse + net1 reverse, Gaussian z), one
                 secret = 24 BPP. HiNet and DeepMIH share module names: time them in separate processes.
                 --model-only skips table 2.
2. pipeline.csv  end-to-end per sequence, every stage timed (wall clock):
                 decode (ffmpeg y4m -> lanczos 256x256 -> rgb24 pipe), secret load + resize,
                 embed (H2D, forward, 8-bit, D2H), FFV1 encode (gbrp, standard settings),
                 FFV1 decode, blind extract. Recovery R-PSNR is checked so the timed path is the real one.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as E  # noqa: E402
from make_stego_set import SEQUENCES  # noqa: E402
from codec_20seq import CODECS  # noqa: E402

S = 256


def sync():
    torch.cuda.synchronize()


def bench(fn, x, warmup, iters):
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        sync()
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(iters):
            sync(); t = time.perf_counter()
            fn(x)
            sync(); ts.append(time.perf_counter() - t)
    ms = np.array(ts) * 1000 / x.shape[0]
    return {"ms_per_frame_mean": ms.mean(), "ms_per_frame_sd": ms.std(ddof=1), "ms_per_frame_median": np.median(ms),
            "ms_per_frame_p95": np.percentile(ms, 95), "FPS": 1000 / ms.mean(),
            "peak_mem_MB": torch.cuda.max_memory_allocated() / 2 ** 20}


def flops(fn, x):
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        fn(x)
    return fc.get_total_flops() / x.shape[0]


def quant(t):
    return torch.round(((t.clamp(-1, 1) + 1) / 2) * 255) / 255 * 2 - 1


def model_rows(args, device):
    rows = []
    model, _ = E.load_model()
    n_par = sum(p.numel() for p in model.parameters())
    embed = lambda x: quant(model(x)[0][:, :3])                                         # noqa: E731
    extract = lambda s: model.inverse(torch.cat([s, torch.zeros_like(s)], 1))[:, 3:]    # noqa: E731
    nets = [("ILWT-INN", n_par, embed, extract, 6, 3)]
    if args.hinet:
        sys.path.insert(0, args.hinet)
        from model import Model
        import modules.Unet_common as common
        dwt, iwt = common.DWT(), common.IWT()
        net = torch.nn.DataParallel(Model().to(device), device_ids=[0])
        ck = torch.load(Path(args.hinet) / "model/released/model.pt", map_location=device)
        net.load_state_dict({k: v for k, v in ck["net"].items() if "tmp_var" not in k}); net.eval()
        h_embed = lambda x: quant(iwt(net(torch.cat((dwt(x[:, :3]), dwt(x[:, 3:])), 1)).narrow(1, 0, 12)))  # noqa: E731
        h_extract = lambda s: iwt(net(torch.cat((dwt(s), torch.randn_like(dwt(s))), 1), rev=True).narrow(1, 12, 12))  # noqa: E731
        nets.append(("HiNet", sum(p.numel() for p in net.parameters()), h_embed, h_extract, 6, 3))
    if args.deepmih:
        for m in ("model", "config", "modules", "hinet", "invblock", "rrdb_denselayer", "denseblock"):
            sys.modules.pop(m, None)
        sys.path.insert(0, args.deepmih)
        from model import Model_1, Model_2
        from imp_subnet import ImpMapBlock
        import modules.Unet_common as common
        dwt, iwt = common.DWT(), common.IWT()
        dm = []
        for i, M in ((1, Model_1), (2, Model_2), (3, ImpMapBlock)):
            n = torch.nn.DataParallel(M().to(device), device_ids=[0])
            sd = torch.load(f"{args.deepmih_weights}{i}.pt", map_location=device, weights_only=True)["net"]
            n.load_state_dict({k: v for k, v in sd.items() if "tmp_var" not in k}); n.eval(); dm.append(n)
        n1, n2, n3 = dm

        def d_embed(x):                                    # x = [cover, secret] in [0,1]; s2 = mid-grey
            c, s1 = x[:, :3], x[:, 3:]
            st1d = n1(torch.cat((dwt(c), dwt(s1)), 1)).narrow(1, 0, 12)
            imp = n3(c, s1, iwt(st1d))
            st2 = iwt(n2(torch.cat((st1d, dwt(imp), dwt(torch.full_like(c, 0.5))), 1)).narrow(1, 0, 12))
            return torch.round(st2.clamp(0, 1) * 255) / 255

        def d_extract(s):
            d = dwt(s)
            r2 = n2(torch.cat((d, torch.randn(d.shape[0], 24, *d.shape[2:], device=d.device)), 1), rev=True)
            return iwt(n1(torch.cat((r2.narrow(1, 0, 12), torch.randn_like(d)), 1), rev=True).narrow(1, 12, 12))
        nets.append(("DeepMIH", sum(p.numel() for n in dm for p in n.parameters()), d_embed, d_extract, 6, 3))
    for name, n_par, emb, ext, cin, cs in nets:
        for bs in (1, 4):
            x = torch.rand(bs, cin, S, S, device=device) * 2 - 1
            s = torch.rand(bs, cs, S, S, device=device) * 2 - 1
            if name == "DeepMIH":                         # DeepMIH works in [0,1]
                x, s = (x + 1) / 2, (s + 1) / 2
            for op, fn, inp in (("embed", emb, x), ("extract", ext, s)):
                r = bench(fn, inp, args.warmup, args.iters)
                rows.append({"model": name, "params": n_par, "op": op, "batch": bs,
                             "GFLOPs_per_frame": round(flops(fn, inp) / 1e9, 3),
                             **{k: round(float(v), 3) for k, v in r.items()}})
                print(rows[-1], flush=True)
    return rows, model


def pipeline_rows(args, model, device):
    rows = []
    secrets = sorted(Path(args.secret_dir).glob("*.png"))
    tmp = Path(args.out) / "tmp"; tmp.mkdir(exist_ok=True)
    for seq, rel in SEQUENCES.items():
        t = {}
        t0 = time.perf_counter()
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(Path(args.videos) / rel),
                              "-vf", f"scale={S}:{S}:flags=lanczos", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                             check=True, stdout=subprocess.PIPE).stdout
        covers = np.frombuffer(raw, np.uint8).reshape(-1, S, S, 3)
        n = len(covers); t["decode"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        secs = np.stack([E.load_secret_u8(secrets[k % len(secrets)]) for k in range(n)])
        t["secret_load"] = time.perf_counter() - t0
        t0 = time.perf_counter(); stego = []
        with torch.no_grad():
            for i in range(0, n, args.batch):
                x = torch.cat([E.u8_to_t(covers[i:i + args.batch], device), E.u8_to_t(secs[i:i + args.batch], device)], 1)
                stego.append(E.t_to_u8(model(x)[0][:, :3]))
        stego = np.concatenate(stego); sync(); t["embed"] = time.perf_counter() - t0
        vid = tmp / f"{seq}.mkv"
        t0 = time.perf_counter()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{S}x{S}",
                        "-framerate", "30", "-i", "-", *CODECS["ffv1"], str(vid)], input=stego.tobytes(), check=True)
        t["encode_ffv1"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(vid), "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                             check=True, stdout=subprocess.PIPE).stdout
        dec = np.frombuffer(raw, np.uint8).reshape(-1, S, S, 3); t["decode_stego"] = time.perf_counter() - t0
        assert np.array_equal(dec, stego), f"{seq}: FFV1 round trip is not lossless"
        t0 = time.perf_counter(); rec = E.recover(model, dec, device, bs=args.batch); sync()
        t["extract"] = time.perf_counter() - t0
        rp = np.mean([E.psnr_u8(a, b) for a, b in zip(secs[::10], rec[::10])])
        tot = sum(t.values())
        row = {"seq": seq, "frames": n, "batch": args.batch, **{f"{k}_s": round(v, 3) for k, v in t.items()},
               "total_s": round(tot, 3), "ms_per_frame": round(1000 * tot / n, 2), "end_to_end_FPS": round(n / tot, 2),
               "embed_side_FPS": round(n / (t["decode"] + t["secret_load"] + t["embed"] + t["encode_ffv1"]), 2),
               "extract_side_FPS": round(n / (t["decode_stego"] + t["extract"]), 2),
               "stego_bytes": vid.stat().st_size, "R_PSNR_check": round(rp, 2)}
        rows.append(row); print(row, flush=True)
        vid.unlink()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(E.ROOT / "data/sd_videos"))
    ap.add_argument("--secret-dir", default=str(E.ROOT / "data/hide300"))
    ap.add_argument("--hinet", default="", help="HiNet repo dir (adds HiNet to the model-only table)")
    ap.add_argument("--deepmih", default="", help="DeepMIH repo dir (adds the DeepMIH cascade to the model-only table)")
    ap.add_argument("--deepmih-weights", default="", help="prefix of model_checkpoint_03000_{1,2,3}.pt")
    ap.add_argument("--model-only", action="store_true", help="skip the end-to-end pipeline table")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--out", default=str(E.ROOT / "runs/timing"))
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0), "torch", torch.__version__, flush=True)
    rows, model = model_rows(args, device)
    E.write_csv(Path(args.out) / "model.csv", rows)
    if args.model_only:
        return
    prow = pipeline_rows(args, model, device)
    E.write_csv(Path(args.out) / "pipeline.csv", prow)
    keys = [k for k in prow[0] if k.endswith("_s")]
    tot = {k: sum(r[k] for r in prow) for k in keys}; nf = sum(r["frames"] for r in prow)
    print(f"\n20 sequences, {nf} frames, batch {args.batch}:")
    for k in keys:
        print(f"  {k:16s} {1000 * tot[k] / nf:7.2f} ms/frame")
    print(f"  end-to-end FPS {nf / tot['total_s']:.2f}")


if __name__ == "__main__":
    main()
