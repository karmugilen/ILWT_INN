#!/usr/bin/env python
"""
real lossless video files at 8 and 16 bits. E14 simulated the bit
depth by rounding; here every stego sequence is actually written to an FFV1 file (standard settings,
codec_20seq.CODECS["ffv1"]; -pix_fmt gbrp for 8 bits, gbrp16le for 16 bits), decoded back, checked to
be bit-exact, and the secret is recovered blindly from the decoded file.

One model per process (HiNet and DeepMIH share module names):
  --model ilwt      the deployed ILWT-INN
  --model hinet     released HiNet (--hinet-dir, --hinet-ckpt), Gaussian z at reverse
  --model deepmih   released DeepMIH full cascade, 1 secret = 24 BPP (s2 mid-grey), Gaussian z
Protocol as E14/E15: all frames of the 20 Xiph sequences (ffmpeg lanczos 256x256), frame i -> hide
sorted[i % 300], torch seed 0 per sequence and depth. Also reported per sequence: FFV1 file size of the
stego at each depth and of the 8-bit cover (file-size growth).

Outputs (--out/<model>): per_frame.csv, files.csv (per sequence and depth: bytes, bits/pixel, growth vs
cover), summary.csv (depth: mean, SD, 95% CI across sequences).
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as E  # noqa: E402
from make_stego_set import SEQUENCES  # noqa: E402
from codec_20seq import CODECS  # noqa: E402

S = 256
METRICS = ("H_PSNR", "H_SSIM", "R_PSNR", "SSIM", "NC", "BER", "E_XAI")
FMT = {8: ("rgb24", np.uint8, 255.0, "gbrp"), 16: ("rgb48le", np.dtype("<u2"), 65535.0, "gbrp16le")}


def covers_of(videos, rel):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(Path(videos) / rel), "-vf",
                          f"scale={S}:{S}:flags=lanczos", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, S, S, 3)


def ffv1_roundtrip(frames, depth, path):
    """frames: (N,S,S,3) integer array at `depth` bits -> (decoded array, file bytes)."""
    in_fmt, dt, _, pix = FMT[depth]
    args = list(CODECS["ffv1"]); args[args.index("-pix_fmt") + 1] = pix
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", in_fmt, "-s", f"{S}x{S}",
                    "-framerate", "30", "-i", "-", *args, str(path)], input=frames.astype(dt).tobytes(), check=True)
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-pix_fmt", in_fmt, "-f", "rawvideo", "-"],
                         check=True, stdout=subprocess.PIPE).stdout
    dec = np.frombuffer(raw, dt).reshape(frames.shape)
    size = path.stat().st_size; path.unlink()
    return dec, size


def build(args, device):
    """-> name, embed(c_u8, s_u8) -> stego in [0,1] (float tensor, NCHW), recover(stego01) -> rec u8 (NHWC)."""
    t01 = lambda u8: torch.from_numpy(np.ascontiguousarray(u8.transpose(0, 3, 1, 2))).to(device).float() / 255  # noqa: E731
    to_u8 = lambda t: (t.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()                    # noqa: E731
    if args.model == "ilwt":
        model, _ = E.load_model()
        emb = lambda c, s: (model(torch.cat([E.u8_to_t(c, device), E.u8_to_t(s, device)], 1))[0][:, :3].clamp(-1, 1) + 1) / 2  # noqa: E731
        rec = lambda st: E.t_to_u8(model.inverse(torch.cat([st * 2 - 1, torch.zeros_like(st)], 1))[:, 3:])        # noqa: E731
        return "ILWT-INN", emb, rec
    for m in ("model", "config", "modules", "hinet", "invblock", "rrdb_denselayer", "denseblock"):
        sys.modules.pop(m, None)
    if args.model == "hinet":
        sys.path.insert(0, args.hinet_dir)
        from model import Model
        import modules.Unet_common as common
        dwt, iwt = common.DWT(), common.IWT()
        net = torch.nn.DataParallel(Model().to(device), device_ids=[0])
        sd = torch.load(args.hinet_ckpt, map_location=device, weights_only=False)["net"]
        net.load_state_dict({k: v for k, v in sd.items() if "tmp_var" not in k}); net.eval()
        emb = lambda c, s: iwt(net(torch.cat((dwt(t01(c)), dwt(t01(s))), 1)).narrow(1, 0, 12)).clamp(0, 1)  # noqa: E731

        def rec(st):
            d = dwt(st)
            return to_u8(iwt(net(torch.cat((d, torch.randn_like(d)), 1), rev=True).narrow(1, 12, 12)))
        return "HiNet released", emb, rec
    sys.path.insert(0, args.deepmih_dir)
    from model import Model_1, Model_2
    from imp_subnet import ImpMapBlock
    import modules.Unet_common as common
    dwt, iwt = common.DWT(), common.IWT()
    nets = []
    for i, M in ((1, Model_1), (2, Model_2), (3, ImpMapBlock)):
        n = torch.nn.DataParallel(M().to(device), device_ids=[0])
        sd = torch.load(f"{args.deepmih_weights}{i}.pt", map_location=device, weights_only=True)["net"]   # downloaded file
        n.load_state_dict({k: v for k, v in sd.items() if "tmp_var" not in k}); n.eval(); nets.append(n)
    n1, n2, n3 = nets

    def emb(c_u8, s_u8):
        c, s1 = t01(c_u8), t01(s_u8)
        st1d = n1(torch.cat((dwt(c), dwt(s1)), 1)).narrow(1, 0, 12)
        imp = n3(c, s1, iwt(st1d))
        return iwt(n2(torch.cat((st1d, dwt(imp), dwt(torch.full_like(c, 0.5))), 1)).narrow(1, 0, 12)).clamp(0, 1)

    def rec(st):
        d = dwt(st)
        r2 = n2(torch.cat((d, torch.randn(d.shape[0], 24, *d.shape[2:], device=device)), 1), rev=True)
        return to_u8(iwt(n1(torch.cat((r2.narrow(1, 0, 12), torch.randn_like(d)), 1), rev=True).narrow(1, 12, 12)))
    return "DeepMIH 1-secret (24 BPP)", emb, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("ilwt", "hinet", "deepmih"))
    ap.add_argument("--videos", default=str(E.ROOT / "data/sd_videos"))
    ap.add_argument("--secret-dir", default=str(E.ROOT / "data/hide300"))
    ap.add_argument("--hinet-dir", default=str(E.ROOT / "baselines/HiNet"))
    ap.add_argument("--hinet-ckpt", default=str(E.ROOT / "baselines/HiNet/model/released/model.pt"))
    ap.add_argument("--deepmih-dir", default=str(E.ROOT / "baselines/DeepMIH"))
    ap.add_argument("--deepmih-weights", default=str(E.ROOT / "baselines/DeepMIH_weights/model_checkpoint_03000_"))
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default=str(E.ROOT / "runs/ffv1_bitdepth"))
    args = ap.parse_args()
    assert torch.cuda.is_available(), "no CUDA device (container lost the GPU?)"
    device = torch.device("cuda")
    name, emb, rec = build(args, device)
    out = Path(args.out) / args.model; out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="ffv1_", dir=out))
    hide = sorted(Path(args.secret_dir).glob("*.png"))[:300]
    rows, files = [], []
    for seq, rel in SEQUENCES.items():
        covers = covers_of(args.videos, rel)[::args.every]
        idx = np.arange(len(covers)) * args.every
        secrets = np.stack([E.load_secret_u8(hide[i % 300]) for i in idx])
        _, cover_bytes = ffv1_roundtrip(covers, 8, tmp / "cover.mkv")
        with torch.no_grad():
            st01 = np.concatenate([emb(covers[b:b + args.batch], secrets[b:b + args.batch]).permute(0, 2, 3, 1).cpu().numpy()
                                   for b in range(0, len(covers), args.batch)])
        for depth in (8, 16):
            mx = FMT[depth][2]
            q = np.round(st01 * mx)
            dec, nbytes = ffv1_roundtrip(q, depth, tmp / f"stego{depth}.mkv")
            assert np.array_equal(dec.astype(np.float64), q), f"{seq} {depth}-bit: FFV1 round trip not lossless"
            torch.manual_seed(0)
            with torch.no_grad():
                recs = np.concatenate([rec(torch.from_numpy(np.ascontiguousarray(dec[b:b + args.batch].transpose(0, 3, 1, 2)))
                                           .to(device).float() / mx) for b in range(0, len(dec), args.batch)])
            st255 = dec.astype(np.float32) * (255.0 / mx)
            for i, c, s_, sec, r in zip(idx, covers, st255, secrets, recs):
                m = E.frame_metrics(sec, r, device)
                rows.append({"model": name, "depth": depth, "seq": seq, "frame": int(i) + 1,
                             "H_PSNR": round(E.psnr_u8(c, s_), 4), "H_SSIM": round(E.ssim_u8(c, s_, device), 6),
                             **{k: round(v, 6) for k, v in m.items()}})
            files.append({"model": name, "depth": depth, "seq": seq, "frames": len(covers), "lossless": True,
                          "stego_bytes": nbytes, "cover8_bytes": cover_bytes,
                          "stego_bpp": round(8 * nbytes / (len(covers) * S * S), 4),
                          "cover8_bpp": round(8 * cover_bytes / (len(covers) * S * S), 4),
                          "growth_vs_cover8_pct": round(100 * (nbytes / cover_bytes - 1), 2)})
            cur = [x for x in rows if x["depth"] == depth and x["seq"] == seq]
            print(f"{seq:13s} {name:26s} {depth:2d}-bit file  H {np.mean([x['H_PSNR'] for x in cur]):.2f}  "
                  f"R {np.mean([x['R_PSNR'] for x in cur]):.2f}  BER {np.mean([x['BER'] for x in cur]):.4f}  "
                  f"{files[-1]['stego_bpp']:.3f} bpp ({files[-1]['growth_vs_cover8_pct']:+.1f}% vs cover)  lossless", flush=True)
        E.write_csv(out / "per_frame.csv", rows); E.write_csv(out / "files.csv", files)
    tmp.rmdir()
    summ = E.summarize(rows, ["model", "depth"], METRICS)
    E.write_csv(out / "summary.csv", [{k: v for k, v in r.items() if not k.startswith("frac_")} for r in summ])
    print("\nmodel                      depth  H-PSNR  H-SSIM  R-PSNR (95% CI)              R-SSIM  NC      BER     file bpp  growth")
    for r in summ:
        fs = [f for f in files if f["depth"] == r["depth"]]
        bpp = 8 * sum(f["stego_bytes"] for f in fs) / sum(f["frames"] * S * S for f in fs)
        gr = 100 * (sum(f["stego_bytes"] for f in fs) / sum(f["cover8_bytes"] for f in fs) - 1)
        print(f"{r['model']:26s} {r['depth']:5d}  {r['H_PSNR']:6.2f}  {r['H_SSIM']:.4f}  {r['R_PSNR']:6.2f} "
              f"{r['R_PSNR_CI95']:20s}  {r['SSIM']:.4f}  {r['NC']:.4f}  {r['BER']:.4f}  {bpp:7.3f}  {gr:+6.1f}%")


if __name__ == "__main__":
    main()
