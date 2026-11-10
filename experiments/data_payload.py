#!/usr/bin/env python
"""
arbitrary-data payload with error correction.

Deployed model, deployed 8-bit storage, covers = 5 evenly spaced frames of each of the 20 Xiph
sequences (standard resize, ffmpeg lanczos 256x256) = 100 frames. The "secret image" carries data bits:

  stage 1  raw bit-plane BER when the secret is not a natural image:
             random       all 8 planes uniform random bits
             msb<k>       hide secret (frame i -> secret[i % 300]) with its top k planes replaced
                          by random bits (k = 1, 2, 3); the lower planes keep the natural image
  stage 2  real error correction, end to end: a random message is Reed-Solomon encoded (RS(255, 255-nsym)
           over GF(256), reedsolo), each coded bit repeated `rep` times (majority-vote inner code),
           bit-interleaved across the whole frame, written into the chosen
           planes of the secret, embedded, stored as 8 bits, recovered blindly, de-interleaved and
           decoded. Reported: raw BER on those planes, byte-error rate, codeword failure rate and
           post-decoding BER, and the error-free payload  3 x planes x (255 - nsym) / 255 / rep  BPP.

Outputs (--out): planes.csv (stage 1), ecc.csv (stage 2).
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from reedsolo import RSCodec, ReedSolomonError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as E  # noqa: E402
from make_stego_set import SEQUENCES  # noqa: E402

N = 255


def covers_of(videos, per_seq):
    out = []
    for rel in SEQUENCES.values():
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(Path(videos) / rel), "-vf",
                              "scale=256:256:flags=lanczos", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                             check=True, stdout=subprocess.PIPE).stdout
        v = np.frombuffer(raw, np.uint8).reshape(-1, 256, 256, 3)
        idx = np.linspace(0, len(v) - 1, per_seq).round().astype(int)
        out += [(int(i), v[i]) for i in idx]
    return out


def roundtrip(model, device, covers, secrets, bs=4):
    stego = []
    with torch.no_grad():
        for b in range(0, len(covers), bs):
            x = torch.cat([E.u8_to_t(covers[b:b + bs], device), E.u8_to_t(secrets[b:b + bs], device)], 1)
            stego.append(E.t_to_u8(model(x)[0][:, :3]))
    return E.recover(model, np.concatenate(stego), device, bs=bs)


def plane_ber(a, b):
    d = np.bitwise_xor(a, b)
    return [float(((d >> p) & 1).mean()) for p in range(8)]


def put_bits(img, planes, bits):
    """Write bits (len = H*W*3*len(planes)) into the given bit planes of img (uint8 HWC)."""
    flat = img.reshape(-1).copy()
    for j, p in enumerate(planes):
        bp = bits[j * flat.size:(j + 1) * flat.size].astype(np.uint8)
        flat = (flat & np.uint8(0xFF ^ (1 << p))) | (bp << p)
    return flat.reshape(img.shape)


def get_bits(img, planes):
    flat = img.reshape(-1)
    return np.concatenate([(flat >> p) & 1 for p in planes]).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--secret-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-seq", type=int, default=5)
    ap.add_argument("--stage", default="1,2")
    ap.add_argument("--configs", default="7:1:64,7:5:128,7:9:64,7:9:128,7:15:64,7:15:128,7:21:64,7:31:32",
                    help="planes:rep:nsym list; 7:15:128 = plane 7, each coded bit repeated 15x (majority vote), "
                         "outer RS(255,127)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model, device = E.load_model()
    cov = covers_of(args.videos, args.per_seq)
    covers = np.stack([c for _, c in cov])
    hide = sorted(Path(args.secret_dir).glob("*.png"))[:300]
    natural = np.stack([E.load_secret_u8(hide[i % len(hide)]) for i, _ in cov])
    rng = np.random.default_rng(0)

    if "1" in args.stage:
        rows = []
        variants = {"natural": natural, "random": rng.integers(0, 256, natural.shape, dtype=np.uint8)}
        for k in (1, 2, 3):
            s = natural.copy()
            planes = list(range(8 - k, 8))
            for n in range(len(s)):
                s[n] = put_bits(s[n], planes, rng.integers(0, 2, s[n].size * k))
            variants[f"msb{k}"] = s
        for name, sec in variants.items():
            rec = roundtrip(model, device, covers, sec)
            ber = np.mean([plane_ber(a, b) for a, b in zip(sec, rec)], axis=0)
            rpsnr = np.mean([E.psnr_u8(a, b) for a, b in zip(sec, rec)])
            rows.append({"secret": name, "frames": len(sec), "R_PSNR": round(rpsnr, 3),
                         **{f"BER_bit{p}": round(ber[p], 5) for p in range(8)}, "BER_all": round(ber.mean(), 5)})
            print(rows[-1], flush=True)
        E.write_csv(out / "planes.csv", rows)

    if "2" in args.stage:
        rows = []
        for cfg in args.configs.split(","):
            pl, rep, nsym = cfg.split(":")
            planes, rep, nsym = [int(c) for c in pl], int(rep), int(nsym)
            rs = RSCodec(nsym, nsize=N)
            nbits = 256 * 256 * 3 * len(planes)
            n_cw = nbits // rep // (8 * N)
            perm = np.random.default_rng(1).permutation(nbits)          # fixed bit interleaver
            secrets, sent = [], []
            for n in range(len(covers)):
                msg = rng.integers(0, 256, n_cw * (N - nsym), dtype=np.uint8).tobytes()
                cw = b"".join(bytes(rs.encode(msg[i:i + N - nsym])) for i in range(0, len(msg), N - nsym))
                bits = np.zeros(nbits, np.uint8)
                bits[:len(cw) * 8 * rep] = np.repeat(np.unpackbits(np.frombuffer(cw, np.uint8)), rep)
                inter = np.empty_like(bits); inter[perm] = bits
                secrets.append(put_bits(natural[n], planes, inter)); sent.append(msg)
            rec = roundtrip(model, device, covers, np.stack(secrets))
            raw_err = byte_err = fails = post_bit_err = total_bits = 0
            for r, s, msg in zip(rec, secrets, sent):
                L = n_cw * N * 8
                got_raw = get_bits(r, planes)[perm][:L * rep]
                ref_raw = get_bits(s, planes)[perm][:L * rep]
                raw_err += int((got_raw != ref_raw).sum())
                got = (got_raw.reshape(L, rep).sum(1) * 2 > rep).astype(np.uint8)   # majority vote
                ref = ref_raw.reshape(L, rep)[:, 0]
                gb, rb = np.packbits(got).tobytes(), np.packbits(ref).tobytes()
                dec = bytearray()
                for i in range(n_cw):
                    c = gb[i * N:(i + 1) * N]
                    byte_err += sum(x != y for x, y in zip(c, rb[i * N:(i + 1) * N]))
                    try:
                        dec += rs.decode(c)[0]
                    except ReedSolomonError:
                        fails += 1; dec += c[:N - nsym]
                post_bit_err += int(np.unpackbits(np.bitwise_xor(np.frombuffer(bytes(dec), np.uint8),
                                                                  np.frombuffer(msg, np.uint8))).sum())
                total_bits += len(msg) * 8
            ncw = n_cw * len(covers)
            row = {"planes": "".join(map(str, planes)), "repetition": rep, "RS": f"({N},{N - nsym})",
                   "rate": round((N - nsym) / N / rep, 5), "frames": len(covers),
                   "raw_BER": round(raw_err / (ncw * N * 8 * rep), 5),
                   "byte_error_rate": round(byte_err / (ncw * N), 5), "codeword_fail_rate": round(fails / ncw, 5),
                   "post_BER": f"{post_bit_err / total_bits:.2e}", "error_free_frames": None,
                   "payload_BPP": round(3 * len(planes) * (N - nsym) / N / rep, 4),
                   "bits_per_frame": n_cw * (N - nsym) * 8}
            row["error_free_frames"] = "all" if post_bit_err == 0 else "no"
            rows.append(row); print(row, flush=True)
            E.write_csv(out / "ecc.csv", rows)


if __name__ == "__main__":
    main()
