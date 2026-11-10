#!/usr/bin/env python
"""
regenerate the XAI attack atlases (fig4_foreman_style_xai / fig6_silent_video_xai,
and the pooled scatter) with the deployed pipeline.

The original atlases were made by VS/explore/ilwt_inn_xai_grid.py with an earlier setup.
This wrapper runs that same script (same attacks, layout and colour scale) and changes only:
  model    deployed ilwt_steganography_model_research.pth (md5 2e3b346e...) instead of the v3 file
  stego    8-bit rounding (the deployed storage) instead of 16-bit for the no-attack row
  NC       mean-centred NCC (eval_common.nc_u8) instead of uncentred cosine
  BER      all 8 bit planes instead of MSB only
  filter   no NC >= 0.90 discovery filter: every candidate attack is kept, so the scatter is unselected
  inputs   covers = frame 1 of Akiyo / Silent with the deployed ffmpeg lanczos 256x256 resize
           (the original cover PNGs are lost); secrets = the same files (74.png, 1000.png),
           resized by the deployed loader (shell.io.load_image)
Badges keep the script's E_XAI (channel-max residual); metrics.csv also reports the
channel-mean E_XAI and the old metric definitions so the change can be explained.

Run (a CPU is enough):  python atlas_deployed.py --out <dir>
"""
import argparse
import csv
import hashlib
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

VS = Path("/home/kar/DontDelete/ramdisk/VS")
R8 = Path("/home/kar/DontDelete/ramdisk/8-bit-ILWT-INN")
VIDEOS = Path("~/ilwt_workspace/data/sd_videos").expanduser()
MODEL = R8 / "ilwt_steganography_model_research.pth"
HIDE = R8 / "Dec-18-E-and-E-V3/hide"

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--frame", type=int, default=1, help="1-based frame index of each cover video")
args = ap.parse_args()
out = Path(args.out).resolve(); (out / "inputs").mkdir(parents=True, exist_ok=True)
assert hashlib.md5(MODEL.read_bytes()).hexdigest().startswith("2e3b346e"), "not the deployed model"

# 1. the original script, with its own packages (attacks, plotting)
sys.path[:0] = [str(VS / "packages/core/src"), str(VS / "packages/shell/src"), str(VS)]
src = (VS / "explore/ilwt_inn_xai_grid.py").read_text()
for old, new in [
    ('    print(f"Loading 16-bit ILWT-INN weights from {WEIGHTS.name} ...")\n'
     '    method = load_ilwt_inn_method(WEIGHTS, size=256, num_blocks=8, hidden_channels=128, device="cpu")\n'
     '    model = method.model  # frozen, eval(), no_grad in all forward/inverse below\n',
     '    print("Deployed ILWT-INN model (8-bit pipeline)")\n    model = DEPLOYED_MODEL\n'),
    ("            sq = (sq * 65535).round() / 65535 * 2.0 - 1.0\n",
     "            sq = (sq * 255).round() / 255 * 2.0 - 1.0  # deployed 8-bit storage\n"),
    ("    render_foreman_scatter(q1_dir, all_rows, baseline)\n",
     "    render_foreman_scatter(q1_dir, all_rows, baseline)\n    _dump_rows(q1_dir, 'akiyo', all_rows, baseline, secret_np)\n"),
    ("    write_bowing_notes(q1_dir, bowing_rows, bowing_baseline)\n",
     "    write_bowing_notes(q1_dir, bowing_rows, bowing_baseline)\n    _dump_rows(q1_dir, 'silent', bowing_rows, bowing_baseline, secret_b)\n"),
]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
mod = types.ModuleType("atlas")
mod.__file__ = str(VS / "explore/ilwt_inn_xai_grid.py")
sys.modules["atlas"] = mod                  # dataclasses look the module up here
G = mod.__dict__
exec(compile(src, mod.__file__, "exec"), G)

# 2. swap in the deployed 8-bit-ILWT-INN code (its own `core` / `shell` packages)
for m in [m for m in sys.modules if m.split(".")[0] in ("core", "shell")]:
    del sys.modules[m]
sys.path[:0] = [str(R8), str(R8 / "scripts")]
import torch  # noqa: E402
from _common import load_pipeline_model  # noqa: E402
from shell.io import load_image  # noqa: E402

model, _, _ = load_pipeline_model(MODEL, 8, 128, 256, device=torch.device("cpu"))
G["DEPLOYED_MODEL"] = model


def ffmpeg_frame(rel, dst):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(VIDEOS / rel), "-vf",
                    f"select=eq(n\\,{args.frame - 1}),scale=256:256:flags=lanczos", "-frames:v", "1",
                    "-pix_fmt", "rgb24", str(dst)], check=True)
    return dst


def secret_png(name):
    t = load_image(HIDE / name, 256)
    u8 = ((t.clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(1, 2, 0).numpy()
    dst = out / "inputs" / f"secret_{name}"
    import cv2
    cv2.imwrite(str(dst), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
    return dst


DS = G["DatasetSpec"]
G["AKIYO_SPEC"] = DS("akiyo", "Akiyo", ffmpeg_frame("Television/akiyo_cif.y4m", out / "inputs/akiyo_frame.png"),
                     secret_png("74.png"))
G["SILENT_SPEC"] = DS("silent", "Silent", ffmpeg_frame("silent_cif.y4m", out / "inputs/silent_frame.png"),
                      secret_png("1000.png"))
G["OUT_DIR"] = out
G["NC_KEEP_THRESHOLD"] = -1.0          # keep every candidate attack (no selection)


def nc_centred(a, b):
    x = a.astype(np.float64).ravel() / 255.0; y = b.astype(np.float64).ravel() / 255.0
    x -= x.mean(); y -= y.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-8))


def ber_all(a, b):
    return float(np.unpackbits(np.bitwise_xor(a, b)).mean())


old_nc, old_ber = G["calc_ncc"], G["calc_ber"]
G["calc_ncc"], G["calc_ber"] = nc_centred, ber_all


def _dump_rows(q1_dir, key, rows, baseline, secret):
    with open(Path(q1_dir) / f"metrics_{key}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["attack", "category", "NC_centred", "BER_all_planes", "E_XAI_badge(channel-max)",
                    "E_XAI_channel_mean", "old_NC_uncentred", "old_BER_MSB"])
        for r in [{**baseline, "id": "none", "cat": "Baseline"}] + list(rows):
            rec = r["rec"]
            w.writerow([r["id"], r["cat"], f"{r['ncc']:.4f}", f"{r['ber']:.4f}", f"{r['xai_energy']:.4f}",
                        f"{np.abs(secret.astype(np.float64) - rec.astype(np.float64)).mean() / 255:.4f}",
                        f"{old_nc(secret, rec):.4f}", f"{old_ber(secret, rec):.4f}"])
    print(f"  ✓ wrote metrics_{key}.csv")


G["_dump_rows"] = _dump_rows
G["run"]()
