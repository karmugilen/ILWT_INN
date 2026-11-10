#!/usr/bin/env python
"""
Multi-seed ablation.

Runs ablation_sweep.py (19 configs, 200 hide images, 100 epochs) unchanged except for the
training seed. The 140/30/30 split stays fixed (default_rng(42) inside ablation_sweep.shared_splits), so
the seed changes only weight init, data order, augmentation and secret pairing. The image list keeps
the glob order of the original run (--order), because ablation_sweep picks images by index
into that order; seed 42 therefore re-runs the original configuration and checks it reproduces.

Outputs: <out>/seed<k>/<config>/test_metrics.json and <out>/seed<k>/ablation_table.{csv,md}
"""
import argparse
import os
import re
import sys
import types
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--code", required=True, help="copy of the ablation repo (ablation_sweep.py, train.py, core/, shell/)")
ap.add_argument("--data", required=True, help="full hide folder")
ap.add_argument("--order", required=True, help="hide file names in the original glob order")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=0, help="override NUM_EPOCHS (smoke tests only; 0 = default 100)")
args, rest = ap.parse_known_args()

import torch  # noqa: E402
assert torch.cuda.is_available(), "no CUDA device (container lost the GPU?) - refusing to train on CPU"

os.chdir(args.code)
sys.path.insert(0, args.code)
import train  # noqa: E402

names = [l.strip() for l in open(args.order) if l.strip()]
_init = train.ImageSteganographyDataset.__init__


def _init_fixed_order(self, image_dir, *a, **k):
    _init(self, image_dir, *a, **k)
    paths = [os.path.join(image_dir, n) for n in names]
    assert sorted(paths) == sorted(self.image_paths), "hide folder does not match the original file list"
    self.image_paths = paths


train.ImageSteganographyDataset.__init__ = _init_fixed_order

src = Path("ablation_sweep.py").read_text()
for pat in (r"torch\.manual_seed\(42\)", r"np\.random\.seed\(42\)", r"(?<![\w.])random\.seed\(42\)"):
    src, n = re.subn(pat, lambda m: m.group(0).replace("42", str(args.seed)), src)
    assert n == 1, pat
S = types.ModuleType("ablation_sweep")
S.__file__ = str(Path("ablation_sweep.py").resolve())
exec(compile(src, S.__file__, "exec"), S.__dict__)
S.OUT_ROOT = Path(args.out) / f"seed{args.seed}"
S.DATA_DIR = args.data
if args.epochs:
    S.NUM_EPOCHS = args.epochs
print(f"seed {args.seed}  out {S.OUT_ROOT}  data {S.DATA_DIR}  ({len(names)} images in original order)", flush=True)
S.main(rest or ["--all", "--no-menu"])
