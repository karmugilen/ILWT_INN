#!/usr/bin/env python
"""
Extra training seeds of the final ILWT-INN model.
Runs the deployed training code (code/train.py, unchanged) in a fresh copy under <out>/run, so nothing in
code/ is touched (code/ilwt_steganography_model_research.pth is the released model and has the same file name).
Same protocol as the released model: 50,000 ImageNet images (data/imagenet256), 70/15/15 split with seed 42, 100 epochs,
batch 1, AMP. The image list is sorted (the original run used file-system order, which cannot be regenerated).
--seed sets the torch / numpy / random seeds (weight init, data order, augmentation); the split stays seed 42.
"""
import argparse, os, random, shutil, subprocess, sys
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--max-images", type=int, default=0, help="smoke test only")
a = ap.parse_args()
R = os.path.expanduser("~/ilwt_workspace")
run = os.path.join(R, a.out, "run")
assert not os.path.exists(run), f"{run} exists - refusing to overwrite"
shutil.copytree(os.path.join(R, "code"), run, symlinks=True, ignore=shutil.ignore_patterns(
    "*.pth", "__pycache__", "Dec-18-*", "*Dataset", "*_output", "checkpoints", "research_*",
    "publication_outputs", "ilwt_test_results", "training_data"))
os.symlink(os.path.join(R, "data/imagenet256"), os.path.join(run, "training_data"))
os.chdir(run); sys.path.insert(0, run)
import numpy as np, torch
assert torch.cuda.is_available(), "no CUDA device"
torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); np.random.seed(a.seed); random.seed(a.seed)
import train
_init = train.ImageSteganographyDataset.__init__
def _init_sorted(self, *x, **k):
    _init(self, *x, **k)
    self.image_paths = sorted(self.image_paths)[: a.max_images or None]
train.ImageSteganographyDataset.__init__ = _init_sorted
_split = train.split_dataset
def _split_then_seed(*x, **k):
    out = _split(*x, **k)                      # split stays seed 42, as for the released model
    np.random.seed(a.seed); random.seed(a.seed)  # augmentation follows the run seed
    return out
train.split_dataset = _split_then_seed
if a.epochs != 100:
    _tm = train.train_model
    train.train_model = lambda m, tr, va, num_epochs=100, **k: _tm(m, tr, va, num_epochs=a.epochs, **k)
print(f"seed {a.seed}  run dir {run}  epochs {a.epochs}  max_images {a.max_images or 'all'}", flush=True)
train.main()
print("final model:", os.path.join(run, "ilwt_steganography_model_research.pth"), flush=True)
