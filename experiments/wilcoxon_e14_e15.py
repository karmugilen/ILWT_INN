"""Paired Wilcoxon (20 per-sequence means) ILWT-INN (E14 bitdepth per_frame) vs DeepMIH (E15 per_frame), per depth."""
import csv, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
a_csv, b_csv = sys.argv[1], sys.argv[2]
def load(p):
    d = defaultdict(list)
    for r in csv.DictReader(open(p)):
        d[(r["model"], int(r["depth"]), r["seq"])].append(r)
    return d
A, B = load(a_csv), load(b_csv)
seqs = sorted({k[2] for k in A})
for name in sorted({k[0] for k in B}):
    for d in (8, 16, 32):
        for m in ("H_PSNR", "R_PSNR", "BER"):
            x = np.array([np.mean([float(r[m]) for r in A[("ILWT-INN", d, s)]]) for s in seqs])
            y = np.array([np.mean([float(r[m]) for r in B[(name, d, s)]]) for s in seqs])
            nA = sum(len(A[("ILWT-INN", d, s)]) for s in seqs); nB = sum(len(B[(name, d, s)]) for s in seqs)
            w, p = wilcoxon(x, y)
            print(f"{name:30s} {d:2d}-bit {m:6s} ours {x.mean():7.3f}  DeepMIH {y.mean():7.3f}  median diff {np.median(x-y):+7.3f}  W {w:5.1f}  p {p:.2g}  frames {nA}/{nB}")
