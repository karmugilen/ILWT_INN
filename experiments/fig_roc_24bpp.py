# pooled out-of-fold ROC curves of the six retrained detectors, three seeds
import csv, sys
from pathlib import Path
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(sys.argv[2] if len(sys.argv) > 2 else "runs/steganalysis_24bpp")   # folder with seed42/, seed43/, seed44/
out = Path(sys.argv[1] if len(sys.argv) > 1 else "roc_24bpp.pdf")
dets = [("xunet", "XuNet"), ("yenet", "YeNet"), ("srnet", "SRNet"),
        ("xunet_color", "XuNet-C"), ("yenet_color", "YeNet-C"), ("srnet_color", "SRNet-C")]
colors = ["#0072B2", "#E69F00", "#009E73", "#0072B2", "#E69F00", "#009E73"]   # Wong palette
plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
for j, (ax, title) in enumerate(zip(axes, ("Luminance input (Y)", "Colour-aware input (YCbCr)"))):
    handles = []
    for (key, name), c in list(zip(dets, colors))[3 * j:3 * j + 3]:
        for k, (seed, ls) in enumerate(((42, "-"), (43, "--"), (44, ":"))):
            r = list(csv.DictReader(open(D / f"seed{seed}" / f"roc_{key}.csv")))
            fpr = [max(float(x["fpr"]), 1e-4) for x in r]; tpr = [float(x["tpr"]) for x in r]
            ln, = ax.plot(fpr, tpr, ls, color=c, lw=1.1)
            if k == 0: ln.set_label(name); handles.append(ln)
    chance = [10 ** (-4 + 4 * i / 400) for i in range(401)]   # y = x sampled densely: a 2-point line is straight on a log axis
    ax.plot(chance, chance, color="0.6", lw=0.7)
    ax.set_xscale("log"); ax.set_xlim(1e-4, 1); ax.set_ylim(0, 1.01)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())   # no minor ticks on the log axis
    ax.set_xlabel("False-alarm rate $P_{FA}$ (log scale)"); ax.set_title(title, fontsize=9)
    ax.grid(True, which="major", lw=0.3, color="0.85")
    ax.legend(handles=handles, loc="lower right", frameon=False, title="solid / dashed / dotted = seed 42 / 43 / 44", title_fontsize=7)
axes[0].set_ylabel("Detection rate $1-P_{MD}$")
fig.tight_layout()
fig.savefig(out); print("wrote", out)
