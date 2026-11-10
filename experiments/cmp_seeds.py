"""3-seed ablation summary: per config mean ± SD of H/R-PSNR, and the paired difference to the proposed
config within each seed (config − proposed, mean ± SD across seeds). Writes <R>/ablation_3seed.csv."""
import csv, sys, statistics as st
R = sys.argv[1]; seeds = sys.argv[2:]
t = {s: {(r["Group"], r["Row"]): r for r in csv.DictReader(open(f"{R}/{s}/ablation_table.csv"))} for s in seeds}
keys = list(t[seeds[0]]); prop = next(k for k in keys if "Proposed" in k[1])
f = lambda s, k, c: float(t[s][k][c])
ms = lambda v: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)
rows = []
print(f"{'Group':13s} {'Row':28s}  H mean±SD     R mean±SD     dH vs prop    dR vs prop   dR>0 seeds")
for k in keys:
    H = [f(s, k, "Hiding PSNR") for s in seeds]; Rr = [f(s, k, "Recovery PSNR") for s in seeds]
    dH = [f(s, k, "Hiding PSNR") - f(s, prop, "Hiding PSNR") for s in seeds]
    dR = [f(s, k, "Recovery PSNR") - f(s, prop, "Recovery PSNR") for s in seeds]
    (hm, hs), (rm, rs), (dhm, dhs), (drm, drs) = ms(H), ms(Rr), ms(dH), ms(dR)
    pos = sum(d > 0 for d in dR)
    print(f"{k[0]:13s} {k[1]:28s}  {hm:5.2f}±{hs:4.2f}   {rm:5.2f}±{rs:4.2f}   {dhm:+5.2f}±{dhs:4.2f}   {drm:+5.2f}±{drs:4.2f}   {pos}/{len(seeds)}")
    rows.append({"Group": k[0], "Row": k[1], "H_mean": round(hm, 3), "H_sd": round(hs, 3), "R_mean": round(rm, 3), "R_sd": round(rs, 3),
                 "dH_mean": round(dhm, 3), "dH_sd": round(dhs, 3), "dR_mean": round(drm, 3), "dR_sd": round(drs, 3),
                 "dR_pos_seeds": pos, "seeds": " ".join(seeds), **{f"R_{s}": f(s, k, "Recovery PSNR") for s in seeds}})
with open(f"{R}/ablation_3seed.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
