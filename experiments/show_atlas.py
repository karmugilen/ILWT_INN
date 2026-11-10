import csv
import sys

base = sys.argv[1]
keep = {"akiyo": ["none", "uniform_noise_5", "sharpen", "crop_fragment_15pct", "translation_10px", "watermark_text", "saturation_1.2"],
        "silent": ["none", "gauss_noise_2", "motion_blur_5", "contrast_1.1", "rotate_0.5deg", "brightness_+30", "histogram_eq"]}
for k, ids in keep.items():
    rows = {r["attack"]: r for r in csv.DictReader(open(f"{base}/q1/metrics_{k}.csv"))}
    print(f"== {k}\n{'attack':22s} NC_new  BER_all | oldNC   BER_MSB")
    for a in ids:
        r = rows[a]
        print(f"{a:22s} {r['NC_centred']:>6s}  {r['BER_all_planes']:>6s} | {r['old_NC_uncentred']:>6s}  {r['old_BER_MSB']:>6s}")
