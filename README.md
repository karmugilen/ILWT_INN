# ILWT-INN: Blind High-Capacity Video Steganography

ILWT-INN hides a full RGB image (256×256, 24 bits per pixel) in every frame of a video and recovers it blindly: extraction needs only the stego video and the trained model, with no cover video and no side information. A learned integer lifting wavelet transform feeds an invertible neural network (8 blocks, 128 hidden channels, about 0.62 M parameters), trained with an 8-bit straight-through estimator so recovery survives storing the stego frame with 8 bits per channel.

This repository contains:

- the model and training code (`core/`, `train.py`),
- the trained model (`ilwt_steganography_model_research.pth`, MD5 `2e3b346e1338…`),
- command-line tools to embed and extract images and videos (`scripts/`, `inference.py`),
- the ablation code (`ablation_sweep.py`),
- the evaluation and experiment scripts (`experiments/`).

---

## Installation

Requirements: Python 3.8+, PyTorch, FFmpeg.

```bash
git clone https://github.com/karmugilen/ILWT_INN.git
cd ILWT_INN
pip install -r requirements.txt
```

The trained weights `ilwt_steganography_model_research.pth` are included in the repository root and are used by default.

---

## Hiding and recovering

### Images

```bash
python scripts/embed_image.py   --cover cover.png --secret secret.png --output stego.png
python scripts/extract_image.py --stego stego.png --output recovered.png
```

### Videos

```bash
python scripts/embed_video.py   --video cover.y4m --hide_dir secrets/ --output stego.mkv
python scripts/extract_video.py --stego stego.mkv --output_dir recovered/
```

Frame *i* carries the *(i mod N)*-th image of `--hide_dir`. Frames are processed at 256×256 (`--size`). The stego video is written losslessly (FFV1, RGB, 8 bits per channel); recovery requires lossless transport, since lossy codecs (H.264/H.265) destroy the hidden image.

`inference.py --y4m video.y4m` runs the full embed–extract pipeline interactively, taking secret images from `secrets/`.

### Notes on use

- Extraction is blind but **keyless**: anyone with the stego video and this public model can recover the hidden image. Encrypt the payload separately if confidentiality is required.
- At 24 BPP the embedding is **detectable** by retrained CNN steganalysers (`experiments/steganalysis_24bpp.py`). The method gives imperceptibility to a human observer, not resistance to steganalysis.

---

## Training

```bash
python train.py
```

Training images go in `training_data/`. The released model was trained for 100 epochs on 35,000 ImageNet images (seed-42 70/15/15 split) at 256×256 with batch size 1 (about 146 h on a Quadro P5000). The seed-42 split used by the experiment scripts is in `experiments/splits/imagenet_split_seed42.json` (ImageNet file names; the images themselves are not redistributed).

---

## Reproducing the experiments

The scripts in `experiments/` expect this workspace layout (paths are set at the top of `experiments/eval_common.py` and in the `--…` arguments of each script):

```text
~/ilwt_workspace/
├── code/        this repository   (or set ILWT_CODE=<path to this repository>)
├── experiments/ the scripts in this folder
├── splits/      experiments/splits/
├── data/
│   ├── sd_videos/       the 20 Xiph.org test sequences (.y4m)
│   ├── hide300/         the 300 secret images of the standard protocol (list: experiments/splits/hide_first300.txt)
│   └── imagenet256/     ImageNet images resized to 256×256
├── baselines/   HiNet and DeepMIH code and released weights (see below)
└── runs/        outputs
```

Evaluation protocol (all scripts): the first 300 frames of each of the 20 Xiph.org sequences, resized to 256×256 with ffmpeg; frame *i* carries secret *(i mod 300)*; the stego frame is rounded to 8 bits per channel and stored as PNG or FFV1; PSNR on the 0–255 scale, SSIM with an 11×11 Gaussian window (σ = 1.5), BER over all eight bit planes.

| Experiment | Script(s) |
|---|---|
| Stego sets of the proposed model | `make_stego_set.py` |
| Same-payload comparison with HiNet and DeepMIH | `hinet_eval_20seq.py`, `deepmih_eval.py`, `make_stego_baseline.py` |
| HiNet and DeepMIH fine-tuned with the 8-bit STE | `hinet_ste_finetune.py`, `extend_hinet100_now.sh`, `deepmih_ste_finetune.py` |
| Paired Wilcoxon tests | `wilcoxon_e14_e15.py` |
| Extra training seeds of the final model | `ilwt_seed_train.py`, `ilwt_seed_eval.py`, `queue_seed.sh`, `cmp_seeds.py` |
| Cross-dataset and native-resolution evaluation | `cross_dataset_eval.py`, `native_res.py` |
| Attacks on all 20 sequences | `attacks_20seq.py` |
| Lossy and lossless codecs | `codec_20seq.py` |
| XAI attack atlases | `atlas_deployed.py`, `show_atlas.py` |
| Timing (model and end-to-end) | `timing.py` |
| Storage bit depth (8-bit, 16-bit, float) | `bitdepth_sweep.py`, `ffv1_bitdepth.py` |
| CNN steganalysis at 24 BPP (6 detectors, 5 folds, 3 seeds) | `steganalysis_24bpp.py`, `fig_roc_24bpp.py` |
| Classical statistics as hypothesis tests | `ks_classical.py` |
| Bit-plane BER, recovery-error distribution, data payload | `error_distribution.py`, `data_payload.py` |
| Invertibility check | `invertibility_check.py` |
| Ablation with three seeds | `ablation_seeds.py`, `ablation_sweep_E07_used.py`, `ablation_hide_glob_order.txt` |

`eval_common.py` holds the shared model loading and metrics.

### Datasets and baseline methods

The datasets and the comparison methods are publicly available from their original sources, so they are linked here rather than copied into this repository:

- **Xiph.org test sequences:** available from the official [media.xiph.org](https://media.xiph.org/video/derf/) collection.
- **ImageNet:** available from the ImageNet project under its terms of access. Only the file lists of the split used here are included (`experiments/splits/`).
- **HiNet:** code and trained model provided by the authors at <https://github.com/TomTomTommi/HiNet>.
- **DeepMIH:** code and trained models provided by the authors at <https://github.com/TomTomTommi/DeepMIH>.

We thank the authors of these datasets and methods for making them available.

---

## Project structure

```text
.
├── core/                     model (ILWT, INN blocks, YCbCr and subband weighting) and settings
├── shell/                    video and image I/O, pipeline runner
├── scripts/                  embed / extract tools for images and videos
├── experiments/              evaluation and experiment scripts (see above)
│   └── splits/               ImageNet split and secret list
├── train.py                  training
├── inference.py              interactive embed–extract pipeline
├── ablation_sweep.py         ablation variants
├── ilwt_steganography_model_research.pth   trained model
└── requirements.txt
```
