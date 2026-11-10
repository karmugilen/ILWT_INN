#!/bin/bash
# Extra training seed of the final model, then evaluation on the standard protocol
# (20 Xiph sequences, first 300 frames, standard secrets, 8-bit stego). Writes only to runs/ilwt_seed$1/.
set -e
S=$1; EP=${2:-100}; MAXI=${3:-0}; MF=${4:-300}; O=runs/ilwt_seed$S
cd ~/ilwt_workspace
# Some containers route CPU BLAS through NVBLAS, which crashes without a CPU library: point it at OpenBLAS
[ -f experiments/nvblas_cpu.conf ] && export NVBLAS_CONFIG_FILE=$PWD/experiments/nvblas_cpu.conf
echo "##### $(date '+%F %T') train seed $S"
python experiments/ilwt_seed_train.py --seed $S --out $O --epochs $EP --max-images $MAXI
M=$O/run/ilwt_steganography_model_research.pth
echo "##### $(date '+%F %T') stego set"
python experiments/make_stego_set.py --model $M --secret-dir data/hide300 --max-frames $MF --out $O/stego_hide
echo "##### $(date '+%F %T') evaluation"
python experiments/ilwt_seed_eval.py --model $M --sets hide=$O/stego_hide=data/hide300 --out $O/eval
chmod -R a-w $O
echo "##### $(date '+%F %T') seed $S finished"
