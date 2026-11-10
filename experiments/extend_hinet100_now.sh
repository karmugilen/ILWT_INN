#!/bin/bash
# Continue the HiNet STE fine-tune from last.pt (same weights, Adam state and learning rate) to epoch 100,
# with the original arguments except --epochs and a time cap that no longer binds.
cd "$HOME/ilwt_workspace"
grep -aq "time cap reached" logs/hinet.log || { echo "hinet did not stop at the time cap; not extending"; exit 1; }
echo "##### $(date '+%F %T') extending the HiNet fine-tune to epoch 100 from last.pt"
for i in 1 2 3; do
  python experiments/hinet_ste_finetune.py --out runs/hinet_ste --epochs 100 --max-hours 200 --lr 5e-7 --beta1 0.9 \
    --lambda-guide 2 --warmup-iters 1000 --lr-halve-every 30 --val-images 1000 --save-every-epoch && break
  echo "=== crashed (try $i), resuming from last.pt in 30 s"; sleep 30
done
echo "##### $(date '+%F %T') extension finished"
