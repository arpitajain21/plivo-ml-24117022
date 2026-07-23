#!/usr/bin/env bash
# Experiment ladder. Run from inside the submission folder with env active.
# Each run writes a checkpoint and prints dev bpb. Log results in RUNLOG.md.
#
#   bash run_experiments.sh
#
# Tip: if a run is too slow on your laptop, drop --steps to 600 for the
# ABLATIONS only (relative ordering holds), then do the FINAL at 2000.

set -e
DATA=../data/train_corpus.txt
DEV=../data/dev_eval.txt

score () {  # $1 = checkpoint
  python evaluate.py --checkpoint "$1" --text_file $DEV
}

echo "=== R0: baseline (original starter code, byte tokenizer) ==="
echo "    run this from a pristine copy of starter/ for the honest baseline"

echo "=== R6: FINAL - full config, 2000 steps ==="
python train.py --data $DATA --steps 2000 --out ckpt.pt \
  --batch 16 --block 512 --lr 3e-3 --warmup 150 --wd 0.1 \
  --n_layer 6 --n_head 6 --n_embd 144 --n_ff 384
score ckpt.pt
