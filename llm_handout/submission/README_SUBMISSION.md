# How to run (from inside this folder, venv active)

    # 1. preflight - catches interface breakage before you burn a run
    python sanity.py

    # 2. train tokenizer (already done: bpe.json is included, ~10s to redo)
    python tokenizer.py --data ../data/train_corpus.txt --vocab 3072 --out bpe.json

    # 3. FINAL run (2000 steps, at the cap)
    python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt \
      --batch 16 --block 512 --lr 3e-3 --warmup 150 --wd 0.1 \
      --n_layer 6 --n_head 6 --n_embd 144 --n_ff 384

    # 4. score
    python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt

## If it is too slow on your CPU
Time step 1..50 first. If >1.5 s/step, drop to --block 256 --batch 16
(halves tokens/step but keeps the step count legal). Do ablations at
--steps 500 and only the FINAL at 2000.

## Baseline number for RUNLOG R0
Run the pristine starter/ copy unmodified to get the honest reference.
