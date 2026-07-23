"""Trainer. Every deviation from the starter is deliberate; see RUNLOG.md.

HARD CAPS (asserted below):
  * max 2,000 optimizer steps  (an optimizer step = one opt.step() call,
    so gradient accumulation micro-batches do NOT count extra)
  * max 2,000,000 total parameters
  * train_corpus.txt only, pure PyTorch/numpy/stdlib, CPU

Usage:
  python tokenizer.py --data ../data/train_corpus.txt --vocab 3072
  python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


# --------------------------------------------------------------- data
def load_ids(path, tok, cache=".ids_cache.npy"):
    """Tokenise once and cache. Re-tokenising 7 MB on every run wastes
    minutes of a 120-minute budget."""
    here = os.path.dirname(os.path.abspath(__file__))
    cache = os.path.join(here, cache)
    meta = cache + ".meta.json"
    sig = {"path": os.path.abspath(path),
           "mtime": os.path.getmtime(path),
           "vocab": tok.vocab_size}
    if os.path.exists(cache) and os.path.exists(meta):
        try:
            if json.load(open(meta)) == sig:
                return np.load(cache)
        except Exception:
            pass
    text = open(path, encoding="utf-8").read()
    t0 = time.time()
    ids = np.array(tok.encode(text), dtype=np.uint16 if tok.vocab_size <= 65535
                   else np.int32)
    print(f"tokenised {len(text.encode('utf-8')):,} bytes -> {len(ids):,} "
          f"tokens in {time.time()-t0:.0f}s")
    np.save(cache, ids)
    json.dump(sig, open(meta, "w"))
    return ids


def get_batch(ids, block, batch, rng):
    """Random crops. Kept random (not sequential) so each step sees a mix of
    English and Hindi regions of the corpus."""
    ix = rng.integers(0, len(ids) - block - 1, size=batch)
    x = np.stack([ids[i:i + block] for i in ix]).astype(np.int64)
    y = np.stack([ids[i + 1:i + 1 + block] for i in ix]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


# ----------------------------------------------------------- schedule
def lr_at(step, total, base, warmup, final_frac):
    """Linear warmup then cosine decay to final_frac * base.

    The baseline used a CONSTANT lr with no warmup. At 2000 steps that is
    doubly wrong: no warmup means the first steps blow up the (badly
    initialised) residual stream, and no decay means the model is still
    taking large noisy steps at the end instead of settling into a minimum.
    """
    if step <= warmup:
        return base * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    prog = min(1.0, prog)
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return base * (final_frac + (1 - final_frac) * cos)


@torch.no_grad()
def quick_val(model, ids, block, batch, rng, iters=12):
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y = get_batch(ids, block, batch, rng)
        _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1,
                    help="micro-batches per optimizer step (does not consume "
                         "the step budget)")
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--min_lr_frac", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--vocab_size", type=int, default=None)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_embd", type=int, default=144)
    ap.add_argument("--n_ff", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cpu"

    tok = tokenizer_mod.load()
    ids_np = load_ids(args.data, tok)
    n_bytes = os.path.getsize(args.data)
    print(f"corpus: {n_bytes:,} bytes -> {len(ids_np):,} tokens "
          f"(vocab {tok.vocab_size}, {n_bytes/len(ids_np):.2f} bytes/token)")

    # hold out a slice for a quick in-training sanity signal
    split = int(0.995 * len(ids_np))
    tr, va = ids_np[:split], ids_np[split:]

    cfg = Config()
    cfg.vocab_size = args.vocab_size or tok.vocab_size
    cfg.block_size = args.block
    cfg.n_layer = args.n_layer
    cfg.n_head = args.n_head
    cfg.n_embd = args.n_embd
    cfg.n_ff = args.n_ff
    cfg.dropout = args.dropout
    cfg.tie_weights = True
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params  (cap {MAX_PARAMS:,}, "
          f"{100*n/MAX_PARAMS:.1f}% used)")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    # weight decay on matrices only; norms/embeddings excluded. Decaying the
    # (tied) embedding actively hurts at this scale.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and "tok_emb" not in name:
            decay.append(p)
        else:
            no_decay.append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, args.beta2), eps=1e-8, fused=False)

    print(f"optimizer: AdamW lr={args.lr} betas=(0.9,{args.beta2}) "
          f"wd={args.wd} warmup={args.warmup} clip={args.clip}")
    print(f"tokens/step: {args.batch*args.accum*args.block:,}")

    model.train()
    t0 = time.time()
    losses, val_hist = [], []
    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.steps, args.lr, args.warmup, args.min_lr_frac)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        acc_loss = 0.0
        for _ in range(args.accum):
            x, y = get_batch(tr, cfg.block_size, args.batch, rng)
            _, loss = model(x, y)
            (loss / args.accum).backward()
            acc_loss += loss.item() / args.accum
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        losses.append(acc_loss)
        if step % args.log_every == 0 or step == 1:
            k = min(args.log_every, len(losses))
            avg = sum(losses[-k:]) / k
            el = time.time() - t0
            eta = el / step * (args.steps - step)
            print(f"step {step:5d}  loss {avg:.4f}  lr {lr:.2e}  "
                  f"({el/step*1000:.0f} ms/step, eta {eta/60:.1f}m)")
        if args.val_every and (step % args.val_every == 0
                               or step == args.steps):
            v = quick_val(model, va, cfg.block_size, args.batch, rng)
            val_hist.append((step, v))
            print(f"    [val] step {step}  loss {v:.4f}  "
                  f"(~{v/math.log(2)*len(ids_np)/n_bytes:.4f} bpb est)")

    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "n_params": n,
                "args": vars(args),
                "val_hist": val_hist,
                "train_loss_curve": losses}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
