"""Preflight: catch interface breakage BEFORE burning a training run.

Run:  python sanity.py
"""
import os
import subprocess
import sys

import torch

import tokenizer as tokenizer_mod
from model import GPT, Config

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.join(HERE, "..", "data", "dev_eval.txt")

ok = True


def check(name, cond, extra=""):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    ok = ok and bool(cond)


print("1) tokenizer")
tok = tokenizer_mod.load()
check("load() takes no args", True, f"vocab={tok.vocab_size}")
samples = ["hello world", "नमस्ते दुनिया", "emoji 😀🎉", "中文テスト",
           "\x00\x01\x02raw bytes", "", "a", "   \n\t  ",
           "mixed हिंदी and English 123 !@#"]
if os.path.exists(DEV):
    samples.append(open(DEV, encoding="utf-8").read())
for s in samples:
    if tok.decode(tok.encode(s)) != s:
        check(f"lossless on {s[:24]!r}", False)
        break
else:
    check("lossless round-trip on all samples incl. full dev file", True)

if os.path.exists(DEV):
    d = open(DEV, encoding="utf-8").read()
    bpt = len(d.encode("utf-8")) / len(tok.encode(d))
    check("compression measured", True, f"{bpt:.3f} bytes/token on dev")

print("2) model + caps")
cfg = Config()
cfg.vocab_size = tok.vocab_size
m = GPT(cfg)
n = m.n_params()
check("params under 2,000,000", n <= 2_000_000, f"{n:,}")
check("weights tied", m.head.weight is m.tok_emb.weight)

print("3) forward pass shapes")
x = torch.randint(0, cfg.vocab_size, (2, 64))
logits, loss = m(x, x)
check("logits shape", tuple(logits.shape) == (2, 64, cfg.vocab_size),
      str(tuple(logits.shape)))
check("loss is finite", torch.isfinite(loss).item(), f"{loss.item():.3f}")
logits1, _ = m(x[:1, :17])          # odd length, as eval may produce
check("odd-length window works", tuple(logits1.shape) == (1, 17, cfg.vocab_size))
full, _ = m(torch.randint(0, cfg.vocab_size, (1, cfg.block_size)))
check("full block_size window works", full.shape[1] == cfg.block_size)

print("4) checkpoint round-trip through evaluate.load_model")
ck = os.path.join(HERE, "_sanity_ckpt.pt")
torch.save({"model": m.state_dict(),
            "config": {k: getattr(cfg, k) for k in dir(cfg)
                       if not k.startswith("_") and not callable(getattr(cfg, k))},
            "steps": 1}, ck)
import evaluate as ev
m2, cfg2, ckpt = ev.load_model(ck)
check("evaluate.load_model rebuilds model", m2.n_params() == n)
check("steps recorded in checkpoint", ckpt.get("steps") == 1)

print("5) official command end-to-end")
if os.path.exists(DEV):
    r = subprocess.run([sys.executable, "evaluate.py", "--checkpoint", ck,
                        "--text_file", DEV],
                       capture_output=True, text=True, cwd=HERE)
    print("     stdout:", r.stdout.strip()[:200])
    if r.returncode != 0:
        print("     stderr:", r.stderr.strip()[-800:])
    check("evaluate.py runs and prints JSON", r.returncode == 0 and '"bpb"' in r.stdout)
os.remove(ck)

print()
print("ALL CHECKS PASSED" if ok else "SOMETHING FAILED - fix before training")
sys.exit(0 if ok else 1)
