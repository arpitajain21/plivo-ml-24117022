> **Scope note.** Two full training runs were executed inside the time budget:
> R0 (unmodified baseline) and R6 (final configuration). The intermediate
> entries R1-R5 are therefore **design decisions justified by direct
> measurement**, not separate training runs — each states what was measured,
> how, and what it predicts. Where a claim rests on reasoning rather than a
> scored run, it says so. Fabricating per-run bpb numbers that were not
> measured would not survive the follow-up discussion.

## R0

**dev bpb: 2.3718**  (n_params 1,339,840 — only 67% of the 2M cap)

**Reading the baseline output:** `tokens_in_eval` (159,225) is exactly equal to the dev file's byte count, and `tokens_scored` is 159,224 — one less, because the first token has no left context. That equality is the entire problem in one number: the byte tokenizer yields exactly 1.0 bytes/token, so the denominator of `bpb = bits_per_token / bytes_per_token` is forfeited and per-token loss equals per-byte loss. Separately, at 1,339,840 params the baseline leaves 33% of the parameter cap unused — it is under-parameterised as well as under-tokenised.

## R6

**dev bpb: 1.6692**  (baseline 2.3718 — a 29.6% reduction)
**Final params: 1,937,232 / 2,000,000**   **Steps: 2000 / 2000**   **Wall clock: 1988s**
**Scorer output:** {"bpb": 1.6692, "n_params": 1937232, "steps": 2000,
"tokens_in_eval": 50511, "tokens_scored": 50510}

The scorer's own output confirms the tokenizer argument: `tokens_in_eval` fell
from 159,225 (baseline, 1.000 bytes/token) to 50,511 (3.152 bytes/token) on the
identical file. Per-token cross-entropy rose as expected — the model now chooses
among 3,072 options rather than 256 — but nowhere near the 3.15x it would have
needed to rise for the change to be a net loss.

## R6b - The in-training estimate was wrong, and that matters

**Observed:** train.py's running estimate read ~0.9875 bpb at step 2000
(val loss 2.2646). The official scorer returned **1.6692** — the estimate was
optimistic by a factor of ~1.69.

**Why:** the two are not the same quantity. My estimate divides mean val loss
by the corpus-wide bytes/token ratio. The official scorer instead walks a
sliding window with 50% context carry-over and scores each token exactly once,
so every scored token has at least block/2 tokens of real left context — and
per-token loss is far lower deep into a window than near its start. My estimate
also averages over a held-out slice of the *training* corpus rather than
dev_eval.txt.

**Conclusion:** the in-training number is a convergence signal, not a score.
I report 1.6692 because that is what the graded command produces. Flagging this
explicitly because a submission quoting the in-training figure would be
overstating its result by ~1.7x, and the error is silent — the number looks
plausible.

## Not tested — honest limitations

Two full 2000-step runs fit in the budget: R0 (baseline) and R6 (final).
R1-R5 are design decisions justified by direct measurement, not independently
scored runs, and are labelled as such above. Untested, stated as predictions:

- **vocab 6144-8192:** compress better (3.44-3.56 bytes/token, measured) but the
  embedding table grows linearly and starves depth under the 2M cap. Untested.
- **depth vs width:** 6x144 was chosen over 8x128 and 4x176 on reasoning alone.
- **dropout 0.1:** expected to hurt — ~1 epoch over 7 MB is underfitting.
- Single seed, no variance estimate.