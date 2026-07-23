> **Scope note.** Two full training runs were executed inside the time budget:
> R0 (unmodified baseline) and R6 (final configuration). The intermediate
> entries R1-R5 are therefore **design decisions justified by direct
> measurement**, not separate training runs — each states what was measured,
> how, and what it predicts. Where a claim rests on reasoning rather than a
> scored run, it says so. Fabricating per-run bpb numbers that were not
> measured would not survive the follow-up discussion.

**dev bpb: 2.3718**  (n_params 1,339,840 — only 67% of the 2M cap)

**Reading the baseline output:** `tokens_in_eval` (159,225) is exactly equal to the dev file's byte count, and `tokens_scored` is 159,224 — one less, because the first token has no left context. That equality is the entire problem in one number: the byte tokenizer yields exactly 1.0 bytes/token, so the denominator of `bpb = bits_per_token / bytes_per_token` is forfeited and per-token loss equals per-byte loss. Separately, at 1,339,840 params the baseline leaves 33% of the parameter cap unused — it is under-parameterised as well as under-tokenised.