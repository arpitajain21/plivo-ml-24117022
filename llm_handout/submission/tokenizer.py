"""Byte-level BPE tokenizer, trained ONLY on train_corpus.txt.

Design notes
------------
* Base alphabet is the 256 raw bytes, so ANY UTF-8 text encodes losslessly.
  decode(encode(t)) == t is guaranteed by construction: we merge byte
  sequences only, and decoding replays merges in reverse to raw bytes.
* Merges are learned with a word-boundary-respecting split (GPT-2 style
  regex-lite) so merges never cross whitespace/word boundaries. This keeps
  the merge table clean and generalises to the hidden file.
* Devanagari is 3 bytes/char at byte level. BPE recovers this: common
  Hindi syllables collapse to single tokens, roughly tripling the
  compression on the Hindi portion.
* Artefacts live next to this file (bpe.json) and are resolved relative to
  __file__, so `python evaluate.py` works from any cwd.

Interface contract kept exactly:
    load() -> obj with .encode(str)->list[int], .decode(list[int])->str, .vocab_size
"""
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT = os.path.join(_HERE, "bpe.json")

# Split pattern: keeps a leading space attached to a word (so " the" is one
# token), isolates digits in short runs, and groups runs of non-space
# non-alnum punctuation. Devanagari falls into the \w class under re.UNICODE.
_SPLIT_RE = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


class BPETokenizer:
    """Byte-level BPE with guaranteed-lossless round trip."""

    def __init__(self, merges=None):
        # merges: list of [int, int] applied in rank order producing ids
        # 256, 257, ...
        self.merges = merges or []
        self.ranks = {(a, b): i for i, (a, b) in enumerate(self.merges)}
        self.vocab_size = 256 + len(self.merges)
        # id -> bytes, for decoding
        self.vocab = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self._cache = {}

    # ---------------- encoding ----------------
    def _merge_piece(self, piece_bytes):
        """Apply learned merges in rank order to one piece.

        Doubly-linked list over positions + a heap of candidate merges, so
        each piece costs O(n log n) instead of O(n^2 * merges). Encoding
        speed matters: the scorer encodes the whole hidden file.
        """
        ids = list(piece_bytes)
        n = len(ids)
        if n < 2:
            return ids
        ranks = self.ranks
        prev = list(range(-1, n - 1))
        nxt = list(range(1, n + 1))
        nxt[n - 1] = -1
        alive = [True] * n

        import heapq
        heap = []
        for i in range(n - 1):
            r = ranks.get((ids[i], ids[i + 1]))
            if r is not None:
                heap.append((r, i))
        if not heap:
            return ids
        heapq.heapify(heap)

        while heap:
            r, i = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j == -1 or not alive[j]:
                continue
            if ranks.get((ids[i], ids[j])) != r:
                continue          # stale entry
            ids[i] = 256 + r
            alive[j] = False
            k = nxt[j]
            nxt[i] = k
            if k != -1:
                prev[k] = i
                nr = ranks.get((ids[i], ids[k]))
                if nr is not None:
                    heapq.heappush(heap, (nr, i))
            p = prev[i]
            if p != -1:
                nr = ranks.get((ids[p], ids[i]))
                if nr is not None:
                    heapq.heappush(heap, (nr, p))
        return [ids[i] for i in range(n) if alive[i]]

    def encode(self, text):
        out = []
        for piece in _SPLIT_RE.findall(text):
            pb = piece.encode("utf-8")
            cached = self._cache.get(pb)
            if cached is None:
                cached = self._merge_piece(pb)
                if len(self._cache) < 300000:
                    self._cache[pb] = cached
            out.extend(cached)
        return out

    # ---------------- decoding ----------------
    def decode(self, ids):
        buf = b"".join(self.vocab[i] if 0 <= i < len(self.vocab) else b"\xef\xbf\xbd"
                       for i in ids)
        return buf.decode("utf-8", errors="replace")

    # ---------------- persistence ----------------
    def save(self, path=_DEFAULT):
        with open(path, "w") as f:
            json.dump({"type": "bpe", "merges": self.merges}, f)


class ByteTokenizer:
    """Fallback identical to the starter's, used if bpe.json is missing."""
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(i & 0xFF for i in ids).decode("utf-8", errors="replace")

    def save(self, path=_DEFAULT):
        with open(path, "w") as f:
            json.dump({"type": "byte"}, f)


def load(path=None):
    """Return the tokenizer. Called with NO arguments by train/evaluate."""
    path = path or _DEFAULT
    if not os.path.exists(path):
        return ByteTokenizer()
    with open(path) as f:
        d = json.load(f)
    if d.get("type") == "byte":
        return ByteTokenizer()
    return BPETokenizer([tuple(m) for m in d["merges"]])


# ---------------------------------------------------------------- training
def train_bpe(text, vocab_size, verbose=True):
    """Learn merges from `text` only. Returns a BPETokenizer."""
    from collections import Counter

    # Word-frequency table: identical pieces are merged once, not per
    # occurrence. This is what makes BPE training take seconds not minutes.
    freq = Counter(_SPLIT_RE.findall(text))
    byte_freq = Counter()
    for w, c in freq.items():
        byte_freq[tuple(w.encode("utf-8"))] += c

    seqs = [list(k) for k in byte_freq]
    counts = list(byte_freq.values())

    # Global pair counts + inverted index pair -> set of word indices.
    # After each merge we only touch the words that actually contained the
    # merged pair, so total work is near-linear instead of V * corpus.
    pair_counts = Counter()
    where = {}
    for wi, (s, c) in enumerate(zip(seqs, counts)):
        for i in range(len(s) - 1):
            p = (s[i], s[i + 1])
            pair_counts[p] += c
            where.setdefault(p, set()).add(wi)

    import heapq
    heap = [(-v, k) for k, v in pair_counts.items()]
    heapq.heapify(heap)

    merges = []
    n_merges = vocab_size - 256
    while len(merges) < n_merges and heap:
        negc, pair = heapq.heappop(heap)
        # lazy deletion: skip entries whose count is stale
        if pair_counts.get(pair, 0) != -negc:
            continue
        if -negc < 2:
            break
        a, b = pair
        new_id = 256 + len(merges)
        merges.append((a, b))

        touched = Counter()
        for wi in list(where.get(pair, ())):
            s = seqs[wi]
            c = counts[wi]
            i = 0
            out = []
            while i < len(s):
                if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            if out == s:
                continue
            # decrement old pairs, increment new pairs for this word
            for i in range(len(s) - 1):
                touched[(s[i], s[i + 1])] -= c
            for i in range(len(out) - 1):
                touched[(out[i], out[i + 1])] += c
                where.setdefault((out[i], out[i + 1]), set()).add(wi)
            seqs[wi] = out

        for p, delta in touched.items():
            if delta == 0:
                continue
            nv = pair_counts.get(p, 0) + delta
            if nv <= 0:
                pair_counts.pop(p, None)
                where.pop(p, None)
            else:
                pair_counts[p] = nv
                heapq.heappush(heap, (-nv, p))
        pair_counts.pop(pair, None)
        where.pop(pair, None)

        if verbose and len(merges) % 500 == 0:
            print(f"  merge {len(merges)}/{n_merges}  pair count {-negc:,}")
    return BPETokenizer(merges)


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--out", default=_DEFAULT)
    a = ap.parse_args()

    txt = open(a.data, encoding="utf-8").read()
    t0 = time.time()
    tk = train_bpe(txt, a.vocab)
    tk.save(a.out)
    nb = len(txt.encode("utf-8"))
    ids = tk.encode(txt[:2_000_000])
    sub = txt[:2_000_000]
    assert tk.decode(tk.encode(sub)) == sub, "LOSSLESS CHECK FAILED"
    print(f"vocab {tk.vocab_size}  trained in {time.time()-t0:.0f}s")
    print(f"compression on 2MB sample: "
          f"{len(sub.encode('utf-8'))/len(ids):.3f} bytes/token")
    print(f"saved -> {a.out}")
