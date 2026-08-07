#!/usr/bin/env python3
"""Test-time ROVER ensemble: align N hypothesis files per utterance, majority-vote.

Rationale (the sub-10 push): greedy streaming decoding commits to one token per frame with
no beam. Substitution errors are largely *unstable under benign perturbation* (a different
silence prefix, different weights precision) while correct words are stable — so a
word-level vote across perturbed passes recovers a slice of the substitutions that no
single deterministic pass can. ROVER-style voting (in the spirit of Fiscus 1997, but a
simplification of it: iterative pairwise alignment against a single consensus
representative per slot, frequency-only voting, no confidence scores, no word transition
network), applied to deployment-mode streaming output. The members are perturbations of
one greedy decoder (quantization / tail / prefix), not independent systems — call this a
perturbation ensemble, not a multi-system ROVER, when writing it up.

Alignment: iterative pairwise Levenshtein against the current consensus (first file seeds
it), then per-slot majority. Ties: the seed system's word wins if it is among the
top-voted; otherwise the first tied word in slot order (deterministic — never Python set
iteration order, which is randomized by string hashing).

usage: rover.py out.tsv in1.tsv in2.tsv [in3.tsv ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_wer import normalize as norm_words   # ONE normalizer for the whole harness

def load(p):
    d = {}
    for line in open(p, encoding="utf-8"):
        if line.startswith("HYP\t"):
            _, f, t = line.rstrip("\n").split("\t", 2)
            d[f] = norm_words(t)
    return d

def align(a, b):
    """Return list of (i, j) pairs / gaps aligning a to b (Levenshtein backtrace)."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1,
                           dp[i-1][j-1] + (a[i-1] != b[j-1]))
    out = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (a[i-1] != b[j-1]):
            out.append((i-1, j-1)); i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            out.append((None, j-1)); j -= 1
        else:
            out.append((i-1, None)); i -= 1
    return out[::-1]

def rover(hyps):
    """hyps: list of word lists; hyps[0] seeds the consensus and wins ties."""
    # consensus = list of slots; each slot = list of votes (word or "" for gap)
    slots = [[w] for w in hyps[0]]
    for r, h in enumerate(hyps[1:], start=1):
        cons = [s[0] if s[0] else (next((w for w in s if w), "")) for s in slots]
        pairs = align(cons, h)
        new_slots = []
        for ci, hj in pairs:
            if ci is not None and hj is not None:
                slots[ci].append(h[hj]); new_slots.append(slots[ci])
            elif ci is not None:
                slots[ci].append(""); new_slots.append(slots[ci])
            else:
                # slot absent from the consensus so far: all r systems already
                # processed implicitly voted "gap" here (a head-insertion used to
                # get a single phantom gap vote regardless of round, letting a
                # 2-of-5 word win the slot)
                new_slots.append([""] * r + [h[hj]])
        slots = new_slots
    out = []
    for s in slots:
        counts = {}
        for w in s:
            counts[w] = counts.get(w, 0) + 1
        top = max(counts.values())
        if counts.get(s[0], 0) == top:
            best = s[0]                     # seed wins any tie it is part of
        else:
            best = next(w for w in s if counts[w] == top)   # slot order, deterministic
        if best:
            out.append(best)
    return out

def main():
    out_path, ins = sys.argv[1], sys.argv[2:]
    systems = [load(p) for p in ins]
    keys = set(systems[0])
    with open(out_path, "w", encoding="utf-8") as f:
        for k in sorted(keys):
            hyps = [s.get(k, []) for s in systems]
            f.write("HYP\t%s\t%s\n" % (k, " ".join(rover(hyps))))
    print(f"rover: {len(keys)} utterances from {len(systems)} systems -> {out_path}")

if __name__ == "__main__":
    main()
