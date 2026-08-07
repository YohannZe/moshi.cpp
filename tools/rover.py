#!/usr/bin/env python3
"""Test-time ROVER ensemble: align N hypothesis files per utterance, majority-vote.

Rationale (the sub-10 push): greedy streaming decoding commits to one token per frame with
no beam. Substitution errors are largely *unstable under benign perturbation* (a different
silence prefix, different weights precision) while correct words are stable — so a
word-level vote across perturbed passes recovers a slice of the substitutions that no
single deterministic pass can. Classic ROVER (Fiscus 1997), applied to deployment-mode
streaming output.

Alignment: iterative pairwise Levenshtein against the current consensus (first file seeds
it), then per-slot majority with ties going to the seed system (our best single config).

usage: rover.py out.tsv in1.tsv in2.tsv [in3.tsv ...]
"""
import re
import sys

def norm_words(s):
    s = s.lower().replace("’", "'")
    s = re.sub(r"([a-zà-ÿ])'", r"\1' ", s)
    return "".join(c if (c.isalnum() or c in "' ") else " " for c in s).split()

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
    for h in hyps[1:]:
        cons = [s[0] if s[0] else (next((w for w in s if w), "")) for s in slots]
        pairs = align(cons, h)
        new_slots = []
        for ci, hj in pairs:
            if ci is not None and hj is not None:
                slots[ci].append(h[hj]); new_slots.append(slots[ci])
            elif ci is not None:
                slots[ci].append(""); new_slots.append(slots[ci])
            else:
                s = [""] * (len(new_slots[-1]) - 1 if new_slots else 1)
                new_slots.append(s + [h[hj]])
        slots = new_slots
    out = []
    for s in slots:
        best, cnt = s[0], 0
        for w in set(s):
            c = s.count(w)
            if c > cnt or (c == cnt and w == s[0]):
                best, cnt = w, c
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
