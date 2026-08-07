#!/usr/bin/env python3
"""Quantify how decorrelated the ensemble members actually are — the evidence behind any
"perturbation-decorrelated" claim. Three views:

1. pairwise disagreement: word-level edit distance between two members' hypotheses,
   normalized by the longer of the two (0 = identical transcripts, 1 = nothing shared);
2. pairwise error correlation: Pearson r between per-utterance error counts vs the
   reference (high r = members fail on the same utterances, votes are redundant);
3. oracle coverage: pooled WER if an oracle picked the best member per utterance —
   the upper bound any per-utterance selection/voting scheme can reach.

usage: ensemble_diversity.py <test.tsv> <sys1.tsv> <sys2.tsv> [...]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rover import load
from score_wer import normalize, wer

def edit(a, b):
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (a[i-1] != b[j-1]))
        prev = cur
    return prev[m]

def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x) ** 0.5
    vy = sum((b - my) ** 2 for b in y) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")

def main():
    test_tsv, paths = sys.argv[1], sys.argv[2:]
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    systems = [load(p) for p in paths]

    refs = {}
    for line in open(test_tsv, encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        if len(c) >= 4:
            refs[c[1]] = normalize(c[3])
    keys = sorted(set(refs) & set.intersection(*(set(s) for s in systems)))

    # per-member, per-utterance error counts
    errs = []   # errs[i][k] = S+I+D of member i on utterance k
    for s in systems:
        e = {}
        for k in keys:
            S, I, D, N = wer(refs[k], s[k])
            e[k] = S + I + D
        errs.append(e)

    print(f"{len(keys)} utterances, {len(systems)} members: {' '.join(names)}\n")
    print("pairwise hypothesis disagreement (norm. edit distance):")
    for i in range(len(systems)):
        row = []
        for j in range(len(systems)):
            if i == j:
                row.append("  --  ")
            else:
                d = sum(edit(systems[i][k], systems[j][k]) for k in keys)
                L = sum(max(len(systems[i][k]), len(systems[j][k]), 1) for k in keys)
                row.append(f"{d / L:6.3f}")
        print(f"  {names[i]:>14} {' '.join(row)}")

    print("\npairwise error-count correlation (Pearson r over utterances):")
    for i in range(len(systems)):
        row = []
        for j in range(len(systems)):
            if i == j:
                row.append("  --  ")
            else:
                r = pearson([errs[i][k] for k in keys], [errs[j][k] for k in keys])
                row.append(f"{r:6.3f}")
        print(f"  {names[i]:>14} {' '.join(row)}")

    N = sum(len(refs[k]) for k in keys)
    for i, nm in enumerate(names):
        print(f"\n{nm}: WER {100 * sum(errs[i].values()) / N:.2f} %", end="")
    oracle = sum(min(e[k] for e in errs) for k in keys)
    print(f"\noracle (best member per utterance): {100 * oracle / N:.2f} %")

if __name__ == "__main__":
    main()
