#!/usr/bin/env python3
"""Confidence-gated ensemble: spend the extra 4 passes ONLY on utterances where the primary
pass hesitated. Produces the cost-vs-WER curve that decides whether adaptive test-time
compute is viable.

Gating signal: per-utterance count of low-margin word decisions (n_low from the CONF lines,
margin < 2 logits), tie-broken by min margin. Hypothesis under test: ensemble corrections
concentrate in low-confidence utterances, so a fraction of the 5x cost buys most of the gain.

usage: gated_ensemble.py <test.tsv> <conf_primary.tsv> <alt1.tsv> <alt2.tsv> ...
"""
import subprocess
import sys
import tempfile
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rover import load, rover   # reuse alignment/vote
import re

def norm_words(s):
    s = s.lower().replace("’", "'")
    s = re.sub(r"([a-zà-ÿ])'", r"\1' ", s)
    return "".join(c if (c.isalnum() or c in "' ") else " " for c in s).split()

def wer_counts(ref, hyp):
    n, m = len(ref), len(hyp)
    if n == 0: return (m, 0)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ref[i-1] != hyp[j-1]))
        prev = cur
    return (prev[m], n)

def main():
    test_tsv, conf_tsv = sys.argv[1], sys.argv[2]
    alt_paths = sys.argv[3:]

    refs = {}
    for line in open(test_tsv, encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        if len(c) >= 4: refs[c[1]] = norm_words(c[3])

    primary, conf = {}, {}
    for line in open(conf_tsv, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if p[0] == "HYP": primary[p[1]] = norm_words(p[2])
        elif p[0] == "CONF":
            # fields: file, min_margin, mean_margin, n_words, n_low
            conf[p[1]] = (int(p[5]) if len(p) > 5 else int(p[4]), float(p[2]))
    alts = [load(p) for p in alt_paths]

    # rank utterances by hesitation: primary key n_low desc, secondary min_margin asc
    ranked = sorted(conf, key=lambda f: (-conf[f][0], conf[f][1]))

    print(f"{'gated %':>8} {'cost x':>7} {'WER %':>7}")
    for pct in (0, 10, 20, 30, 50, 75, 100):
        k = int(len(ranked) * pct / 100)
        gated = set(ranked[:k])
        errs = words = 0
        for f, ref in refs.items():
            if f not in primary: continue
            hyp = rover([primary[f]] + [a.get(f, []) for a in alts]) if f in gated \
                  else primary[f]
            e, n = wer_counts(ref, hyp)
            errs += e; words += n
        cost = 1 + len(alts) * pct / 100.0
        print(f"{pct:>7}% {cost:>6.2f}x {100.0*errs/words:>6.2f}")

if __name__ == "__main__":
    main()
