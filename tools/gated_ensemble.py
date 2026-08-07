#!/usr/bin/env python3
"""Confidence-gated ensemble: spend the extra 4 passes ONLY on utterances where the primary
pass hesitated. Produces the cost-vs-WER curve that decides whether adaptive test-time
compute is viable.

Gating signal: per-utterance count of low-margin word decisions (n_low from the CONF lines,
margin < 2 logits), tie-broken by min margin. --rank frac ranks by n_low/n_words instead
(n_low alone is confounded with utterance length). In BOTH modes, utterances whose primary
hypothesis is EMPTY are escalated first: they have zero word decisions, hence n_low = 0,
and would otherwise be unreachable by the gate even though total deletion is the worst
failure mode.

The printed cost is ANALYTIC: 1 + n_alts * gated_fraction, counting each pass as equal.
It is not wall-clock, and members do not actually cost the same (an F16 pass is ~1.8x a
Q4_K pass in compute) — weight accordingly before quoting a deployed cost.

Scoring imports score_wer's normalize/wer so this curve and score_wer.py cannot disagree.

usage: gated_ensemble.py <test.tsv> <conf_primary.tsv> <alt1.tsv> <alt2.tsv> ...
                         [--rank nlow|frac]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rover import load, rover           # alignment + vote (deterministic)
from score_wer import normalize, wer    # the one true scorer

def main():
    args = sys.argv[1:]
    rank_mode = "nlow"
    if "--rank" in args:
        i = args.index("--rank")
        rank_mode = args[i + 1]
        del args[i:i + 2]
    test_tsv, conf_tsv = args[0], args[1]
    alt_paths = args[2:]

    refs = {}
    for line in open(test_tsv, encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        if len(c) >= 4:
            refs[c[1]] = normalize(c[3])

    primary, conf = {}, {}
    for line in open(conf_tsv, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if p[0] == "HYP":
            primary[p[1]] = normalize(p[2])
        elif p[0] == "CONF":
            # fields: CONF, file, min_margin, mean_margin, n_words, n_low
            if len(p) <= 5:
                sys.exit(f"CONF line without n_low field: {line.rstrip()}")
            conf[p[1]] = (int(p[5]), float(p[2]), int(p[4]))
    alts = [load(p) for p in alt_paths]

    # hesitation ranking; empty primary hypotheses first in every mode
    def key(f):
        n_low, min_margin, n_words = conf[f]
        empty = 0 if primary.get(f) else 1
        if rank_mode == "frac":
            return (-empty, -(n_low / max(1, n_words)), min_margin)
        return (-empty, -n_low, min_margin)
    ranked = sorted(conf, key=key)
    n_empty = sum(1 for f in conf if not primary.get(f))

    print(f"rank={rank_mode}  empty-primary escalated first: {n_empty}")
    print(f"{'gated %':>8} {'cost x':>7} {'WER %':>7}")
    for pct in (0, 10, 20, 30, 50, 75, 100):
        k = int(len(ranked) * pct / 100)
        gated = set(ranked[:k])
        errs = words = 0
        for f, ref in refs.items():
            if f not in primary:
                continue
            hyp = rover([primary[f]] + [a.get(f, []) for a in alts]) if f in gated \
                  else primary[f]
            S, I, D, N = wer(ref, hyp)
            errs += S + I + D
            words += N if N else 0
        cost = 1 + len(alts) * pct / 100.0
        print(f"{pct:>7}% {cost:>6.2f}x {100.0*errs/words:>6.2f}")
    print("note: cost is analytic (equal-cost passes), not wall-clock")

if __name__ == "__main__":
    main()
