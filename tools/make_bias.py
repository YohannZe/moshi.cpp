#!/usr/bin/env python3
"""Build per-utterance hotword bias lists, standard-protocol.

Entities are mid-sentence capitalized words from the REFERENCE (the oracle list — this is
how the contextual-biasing literature evaluates: the bias list contains the true entities
plus distractors, since in deployment the list comes from user context, not from thin air).
Each list gets K distractor entities sampled from OTHER utterances, so false-trigger damage
is part of the measurement, not hidden.

usage: make_bias.py <ref.tsv> <out.tsv> [K_distractors] [seed]
"""
import random
import re
import sys

ref_tsv, out_path = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 10
rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 1234)

# column 2 (0-indexed) is the CASED transcription — the normalized column that scoring uses
# is lowercased and carries no entity signal.
refs = {}
for line in open(ref_tsv, encoding="utf-8"):
    c = line.rstrip("\n").split("\t")
    if len(c) >= 4:
        refs[c[1]] = c[2]

entities = {}
for f, text in refs.items():
    words = text.split()
    ents = []
    for i, w in enumerate(words):
        wc = w.strip(".,;:!?()«»\"'")
        # mid-sentence capitalized, not an acronym-less single letter, not a number
        if i > 0 and len(wc) > 1 and wc[0].isupper() and not wc.isupper() \
           and not any(ch.isdigit() for ch in wc):
            ents.append(wc.lower())
    entities[f] = sorted(set(ents))

pool = sorted({e for es in entities.values() for e in es})
print(f"{sum(map(len, entities.values()))} entity slots, {len(pool)} unique entities",
      file=sys.stderr)

with open(out_path, "w", encoding="utf-8") as out:
    for f, ents in entities.items():
        distract = [e for e in rng.sample(pool, min(K + len(ents), len(pool)))
                    if e not in ents][:K]
        words = ents + distract
        if words:
            out.write(f"{f}\t{' '.join(words)}\n")
