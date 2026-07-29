#!/usr/bin/env python3
"""WER scoring for stt_eval output against FLEURS references.

usage: score_wer.py <fleurs test.tsv> <hyp.tsv> [--dump-worst N]

hyp.tsv is stt_eval's stdout: HYP\t<basename.wav>\t<text>

Normalization (applied identically to both sides, and stated because it decides the
number): lowercase; French elisions split (l'accident -> l' accident, matching FLEURS'
own normalized column); punctuation stripped; digits kept as-is (the model writes
numbers as digits and FLEURS references mostly do too — mismatches there are real
errors, not formatting); whitespace collapsed.

FLEURS tsv columns: id, filename, raw_transcription, transcription (already
lowercased/unpunctuated), words, chars, gender. We score against column 3 but
re-normalize it anyway so both sides pass through exactly the same function.
"""
import re
import sys
import unicodedata

def normalize(s: str) -> list[str]:
    s = s.lower()
    s = s.replace("’", "'").replace("`", "'")
    # split elisions the way FLEURS' normalized column does: l'accident -> l' accident
    s = re.sub(r"([a-zà-ÿ])'", r"\1' ", s)
    # strip everything that is neither letter, digit, apostrophe nor space
    s = "".join(c if (c.isalnum() or c in "' ") else " " for c in s)
    return s.split()

def wer(ref: list[str], hyp: list[str]):
    n, m = len(ref), len(hyp)
    if n == 0:
        return (0, 0, 0, 0) if m == 0 else (0, m, 0, 0)
    prev = list(range(m + 1))
    # track ops via full DP for S/I/D breakdown
    dp = [prev]
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(dp[i-1][j] + 1, cur[j-1] + 1,
                         dp[i-1][j-1] + (ref[i-1] != hyp[j-1]))
        dp.append(cur)
    # backtrack
    i, j = n, m
    S = I = D = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (ref[i-1] != hyp[j-1]):
            if ref[i-1] != hyp[j-1]:
                S += 1
            i, j = i-1, j-1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            I += 1; j -= 1
        else:
            D += 1; i -= 1
    return (S, I, D, n)

def main():
    tsv_path, hyp_path = sys.argv[1], sys.argv[2]
    dump_worst = 0
    if "--dump-worst" in sys.argv:
        dump_worst = int(sys.argv[sys.argv.index("--dump-worst") + 1])

    refs = {}
    for line in open(tsv_path, encoding="utf-8"):
        cols = line.rstrip("\n").split("\t")
        if len(cols) >= 4:
            refs[cols[1]] = cols[3]     # filename -> normalized transcription

    total_S = total_I = total_D = total_N = 0
    scored, missing = 0, 0
    per_file = []
    for line in open(hyp_path, encoding="utf-8"):
        if not line.startswith("HYP\t"):
            continue
        _, fname, text = line.rstrip("\n").split("\t", 2)
        if fname not in refs:
            missing += 1
            continue
        r, h = normalize(refs[fname]), normalize(text)
        S, I, D, N = wer(r, h)
        total_S += S; total_I += I; total_D += D; total_N += N
        scored += 1
        if N:
            per_file.append(((S + I + D) / N, fname, refs[fname], text))

    if not total_N:
        print("nothing scored"); return
    w = (total_S + total_I + total_D) / total_N
    print(f"files scored : {scored}  (missing refs: {missing})")
    print(f"ref words    : {total_N}")
    print(f"S/I/D        : {total_S}/{total_I}/{total_D}")
    print(f"WER          : {100*w:.2f} %")

    if dump_worst:
        per_file.sort(reverse=True)
        print(f"\n-- {dump_worst} worst --")
        for werr, fname, ref, hyp in per_file[:dump_worst]:
            print(f"[{100*werr:.0f}%] {fname}\n  REF {ref}\n  HYP {hyp.strip()}")

if __name__ == "__main__":
    main()
