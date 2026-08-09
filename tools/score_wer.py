#!/usr/bin/env python3
"""WER scoring for stt_eval output against FLEURS references.

usage: score_wer.py <fleurs test.tsv> <hyp.tsv> [--dump-worst N] [--ci [B]]
                    [--compare other_hyp.tsv]

hyp.tsv is stt_eval's stdout: HYP\t<basename.wav>\t<text>

--ci [B]              utterance-level percentile bootstrap (default B=2000, fixed seed
                      1234) around the pooled WER. Utterances are the resampling unit,
                      so word-level clustering is respected.
--compare other.tsv   paired bootstrap of (this hyp − other hyp) on the intersection of
                      scored utterances: 95 % CI of the WER difference and a two-sided
                      bootstrap p-value. This is the test to cite when claiming one
                      config beats another.

Normalization (applied identically to both sides, and stated because it decides the
number): lowercase; French elisions split (l'accident -> l' accident, matching FLEURS'
own normalized column); punctuation stripped; digits kept as-is (the model writes
numbers as digits and FLEURS references mostly do too — mismatches there are real
errors, not formatting); whitespace collapsed.

FLEURS tsv columns: id, filename, raw_transcription, transcription (already
lowercased/unpunctuated), words, chars, gender. We score against column 3 but
re-normalize it anyway so both sides pass through exactly the same function.
"""
import random
import re
import sys

_UNITS = ["zéro","un","deux","trois","quatre","cinq","six","sept","huit","neuf","dix",
          "onze","douze","treize","quatorze","quinze","seize","dix-sept","dix-huit","dix-neuf"]
_TENS = {20:"vingt",30:"trente",40:"quarante",50:"cinquante",60:"soixante",
         70:"soixante-dix",80:"quatre-vingts",90:"quatre-vingt-dix"}

def _fr_num(n):
    """French spelling of 0..9999 — enough to cover every digit token in FLEURS/CV."""
    if n < 20:
        return _UNITS[n]
    if n < 100:
        t, u = (n // 10) * 10, n % 10
        if t in (70, 90):
            t -= 10
            u += 10
        base = _TENS[t].rstrip("s") if (t == 80 and u) else _TENS[t]
        if not u:
            return base
        joiner = " et " if u in (1, 11) and t not in (80, 90) else " "
        return base + joiner + (_UNITS[u] if u < 20 else _fr_num(u))
    if n < 1000:
        c, r = divmod(n, 100)
        head = "cent" if c == 1 else _UNITS[c] + " cents"
        if r:
            head = head.rstrip("s") if c > 1 else head
            return head + " " + _fr_num(r)
        return head
    m, r = divmod(n, 1000)
    head = "mille" if m == 1 else _fr_num(m) + " mille"
    return head + (" " + _fr_num(r) if r else "")

def spell_numbers(words):
    """Replace pure-digit tokens with their French spelling, splitting the result into
    words. Secondary-metric use only: it must be applied to BOTH sides, never one."""
    out = []
    for w in words:
        if w.isdigit() and len(w) <= 4:
            out.extend(_fr_num(int(w)).replace("-", " ").split())
        else:
            out.append(w)
    return out

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

def score_file(refs, hyp_path, spell=False):
    """Score one hyp file. Returns (per_utt, missing) where per_utt maps
    fname -> (S, I, D, N, ref_text, hyp_text)."""
    per_utt, missing = {}, 0
    for line in open(hyp_path, encoding="utf-8"):
        if not line.startswith("HYP\t"):
            continue
        _, fname, text = line.rstrip("\n").split("\t", 2)
        if fname not in refs:
            missing += 1
            continue
        r, h = normalize(refs[fname]), normalize(text)
        if spell:
            r, h = spell_numbers(r), spell_numbers(h)
        S, I, D, N = wer(r, h)
        per_utt[fname] = (S, I, D, N, refs[fname], text)
    return per_utt, missing

def pooled_wer(per_utt, keys=None):
    keys = per_utt if keys is None else keys
    E = N = 0
    for k in keys:
        S, I, D, n = per_utt[k][:4]
        E += S + I + D; N += n
    return E / N if N else 0.0

BOOT_SEED, BOOT_B = 1234, 2000

def bootstrap_ci(per_utt, B=BOOT_B):
    """Percentile bootstrap of the pooled WER, resampling utterances."""
    keys = sorted(per_utt)
    rng = random.Random(BOOT_SEED)
    stats = []
    for _ in range(B):
        sample = [keys[rng.randrange(len(keys))] for _ in keys]
        stats.append(pooled_wer(per_utt, sample))
    stats.sort()
    return stats[int(0.025 * B)], stats[int(0.975 * B)]

def paired_bootstrap(pa, pb, B=BOOT_B):
    """Paired bootstrap of pooled WER(a) - WER(b) over the common utterances.
    Returns (delta, lo, hi, p_two_sided)."""
    keys = sorted(set(pa) & set(pb))
    rng = random.Random(BOOT_SEED)
    delta = pooled_wer(pa, keys) - pooled_wer(pb, keys)
    deltas = []
    for _ in range(B):
        sample = [keys[rng.randrange(len(keys))] for _ in keys]
        deltas.append(pooled_wer(pa, sample) - pooled_wer(pb, sample))
    deltas.sort()
    lo, hi = deltas[int(0.025 * B)], deltas[int(0.975 * B)]
    # two-sided: how often does the resampled difference cross zero
    if delta > 0:
        opposite = sum(1 for d in deltas if d <= 0)
    elif delta < 0:
        opposite = sum(1 for d in deltas if d >= 0)
    else:
        opposite = B // 2
    p = min(1.0, 2 * opposite / B)
    return delta, lo, hi, len(keys), p

def main():
    tsv_path, hyp_path = sys.argv[1], sys.argv[2]
    dump_worst = 0
    if "--dump-worst" in sys.argv:
        dump_worst = int(sys.argv[sys.argv.index("--dump-worst") + 1])
    want_ci = "--ci" in sys.argv
    B = BOOT_B
    if want_ci:
        nxt = sys.argv.index("--ci") + 1
        if nxt < len(sys.argv) and sys.argv[nxt].isdigit():
            B = int(sys.argv[nxt])
    compare_path = None
    if "--compare" in sys.argv:
        compare_path = sys.argv[sys.argv.index("--compare") + 1]
    spell = "--spell-numbers" in sys.argv

    refs = {}
    for line in open(tsv_path, encoding="utf-8"):
        cols = line.rstrip("\n").split("\t")
        if len(cols) >= 4:
            refs[cols[1]] = cols[3]     # filename -> normalized transcription

    per_utt, missing = score_file(refs, hyp_path, spell)
    total_S = sum(v[0] for v in per_utt.values())
    total_I = sum(v[1] for v in per_utt.values())
    total_D = sum(v[2] for v in per_utt.values())
    total_N = sum(v[3] for v in per_utt.values())
    scored = len(per_utt)

    if not total_N:
        print("nothing scored"); return
    w = (total_S + total_I + total_D) / total_N
    print(f"files scored : {scored}  (missing refs: {missing})")
    print(f"ref words    : {total_N}")
    print(f"S/I/D        : {total_S}/{total_I}/{total_D}")
    print(f"WER          : {100*w:.2f} %")

    if want_ci:
        lo, hi = bootstrap_ci(per_utt, B)
        print(f"95% CI       : [{100*lo:.2f}, {100*hi:.2f}] %"
              f"  (utterance bootstrap, B={B}, seed={BOOT_SEED})")

    if compare_path:
        pb, _ = score_file(refs, compare_path, spell)
        delta, lo, hi, n_common, p = paired_bootstrap(per_utt, pb, B)
        print(f"vs {compare_path}  (n={n_common} common utterances)")
        print(f"  other WER  : {100*pooled_wer(pb, sorted(set(per_utt) & set(pb))):.2f} %")
        print(f"  delta      : {100*delta:+.2f} pt  95% CI [{100*lo:+.2f}, {100*hi:+.2f}]"
              f"  p={p:.4f}  (paired bootstrap, B={B}, seed={BOOT_SEED})")

    if dump_worst:
        per_file = [((S + I + D) / N, f, ref, hyp)
                    for f, (S, I, D, N, ref, hyp) in per_utt.items() if N]
        per_file.sort(reverse=True)
        print(f"\n-- {dump_worst} worst --")
        for werr, fname, ref, hyp in per_file[:dump_worst]:
            print(f"[{100*werr:.0f}%] {fname}\n  REF {ref}\n  HYP {hyp.strip()}")

if __name__ == "__main__":
    main()
