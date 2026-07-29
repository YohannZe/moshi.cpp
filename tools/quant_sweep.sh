#!/usr/bin/env bash
# Score quantizations of the LM against the F16 transcript of the SAME model.
#
# Why F16 and not a human reference: the question here is "how much does quantization cost",
# not "how good is kyutai stt-1b". Scoring against F16 isolates exactly the damage introduced
# by the quantizer; a human reference would fold in the model's own errors and hide it.
#
# The model is bandwidth-bound on its own weights at batch 1, so bits-per-weight is the main
# lever left on speed — hence sweeping it rather than guessing.
#
# usage: ./tools/quant_sweep.sh [fixture.wav ...]
set -euo pipefail

cd "$(dirname "$0")/.."
HERE="$PWD"
MODEL_DIR="$HERE/models/Codes4Fun/stt-1b-en_fr-GGUF"
BENCH="$HERE/build/bin/stt_bench"
THREADS="${THREADS:-6}"
OUT="${OUT:-$HERE/quant_sweep_out}"

export LD_LIBRARY_PATH="$HERE/build/bin:$HERE/../voxtral.cpp/build-host/src:${LD_LIBRARY_PATH:-}"

QUANTS="${QUANTS:-model-f16.gguf model-q8_0.gguf model-q4_k.gguf model-iq4_xs.gguf model-q3_k.gguf}"
FIXTURES=("$@")
if [ "${#FIXTURES[@]}" -eq 0 ]; then
  FIXTURES=("$HERE/../voxtral.cpp/tests/fixtures/test_16k.wav"
            "$HERE/../voxtral.cpp/tests/fixtures/test_speech_90s.wav")
fi

[ -x "$BENCH" ] || { echo "build first: ./build-host.sh" >&2; exit 1; }
mkdir -p "$OUT"

for fx in "${FIXTURES[@]}"; do
  fxname="$(basename "$fx" .wav)"
  echo
  echo "════ $fxname"
  for q in $QUANTS; do
    [ -f "$MODEL_DIR/$q" ] || { echo "  skip $q (absent)"; continue; }
    log="$OUT/$fxname.$q.log"
    "$BENCH" "$MODEL_DIR" "$fx" "$THREADS" "$q" > "$log" 2>&1 || {
      echo "  FAIL $q — see $log"; continue; }
    # The bench prints "full text: ..." once, and a machine-readable RESULT line.
    sed -n 's/^full text: *//p' "$log" > "$OUT/$fxname.$q.txt"
    grep '^RESULT' "$log" > "$OUT/$fxname.$q.result" || true
  done

  python3 - "$OUT" "$fxname" $QUANTS <<'PY'
import re, sys, os
out, fxname, quants = sys.argv[1], sys.argv[2], sys.argv[3:]

def words(p):
    if not os.path.exists(p): return None
    return re.findall(r"[\w'’]+", open(p, encoding='utf-8').read().lower())

def wer(ref, hyp):
    # Plain Levenshtein over words. Substitutions+insertions+deletions / len(ref).
    n, m = len(ref), len(hyp)
    if n == 0: return 0.0 if m == 0 else 1.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j-1] + 1,
                         prev[j-1] + (ref[i-1] != hyp[j-1]))
        prev = cur
    return prev[m] / n

ref = words(f"{out}/{fxname}.model-f16.gguf.txt")
print(f"\n{'weights':<20} {'MB':>6} {'compute ms':>11} {'rtf':>7} {'chars':>7} {'WER vs f16':>11}")
print("-" * 68)
for q in quants:
    rp = f"{out}/{fxname}.{q}.result"
    if not os.path.exists(rp): continue
    r = open(rp).read()
    def g(k, cast=float):
        m = re.search(rf"{k}=([0-9.]+)", r)
        return cast(m.group(1)) if m else 0
    hyp = words(f"{out}/{fxname}.{q}.txt")
    w = "—" if ref is None or hyp is None else f"{100*wer(ref, hyp):.2f} %"
    mb = os.path.getsize(f"{os.path.dirname(out)}/models/Codes4Fun/stt-1b-en_fr-GGUF/{q}") / 2**20 \
         if os.path.exists(f"{os.path.dirname(out)}/models/Codes4Fun/stt-1b-en_fr-GGUF/{q}") else 0
    print(f"{q.replace('model-','').replace('.gguf',''):<20} {mb:6.0f} "
          f"{g('compute_ms'):11.0f} {g('rtf'):7.3f} {g('chars', int):7d} {w:>11}")
PY
done

echo
echo "transcripts and RESULT lines in $OUT"
