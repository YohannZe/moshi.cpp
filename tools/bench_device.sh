#!/usr/bin/env bash
# Run stt_bench on the device with a thermal cooldown before every measurement.
#
# This exists because the phone throttles hard: back-to-back runs of the same
# workload drifted from RTF 1.27 to 2.86 as thermal_zone0 climbed to 79-82 C.
# Any A/B measured without cooldown is noise, and the second variant always
# looks worse. Do not "just run it twice quickly".
#
# usage: bench_device.sh <fixture.wav> [weights...]
#   e.g. bench_device.sh test_16k.wav model-q4_k.gguf model-q4_0.gguf
set -uo pipefail

ADB="${ADB:-$HOME/Android/Sdk/platform-tools/adb}"
D="${D:-/data/local/tmp/kyutai_bench}"
MODEL_DIR="${MODEL_DIR:-Codes4Fun/stt-1b-en_fr-GGUF}"
THREADS="${THREADS:-6}"
COOL_MC="${COOL_MC:-50000}"   # millicelsius; wait until below this before measuring
COOL_MAX_S="${COOL_MAX_S:-600}"

FIXTURE="${1:?usage: bench_device.sh <fixture.wav> [weights...]}"
shift
WEIGHTS=("$@")
[ ${#WEIGHTS[@]} -eq 0 ] && WEIGHTS=("")

temp_mc() { "$ADB" shell 'cat /sys/class/thermal/thermal_zone0/temp' | tr -dc '0-9'; }

cooldown() {
  local t waited=0
  t=$(temp_mc)
  while [ -n "$t" ] && [ "$t" -ge "$COOL_MC" ] && [ "$waited" -lt "$COOL_MAX_S" ]; do
    printf '\r  cooling: %d C (target < %d C, %ds elapsed)   ' \
      $((t/1000)) $((COOL_MC/1000)) "$waited" >&2
    sleep 15; waited=$((waited+15)); t=$(temp_mc)
  done
  [ "$waited" -gt 0 ] && printf '\r  cooled to %d C after %ds%*s\n' \
    $((t/1000)) "$waited" 20 '' >&2
  echo "$t"
}

for w in "${WEIGHTS[@]}"; do
  label="${w:-<config.json default>}"
  echo "=== $label  ($FIXTURE, $THREADS threads) ==="
  t0=$(cooldown)
  out=$("$ADB" shell "cd $D && LD_LIBRARY_PATH=. ./stt_bench $MODEL_DIR $FIXTURE $THREADS $w 2>&1")
  t1=$(temp_mc)
  echo "$out" | grep -E "^load:|^  mimi|^  lm|^per-frame|^compute:|^RESULT" \
    || { echo "  FAILED:"; echo "$out" | tail -5; }
  echo "  thermal: ${t0:0:2}C -> ${t1:0:2}C"
  echo
done
