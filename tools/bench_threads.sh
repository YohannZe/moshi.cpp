#!/usr/bin/env bash
# Sustained-throughput sweep over thread count, under the thermal cap.
#
# The existing "6 threads is optimal" figure was measured on a COOL phone. That is the wrong
# regime for this app, which runs for tens of minutes: the platform caps scaling_max_freq from
# 3628 MHz to 1440 MHz once the SoC passes ~65 C, and power scales superlinearly with frequency
# but only linearly with core count. So under a power budget, fewer threads held at a higher
# sustained clock can beat more threads that are throttled — and nothing here had ever tested
# the steady state rather than the cold start.
#
# Protocol: cool to a fixed temperature before each arm so every arm starts equal, then run the
# fixture TWICE back to back with no cooldown between. The first pass warms the SoC; the second
# is the steady state, which is the number that matters for a real session.
#
# usage: ./tools/bench_threads.sh [fixture_on_device.wav] [weights.gguf] [threads...]
set -euo pipefail

ADB="${ADB:-$HOME/Android/Sdk/platform-tools/adb}"
DEV_DIR="${DEV_DIR:-/data/local/tmp/kyutai_bench}"
MODEL_DIR="${MODEL_DIR:-/data/local/tmp/katarina_stt/kyutai}"
FIXTURE="${1:-test_speech_90s.wav}"
WEIGHTS="${2:-model-q4_k.gguf}"
shift 2 2>/dev/null || true
THREADS=("$@")
[ "${#THREADS[@]}" -eq 0 ] && THREADS=(4 6 8)

COOL_TO="${COOL_TO:-42000}"      # millidegrees
COOL_TIMEOUT="${COOL_TIMEOUT:-600}"

# Every adb read is checked. A previous run of this sweep spun for ten minutes against a
# disconnected phone because the cooldown loop treated "no devices/emulators found" as
# "still too hot" — an empty string compared as an integer, silently.
need_device() {
  "$ADB" shell true >/dev/null 2>&1 || { echo "no device" >&2; exit 1; }
}

temp() {
  local t
  t=$("$ADB" shell cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null | tr -d '\r')
  case "$t" in
    ''|*[!0-9]*) return 1 ;;
    *) echo "$t" ;;
  esac
}

cap() {
  "$ADB" shell cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null | tr -d '\r'
}

cool() {
  local waited=0 t
  while :; do
    need_device
    t=$(temp) || { echo "cannot read thermal zone" >&2; exit 1; }
    [ "$t" -lt "$COOL_TO" ] && return 0
    [ "$waited" -ge "$COOL_TIMEOUT" ] && {
      echo "still $((t/1000))C after ${COOL_TIMEOUT}s — giving up on cooling" >&2; return 1; }
    sleep 15; waited=$((waited + 15))
  done
}

need_device
echo "fixture=$FIXTURE weights=$WEIGHTS threads=${THREADS[*]}"
echo

for n in "${THREADS[@]}"; do
  cool || true
  echo "── $n threads (start $(( $(temp) / 1000 ))C)"
  for pass in 1 2; do
    printf '   pass%d  ' "$pass"
    "$ADB" shell "cd $DEV_DIR && LD_LIBRARY_PATH=. ./stt_bench $MODEL_DIR $FIXTURE $n $WEIGHTS 2>&1 \
                  | grep '^RESULT'" | tr -d '\r' \
      | grep -oE "mimi_ms=[0-9]+|lm_ms=[0-9]+|rtf=[0-9.]+|chars=[0-9]+" | tr '\n' ' '
    printf 'temp=%dC cap=%s\n' "$(( $(temp) / 1000 ))" "$(cap)"
  done
done

echo
echo "pass2 is the steady state; pass1 only warms the SoC."
