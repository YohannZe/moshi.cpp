#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
D=eval-data/fleurs
M=models/Codes4Fun/stt-1b-en_fr-GGUF
export LD_LIBRARY_PATH=$PWD/build/bin:$PWD/../voxtral.cpp/build-host-lf/src
run() { local n=$1 w=$2; shift 2; [ -s "$D/s2_$n.tsv" ] && return
  env "$@" ./build/bin/stt_eval "$M" "$D/subset_24k.lst" 6 "$w" > "$D/s2_$n.tsv" 2>>"$D/sweep.log"; }
run t16        model-q4_k.gguf STT_TAIL_EXTRA=16
run t16p4      model-q4_k.gguf STT_TAIL_EXTRA=16 STT_PREFIX=4
run t16p6      model-q4_k.gguf STT_TAIL_EXTRA=16 STT_PREFIX=6
run t16p8      model-q4_k.gguf STT_TAIL_EXTRA=16 STT_PREFIX=8
run t16p6_f16  model-f16.gguf  STT_TAIL_EXTRA=16 STT_PREFIX=6
# ROVER ensemble inputs: perturbed passes (gain via AGC target is fixed; perturb with prefix
# and a deliberate small gain change through pre-scaled lists is heavier — use prefix+weights
# diversity: q4_k/p6, f16/p6, q4_k/p4 are three decorrelated systems)
echo "SWEEP2 DONE" >> "$D/sweep.log"
