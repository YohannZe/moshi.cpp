#!/usr/bin/env bash
# Serving-protocol sweep on the 113-file FLEURS subset. Each arm changes ONE thing against
# the baseline (Q4_K, tail=+8, prefix=0, no AGC, normalized audio). Detached + idempotent:
# arms that already have a hypothesis file are skipped.
set -u
cd "$(dirname "$0")/.."
D=eval-data/fleurs
M=models/Codes4Fun/stt-1b-en_fr-GGUF
export LD_LIBRARY_PATH=$PWD/build/bin:$PWD/../voxtral.cpp/build-host/src

run() { # name, list, weights, env...
  local name=$1 list=$2 weights=$3; shift 3
  [ -s "$D/sweep_$name.tsv" ] && return
  echo "── $name" >> "$D/sweep.log"
  env "$@" ./build/bin/stt_eval "$M" "$D/$list" 6 "$weights" \
    > "$D/sweep_$name.tsv" 2>> "$D/sweep.log"
}

# The serving-parameter arms (one variable each)
run base         subset_norm.lst model-q4_k.gguf
run tail0        subset_norm.lst model-q4_k.gguf STT_TAIL_EXTRA=0
run tail16       subset_norm.lst model-q4_k.gguf STT_TAIL_EXTRA=16
run prefix6      subset_norm.lst model-q4_k.gguf STT_PREFIX=6
run prefix12     subset_norm.lst model-q4_k.gguf STT_PREFIX=12
# The AGC thesis: raw -46 dBFS audio, with and without causal AGC
run raw_noagc    subset_raw.lst  model-q4_k.gguf
run raw_agc      subset_raw.lst  model-q4_k.gguf STT_AGC=1
run norm_agc     subset_norm.lst model-q4_k.gguf STT_AGC=1
# Engine variants (quality side)
run lm_imat      subset_norm.lst model-q4_k-imat.gguf
run encq8        subset_norm.lst model-q4_k.gguf STT_MIMI=mimi-encq8.gguf

echo "SWEEP DONE $(date)" >> "$D/sweep.log"
