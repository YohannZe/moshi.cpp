#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
D=eval-data/fleurs
M=models/Codes4Fun/stt-1b-en_fr-GGUF
export LD_LIBRARY_PATH=$PWD/build/bin:$PWD/../voxtral.cpp/build-host-lf/src
until grep -q "SWEEP2 DONE" "$D/sweep.log"; do sleep 20; done
run() { local n=$1 w=$2; shift 2; [ -s "$D/s3_$n.tsv" ] && return
  env "$@" ./build/bin/stt_eval "$M" "$D/subset_24k.lst" 6 "$w" > "$D/s3_$n.tsv" 2>>"$D/sweep.log"; }
# prefix on the RIGHT tail (8, the sinc-base optimum)
run t8p4   model-q4_k.gguf STT_PREFIX=4
run t8p6   model-q4_k.gguf STT_PREFIX=6
# F16 on the best base
run t8_f16 model-f16.gguf
# bigram fusion at two lambdas on the best base
run t8_bias03 model-q4_k.gguf MOSHI_TEXT_BIAS=eval-data/fleurs/bigram_fr.txt MOSHI_TEXT_BIAS_LAMBDA=0.3
run t8_bias05 model-q4_k.gguf MOSHI_TEXT_BIAS=eval-data/fleurs/bigram_fr.txt MOSHI_TEXT_BIAS_LAMBDA=0.5
echo "SWEEP3 DONE" >> "$D/sweep.log"
