#!/usr/bin/env bash
cd "$(dirname "$0")/.."
D=eval-data/fleurs
export LD_LIBRARY_PATH=$PWD/build/bin:$PWD/../voxtral.cpp/build-host/src
[ -s $D/sweep_tail24.tsv ] || STT_TAIL_EXTRA=24 ./build/bin/stt_eval models/Codes4Fun/stt-1b-en_fr-GGUF $D/subset_norm.lst 6 model-q4_k.gguf > $D/sweep_tail24.tsv 2>>$D/sweep.log
[ -s $D/hyp_combo676.tsv ] || STT_TAIL_EXTRA=16 STT_PREFIX=6 STT_AGC=1 ./build/bin/stt_eval models/Codes4Fun/stt-1b-en_fr-GGUF $D/full.lst 6 model-q4_k.gguf > $D/hyp_combo676.tsv 2>>$D/sweep.log
echo "COMBO676 DONE" >> $D/sweep.log
