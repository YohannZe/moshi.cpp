#!/usr/bin/env bash
# Detached, resumable driver for the three FLEURS runs. Launched with setsid nohup so it
# survives the death of whatever launched it — two runs have already been killed tonight,
# once by a battery-dead host and once by the harness exiting.
#
# Resume at the shell level for stt_eval (which appends HYP lines to stdout): the remaining
# list is full.lst minus basenames already present in the hyp file. ref_eval.py resumes by
# itself.
set -u
cd "$(dirname "$0")/.."   # moshi.cpp
D=eval-data/fleurs
export LD_LIBRARY_PATH=$PWD/build/bin:$PWD/../voxtral.cpp/build-host/src

remaining() { # $1 = hyp file; prints paths from full.lst not yet done
  if [ -s "$D/$1" ]; then
    awk -F'\t' 'NR==FNR && /^HYP/ {done[$2]=1; next}
                {n=split($0,a,"/"); if (!(a[n] in done)) print}' "$D/$1" "$D/full.lst"
  else
    cat "$D/full.lst"
  fi
}

run_gguf() { # $1 = weights, $2 = hyp file, $3 = log
  remaining "$2" > "$D/.rem.$$"
  local n; n=$(wc -l < "$D/.rem.$$")
  if [ "$n" -gt 0 ]; then
    echo "[$1] $n files remaining" >> "$D/$3"
    ./build/bin/stt_eval models/Codes4Fun/stt-1b-en_fr-GGUF "$D/.rem.$$" 6 "$1" \
      >> "$D/$2" 2>> "$D/$3"
  fi
  rm -f "$D/.rem.$$"
}

# Reference in parallel (6 torch threads + 6 ggml threads fits in 16 cores).
eval-data/venv/bin/python tools/ref_eval.py "$D/full.lst" "$D/hyp_ref.tsv" \
  2>> "$D/ref_eval.log" &
REF_PID=$!

run_gguf model-q4_k.gguf hyp_q4k_norm.tsv eval_q4k.log
run_gguf model-f16.gguf  hyp_f16_norm.tsv eval_f16.log

wait $REF_PID
echo "ALL DONE $(date)" >> "$D/run_all.done"
