#!/usr/bin/env bash
# Full-676 confirmation of the sub-10 ROVER: sinc-resample everything, run the 5 systems,
# vote. Detached + resumable (per-arm skip). ~3.5 h.
set -u
cd "$(dirname "$0")/.."
D=eval-data/fleurs
M=models/Codes4Fun/stt-1b-en_fr-GGUF
export LD_LIBRARY_PATH=$PWD/build/bin:$PWD/../voxtral.cpp/build-host-lf/src
# 1) full-set sinc resample (julius), idempotent
eval-data/venv/bin/python - <<'PY'
import glob, os, struct, array
import torch, julius, sphn
S = "eval-data/fleurs"
os.makedirs(f"{S}/test_pcm24k", exist_ok=True)
n = 0
for p in sorted(glob.glob(f"{S}/test_pcm/*.wav")):
    dst = f"{S}/test_pcm24k/" + os.path.basename(p)
    if os.path.exists(dst): continue
    a, sr = sphn.read(p)
    t = julius.resample_frac(torch.from_numpy(a), sr, 24000)
    pcm = array.array('h', (max(-32768, min(32767, int(x*32767))) for x in t[0].tolist()))
    d = pcm.tobytes()
    hdr = (b'RIFF'+struct.pack('<I',36+len(d))+b'WAVEfmt '+
           struct.pack('<IHHIIHH',16,1,1,24000,48000,2,16)+b'data'+struct.pack('<I',len(d)))
    open(dst,'wb').write(hdr+d); n += 1
print(n, "resampled")
PY
ls $D/test_pcm24k/*.wav > $D/full24k.lst
run() { local n=$1 w=$2; shift 2; [ -s "$D/f676_$n.tsv" ] && [ "$(grep -c '^HYP' $D/f676_$n.tsv)" -ge 676 ] && return
  rm -f "$D/f676_$n.tsv"
  env "$@" ./build/bin/stt_eval "$M" "$D/full24k.lst" 6 "$w" > "$D/f676_$n.tsv" 2>>"$D/sweep.log"; }
run t16p8     model-q4_k.gguf STT_TAIL_EXTRA=16 STT_PREFIX=8
run t16p6f16  model-f16.gguf  STT_TAIL_EXTRA=16 STT_PREFIX=6
run t8f16     model-f16.gguf
run t8p4      model-q4_k.gguf STT_PREFIX=4
run t16p4     model-q4_k.gguf STT_TAIL_EXTRA=16 STT_PREFIX=4
python3 tools/rover.py $D/f676_rover.tsv $D/f676_t16p8.tsv $D/f676_t16p6f16.tsv \
  $D/f676_t8f16.tsv $D/f676_t8p4.tsv $D/f676_t16p4.tsv >> "$D/sweep.log" 2>&1
echo "FULL676 DONE" >> "$D/sweep.log"
