#!/usr/bin/env python3
"""Whisper baseline on our own FLEURS list, scored by our own scorer.

Deliberately not a citation. Cross-paper WER numbers are confounded by text normalization —
the Whisper paper's own appendix moves by more than a point depending on the normalizer, and
its per-language FLEURS table has rotated headers that do not survive PDF extraction. Running
it here means the comparison differs only in the system under test.

Whisper is NOT a like-for-like competitor and the paper must say so: it is non-streaming
(30 s windows, no incremental output), so it answers a different question. It is here as the
accuracy reference point every reader will ask about.

usage: whisper_eval.py <list.txt> <out.tsv> [model] [device]
Resumes: already-transcribed basenames in out.tsv are skipped.
"""
import os
import sys
import time

from faster_whisper import WhisperModel

list_path, out_path = sys.argv[1], sys.argv[2]
model_name = sys.argv[3] if len(sys.argv) > 3 else "small"
device = sys.argv[4] if len(sys.argv) > 4 else "cpu"

done = set()
if os.path.exists(out_path):
    for line in open(out_path, encoding="utf-8"):
        if line.startswith("HYP\t"):
            done.add(line.split("\t", 2)[1])
    print(f"resuming: {len(done)} done", file=sys.stderr)

files = [l.strip() for l in open(list_path) if l.strip()]
files = [f for f in files if os.path.basename(f) not in done]
print(f"{len(files)} to go, model={model_name}", file=sys.stderr)

model = WhisperModel(model_name, device=device, compute_type="int8", cpu_threads=6)
out = open(out_path, "a", encoding="utf-8")
t0, audio_s = time.time(), 0.0

for i, path in enumerate(files):
    segments, info = model.transcribe(path, language="fr", beam_size=5)
    text = " ".join(s.text.strip() for s in segments)
    audio_s += info.duration
    out.write(f"HYP\t{os.path.basename(path)}\t{text}\n")
    out.flush()
    if (i + 1) % 25 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(files)}  rtf={el/audio_s:.3f}", file=sys.stderr)

el = time.time() - t0
print(f"RESULT whisper model={model_name} files={len(files)} audio_s={audio_s:.0f} "
      f"wall_s={el:.0f} rtf={el/max(audio_s,1):.3f}", file=sys.stderr)
