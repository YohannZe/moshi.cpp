#!/usr/bin/env python3
"""Prepare the missing-octave experiment from Common Voice French.

The question FLEURS cannot answer: does the 8-12 kHz octave — empty in every 16 kHz corpus,
present in what the model was trained on (24 kHz) and in what Katarina captures — carry WER?
CV clips are browser recordings stored at 32-48 kHz, so the octave is present at the source.

Design: paired conditions on the SAME clips,
    A (wideband) : native -> soxr 24 kHz            -> what a 24 kHz product hears
    B (telephone-ish, FLEURS regime) : native -> soxr 16 kHz -> fed as 16 kHz
      (stt_eval's own windowed-sinc then does 16->24, byte-identical to the FLEURS path)
Both peak-normalized to -3 dBFS AFTER resampling. The only difference reaching the model is
content above 8 kHz. Paired bootstrap on the WER delta.

Selection guards, each excluding a real confound:
  - duration 3-15 s (matches FLEURS shape)
  - genuine HF content: mean energy in 9-15 kHz at least -55 dB relative to 1-4 kHz band —
    browsers upsample 16 kHz headsets and the metadata cannot tell; the spectrum can.
  - sentence has >= 3 words (CV has single-word clips)

usage: cv_prep.py <cv_dir> <n_clips> <out_dir>
  expects <cv_dir>/clips/*.mp3 and <cv_dir>/test.tsv
"""
import csv
import os
import subprocess as sp
import sys

import numpy as np

cv_dir, n_want, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
os.makedirs(f"{out_dir}/wide", exist_ok=True)
os.makedirs(f"{out_dir}/narrow", exist_ok=True)

rows = []
with open(f"{cv_dir}/test.tsv", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if len(r["sentence"].split()) >= 3:
            rows.append((r["path"], r["sentence"]))
print(f"{len(rows)} candidate rows", file=sys.stderr)

def decode(path, rate):
    p = sp.run(["ffmpeg", "-v", "quiet", "-i", path, "-ac", "1", "-ar", str(rate),
                "-af", "aresample=resampler=soxr", "-f", "f32le", "-"],
               stdout=sp.PIPE, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32)

def band_db(x, sr, lo, hi):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    m = spec[(freqs >= lo) & (freqs < hi)].mean()
    return 10 * np.log10(m + 1e-20)

def write_wav(path, x, sr):
    import struct
    x = x / (np.abs(x).max() + 1e-9) * (10 ** (-3 / 20))     # peak -3 dBFS
    q = np.clip(x * 32767, -32768, 32767).astype("<i2").tobytes()
    hdr = (b"RIFF" + struct.pack("<I", 36 + len(q)) + b"WAVEfmt " +
           struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) +
           b"data" + struct.pack("<I", len(q)))
    open(path, "wb").write(hdr + q)

kept, seen = [], 0
ref_out = open(f"{out_dir}/cv_ref.tsv", "w", encoding="utf-8")
for path, sentence in rows:
    if len(kept) >= n_want:
        break
    mp3 = f"{cv_dir}/clips/{path}"
    if not os.path.exists(mp3):
        continue
    seen += 1
    try:
        x48 = decode(mp3, 48000)
    except sp.CalledProcessError:
        continue
    dur = len(x48) / 48000
    if not (3.0 <= dur <= 15.0):
        continue
    hf = band_db(x48, 48000, 9000, 15000) - band_db(x48, 48000, 1000, 4000)
    if hf < -55:
        continue                     # upsampled-16k mic in disguise
    base = os.path.splitext(path)[0] + ".wav"
    write_wav(f"{out_dir}/wide/{base}", decode(mp3, 24000), 24000)
    write_wav(f"{out_dir}/narrow/{base}", decode(mp3, 16000), 16000)
    ref_out.write(f"cv\t{base}\tfr\t{sentence}\n")
    kept.append(base)
    if len(kept) % 50 == 0:
        print(f"  kept {len(kept)} / probed {seen}", file=sys.stderr)

for cond in ("wide", "narrow"):
    with open(f"{out_dir}/{cond}.lst", "w") as f:
        for b in kept:
            f.write(os.path.abspath(f"{out_dir}/{cond}/{b}") + "\n")
print(f"kept {len(kept)} of {seen} probed", file=sys.stderr)
