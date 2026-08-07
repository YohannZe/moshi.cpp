#!/usr/bin/env python3
"""Download and prepare the FLEURS fr_fr test split for stt_eval.

Committed because the first version lived inline in a shell one-liner inside a tmpfs
scratchpad, and a battery-death reboot erased the dataset, the hypothesis files and six
hours of reference-baseline compute along with it. Everything an evaluation needs to be
re-runnable belongs in the repo.

Two things this does beyond downloading:
- FLEURS ships IEEE-float32 WAVs (fmt tag 3); the C tools read 16-bit PCM. Converted here.
- FLEURS audio peaks around -46 dBFS, and a neural-codec STT is input-gain sensitive
  (measured: 17.62 % WER raw vs 11.40 % normalized, 29 vs 3 empty transcripts). Each file
  is peak-normalized to -3 dBFS. This is stated in the eval protocol.

usage: fleurs_prepare.py <out_dir> [split]
split defaults to "test". Use "dev" to prepare a tuning set that is disjoint from the
reported test set — serving knobs, ensemble membership and gate thresholds must be
selected on dev, never on test (the 2026-08 sub-10 sweep tuned on a subset of test,
which is the single most attackable choice in the eval; do not repeat it).
creates <out_dir>/{<split>.tsv, <split>_pcm/*.wav, full.lst (test) / <split>.lst}
"""
import array
import glob
import os
import struct
import subprocess
import sys
import tarfile

BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/fr_fr"

def main():
    out = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
    os.makedirs(out, exist_ok=True)

    tsv = f"{out}/{split}.tsv"
    if not os.path.exists(tsv):
        subprocess.run(["curl", "-sL", "-o", tsv, f"{BASE}/{split}.tsv"], check=True)
    tar = f"{out}/{split}.tar.gz"
    if not os.path.isdir(f"{out}/{split}"):
        if not os.path.exists(tar):
            print("downloading audio (~350 MB)...", file=sys.stderr)
            subprocess.run(["curl", "-sL", "-o", tar, f"{BASE}/audio/{split}.tar.gz"],
                           check=True)
        with tarfile.open(tar) as t:
            t.extractall(out)

    os.makedirs(f"{out}/{split}_pcm", exist_ok=True)
    n = 0
    for p in sorted(glob.glob(f"{out}/{split}/*.wav")):
        dst = f"{out}/{split}_pcm/" + os.path.basename(p)
        if os.path.exists(dst):
            n += 1
            continue
        raw = open(p, "rb").read()
        assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", p
        i, fmt, data, sr, ch = 12, None, None, 16000, 1
        while i + 8 <= len(raw):
            cid = raw[i:i+4]
            sz = struct.unpack("<I", raw[i+4:i+8])[0]
            if cid == b"fmt ":
                fmt, ch, sr = struct.unpack("<HHI", raw[i+8:i+16])
            elif cid == b"data":
                data = raw[i+8:i+8+sz]
            i += 8 + sz + (sz & 1)
        if fmt == 3:
            f = array.array("f"); f.frombytes(data[:len(data)//4*4])
        elif fmt == 1:
            pcm16 = array.array("h"); pcm16.frombytes(data[:len(data)//2*2])
            f = array.array("f", (x / 32768.0 for x in pcm16))
        else:
            print(f"skip {p} (fmt {fmt})", file=sys.stderr)
            continue
        if ch == 2:
            f = array.array("f", ((f[j] + f[j+1]) / 2 for j in range(0, len(f)-1, 2)))
        peak = max(abs(x) for x in f) or 1.0
        g = 0.7 / peak
        pcm = array.array("h", (max(-32768, min(32767, round(x*g*32767))) for x in f))
        d = pcm.tobytes()
        hdr = (b"RIFF" + struct.pack("<I", 36+len(d)) + b"WAVEfmt " +
               struct.pack("<IHHIIHH", 16, 1, 1, sr, sr*2, 2, 16) +
               b"data" + struct.pack("<I", len(d)))
        open(dst, "wb").write(hdr + d)
        n += 1

    lst = f"{out}/full.lst" if split == "test" else f"{out}/{split}.lst"
    with open(lst, "w") as fl:
        for p in sorted(glob.glob(f"{out}/{split}_pcm/*.wav")):
            fl.write(os.path.abspath(p) + "\n")
    print(f"{n} files ready in {out}/{split}_pcm, list in {lst}")

if __name__ == "__main__":
    main()
