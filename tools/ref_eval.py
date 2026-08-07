#!/usr/bin/env python3
"""Batch FLEURS through kyutai's REFERENCE PyTorch implementation, for the
"our port vs the reference" table.

Mirrors scripts/stt_from_file_pytorch.py from kyutai-labs/delayed-streams-modeling
(model loading, silence prefix, delay-flush suffix, token filtering), but loads the
model once and iterates a file list. Streaming contexts are opened FRESH per file —
the reference implementation's intended per-utterance semantics, the right baseline
against our deployment-mode numbers (context persisting) and the patched
utterance-independent estimate.

Committed to the repo this time: the first copy lived in a tmpfs scratchpad and died
with a reboot, taking a 6-hour run's harness with it.

setup:  python3 -m venv venv && venv/bin/pip install moshi julius soundfile
usage:  ref_eval.py <list.txt> <out.tsv> [device] [dtype]

Serving knobs, for the apples-to-apples grid against the port (the headline comparison
confounds engine, precision, tail flush and LM context unless these are matched):
  REF_TAIL_EXTRA=N    extra silence frames appended AFTER the default delay flush of
                      ceil(delay*fps) = 7. The port's default flush is int(delay*fps)+8
                      = 14 frames, so REF_TAIL_EXTRA=7 matches the port's default and
                      REF_TAIL_EXTRA=15 matches the port's tail16 serving config.
  REF_PREFIX_FRAMES=N override the silence prefix in frames (default: config prefix_s,
                      0.0 s for this model). The port's prefix6 = REF_PREFIX_FRAMES=6.

Hypotheses append to <out.tsv> with flush after each file, and files already present
in <out.tsv> are skipped — so an interrupted run RESUMES instead of restarting.
"""
import itertools
import math
import os
import sys
import time

import julius
import sphn
import torch
import moshi.models

list_path, out_path = sys.argv[1], sys.argv[2]
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
# float32 on x86 CPU: torch's bf16 CPU path is slower than fp32+AVX512 here, and this
# run is about QUALITY of the reference, not its speed.
dtype = torch.float32 if len(sys.argv) <= 4 else getattr(torch, sys.argv[4])

# REF_THREADS: torch CPU streaming at batch 1 usually saturates around 6-8 threads, but
# with the machine to itself 12 is worth measuring — resume makes the experiment free.
torch.set_num_threads(int(os.environ.get("REF_THREADS", "6")))

done = set()
if os.path.exists(out_path):
    for line in open(out_path):
        if line.startswith("HYP\t"):
            done.add(line.split("\t", 2)[1])
    print(f"resuming: {len(done)} already done", file=sys.stderr)

info = moshi.models.loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")
mimi = info.get_mimi(device=device)
tokenizer = info.get_text_tokenizer()
lm = info.get_moshi(device=device, dtype=dtype)
lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)

prefix_s = info.stt_config.get("audio_silence_prefix_seconds", 1.0)
delay_s = info.stt_config.get("audio_delay_seconds", 5.0)
pad_id = info.raw_config.get("text_padding_token_id", 3)
tail_extra = int(os.environ.get("REF_TAIL_EXTRA", "0"))
prefix_override = os.environ.get("REF_PREFIX_FRAMES")
print(f"prefix={prefix_s}s delay={delay_s}s pad_id={pad_id} dtype={dtype} "
      f"tail_extra={tail_extra} prefix_frames={prefix_override or 'config'}",
      file=sys.stderr)

files = [l.strip() for l in open(list_path) if l.strip()]
files = [f for f in files if os.path.basename(f) not in done]
out = open(out_path, "a")
total_audio, t_start = 0.0, time.time()

for fi, path in enumerate(files):
    audio, sr = sphn.read(path)
    audio = torch.from_numpy(audio).to(device)
    audio = julius.resample_frac(audio, sr, mimi.sample_rate)
    if audio.shape[-1] % mimi.frame_size != 0:
        audio = torch.nn.functional.pad(
            audio, (0, mimi.frame_size - audio.shape[-1] % mimi.frame_size))
    total_audio += audio.shape[-1] / mimi.sample_rate

    n_prefix = int(prefix_override) if prefix_override is not None \
        else math.ceil(prefix_s * mimi.frame_rate)
    n_suffix = math.ceil(delay_s * mimi.frame_rate) + tail_extra
    silence = torch.zeros((1, 1, mimi.frame_size), dtype=torch.float32, device=device)
    chunks = itertools.chain(
        itertools.repeat(silence, n_prefix),
        torch.split(audio[:, None], mimi.frame_size, dim=-1),
        itertools.repeat(silence, n_suffix),
    )

    tokens = []
    with torch.no_grad(), mimi.streaming(1), lm_gen.streaming(1):
        for chunk in chunks:
            audio_tokens = mimi.encode(chunk)
            text_tokens = lm_gen.step(audio_tokens)
            if text_tokens is not None:
                t = text_tokens[0, 0, 0].cpu().item()
                if t > pad_id:
                    tokens.append(t)

    text = tokenizer.decode(tokens)
    out.write(f"HYP\t{os.path.basename(path)}\t{text}\n")
    out.flush()
    if (fi + 1) % 10 == 0:
        el = time.time() - t_start
        print(f"  {fi+1}/{len(files)}  rtf={el/total_audio:.2f}", file=sys.stderr)

el = time.time() - t_start
print(f"RESULT ref files={len(files)} audio_s={total_audio:.0f} "
      f"wall_s={el:.0f} rtf={el/total_audio:.3f}", file=sys.stderr)
