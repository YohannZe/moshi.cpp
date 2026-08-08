# What is left before the paper — actionable plan (2026-08-08)

Companion to BENCH.md. BENCH.md records *what was measured*; this file records *what still
has to be done*, in execution order, with the command and the claim each task buys.

Status legend: **P0** = a reviewer rejects without it · **P1** = turns a defensible paper
into a strong one · **P2** = blocked on the phone · **P3** = writing/artifact.

---

## 0. Where we actually stand

Full FLEURS-fr test set (676 utterances, 17 997 ref words), deployment mode, same scorer
(`tools/score_wer.py`, symmetric normalization), paired bootstrap B=2000 seed 1234:

| system | WER | 95 % CI | note |
|---|---|---|---|
| kyutai reference (PyTorch F32, fresh state, tail 7) | 12.62 % | — | host RTF 2.95 |
| this port, F16, linear resample, tail 14 | 11.29 % | [10.38, 12.26] | vs ref: −1.33 pt, **p=0.005** |
| this port, Q4_K, same | 11.61 % | [10.69, 12.57] | vs F16: +0.32, p=0.10 (n.s.) |
| **this port, F16, sinc + tail16/prefix6** | **10.66 %** | [9.80, 11.59] | vs linear baseline: −0.62 pt, **p=0.012** |
| + 5-way ROVER (5× compute) | 10.47 % | — | vs best single: −0.19 pt, p=0.24 (**n.s.**) |

Device (Poco F8 Ultra, SM8850): 11.58 % on the 113-file subset, RTF 0.68–0.98 saturated,
0.75–0.78 in real app sessions, 1087 MB peak RSS.

What the 2026-08-07/08 audit + full-676 run changed:
- **sub-10 is withdrawn.** 9.67 % was measured on the 113 files that had also selected the
  configs. Selection bias ≈ 0.8–1.3 pt, and it picked the wrong winner (subset said
  Q4_K/t16p8, full set says F16/t16p6).
- **Test-time ensembling is measured flat** (p=0.24 at 5× compute). The gated-ensemble
  1.4× curve inherits the same subset bias and is withdrawn pending dev-set measurement.
- **The front end is where the wins are**: gain (+11 pt recovered by AGC at −46 dBFS) and
  resampler imaging (−0.62 pt, significant). Nothing else this campaign moved WER
  significantly.

---

## 1. Re-scope the paper first — everything below depends on it

The original pitch ("we reach sub-10 / we beat the reference") no longer survives its own
statistics. The result that *does* survive is more interesting and harder to attack:

> **Thesis: for streaming neural-codec ASR, WER is dominated by the acoustic front end and
> the serving protocol, not by the decoder or by test-time compute.** On FLEURS-fr with
> kyutai stt-1b: input level is worth up to 11 WER points, resampler imaging 0.6 pt,
> flush/prefix protocol ~0.5 pt — while a 5× decoding ensemble buys 0.19 pt (not
> significant) and explicit LM fusion hurts. This class of model (raw-waveform encoder +
> RVQ) is *structurally* more input-sensitive than log-mel systems, and nobody documents it.

Secondary contributions, all defensible as stated:
1. First public FLEURS-fr numbers for kyutai stt-1b (they publish none; DSM paper is
   English-only ASR).
2. A ggml/C++ port that matches-or-exceeds the reference implementation's accuracy at
   **8.6× its speed on the same CPU**, and runs in real time on a phone (RTF 0.32–0.78).
   → conditional on task **E1** below, which decomposes the gap.
3. Rigorous negative results: ROVER ensembling flat, n-gram shallow fusion harmful,
   imatrix harmful on the LM, subset overfitting quantified with the wrong-winner example.
4. An engineering post-mortem worth reading: a dtype that silently disarmed flash
   attention (LM ×1.85), a 40 ms capture ring buffer that produced zero text with healthy
   levels.

**Task W0 (P0):** write the thesis + contributions list, one page, before running anything
else. Every experiment below is then either "supports a stated claim" or "cut it".

---

## 2. P0 — blocking

### T1. Rebuild `stt_eval` and verify the in-code sinc == julius
The sinc resampler now lives in `tools/stt_eval.cpp` (default; `STT_RESAMPLE=linear`
reverts), but `build/bin/stt_eval` is the pre-change binary and every sinc number so far
came from julius-prepped 24 kHz WAVs. Until these agree, the harness is not self-contained.

```bash
cd ~/kata/moshi.cpp && ./build-host.sh          # or: cmake --build build -j$(nproc)
D=eval-data/fleurs
# same 113 files, once from pre-resampled 24 kHz (julius) and once from 16 kHz in-code
env STT_TAIL_EXTRA=16 STT_PREFIX=6 ./build/bin/stt_eval models/Codes4Fun/stt-1b-en_fr-GGUF \
    $D/subset_24k.lst 6 model-f16.gguf > $D/chk_julius.tsv
env STT_TAIL_EXTRA=16 STT_PREFIX=6 ./build/bin/stt_eval models/Codes4Fun/stt-1b-en_fr-GGUF \
    $D/device_subset.lst 6 model-f16.gguf > $D/chk_incode.tsv
python3 tools/score_wer.py $D/test.tsv $D/chk_incode.tsv --ci --compare $D/chk_julius.tsv
```
**Done when:** the two WERs differ by less than the run-to-run floor (they should be within
~0.05 pt; identical transcripts would be better still). If they diverge, the kernel
parameters differ from julius' and the −0.62 pt result needs re-attribution.
**Cost:** ~2 × 12 min. **Buys:** the front-end result becomes reproducible from this repo
alone, with no PyTorch in the loop.

### T2. Apples-to-apples grid against the reference implementation
The −1.33 pt headline currently confounds four variables: engine, precision, tail flush
(7 vs 14), and warm vs cold LM context. `tools/ref_eval.py` now takes the knobs.

```bash
D=eval-data/fleurs
# arm A: reference with OUR flush (matches the port's default tail=14)
REF_TAIL_EXTRA=7 eval-data/venv/bin/python tools/ref_eval.py $D/full.lst $D/hyp_ref_tail14.tsv
# arm B: reference with our full serving protocol
REF_TAIL_EXTRA=9 REF_PREFIX_FRAMES=6 eval-data/venv/bin/python tools/ref_eval.py \
    $D/full.lst $D/hyp_ref_tail16p6.tsv
```
Both resume automatically (skip files already in the output).

**Arm C — the port with fresh state per utterance** needs a harness: `moshi_lm_start` is
not re-entrant and leaks ~100 MB per call, so run process-per-file rather than fixing it:
```bash
# tools/eval-scripts/fresh_state_676.sh  (to write, ~15 lines)
while read w; do echo "$w" > /tmp/one.lst
  env STT_TAIL_EXTRA=16 STT_PREFIX=6 ./build/bin/stt_eval $M /tmp/one.lst 6 model-f16.gguf
done < $D/full24k.lst > $D/f676_fresh.tsv
```
**Done when:** a 2×2 table exists — {reference, port} × {tail 7, tail 14} plus the
port's {persistent, fresh} context — so the gap is decomposed into "protocol" and "engine"
shares with a p-value on each.
**Cost:** ref ≈ 5.8 h per arm (overnight, resumable); fresh-state port ≈ 1.5 h
(676 model loads at ~4 s + 40 min inference). **Buys:** the difference between "our system
beats theirs" (weak, confounded) and "here is exactly why, and X pt of it is the engine"
(the paper's best ablation).

### T3. Re-tune on dev, confirm once on test
Every serving knob, ensemble member and gate threshold was selected on a subset of test.
The dev split is downloaded: `eval-data/fleurs/dev.lst`, **289 files / 2 863 s**, manifest
`dev.tsv` (score against it, not test.tsv).

```bash
D=eval-data/fleurs; M=models/Codes4Fun/stt-1b-en_fr-GGUF
for arm in "t8:STT_TAIL_EXTRA=8" "t16:STT_TAIL_EXTRA=16" "t24:STT_TAIL_EXTRA=24" \
           "t16p4:STT_TAIL_EXTRA=16 STT_PREFIX=4" "t16p6:STT_TAIL_EXTRA=16 STT_PREFIX=6" \
           "t16p8:STT_TAIL_EXTRA=16 STT_PREFIX=8"; do
  n=${arm%%:*}; env ${arm#*:} ./build/bin/stt_eval $M $D/dev.lst 6 model-f16.gguf > $D/dev_$n.tsv
  python3 tools/score_wer.py $D/dev.tsv $D/dev_$n.tsv --ci
done
```
Then run **one** confirmation of the dev-selected config on test, and report that single
number as the tuned result.
**Done when:** the paper's tuned config is justified by dev WER and confirmed once on test.
**Cost:** ~16 min per arm on dev (6 arms ≈ 1.6 h) + one test run (40 min).
**Buys:** removes the audit's single most attackable finding. Also lets you *keep* the
subset-overfitting story as a deliberate, quantified methodological result rather than an
accident (see W2).

### T4. Same-scorer external baselines
The Whisper context numbers (zero-shot FLEURS-fr: small 15.0 / medium 8.7 / large 7.7 /
large-v2 8.3, Table 13) use Whisper's own text normalizer — not comparable to ours. Run
whisper-small and whisper-medium over the same 676 files and score with `score_wer.py`.
```bash
eval-data/venv/bin/pip install faster-whisper
# tools/whisper_eval.py (to write): iterate full.lst, emit HYP\t<basename>\t<text>
```
**Done when:** one table, one scorer, our system + 2 Whisper sizes + the reference.
**Cost:** ~1–2 h CPU for small, ~3–5 h for medium (or skip medium if time is short).
**Buys:** kills the "your WER is not comparable to anything" objection, and positions the
model honestly between small and medium *measured*, not asserted.

### T5. Propagate the withdrawal
`CLAUDE.md`, the memory index, and two kata commit messages still headline "sub-10 /
9.67 % / sub-10 at 1.4x". BENCH.md is corrected; nothing else is.
**Done when:** no document in either repo states sub-10 without the "subset only,
withdrawn at full scale" qualifier. **Cost:** 15 min.

---

## 3. P1 — the experiments that make the thesis

### E1. Front-end sensitivity study (the paper's core figure)
Two anecdotes (gain, imaging) become one systematic study. All on **dev**, all cheap.

**(a) Input level × AGC.** Prepare dev at target peaks −46, −40, −30, −20, −10, −3 dBFS
(extend `fleurs_prepare.py` with a `--peak-dbfs` flag, or a small scaling script), then run
each level with `STT_AGC=0` and `STT_AGC=1`.
→ Figure: WER vs input level, two curves. Expected shape: a cliff below ~−30 dBFS with AGC
off, flat with AGC on. Add the empty-transcript count as a second axis — that is the
mechanism (RVQ stages starving), not just a WER bump.
**Cost:** 12 arms × 16 min ≈ 3.2 h.

**(b) Resampler quality.** `STT_RS_ZEROS` and `STT_RS_ROLLOFF` are now sweepable (that was
the right instinct — the imaging cut is a free quality knob):
```bash
for z in 4 8 16 24 64; do env STT_RS_ZEROS=$z STT_TAIL_EXTRA=16 STT_PREFIX=6 \
  ./build/bin/stt_eval $M $D/dev.lst 6 model-f16.gguf > $D/dev_z$z.tsv; done
for r in 0.85 0.90 0.945 0.99; do ... STT_RS_ROLLOFF=$r ... ; done
env STT_RESAMPLE=linear ... > $D/dev_linear.tsv
```
→ Figure: WER vs kernel length / rolloff, with the linear point marked. The interesting
claim is *where it saturates*: if 8 taps already recovers most of the 0.62 pt, the finding
is "any anti-imaging filter suffices", which is a stronger and more useful statement than
"use julius".
**Cost:** 10 arms ≈ 2.7 h.

**(c) Optional, high value: band-limiting / additive noise.** FLEURS is clean read speech.
One noise axis (SNR 20/10/5 dB, additive babble or white) or a telephone-band lowpass would
show whether the front-end sensitivity is a gain artifact or a general fragility of the
codec's input representation. This is the difference between "we found a bug in our data
loading" and "here is a property of neural-codec ASR". Cost ≈ 3 h on dev.

**Why this matters most:** log-mel systems are largely immune to input gain; a raw-waveform
RVQ encoder is not. That comparison — same audio, same levels, Whisper vs stt-1b — is
directly runnable once T4 exists, and it is the single most publishable figure in the
project. Add it as **E1(d)**: Whisper-small at the same 6 levels, no AGC. If Whisper is
flat where stt-1b cliffs, the thesis is proven, not argued.

### E2. Error-position analysis (the "why" of the reference gap)
BENCH.md asserts "the reference truncates 112/674 utterances (219 final words); we truncate
89 (160)" with no script and an unexplained 674 denominator.
**Do:** write `tools/error_analysis.py` — align hyp/ref, bucket S/I/D by relative position
in the utterance (deciles), and report deletions in the final decile per system.
**Done when:** a committed script reproduces the truncation counts and produces a
position-histogram figure. **Cost:** 1 h of coding, seconds to run.
**Buys:** turns the flush explanation from a plausible story into a measurement, and
supports "serving protocol dominates" with a mechanism.

### E3. Ensemble: report it as a negative, but make the negative solid
The −0.19 pt / p=0.24 result is a *contribution* if it is presented well.
- Re-run the 5 members with **`STT_AGC=1`** (they ran AGC off; the app ships AGC on) so the
  ensemble is evaluated on the deployed input conditioning.
- Report the diversity table from `tools/ensemble_diversity.py` (error correlation
  0.81–0.92, oracle 8.09 %) as the explanation: the oracle says 2.4 pt is theoretically
  available per-utterance, and voting captures 0.19 of it — the members fail together.
- Re-derive the gated curve on dev, or drop the 1.4× claim entirely.
**Cost:** 5 arms × 40 min on test, or 5 × 16 min on dev.
**Buys:** "we tried the obvious test-time-compute lever, measured it properly, and it does
not work here — here is why" is a better section than a marginal positive.

### E4. Statistical hygiene pass over every remaining table
`score_wer.py --ci --compare` exists; use it for every A/B that appears in the paper,
including the quantization claims (Q4_K vs F16 is **n.s.**, so "quantization costs 0.11 pt"
must become "quantization is free within measurement noise"). **Cost:** an afternoon of
re-scoring committed artifacts.

---

## 4. P2 — device and deployment (blocked: phone not connected)

`adb devices` is empty. Nothing here can proceed until it is plugged in.

| id | task | why it matters |
|---|---|---|
| D1 | `./build.sh --install` and A/B the **JNI windowed-sinc** resampler on device | shipped but never device-validated; only affects the 16 kHz fallback path, but it is new code in the hot path |
| D2 | Verify the **AGC toggle** now actually toggles (log line `agc on/off`, and `agc=jni:true/false` at session start) | the double-AGC fix is unverified on hardware |
| D3 | Measure **spec=2 on device** (interleaved, cooled, ≥3 rounds) | BENCH.md's "−8 % device" is a host extrapolation; either measure it or soften the claim in the paper |
| D4 | **Battery**: one real session, hours-to-empty at the deployed config | "nobody ships an RTF, they ship hours" — the last item on the original paper checklist, and the only deployment number a reader actually feels |
| D5 | Re-measure **context 750 vs 375** post-flash + llamafile | 375 ships on a stale −24 % whose mechanism the flash fix removed, while its quality cost (10.32 → 10.91 % host) is still being paid. This may be a free 0.6 pt. |
| D6 | Device WER at the **final tuned config** on the 113-file subset (or the full 676 if patient) | the paper's device row should match the paper's config |
| D7 | Re-test the pre-flash verdicts that still gate the shipped build: OpenMP, +fp16 kernels, thread count, repack, GPU offload | all were closed on a machine that the flash fix made obsolete; only 3 of ~9 were revisited |

---

## 5. P3 — writing and artifact

| id | task |
|---|---|
| W0 | Thesis + contributions page (see §1) — **do first** |
| W1 | Method section: the exact serving protocol (frame rate, 0.5 s delay, tail/prefix, codec reset between utterances, LM context persistence), stated as a protocol so others can match it. This is the paper's most reusable content. |
| W2 | A short methodology section on **subset overfitting**, using our own failure: 113 files selected configs, cost 0.8–1.3 pt at full scale, and picked the wrong winner. Self-reported failures of this kind are credited by reviewers and it costs nothing to include. |
| W3 | Reproducibility appendix: `eval-artifacts/` maps file → claim → caveat; `tools/eval-scripts/` holds the drivers; determinism statement (two F16 runs 24 h apart, identical S/I/D) **with its caveat** (holds only at fixed session structure; ±0.2 pt from segmentation). |
| W4 | Related work: kyutai DSM paper, Whisper (Table 13 numbers, verified), ROVER (Fiscus 1997 — and say explicitly that ours is a simplification: iterative pairwise alignment, frequency vote, no confidence, no WTN), NGPU-LM fusion (arXiv:2505.22857, our negative result). |
| W5 | Licensing/data statement: FLEURS is CC-BY-4.0; check and state the kyutai stt-1b model license and this fork's license. |
| W6 | Threats to validity: single language, single dataset (read speech, clean), single device, deployment-mode session variance ±0.2 pt, cross-scorer comparability. |

---

## 6. Claims to withdraw or restate — do not let these into the draft

| claim as written | corrected status |
|---|---|
| "sub-10 reached (9.67 % / 9.94 %)" | **withdrawn** — subset only; 10.47 % / 10.66 % at full scale |
| "confidence-gated ensemble: sub-10 at 1.4×" | **withdrawn** pending dev-set re-derivation; the underlying ensemble gain is n.s. |
| "quantization costs 0.11 WER points" | +0.32 pt, p=0.10 → "free within noise" (and the artifact on disk is the 11.61 % resumed chain; the 11.40 % hyp file was lost) |
| "the port beats the reference by 1.33 pt" | true and significant (p=0.005), but a **system** claim until T2 decomposes it |
| "speculation −8 % device compute" | host-only measurement; device unmeasured |
| "device RTF ≈ 0.45 for the 1.4× gated ensemble" | arithmetic, never measured |
| "5 decorrelated systems" | perturbation ensemble of one decoder; error correlation 0.81–0.92 |
| "linear resampling folds aliasing into the band" | imaging, not aliasing (16→24 kHz is upsampling) — already fixed in BENCH.md |
| "ROVER (Fiscus 1997)" | a simplification of it; say so |

---

## 7. Known-open discrepancies — fix or disclose

- **Padding rate**: 55.5 % (measured) vs "roughly 65 %" (later section) for the same
  fixture family. Re-measure with `STT_DUMP_TOKENS` before either number is cited.
- **Q4_K vs Q4_0**: claimed at 3 %, retracted as inside the ±7 % noise floor, then
  re-asserted at 17 % under tinyBLAS with "same order as before" — the last sentence
  contradicts the retraction. The tinyBLAS margin is real; the pre-tinyBLAS ordering
  stays retracted.
- **The 11.40 % unbroken-chain hypothesis file is lost**; the committed artifact scores
  11.61 %. Either re-run it or quote 11.61 with the ±0.2 pt segmentation caveat.
- **AGC envelope persists across files** in the eval, so AGC-arm WER depends on list order.
  `STT_AGC_RESET=1` gives the order-independent variant — measure the delta once and state it.
- **The 674 denominator** in the end-truncation analysis is unexplained (676 files exist).
  E2 replaces this whole paragraph with a script.

---

## 8. Suggested order

**Session 1 (host, ~1 day, no device needed)**
1. W0 — thesis page (1 h).
2. T1 — rebuild + sinc equivalence (30 min).
3. Launch T2 arm A overnight (reference, tail 14) — it is the long pole at ~6 h.
4. While it runs: T3 dev sweep (1.6 h), then E2 error-analysis script (1 h).

**Session 2**
5. T2 arm C (fresh-state port, 1.5 h) + assemble the decomposition table.
6. E1(a) and E1(b) front-end sweeps on dev (~6 h wall, unattended).
7. T4 Whisper-small baseline (1–2 h), then **E1(d)** — Whisper vs stt-1b across levels.
   This is the money figure; if it works, restructure the paper around it.

**Session 3 (needs the phone)**
8. D1–D4 in one sitting: install, verify AGC toggle, spec=2 A/B, then leave the battery
   session running while writing.
9. D5 (context 750 vs 375) — potentially a free 0.6 pt, decide before freezing the config.

**Session 4 — writing**
10. W1–W6, with E4 (statistics pass) applied to every table as it is written.

---

## 9. One-line reference for the harness

```bash
# score anything, with error bars and a paired test
python3 tools/score_wer.py <manifest.tsv> <hyp.tsv> --ci --compare <other_hyp.tsv>

# stt_eval knobs: STT_TAIL_EXTRA, STT_PREFIX, STT_AGC, STT_AGC_RESET,
#                 STT_RESAMPLE=linear, STT_RS_ZEROS, STT_RS_ROLLOFF, STT_MIMI, MOSHI_MARGIN
# ref_eval knobs: REF_TAIL_EXTRA, REF_PREFIX_FRAMES, REF_THREADS
# ensemble:       tools/rover.py, tools/gated_ensemble.py [--rank nlow|frac],
#                 tools/ensemble_diversity.py
# data:           tools/fleurs_prepare.py <dir> [test|dev]
```
