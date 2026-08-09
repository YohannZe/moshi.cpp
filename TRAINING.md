# Pushing past 10.66 % — the model-side map

Serving is exhausted (measured: every remaining serving lever is null or negative — BENCH.md).
This file is the other half: what to do **to the weights**, ranked by evidence and cost. Every
claim below cites a measurement from this repo, not intuition.

---

## 1. The constraints any improvement must respect (all measured here)

| finding | number | consequence |
|---|---|---|
| errors are *hesitant*, not confident (margin 4.45 vs 8.34 in wrong vs right words) | 3.7× near-ties | decisions sit near boundaries → **small weight movements can flip them**; downstream rescues cannot (5 tried, 5 lost to calibration) |
| 20 % of substitutions are true homophones (leur/leurs, au/aux) | 275/1368 | pure grammar — only *text-side knowledge* fixes them |
| 80 % are acoustically distinct (des/les, est/et) | 1093/1368 | information lost before the decision — front-end/encoder side |
| model exploits only bands its input reliably contains (removing 6–8 kHz: +0.85 pt, p=0.016; restoring 8–12 kHz: −0.37 pt n.s.) | | training-data bandwidth mix defines what the model can use at inference |
| input gain sensitivity: −46 dBFS costs 11 pt without AGC | 22.36→11.41 % | the codec front end has **no level invariance** unless trained in |
| serving protocol worth 1.96 pt on their own weights | 12.62→10.66 % | the *training-time* flush/padding distribution defines the serving optimum; nobody publishes it |
| calibration is excellent: cumulative logprob ranks greedy above every forced alternative | beam probe | temperature-0 calibration is an asset — do not trade it away |
| Q4_K costs +0.11 pt, reproduced on two disjoint sets | dev AND test | quantization is a solved cost here; QAT is not the bottleneck |

---

## 2. Pushing **this** model with light training — the realistic moonshot

### 2.1 LoRA French adaptation (the one with a credible path to sub-10)

**Why it should work when every inference-side idea failed:** the failures were all attempts
to overrule a well-calibrated model from outside. Training moves the decision boundary
itself — and the margin analysis says the erroneous decisions are *close* (median 4.45), so
they are cheap to flip. The two dominant error classes both map to trainable capacity:
agreement/homophones (20 %) are text-side knowledge, and the des/les class points at
acoustic-to-text mapping sharpness, both of which LoRA on the LM reaches.

**Recipe** (adapting [kyutai-labs/moshi-finetune](https://github.com/kyutai-labs/moshi-finetune),
which fine-tunes the same multistream backbone for the dialogue model):

- **Changes needed**: STT consumes mono audio + emits delayed text, vs Moshi's stereo
  dialogue format. The dataset loader must produce (Mimi tokens, framewise text tokens with
  the 6-frame delay) pairs instead of dual-stream audio. The text stream alignment comes
  from timestamped transcripts (whisperx or CV's own alignment).
- **Data**: Common Voice fr train — accessible via the `fsicoli/common_voice_17_0` mirror we
  already use (no gate), with our spectral HF filter and peak-norm pipeline (tools/cv_prep.py).
  Add MLS-French for volume. 50–200 h is the LoRA regime.
- **Config** (from moshi-finetune's published example): LoRA rank 128, scale 2, ~2000 steps,
  batch 16 × 100 s. Target the text head + attention projections; freeze Mimi entirely
  (our codec findings say it is not the residual bottleneck).
- **Hardware**: 1×H100 ≈ 40 GB peak (their numbers). Rented: hours, ~10–30 €. **Not
  runnable on this laptop** (Ryzen 7840U, no CUDA — verified).
- **Success criterion, pre-registered**: −0.7 pt on FLEURS-fr test (676, our scorer, one
  confirmation run, paired bootstrap vs 10.66 %). That is half the agreement class; if
  agreement errors do not move, stop — the class was memorization, not knowledge.
- **Risks**: English degradation (acceptable — Katarina is French-first); serving optimum
  shifts (re-run the tail/prefix sweep on dev after — cheap); CV read-speech domain
  transfer (mitigate by mixing in the MLS audiobook register).

### 2.2 Flush-invariance micro-tune (tiny, targeted, novel)

The 1.96 pt serving gap exists because the model's end-of-stream behaviour was shaped by an
undocumented training-time padding distribution — the reference's own script under-flushes it
(112/674 truncated utterances). A micro-fine-tune with **randomized trailing-silence lengths**
would make the delay line drain correctly under any flush, deleting the serving trap for
every downstream user. Hundreds of steps, same rig as 2.1. Publishable as its own finding:
"serving-protocol robustness is trainable".

### 2.3 Not worth training

- **QAT**: the whole quantization gap is +0.11 pt. Nothing to win.
- **Codec (Mimi) fine-tuning**: our octave and low-pass results bound the front end's
  residual contribution; the codec transmits what matters at F16. Frozen it stays.

---

## 3. How the *next* model should be trained — recommendations priced by our pathologies

For kyutai or anyone training streaming codec STT. Each line pairs a measured pathology with
the training-time fix; augmentations are data-pipeline changes, near-free at training scale.

1. **Gain augmentation** (random −40…0 dBFS per sample). Pathology: 11 pt collapse at
   −46 dBFS. A codec front end sees raw amplitude; nothing teaches it level invariance
   unless the data does. This is one line in a data loader and removes an entire deployment
   trap (our in-app AGC would become unnecessary).
2. **Resampler/bandwidth augmentation** (mix native 24 kHz with 16 kHz-upsampled — both
   linear and sinc). Pathology: sinc-vs-linear is −0.62 pt at inference (p=0.012), i.e. the
   model is sensitive to imaging artifacts it never saw in training; and it cannot use bands
   its inputs lack. Make the training distribution span deployment front ends and the
   sensitivity disappears — plus every 16 kHz benchmark stops being an out-of-distribution
   test.
3. **Randomized end-of-stream padding**. Pathology: the flush window (§2.2). Also fixes the
   reference implementation's own 112/674 truncations without anyone touching serving code.
4. **Publish the serving contract with the weights**: expected input level, flush depth,
   warm-up prefix, resampler. Two numbers justify this: 12.62 % vs 10.66 % is the same model
   under two protocols, and every mechanism involved was discoverable only by black-box
   sweeps. A half-page model-card section makes 2 WER points reproducible for everyone.
5. **Text-side French grammar capacity**. Pathology: 20 % of residual substitutions are
   agreement/homophones that only grammar decides — and the model's *implicit* LM already
   beats explicit fusion everywhere (bigram fusion: +1 pt, measured), so the fix is more/better
   text in the multistream pretraining mix (or text-LM distillation), not decode-time fusion.
6. **Stop-consonant discrimination at 12.5 Hz** *(hypothesis, not measured)*: des/les-class
   confusions concentrate on short consonant releases; cues survive below 8 kHz (low-pass
   asymmetry) yet still get lost, pointing at the 80 ms frame grid. An auxiliary phoneme
   objective on the encoder, or a finer-rate encoder distilled into the 12.5 Hz one, is the
   experiment. Flagged speculative — the only line here without a number behind it.
7. **Preserve temperature-0 calibration.** The model's margins rank its own paths perfectly
   (beam probe: greedy is both best-WER and best-logprob at every forced divergence). This
   property is what makes confidence-gated serving possible at all; RLHF-style or
   sequence-level objectives that distort token-level calibration would cost more than they
   bring.

---

## 4. Decision table

| action | cost | expected | decision |
|---|---|---|---|
| LoRA fr (2.1) | rented H100, hours, ~20 € + ~2 days adaptation work | −0.7…−1.3 pt, credible sub-10 | **the** next move; needs a GPU budget |
| flush-invariance tune (2.2) | same rig, smaller | robustness + a novel publishable claim | ride along with 2.1 |
| QAT | same rig | +0.11 pt ceiling | no |
| next-model recs (§3) | a README section | community value; strengthens the paper's §"implications" | free — fold into PAPER.md |

---

## 5. What the DSM paper itself says (read 2026-08-09, arXiv:2509.08753) — and the lever kyutai kept

Read against our measurements, three things stand out:

1. **Delay conditioning is the paper's best ASR trick — and it is not in the released 1B.**
   They train DSM-ASR over *random* delays with a cosine embedding of the target delay summed
   into the inputs; one conditioned model **outperforms every fixed-delay variant**, and the
   effective delay tracks the conditioning within ~300 ms. Our checkpoint's config says
   `"conditioners": {}` (verified in the GGUF and the HF config): stt-1b-en_fr is fixed at
   τ=0.5 s. So the quality-vs-latency dial exists in the method but not in the shipped
   weights — for Katarina, which buffers transcripts and could happily absorb 1–2 s of extra
   delay, a conditioned checkpoint would be free quality. **Add to §2.1's fine-tune: train
   the LoRA with delay conditioning** (the paper proves it composes with the task), or ask
   kyutai to release the conditioned variant.
2. **Their training pipeline validates the §2.1 plan**: pseudo-labels from Whisper first,
   then fine-tune on ground-truth transcripts (WER 6.4 % after fine-tuning stage), with
   codebook-dropout augmentation. Our LoRA plan is a miniature of their own second stage.
3. **PAD/WORD is the whole text-stream protocol** (word start index = floor(s·fr), tokens
   follow, PAD elsewhere; loss on PAD/WORD trains word boundaries). This explains a week of
   our serving results mechanically: the flush window works because trailing text is still
   scheduled behind τ; the 12-frame prefix hurt because PAD-heavy context is itself a
   trained signal ("nobody is speaking"), not neutral filler.

---

## 6. What training THIS model actually cost kyutai — and the three tiers open to us

From the DSM paper (§4.2), the real recipe for stt-1b:

| stage | data | compute |
|---|---|---|
| pretraining (hard distillation of whisper-timestamped) | **2.5 M hours** en+fr, pseudo-labels, 90 s segments | 1.6 M steps on **48 H100** |
| fine-tune | 28 k h ground-truth public data + codebook dropout | 100 k updates, batch 128, **16 H100** |
| long-form adaptation | long-form mixture | 25 k updates, batch 32, 16 H100 |

Order-of-magnitude bill: pretraining alone is tens of thousands of H100-hours (~10⁵ €);
the fine-tune stages a few thousand more. Mimi is reused frozen — "training a model like
this" means training the 1B backbone, not the codec.

### The three tiers, priced

**Tier A — LoRA French (the one to actually do).** stt-1b is 7× smaller than the Moshi
model moshi-finetune quotes 39.6 GB for — a LoRA fits a rented RTX 4090 (24 GB) at
~0.5 €/h, an H100 makes it comfortable. Data: CV-fr (≥1000 h validated, free, already
pipelined in this repo) — the LoRA regime needs 50–200 h. Wall time hours, **total
10–30 €**. Expected from the error taxonomy: half the agreement class ≈ −0.7 pt → ~9.9 %,
plus delay conditioning and flush invariance trained in (§5). Pre-registered criterion in
§2.1 still stands.

**Tier B — full French fine-tune (reproduce their stage 2, French-only).** All 1B params,
1–5 k h public French (MLS-fr 1.1 k h + CV-fr + VoxPopuli-fr), 1–2×H100 for a few days,
**~200–500 €**. This is the "French specialist stt-1b" — plausibly −1.5…−2 pt and a model
worth releasing on its own. The paper's own stage-2 hyperparameters transfer directly.

**Tier C — from scratch.** 2.5 M hours and 48 H100 for weeks: **~10⁵ €**, out of scope. The
paper's own "public data only" reproduction (88 k h, done for TTS) shows a legitimate
~10–20 k€ path, but Tier B captures most of the value for 2 % of the price.

The through-line: every Tier A/B ingredient is already de-risked by this repo — the data
pipeline exists (cv_prep.py), the eval harness is trustworthy (reproducibility check passed),
the error taxonomy says exactly what to train against, and the serving protocol to re-tune
afterwards is documented.
