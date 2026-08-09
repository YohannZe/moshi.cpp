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
