# Spike B.1 — kyutai stt-1b-en_fr via moshi.cpp, 2026-07-27

Evaluating [`Codes4Fun/moshi.cpp`](https://github.com/Codes4Fun/moshi.cpp) as a
replacement STT engine for Katarina, against the measured Voxtral baseline in
`voxtral.cpp/tests/baselines/2026-07-27-513822e-*.md`.

Upstream pinned at `f1fabbd` (2026-02-18).

## Verdict: right model, wrong runtime — and neither engine is viable today

kyutai is **correct where Voxtral is broken** (no blackout, ~100 % coverage) and
**1.8 GB lighter**, but **2.2x slower on device** as shipped. The slowness is a
runtime inefficiency, not physics — it reproduces on host, so it isn't ARM.

| | Voxtral 4B Q4_0 | kyutai stt-1b q4_k |
|---|---|---|
| **device RTF** (cooled, 6 threads) | **1.27–1.40** | **2.99** |
| host RTF | 1.05–1.17 | 1.04–1.09 |
| 90 s continuous-speech coverage | **61 %** (16 s blackout) | **102 %** (no blackout) |
| **device peak RSS** | **4484 MB** | **2677 MB** |
| French error profile | correct on "l'IA générative" | writes "liage génératif" |

### Neither reaches real time on this phone, and thermal is the harder ceiling

Voxtral measures 1.27 cold but drifts to **2.86 after a few minutes of sustained
load** (79–82 °C). kyutai is 2.99. **Both are above 1.0 even before throttling**, and
sustained transcription is precisely the workload. On-device real-time French STT is
not achievable on this device with either engine at these implementations — the
question is which one has the shorter path to it.

### Where kyutai's 2.99 actually goes (measured, not guessed)

Per-frame at q4_k on a cooled device, 80 ms budget:

| stage | ms/frame | share |
|---|---|---|
| Mimi encoder | 38.9 | 17 % |
| **LM (1B)** | **191.6** | **83 %** |

**191 ms for a 1B model at q4_k is ~10x off.** Reference points: llama.cpp reports
Llama-3.2-1B Q4_0 at ~51 tok/s (≈19 ms/token) on Snapdragon, and Voxtral's own
*3.4B* decoder runs ~52 ms/step. A 1B should not cost 4x a 3.4B.

Two observations pin it to implementation rather than hardware:
1. **It reproduces on host.** LM = 49 ms/frame at q4_k on an AVX-512 desktop, where
   ~5–10 ms is the expectation. Same ~5–10x gap.
2. **Negative thread scaling.** threads=16 is **2.2x slower** than threads=6 on host
   (177.9 vs 80.3 ms/frame). Compute-bound work does not behave like that; this is the
   signature of per-op synchronisation overhead dominating tiny batch-1 kernels.

**The obvious cause is ruled out:** the graph is *not* rebuilt per frame —
`context.h:487` caches it (`if (!gf) gf = ggml_new_graph_custom(...)`) and
`ggml_backend_graph_compute` runs the cached graph. So the cost is inside the graph:
plausibly a very high node count for batch-1 (16 layers, plus **32 audio-codebook
embedding lookups per frame**, plus the extra VAD heads), where ggml's per-op thread
barrier dominates. Confirming that needs a node count / per-op profile — not done.

**So the port is not "build it for arm64 and go".** That part took an afternoon and
works. Making it *fast* is open-ended optimisation inside a 36-star, single-author
codebase, with the cheapest hypothesis already eliminated.

## Original comparison detail

Host x86, 6 threads, CPU backend, same fixtures, same RTF definition
(compute_ms / audio_ms):

| | Voxtral 4B Q4_0 | kyutai stt-1b f16 |
|---|---|---|
| `test_16k` (30 s) chars | 501 | **572** |
| `test_speech_90s` chars | 1048 | **1744** (expected ~1716) |
| **blackout on 90 s** | **16 s of nothing** | **none** |
| host RTF (30 s) | 1.05–1.17 | 1.037 |
| host RTF (90 s) | 1.305 | **1.153** |
| model on disk | 2.5 GB (Q4_0) | 1.98 GB (**f16, unquantized**) |
| resident (device) | **4484 MB** | not yet measured |

`test_speech_90s` is `test_16k` repeated 3x, so the expected output is ~3 × 572.
kyutai returns 1744 (≈102 % coverage, the same passage cleanly three times).
Voxtral returns 1048 (≈61 %), losing a contiguous 16-second span to its pipeline
reset. **That is the decisive difference**: for a rolling-transcript assistant, a
16 s hole means the answer to "what did he just say?" is not in the buffer.

Note kyutai achieves this **unquantized (f16) against Voxtral's Q4_0**, and on
16 kHz audio upsampled to the 24 kHz the codec expects — both handicaps. Its
quality here is a floor, not a ceiling.

## Architecture facts confirmed (answers the spike questions)

- **`"dep_q": 0` in the model config ⇒ the depformer is not used for STT.** It is
  declared (dim 1024, 6 layers) because Moshi shares one config across directions,
  but `dep_q` is the number of audio codebooks generated, and STT generates none.
  Confirmed from config, not inferred.
- **`"context": 750` frames at 12.5 Hz = 60 s of sliding context.** Compare Voxtral,
  whose encoder context is clamped to `ENC_KV_TARGET = 100` ≈ 2 s (against a trained
  750-frame / 15 s window).
- **Audio input is discrete RVQ codes** from Mimi (`n_q = 32`, `card = 2048`), not a
  continuous latent.
- **No reset logic anywhere in the loop.** Streaming by design, as the DSM paper claims.
- **VAD comes free**: `moshi_lm_receive2` returns a per-frame VAD score alongside the
  text token. Directly useful for the wake-word / end-of-question flow, which
  currently uses a hand-rolled RMS VAD in `MicListenerService`.
- **The model path needs neither SDL nor FFmpeg** — those are only used by the demo
  tools for I/O. `tools/stt_bench.cpp` (added here) drives the engine with nothing but
  libmoshi + ggml, which is exactly the shape of the future JNI layer.

## The streaming API — and why the port is small

```cpp
mimi_encode_send(enc, frame);              // frame_size floats @ 24 kHz
mimi_encode_receive(enc, tokens.data());   // -> n_q RVQ codes
moshi_lm_send2(gen, tokens);
moshi_lm_receive2(gen, text_token, vad);   // -> one text token + VAD, per frame
```

One frame in, one token out, 12.5 Hz. No chunk-size heuristics, no grid-alignment
trap (contrast Voxtral, where a 3 s chunk silently yields zero text because
48000 samples is 37.5 audio tokens — see the device baseline).

## Known caveats and open risks

1. **Frame-time jitter.** Average 80.0 ms/frame against an 80 ms budget, but worst
   frame is 167–234 ms. Real-time use needs a buffer absorbing ~3x budget spikes.
2. **RTF creeps with context**: 1.037 at 30 s → 1.153 at 90 s as the 750-frame window
   fills. Should plateau at ~60 s of audio; not yet verified beyond 90 s.
3. **q4_k was slightly *slower* than f16 on host** (1.086 vs 1.037) — dequantization
   costs more than it saves on an AVX-512 host with ample bandwidth. This should
   invert on the phone, where bandwidth is the binding constraint. **Untested on
   device — this is the key remaining measurement.**
4. **French error profile differs.** kyutai writes "liage génératif" for
   "l'IA générative" (consistently, 6x) and "On a quand même" for "on est quand même";
   Voxtral gets both right. Comparable overall quality, different failure modes.
5. **24 kHz vs 16 kHz.** The app captures 16 kHz; Mimi wants 24 kHz. The bench
   upsamples linearly, which adds no content above 8 kHz — a domain shift against the
   model. Capturing natively at 24/48 kHz is possible via AudioPlaybackCapture and
   would likely improve quality. A fair quality verdict needs a natively-24 kHz fixture.
6. **Project health**: 36 stars, 83 commits, one author, last push 2026-02-18. No
   Android build. Works, but there is no maintenance guarantee — porting means owning it.
7. **sentencepiece is genuine build friction**, exactly as the author's own TODO says
   ("wrap sentencepiece into its own dynamic library or externalize it"). It needs
   `src/` + repo root + bundled `abseil-cpp` on the include path, and linking needs
   93 absl static libs. Cross-compiling this for Android is the main port cost.

## Reproducing

```
# sentencepiece (headers are NOT installed; they live in src/, and third_party/
# abseil-cpp must also be on the include path)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DSPM_ENABLE_SHARED=OFF -DSPM_ENABLE_TCMALLOC=OFF
cmake --build build -j$(nproc)          # full build, so the absl static libs exist

# libmoshi, reusing voxtral.cpp's ggml so both engines share one backend
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DMOSHI_BUILD_TOOLS=OFF \
  -DGGML_INCLUDE_DIR=../voxtral.cpp/ggml/include \
  -DGGML_LIBRARY_DIR=../voxtral.cpp/build-x86/ggml/src \
  -DSentencePiece_INCLUDE_DIR=$SP/src -DSentencePiece_LIBRARY_DIR=$SP/build/src \
  -DCMAKE_CXX_FLAGS="-I$SP -I$SP/third_party/abseil-cpp -I$SP/third_party/protobuf-lite"
cmake --build build -j$(nproc)

# bench (see the g++ line in git history for the full absl link group)
./build/stt_bench models/Codes4Fun/stt-1b-en_fr-GGUF <audio.wav> [threads] [quant]
```

Models (verified by sha256): `Codes4Fun/stt-1b-en_fr-GGUF/{config.json,model.gguf}`,
`Codes4Fun/moshi-common/{mimi-e351c8d8-125.gguf,tokenizer_spm_8k_en_fr.model}` —
2.33 GB total, download lists in `tools/Codes4Fun_*.txt`.

## Next

B.2 — cross-compile sentencepiece + libmoshi + stt_bench for arm64-v8a and measure
on the device (SM8850). Two things to settle there: q4_k vs f16 under real bandwidth
pressure, and resident memory vs Voxtral's measured 4484 MB.

⚠️ On-device RTF must be measured **with cooldown between runs**: the phone throttles
hard under sustained load. Back-to-back Voxtral runs drifted 1.27 → 2.86 RTF at
79–82 °C. See the device baseline notes.
