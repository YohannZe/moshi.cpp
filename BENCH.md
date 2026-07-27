# Benchmarks — kyutai stt-1b-en_fr on moshi.cpp

Every row is a measurement, not an estimate. Device rows are taken with a thermal
cooldown to < 50 °C before the run (`tools/bench_device.sh`) — the phone throttles from
RTF 1.27 to 2.86 as it climbs to 82 °C, so back-to-back A/B is worthless.

- Device: Xiaomi `25102PCBEG`, **SM8850 = Snapdragon 8 Elite Gen 5**, Android 16, 6 threads
- Host: x86_64 AVX-512, 6 threads
- Fixture `test_16k.wav`: 30 s of French speech (16 kHz, upsampled to Mimi's 24 kHz)
- RTF = compute_ms / audio_ms. Frame budget is 80 ms (12.5 Hz).
- Baseline to beat: Voxtral Mini 4B Q4_0 = **RTF 1.40**, 4484 MB RSS, and a **16 s
  blackout** on continuous speech (`voxtral.cpp/tests/baselines/`).

---

## P1 — Real quantization (2026-07-27)

### The bug being fixed

`moshi_lm_quantize(lm, "q4_k")` is a **silent no-op for GGUF inputs**: both overloads of
`WeightLoader::fetch()` (`src/loader.h:149-153`, `:191-195`) return `get_tensor(name)`
without ever consulting `dst_type`/`qtype`. The published `model.gguf` is **102 BF16
tensors (989.2 M params) + 33 F32**, so every `-q q4_k` run ever made against it measured
BF16. Fixed offline by `tools/requantize_gguf.cpp`.

The second half of the bug is ARM-specific: **`ggml_vec_dot_bf16` has no ARM path at all**
(AVX512BF16 / AVX512F / AVX2 / RISC-V / POWER9 only), so on the phone every MAC falls
through to a scalar loop with two bf16→f32 conversions. BF16 is the worst 16-bit format to
ship for aarch64, and it is what upstream ships.

### Device (SM8850, cooled)

| weights | size | LM ms/frame | Mimi ms/frame | RTF | chars |
|---|---|---|---|---|---|
| `model.gguf` (BF16, upstream) | 1979 MB | 168.02 | 35.85 | **2.644** | 573 |
| `model-f16.gguf` | 1979 MB | 64.34 | 37.26 | **1.317** | 573 |
| **`model-q4_k.gguf`** | **773 MB** | **41.00** | 34.83 | **0.983** | 571 |
| `model-q4_0.gguf` | 773 MB | 42.21 | 34.59 | 0.996 | 573 |

### Host (x86 AVX-512)

| weights | LM ms/frame | Mimi ms/frame | RTF | chars |
|---|---|---|---|---|
| `model.gguf` (BF16) | 59.73 | 35.11 | 1.230 | 572 |
| `model-f16.gguf` | 60.46 | 36.42 | 1.256 | 572 |
| **`model-q4_k.gguf`** | **29.06** | 38.08 | **0.871** | 571 |
| `model-q4_0.gguf` | 31.60 | 39.99 | 0.928 | 572 |

### What this proves

1. **BF16 → F16 alone is 2.6x on the LM on ARM (168 → 64 ms) at identical byte count.**
   This is a clean controlled experiment — same file size, same values to within F16
   rounding, only the kernel differs. It isolates the missing-ARM-kernel effect from the
   bandwidth effect, and on host the same swap changes nothing (59.7 → 60.5), exactly as
   predicted since x86 has AVX512F paths for both.
2. **Q4_K then adds 1.57x** (64.3 → 41.0). Total LM speedup **4.1x**, RTF **2.644 → 0.983**.
3. **Q4_K ≥ Q4_0** on both platforms, marginally (41.0 vs 42.2 device; 29.1 vs 31.6 host).
   Identical file size — both are 4.5 bits/weight. Q4_0's advantage would be the i8mm
   repack path, which is not yet enabled (see P6).
4. **Quality is unaffected**: 571–573 chars against 572 for BF16, same transcript.
5. **Model load got 3.1x faster** as a side effect (3085 → 979 ms), which matters for app
   startup.
6. **Mimi is now the bottleneck**: 34.8 ms/frame = 46 % of compute, and it is nearly
   identical on phone and desktop (34.8 vs 38.1) — the signature of an overhead-bound
   stage, not a compute-bound one. That is P2.

### Type policy

Not everything should be quantized. `tools/requantize_gguf.cpp` applies:

| tensors | target | why |
|---|---|---|
| 69 matmul weights (incl. `text_linear` 2048×8001) | Q4_K / Q4_0 | the traffic |
| 33 lookup tables (32 audio codebooks + `text_emb`) | **F16** | `ggml_get_rows` reads one row per frame — negligible bandwidth, so spend the bits on precision |
| 33 norms / 1-D params (0.1 M params total) | **F32** | free, and quantizing per-channel scales is actively harmful |
| anything whose row isn't a block multiple | F16 + warning | correctness |

Never BF16 on output, whatever the input was.

That policy is why the file is 2.56x smaller rather than 3.55x: the 33 F16 lookup tables
are 277 MB of the 773 MB. Quantizing them too (`--quantize-all`) would reach ~500 MB and
is worth measuring against quality if RSS becomes binding.

### Caveat

Every device run ends at 83–84 °C regardless of variant. **At RTF 0.983 the device is
still thermally saturated**, so sustained transcription will throttle below real time.
Headroom, not just parity, is required — hence P2.

---

## P2 — RVQ search by matvec (2026-07-27)

`moshi_EuclideanCodebook_encode` materialised the full [D, card] difference matrix per
codebook — `repeat_4d(x)` + `repeat_4d(embedding)` + `sub` + `mul` + `sum_rows`. At
card=2048, D=256 that is ~2 MiB per intermediate and ~18 MiB per codebook, times 32
codebooks times 12.5 frames/s. `ggml_repeat_4d` always emits a REPEAT node even when the
shape is unchanged, so the embedding repeat was a 2 MiB memcpy of **static weights** every
frame. And `GraphContext::alloc()` uses `ggml_backend_alloc_ctx_tensors` rather than a
gallocr, so those intermediates pinned ~256 MiB of permanently resident graph buffer.

Replaced by `argmin ||x-e||² = argmax(x·e - ||e||²/2)`: one matvec plus a broadcast
subtract, with `||e||²/2` precomputed once at graph-build time.

| | Mimi ms/frame | RTF |
|---|---|---|
| device, before | 34.83 | 0.983 |
| **device, after** | **13.79** | **0.685** |
| host, before (same session) | 32.70 | 0.735 |
| **host, after** | **10.70** | **0.459** |

**Transcript identical to the character.** Also numerically better than what it replaced:
the old code turned argmin into argmax via `1/(1+d)`, whose derivative `-dd/(1+d)²`
collapses distance gaps below ~2e-6 into F32 rounding at the d≈50 typical here — exactly
the near-tie regime that decides centroid choice — and inherited
`ggml_vec_argmax_f32`'s last-maximum tie-break where `torch.argmin` takes the first.

## P3 — attention mask fix (2026-07-27)

`bias_pattern_index` omitted the block width `t` in its wrapped branch, so from
`offset == capacity+1` onward each step was denied the slot holding the preceding block
and allowed a slot already overwritten with the *current* block: a `t`-frame lookahead.
Only bites for `t > 1`, which is why the Mimi transformers (t=2, capacity=250) were wrong
from ~10 s of audio while the LM (t=1, capacity=750) was accidentally fine — at t=1 the
window equals the capacity, so once the ring is full every slot is legitimately in range
and the mask is all-ones.

Measured on `test_speech_90s` (one 30 s passage × 3, so the ideal output is the 30 s
reference × 3 and repetitions 2–3 lie entirely in the broken regime):

| | words matched | word error rate |
|---|---|---|
| before | 273/297 | **8.08 %** |
| **after** | **297/297** | **0.00 %** |

Speed unaffected. This is why the 90 s fixture exists.

## P4/P5 — KV cache F16 + flash attention (2026-07-27)

KV cache BF16 → F16: LM 24.71 → 23.48 ms/frame on host, and the transcript got *better*
(583 vs 546 chars; "d'abord dans des investissements dans les infrastructures" correct
where BF16 gave "dans les investissements, dans les infrastructures"). 10 mantissa bits
against 7, at the same 2 bytes.

Flash attention: **~2 %** (lm_ms 9492 → 9308, same session, identical text) — **not** the
10–40 ms an earlier derived estimate suggested. The V-cache transpose is ~98 MB/frame
across 16 layers ≈ 1.2 GB/s at 12.5 fps, a few percent of a 34 GB/s bus. The estimate was
simply wrong; kept the change for the correctness guard, fewer nodes and simpler call
sites.

⚠️ **ggml footgun found here.** `ggml_flash_attn_ext` asserts only
`ggml_is_contiguous(mask)`, but its CPU kernel reads the mask as `ggml_fp16_t`
unconditionally. Passing the existing F32 mask produced garbage **silently** — and
garbage that ran **45 % faster** (RTF 0.291 vs 0.448), because the reinterpreted bytes
looked like `-inf` so the kernel skipped most positions. Only comparing transcripts caught
it: 388 characters of pure whitespace. `create_bias_pattern` now emits F16 and the flash
path asserts the dtype itself.

## The LM is now bandwidth-bound on its own weights

Host LM time against matmul bytes per parameter, all else equal:

| quant | B/param | bytes vs Q4_K | LM ms | time vs Q4_K |
|---|---|---|---|---|
| Q4_K | 0.5625 | 1.00 | 25.01 | 1.00 |
| Q8_0 | 1.0625 | 1.89 | 47.54 | **1.90** |
| F16 | 2.0 | 3.56 | 69.60 | 2.78 |

Q8_0 predicted 1.89x, measured 1.90x. 839 M matmul params at Q4_K is ~472 MB/frame, i.e.
~18.9 GB/s at 25 ms. **Further LM gains therefore need either fewer bytes (quality cost)
or better kernels per byte** — not graph restructuring.

## Reverted: repack buffer type — and a measurement lesson

Routing weights through `ggml_backend_dev_get_extra_bufts` (the CPU repack buffer, which
rewrites quantized weights into the blocked layouts the wide i8mm/dotprod GEMM kernels
want) looked like a 14 % win on Q4_K. **It was not measured.** The A/B passed
`MOSHI_NO_REPACK=` (empty) for the "on" arm, and `getenv()` returns non-NULL for an empty
value, so repack was disabled in *both* arms — the 14 % was noise between two identical
configurations. Exactly the failure mode this branch opened by fixing upstream's silent
`-q` no-op. Guard on the value, not on presence.

Once genuinely enabled it **segfaults**, including with F16 weights repack should not
touch. Most likely an interaction with P2: `moshi_EuclideanCodebook_half_sqnorm` calls
`ggml_backend_tensor_get` on the codebook embedding, and a repack buffer does not
necessarily support reading back. Reverted rather than debugged, because the payoff is an
aarch64 story (x86 gets AVX either way) and the device was disconnected. **Worth
revisiting on device**, moving the codebook read off the repack path first.

## Status — host, all fixtures, Q4_K

| fixture | RTF | chars | expected |
|---|---|---|---|
| `test_16k` (30 s speech) | 0.571 | 583 | — |
| `test_speech_90s` | **0.498** | **1735** | ≈3×583, no blackout ✅ |
| `test_speech_music` | 0.506 | 574 | no regression ✅ |
| `test_music` | 0.536 | **0** | no hallucination ✅ |

Device figures are from before P3/P4/P5 (USB dropped mid-session): **RTF 0.685** after P1+P2,
against Voxtral's 1.40. The post-P4/P5 device numbers and peak RSS still need taking.

## Reproducing

```
# quantize (host)
./build/requantize_gguf models/.../model.gguf models/.../model-q4_k.gguf q4_k

# host bench
./build/stt_bench models/Codes4Fun/stt-1b-en_fr-GGUF <fixture.wav> 6 model-q4_k.gguf

# device bench, with cooldown between runs
./tools/bench_device.sh test_16k.wav model-q4_k.gguf model-q4_0.gguf
```
