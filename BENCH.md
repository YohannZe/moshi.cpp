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

## Reproducing

```
# quantize (host)
./build/requantize_gguf models/.../model.gguf models/.../model-q4_k.gguf q4_k

# host bench
./build/stt_bench models/Codes4Fun/stt-1b-en_fr-GGUF <fixture.wav> 6 model-q4_k.gguf

# device bench, with cooldown between runs
./tools/bench_device.sh test_16k.wav model-q4_k.gguf model-q4_0.gguf
```
