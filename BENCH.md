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

## FINAL — device (SM8850), all fixtures, Q4_K weights

| fixture | RTF | chars | expected |
|---|---|---|---|
| `test_16k` (30 s speech) | **0.685** | 583 | — |
| `test_speech_90s` | **0.695** | **1735** | ≈3×583, no blackout ✅ |
| `test_speech_music` | 0.695 | 574 | no regression ✅ |
| `test_music` | 0.697 | **0** | no hallucination ✅ |
| **peak VmRSS** | **1352 MB** | | < 1500 MB target ✅ |

Character counts match host exactly, so the mask fix and the numerics hold on aarch64.

### Sustained load — the test that actually matters

6 minutes of continuous audio, four 90 s runs back to back with **no cooldown**:

| run | RTF | chars | temp after |
|---|---|---|---|
| 1 | 0.706 | 1735 | 67 °C |
| 2 | 0.687 | 1735 | 66 °C |
| 3 | 0.709 | 1735 | 60 °C |
| 4 | 0.938 | 1735 | 58 °C |

**It no longer thermally saturates.** Peak 67 °C against 83–84 °C for every pre-P2 run,
because there is simply far less work per frame. RTF stays under 1.0 throughout and the
output is bit-stable across all four runs. The run-4 excursion is not thermal (58 °C) —
most likely background activity on the phone.

Compare where this started, and the incumbent:

| | RTF (device) | peak RSS | 90 s coverage |
|---|---|---|---|
| Voxtral Mini 4B Q4_0 (incumbent) | 1.40 | 4484 MB | 61 %, **16 s blackout** |
| kyutai stt-1b, as shipped upstream | 2.99 | 2677 MB | — |
| **kyutai stt-1b, this branch** | **0.685** | **1352 MB** | **~100 %, no blackout** |

**4.4x faster than upstream, 2x faster and 3.3x lighter than Voxtral**, with the 16 s
transcript hole gone.

## Earlier status — host, all fixtures, Q4_K

| fixture | RTF | chars | expected |
|---|---|---|---|
| `test_16k` (30 s speech) | 0.571 | 583 | — |
| `test_speech_90s` | **0.498** | **1735** | ≈3×583, no blackout ✅ |
| `test_speech_music` | 0.506 | 574 | no regression ✅ |
| `test_music` | 0.536 | **0** | no hallucination ✅ |

(Host figures, for reference. Device figures above.)

## Reproducing

```
# quantize (host)
./build/requantize_gguf models/.../model.gguf models/.../model-q4_k.gguf q4_k

# host bench
./build/stt_bench models/Codes4Fun/stt-1b-en_fr-GGUF <fixture.wav> 6 model-q4_k.gguf

# device bench, with cooldown between runs
./tools/bench_device.sh test_16k.wav model-q4_k.gguf model-q4_0.gguf
```

---

## The text stream is 55 % padding — and that is exploitable

Measured on `test_speech_90s` (1139 frames, `STT_DUMP_TOKENS=1`):

| | |
|---|---|
| productive frames (a real text token) | 507 (44.5 %) |
| padding frames | 632 (**55.5 %**) |
| padding runs | 171, mean length 3.70 |
| P(next frame is padding) | **55.4 %** |
| P(next 2 frames padding) | 40.5 % |
| P(next 3 frames padding) | 29.3 % |

Why this matters: the LM does 12.5 full forward passes per second of audio, and each pass
reads all 472 MB of matmul weights (we are bandwidth-bound — see above). The output rate is
fixed by the audio, so classic speculative decoding buys nothing: you cannot produce fewer
passes by producing more tokens.

But the *input* to frame t+1 includes frame t's output text token, and that token is padding
55 % of the time. Speculate it, batch frames (t, t+1) through the transformer in one pass,
and verify:

  expected passes per 2 frames = 0.554x1 + 0.446x2 = 1.446  vs 2
  => ~28 % fewer weight-loading passes

Batch-2 costs 2x the FLOPs, which is ~free on a bandwidth-bound workload — the weights are
read once for both positions. Batching deeper is worse: P(3 padding) is only 29 % and the
rollback cost grows, so 2 is the sweet spot.

This is speculative decoding pointed at a different target: not "more tokens per pass" but
"more audio frames per weight load". The hard part is rollback — undoing the KV write and
RoPE position for frame t+1 when the guess was wrong. Not attempted.

---

## Repack, properly tested this time: it is a PREFILL optimization and hurts us

The earlier repack attempt was reverted for a bad measurement (empty env var) and a
segfault. Both are now fixed and the change was measured honestly. **It is still a loss.**

Device (SM8850, cooled, Q4_K, 6 threads):

| | Mimi ms | LM ms | RTF | chars |
|---|---|---|---|---|
| repack OFF | 6360 | 14692 | **0.702** | 583 |
| repack ON | 8067 | 14845 | 0.764 | 583 |

Host x86: 9317 vs 9338 ms — no difference at all.

**Why, and this is the useful part:** the repacked kernels (`q4_K_8x8`, `q4_0_4x8` ...) are
wide GEMM kernels. They need ~8 columns of activations to fill a tile. We run **batch 1** —
every matmul is a matvec, one token at a time — so the tiles cannot be filled and there is
nothing to win. This is consistent with llama.cpp reporting repack gains on *prompt
processing* and not on decode. Mimi additionally got 27 % slower, presumably locality or
dispatch differences from living in an extra buffer type.

**Consequence for what to do next:** the only way to make wide kernels (and any GEMM-shaped
optimization) pay here is to stop being batch 1. That is exactly what the text-stream
speculation above buys — it turns two audio frames into one batch-2 pass. The two ideas are
synergistic rather than alternatives, and batching is the prerequisite.

Two ggml bugs were found and fixed while getting this far, and they are worth upstreaming
independently of the negative result:

1. `ggml_backend_cpu_repack_buffer_set_tensor` dereferenced `tensor->extra` unconditionally,
   but `init_tensor` sets it to null for every type the buffer cannot repack (F32, F16, ...).
   So a weight context mixing quantized matmuls with F32 norms segfaulted on the first
   non-repackable tensor. llama.cpp never hits this because its loader assigns a buffer type
   per tensor; any embedder allocating a whole context at once does. Fixed with a plain
   memcpy fallback — correct because `get_alloc_size` is null for this buft (so allocation is
   `ggml_nbytes` either way) and mul_mat dispatch already keys off `extra`.
2. `iface.get_tensor` was left null, so any `ggml_backend_tensor_get` on a tensor in this
   buffer jumped to address 0. Implemented symmetrically: memcpy for un-repacked tensors,
   and an assert for repacked ones, whose layout genuinely cannot be read back.

---

## Batching is nearly free on this CPU — measured, and it redirects the strategy

`tools/bench_batch.cpp` times the LM's real weight shapes in isolation from moshi.cpp, to
test the load-bearing assumption behind speculation before building any of it.

Device (SM8850, cooled, q4_K, 6 threads, 200 iters), one transformer layer:

| shape | batch1 | batch2 | batch4 | b2/b1 |
|---|---|---|---|---|
| `self_attn.in_proj [2048x6144]` | 0.835 | 0.937 | 0.966 | 1.12 |
| `self_attn.out_proj [2048x2048]` | 0.359 | 0.320 | 0.507 | 0.89 |
| `gating.linear_in [2048x8448]` | **1.293** | **0.788** | 0.838 | **0.61** |
| `gating.linear_out [4224x2048]` | 0.482 | 0.548 | 0.608 | 1.14 |
| **TOTAL** | **2.968** | **2.593** | **2.920** | **0.87** |

**Four frames cost the same as one.** And batch 2 is *cheaper in absolute terms* than batch 1
— `gating.linear_in` is 40 % faster at batch 2 for identical weights. So this is not only the
weight-load amortization predicted earlier: at batch 1 ggml takes a **matvec** path, and at
batch >= 2 it switches to a properly vectorized matmul kernel. The earlier ~28 % estimate was
too conservative.

### Consequence: batch-4 speculation with prefix acceptance, not batch-2

Speculate the next N-1 text tokens as padding, run one batch-N pass, then accept the longest
correct prefix. Frame t is *always* correct — the mask is causal, so position t cannot see the
speculated positions. Frame t+k is correct iff every output before it was padding.

    E[accepted frames per pass] = 1 + P(pad) + P(2 pad) + P(3 pad)
                                = 1 + 0.554 + 0.405 + 0.293 = 2.252
    cost/frame = 2.920 / 2.252 = 1.297 ms   vs 2.968 unbatched   =>  -56%

Batch 2 by comparison: 0.554x2.593 + 0.446x(2.593+2.968) = 3.917 ms per 2 frames vs 5.936,
i.e. **-34 %**. Batch 4 wins because the pass is nearly free while acceptance decays slowly —
padding runs average 3.70 frames.

Projected: LM matmul time -56 % => device RTF 0.702 -> ~0.45, in-app 0.92 -> ~0.60.

### Why rollback is nearly free (the design insight)

The autoregressive dependency is a single integer: `state->cache[position][0] = text_token`
(`lm.h:937`) is the only thing frame t+1 inherits from frame t. And the KV cache is a ring
written by `ggml_set_rows` at `(offset+i) % capacity`. So rejecting speculated frames means
**not advancing `state->offset`** — the next real pass overwrites those slots. No undo, no
copy, no bookkeeping.

### What remains (not implemented)

1. A second graph built for T=N. The embedding path is structurally T=1: each codebook is a
   `get_rows` with one scalar index (`moshi_lmmodel_text_token_embed_step`, `lm.h:586`), summed
   into one `[dim]` vector. It needs N indices producing `[dim, N]`.
2. Per-T bias patterns. `create_bias_pattern` stores the pattern on the *model*
   (`moshi_streaming_transformer_t::pattern`), and `own_ctx_tensor::new_tensor` frees the
   previous one — so a T=1 and a T=N graph coexisting would free each other's mask. This is a
   real lifetime bug that only shows up once a second T exists.
3. The speculate / verify / accept-prefix loop, and advancing `offset` by the accepted count.

Estimated a few hours of careful surgery in a codebase that has already yielded two lifetime
bugs. The premise and the design are validated; the implementation is not started.

---

## Speculative batched decoding: implemented, correct, and NOT faster on ARM

`STT_SPEC_N=k` runs k audio frames per LM pass, speculating that every text token after the
first is padding, and keeps the longest prefix that held. Default 1 (off).

**Correctness: it works.** Transcript byte-identical to sequential decoding at k = 1, 2, 3, 4,
6, 8 on the 30 s French fixture. That is a strong validation of the whole batched path.

**Speed: no.**

Host x86 (same session):

| k | LM ms | frames/pass | chars |
|---|---|---|---|
| 1 | 10831 | 1.00 | 583 |
| **2** | **9596** | 1.26 | 583 |
| 3 | 9999 | 1.42 | 583 |
| 4 | 10676 | 1.46 | 583 |
| 6 | 11924 | 1.55 | 583 |
| 8 | 13329 | 1.56 | 583 |

Device (SM8850), interleaved with cooldown before every run:

| k | LM ms (run 1) | LM ms (run 2) |
|---|---|---|
| 1 | 14721 | 14583 |
| 2 | 14523 | 16342 |

So -11 % on host at k=2, and **nothing measurable on ARM** — within a ±15 % run-to-run
spread that persists even with cooldown.

### Why the earlier -56 % projection was wrong, twice over

1. **Statistical error.** The projection used marginal padding probabilities
   (1 + 0.554 + 0.405 + 0.293 = 2.25 frames/pass). But a pass *stops* precisely because a
   token was not padding, so the next pass systematically starts right after a non-padding
   token — where the conditional probability of padding is well below the unconditional
   55 %. Measured acceptance at k=4 is **1.46**, not 2.25.
2. **The microbenchmark measured the wrong thing.** `bench_batch` timed matmuls in isolation,
   where 4 positions genuinely cost the same as 1. The real pass also does attention over
   750 KV positions **per query**, which grows linearly with k and turns compute-bound as k
   rises. That is what makes k >= 3 actively worse.

On ARM specifically, the q4_K matvec path is already well optimized (dotprod/i8mm), so the
matvec→matmul kernel switch that gave 40 % on `gating.linear_in` in isolation does not survive
integration. Third time in this work that an isolated measurement overpredicted the integrated
result.

Kept rather than reverted, unlike the repack attempt: this one is *correct* and validated, the
per-T infrastructure is needed for any future prefill path, and if `context` were reduced from
750 the attention term would shrink and the trade could turn. Default off.

### Four latent T=1-only bugs found by building it

All invisible while the LM only ever ran one position, and all fatal to anyone attempting a
prefill or batched path:

1. **`moshi_rms_norm` had its `ggml_mul` operands reversed.** `ggml_mul(alpha, y)` asserts
   `ggml_can_repeat(y, alpha)`, i.e. only the *second* operand may broadcast. alpha is [dim,1]
   and y is [dim,T], so `1 % T` fails for every T > 1. Worked at T=1 by coincidence.
2. **The gating half-views were unreadable by `ggml_silu` at T > 1.** They were 4-D views
   parking T in dim 2 with correct strides, but silu walks `ggml_nrows` rows using `nb[1]`
   alone, so it read position p's second half as position p+1's first half. Replaced with
   plain 2-D views made contiguous.
3. **The token cache held only 2 time slots.** `cache_capacity = max_delay + 2`, and this model
   has all-zero delays, so a batch of 4 aliased frames 2 and 3 onto 0 and 1 — the cause of the
   all-padding output at k=4 that took a while to find. Now sized for `MOSHI_MAX_SPEC_N + 1`.
4. **Masks and graph state were single-slot on the model.** Fixed in the preceding commit.

---

## Adreno 840 via OpenCL: the LM runs 1.64x faster on the GPU

The CPU was only achieving **12.5 GB/s** of a bus worth roughly 77 GB/s (472 MB of matmul
weights per frame at 37.8 ms). So the bottleneck was never the memory bus — it was the CPU's
ability to saturate it. A GPU is built for exactly that.

Device (SM8850 / Adreno 840, cooled, **q4_0**, one 30 s fixture):

| config | Mimi ms | LM ms | RTF | chars |
|---|---|---|---|---|
| CPU, 6 threads | 6463 | 15222 | 0.723 | 573 |
| **GPU, 6 threads** | 10251 | **9260** | **0.650** | 573 |
| GPU, 4 threads | 11117 | 9313 | 0.681 | 573 |
| GPU, 2 threads | 15674 | 9664 | 0.845 | 573 |

**LM: 15222 → 9260 ms, 1.64x.** Transcript identical in every config, so the Adreno path is
numerically correct.

Net RTF is only 0.723 → 0.650 (10 %) because **Mimi gets 1.59x slower on the CPU whenever the
GPU is running** (6463 → 10251 ms). Lowering the thread count makes it worse, so this is not
thread oversubscription: ggml's OpenCL backend burns CPU waiting on GPU completion. That
contention is now the limiter, not the LM.

### Three things that had to be fixed to get here

1. **The codec must stay on the CPU.** Mimi's SEANet stack uses **ELU**, which ggml's OpenCL
   backend does not implement, and moshi runs a whole graph on one backend with no scheduler
   to split it — so offloading the codec fails outright with "op not supported (UNARY)". Added
   `moshi_context_t::backend_codec` (CPU) alongside the LM's. Safe because the two never share
   tensors: `mimi_encode_receive` copies the RVQ codes out to host ints.
2. **Q4_K crashes the Adreno driver.** `clSetKernelArg` segfaults inside `libCB.so` on a plain
   Q4_K matmul. Q4_0, Q8_0 and F16 all work. The Adreno kernels are documented as tuned for
   Q4_0, so this is consistent — and we already had a Q4_0 build measured at parity with Q4_K
   on CPU. Isolated matmul throughput, one layer, batch 1: **CPU q4_K 2.548 ms, GPU q4_0
   1.324 ms**.
3. **ARGMAX is not implemented in the OpenCL backend.** Greedy sampling now argmaxes on the
   host: one ~32 KB readback per frame against ~38 ms of compute, so free either way, and it
   removes a graph node. `moshi_argmax_host`.

Note also that on GPU, **batch 1 is the sweet spot** — batch 2 costs 2.41x batch 1 there,
the opposite of the CPU. Which suits this workload, since it is naturally batch 1.

### The next lever, and it is well defined

Implementing **ELU in ggml's OpenCL backend** would let the codec run on the GPU too, removing
the CPU contention that is currently eating two thirds of the LM's gain. ELU is
`x > 0 ? x : alpha*(expm1(x))`, and the backend already has GELU / SILU / EXPM1 kernels to
copy from, plus a `supports_op` case to add. If Mimi went to the GPU at anything like the LM's
ratio, RTF would land near **0.35–0.40**.

### Correction: the GPU's real margin is ~9%, and the CPU noise floor is +/-7%

The table above compares q4_0 against q4_0, but the app runs **q4_K**. Measured in one session
with a cooldown before each run:

| config | LM ms | RTF | chars |
|---|---|---|---|
| CPU q4_K | 15236 | 0.734 | 583 |
| CPU q4_0 | 15377 | 0.720 | 573 |
| **GPU q4_0** | **9513** | **0.627** | 573 |
| CPU q4_K (repeat) | 14537 | **0.686** | 583 |

So against the *best* CPU run (0.686, consistent with the 0.685 measured earlier), the GPU is
**~9 %** better, not the 10 % implied by comparing to a q4_0 CPU baseline.

More important: **CPU q4_K varies 0.686–0.734 run to run** — same binary, same weights, cooled
each time. That +/-7 % spread is comparable to the GPU's whole advantage. The LM figure itself
is unambiguous (14537 -> 9513 ms, **1.53x**); it is the *total* RTF gain that drowns in the CPU
contention the OpenCL driver creates.

Also retracting an earlier claim: "q4_K is faster than q4_0 on CPU" rested on 41.00 vs
42.21 ms from the quantization session — a 3 % difference, i.e. below this noise floor. They are
equivalent on CPU. The real differences are that q4_K yields 583 characters against 573, and
that q4_K crashes the Adreno driver.

---

## The best CPU win of all, and it is one config field: narrow the attention window

`context` is the attention window in frames at 12.5 Hz, shipped at **750 = 60 s**. Attention
reads the whole window every frame: `750 x 128 x 16 heads x 2 (K,V) x 2 bytes x 16 layers`
= **98 MB/frame, ~20 % of all traffic**. Halving it is nearly free in quality and large in speed.

Device (SM8850, cooled, q4_K, interleaved):

| context | LM ms | RTF (30 s) | chars |
|---|---|---|---|
| 750 | 15344 | 0.734 | 583 |
| **375** | **10282** | **0.555** | 569 |

**-32 % on the LM, -24 % RTF.** And on 90 s of continuous speech, context=375 yields **1735
characters — exactly the same as 750**, so no coverage is lost. On host, word error over the
90 s fixture moves 10.32 % -> 10.91 %: two words out of 339.

375 is also markedly more *stable*: two runs gave 10282 and 10536 ms (2.5 % apart) where 750
gave 15344 and 28147. Less memory traffic, less thermal and contention sensitivity.

**250 is a trap**: RTF 0.369 on host looks better still, but word error jumps to 14.45 %
(+4 points). That is a real loss. 375 is the knee.

Why this is safe for Katarina specifically: the model needs this window only for linguistic
continuity. The user's questions are answered from the app's own 3000-character rolling text
buffer, not from the model's acoustic memory. The KV cache halves as a bonus (98 -> 49 MB).

Mask correctness re-verified by simulation at capacity 250, 375 and 750 for t=1 and t=2 — the
wrapped-branch fix holds at every window size, not just the one it was found at.

**This beats the GPU while staying on the CPU**: 0.555 against the Adreno's 0.627. Deployed by
`deploy_kyutai_model.sh` (override with `CONTEXT=`).

### Where that leaves the ranking of remaining ideas

Measured or reasoned, best ratio first:

1. ~~Narrow the context~~ — **done, -24 %.**
2. **ELU in ggml's OpenCL backend**, which would move Mimi to the GPU and remove the CPU
   contention that ate two thirds of the LM's 1.53x GPU gain. Bounded, no ML risk, upstreamable.
3. **Early exit on padding frames.** 55 % of frames emit padding; a probe on layer-4
   activations predicting that would skip 12 of 16 layers on those frames (~-41 % in the limit).
   Needs a trained probe, and skipping layers leaves their KV unwritten for that position, which
   the literature handles by propagating state — an approximation with real degradation.
4. Q3_K / IQ4_XS: ~-25 % bytes, needs an audio-representative imatrix nobody has published.

### Two algebraic ideas that do NOT work here, with the reason

- **Low-rank factorization of the weights.** W[k x n] -> U[k x r]V[r x n] saves bytes only when
  `r < kn/(k+n)`. For `gating.linear_in` [2048 x 8448] that threshold is **r = 1648 against a
  maximum rank of 2048** — a 20 % rank truncation just to break even, on FFN weights that are
  famously near full rank. The matrix shape is wrong for it.
- **O(1) incremental attention.** Tempting, since only one position enters and one leaves the
  window per frame, so running accumulators for `sum exp(s_j) V_j` and `sum exp(s_j)` look
  updatable. They are not: `s_j = q_t . k_j` depends on the *current* query, so every `exp(s_j)`
  changes each frame. This is exactly why softmax attention is irreducibly O(context) per token;
  only linear attention gives a recurrent state, and that means retraining.

---

## The codec was never quantized: F32 -> F16 is free, Q4_K is not

The requantizer was only ever pointed at the LM. The Mimi GGUF ships **310 MB of F32 out of
347**: its transformer (33.6 M params), its 32 RVQ codebooks (16.8 M — read by the matvec search
on every frame), and more.

Converting the codec to F16 (347 -> 192 MB):

| codec | Mimi ms (device) | RTF | word error (90 s) |
|---|---|---|---|
| F32 (shipped) | 5586 | 0.512 | 10.91 % |
| **F16** | **4858** | **0.482** | **10.32 %** |
| Q4_K | 4445 | 0.460 | **15.93 %** |

F16 is a pure win: 13–20 % faster and word error *improves* slightly. Peak RSS drops
**1352 -> 1087 MB**.

**Q4_K on the codec must be rejected** despite being faster still, and the reason is structural
rather than incidental: **the codec feeds a hard argmax cascade** — 32 residual stages each
picking among 2048 centroids, where a small encoder error flips a code and the residual carries
the mistake into every later stage — while **the LM feeds a soft distribution** (softmax over
8000 tokens, where small logit perturbations rarely change the argmax). Quantize the LM
aggressively; keep the codec at F16.

### ⚠️ Methodological warning about device RTF on this phone

While measuring the above, three consecutive runs of the *identical* configuration, each started
below 45 °C, gave RTF **0.532, 0.724, 1.244** — monotonically worse. Investigated:

- the real per-core sensors (`cpu-0-*`, `cpu-1-*`) read 37–38 °C, so the cores were genuinely
  cool and `thermal_zone0` (`cpullc`) was not misleading me;
- `cpu-hw-trip-*` reporting 95 °C is a trip *threshold*, not a reading;
- memory was not tight (6.3 GB available, 4.8 GB cached).

Most likely the platform demoting a long-running `adb shell` process into a restricted cpuset
(cpu0/cpu4 observed at 384 MHz of 3628 while cpu7 was at 883 of 4608). **Consequence: absolute
device RTF from a long benchmarking session is not trustworthy, and only interleaved A/Bs taken
close together are.** The quality figures above are host-side and unaffected. Treat every
absolute device RTF in this file as a lower bound with a wide error bar.

---

# The app produced no text while the engine was fine (2026-07-29)

Whole in-app sessions transcribed **nothing** — one ran 11 minutes at RTF 0.74 and emitted zero
characters — while audio levels looked healthy. I first explained this away as instrumental
music. That was wrong, and the correction is the most useful thing in this section.

## Grouping the log by pid settled it in one command

| pid | audio captured | text emitted |
|---|---|---|
| 15638 | 11:23:30 → 11:34:45 (**11 min**) | 0 |
| 16385 | 70 s | 0 |
| 25869 | 55 s | 0 |
| 26975 | 78 s | **46 outputs** |

Eleven minutes of continuous audio at rms ~5000 is not instrumental. And pid 15638 was healthy
at RTF 0.74 for its first 75 chunks and *still* emitted nothing, so it was never a speed problem.
It then collapsed to **RTF 4.54 with 383 dropped chunks**.

Having a fixture that reproduced the *benign* explanation (`test_music` → 0 chars) made me more
confident, not less. That is exactly backwards: a hypothesis that explains the symptom is not
evidence for itself.

## The engine is innocent — and deterministic

The app now dumps the PCM it feeds. Replaying that dump (30 s of real captured audio) through
`stt_bench`, five times:

```
chars=469  rtf=1.210   <- first run, phone warm from a previous session
chars=469  rtf=0.521
chars=469  rtf=0.517
chars=469  rtf=0.523
chars=469  rtf=0.528
```

Byte-identical transcripts, 5/5. So the model, the weights, the quantization and the JNI feed
loop are all fine on exactly the bytes the app captured.

## Refuted: "the LM stays wedged across sessions"

The app calls `streamFlush()` (which pushes ~0.5 s of silence) at session end and then
`streamReset()` (codec only, LM context deliberately kept) at the next session start. Plausible
attractor: the LM conditions on its own emitted text tokens, so a run of padding could
self-reinforce and never recover — matching "never recovers".

`STT_SESSIONS=4` replays the audio through one context with exactly that sequence:

```
chars per session: 469 480 480 480     rtf=0.555 over 120 s
```

**Refuted.** No degradation, and RTF is flat over 4 sessions — so the engine also does not drift
over minutes, which matters for the collapse below.

## Root cause: the app was losing captured audio, unreported

`AudioRecord`'s ring buffer was `getMinBufferSize()` — **1920 bytes, 40 ms at 24 kHz**. The
reader thread has that long to come back before AudioRecord overwrites unread samples, and it
was doing far more than 40 ms of work per chunk: a 24000-sample byte conversion, a second
24000-sample pass for RMS, and a **blocking file write** for the PCM dump — which, when it
failed with EACCES, logged a 25-line stack trace *every second*. The diagnostic was causing the
loss it existed to diagnose.

AudioRecord never reports this. So the stream fed to the model had holes in it while the levels
still looked fine — and for a model with per-frame convolution and KV state, a spliced stream is
state corruption, not a small gap. That is the mechanism, and it explains every feature of the
symptom: intermittent (scheduling luck), levels healthy, never recovers, worse under load.

Fixes: ring ≥ 2 s (96 kB); ring size split from read granularity (one variable served both, so
the ring could not be grown without making each read block for a second); dump moved to the
inference thread; a dropped chunk now triggers a codec reset, because the next chunk is not
contiguous with the last.

And the instrument that was missing — the reader compares samples collected against the wall
clock (the source runs at exactly `captureSampleRate`, so any shortfall is audio that no longer
exists) and logs `capture LOST <n>ms`. Plus a JNI health line every 10 s: frames, share emitting
a real token, post-resample RMS, max VAD, which separates "handed silence" from "LM only emits
padding" from "detokenizer drops everything".

## Still open: the in-app RTF collapse

pid 15638 went from RTF 0.74 to **4.54** with 383 dropped chunks after ~98 s. The engine is flat
at 0.555 over 120 s and 4 sessions, so this is contention, thermal, or cpuset demotion — not the
model. Unexplained, and not claimed as fixed by the above.

---

# The quantization lever is exhausted, at both ends (2026-07-29)

Established earlier that this workload is bandwidth-bound on its own weights at batch 1
(Q8_0 predicted 1.89x the bytes of Q4_K and measured 1.90x the time). The obvious follow-up
is fewer bits. It does not work, in either the LM or the codec, and both failures are
measured rather than argued.

## LM below Q4_K: slower despite fewer bytes

Added `q3_k` and `iq4_xs` to `tools/requantize_gguf`. On the 69 matmul tensors — the only ones
whose size matters, since the 33 embedding tables are `get_rows` at one row per frame — Q4_K is
4.5 bpw and Q3_K is 3.44, so **24 % less traffic**.

Host, `test_16k`, 3 interleaved rounds (medians):

| weights | file | host rtf | WER vs f16 |
|---|---|---|---|
| q4_k | 737 MB | **0.605** | 0.00 % |
| iq4_xs | 712 MB | 0.677 | 1.00 % |
| q3_k | 631 MB | 0.595 | 1.00 % |

Device, `test_16k`, 3 interleaved rounds with a <45 °C cooldown before each, comparing `lm_ms`
(Mimi is unchanged, so this isolates the effect):

| weights | run 1 | run 2 | run 3 | median |
|---|---|---|---|---|
| q4_k | *28973* ⚠ | 10493 | 10880 | **10687 ms** |
| q3_k | 12589 | 12249 | 16284 | **12589 ms** |

⚠ first run is a cold-start outlier (`mimi_ms` 14858 vs ~5600 in every other run — the model
is being faulted in from storage).

**Q3_K is 16 % slower on device** while reading 24 % fewer bytes, and its best run (12249) is
worse than the worst valid Q4_K run (10880). ARM's i8mm path for Q4_K beats Q3_K's, and the
extra unpacking work costs more than the traffic saved. IQ4_XS is worse still, ~10 % slower on
host, presumably for the same reason plus its own search.

**Q4_K is the optimum on this hardware.** Bits-per-weight is only a proxy for time while the
dequantization kernel stays as cheap; below 4.5 bpw it does not.

## Codec below F16: even Q8_0 is a bad trade

`STT_MIMI` was added to `stt_bench` so codec variants can be A/B'd without rewriting
config.json. Mimi is ~35 % of per-frame compute, so it was worth sweeping separately.

Host, 3 interleaved rounds, LM fixed at q4_k. `mimi_ms` medians: F16 4196, Q8_0 4005,
Q6_K 3990 — about **5 % of Mimi, i.e. 1.7 % of total**. Against that:

| codec | WER vs F16 (test_16k) | WER vs F16 (90 s) |
|---|---|---|
| Q8_0 | **3.03 %** | **2.01 %** |
| Q6_K | 6.06 % | 6.02 % |

Q8_0 is near-lossless on an ordinary transformer. Costing 2–3 % here **confirms the structural
asymmetry** already recorded for Q4_K, and with a much gentler quantization: the codec feeds a
hard argmax cascade (32 residual stages over 2048 centroids, where one flipped code propagates
through every later stage), while the LM feeds a softmax over 8000 tokens that absorbs noise.
1.7 % of speed for 2 % of words is not a trade worth making.

**Keep the LM at Q4_K and the codec at F16.** Both are now measured optima, not defaults.

## Two notes on the host harness

- `config.json` in the repo points at `../moshi-common/mimi-e351c8d8-125.gguf` — the **F32**
  codec — and sets `context: 750`. The device gets `mimi-f16.gguf` and `context: 375` because
  `deploy_kyutai_model.sh` rewrites both. So host and device absolute RTF are not comparable;
  only within-sweep comparisons are, and every sweep above holds its conditions fixed across
  arms.
- `tools/quant_sweep.sh` scores each variant's transcript against the **F16 transcript of the
  same model**, not a human reference. That isolates quantization damage; a human reference
  folds in the model's own errors and hides it.

## What this closes

Reducing weight traffic was the last cheap lever on CPU. It is now measured shut. The binding
constraint is thermal: the platform caps `scaling_max_freq` from 3628 MHz to 1440 MHz once the
phone passes ~65 °C, and RTF goes from 0.55 to over 1. No arrangement of bits changes the
thermal budget — only doing less total work would, and the per-frame work is inherent (see the
note on why padding frames cannot be skipped).

## Why "early exit on padding frames" is not available

Worth recording because it looks like a large win and is not one. Roughly 65 % of frames emit
padding rather than a text token, which invites skipping their LM pass. It cannot be done: the
model is autoregressive at a fixed rate, every frame must advance the LM's KV cache and Mimi's
convolution ring buffers, and **it is the forward pass itself that determines the output is
padding**. The 65 % is not waste, it is the cost of maintaining state. Skipping a frame
corrupts everything after it. What remains is not computing the 8001-wide output head on
padding frames, which is ~2 % of traffic.
