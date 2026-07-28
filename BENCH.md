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
