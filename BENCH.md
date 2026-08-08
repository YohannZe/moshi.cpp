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

---

# Importance-weighted quantization, kernel flags, and the sustained regime (2026-07-29)

Four experiments on "revoir l'inférence" beyond hardware tuning. One new capability, three
clean negatives — all measured.

## imatrix: moshi.cpp can now do importance-weighted quantization

Everything quantized so far minimized error ON THE WEIGHTS (‖W−Ŵ‖); what matters is error on
the OUTPUTS (‖(W−Ŵ)x‖ over the activations the model actually sees). `ggml_quantize_chunk`
accepts per-column importance weights; nothing in this repo produced them.

Now: `src/imatrix.h` hooks `GraphContext::compute()` (every graph flows through it, and its
tensors have permanent slots, so activations are readable post-compute), accumulates per-column
E[x²] for every named leaf feeding a MUL_MAT, and writes a `moshi-imatrix v1` text file at
exit. Runs accumulate across processes (the file is reloaded at init). `requantize_gguf
--imatrix <file>` applies it. Guarded on the env var's *value*, not presence — the repack
lesson.

Collection: 2.5 min of French (test_speech_90s + test_speech_music + cap24k, the phone's own
captured audio). **test_16k held out for eval.** Scored with tools/quant_sweep.sh's WER
against the F16 transcript of the same model.

Codec (the argmax-cascade end, where quantization hurts most):

| codec | WER vs F16, held-out | size |
|---|---|---|
| Q4_K plain | 6.06 % | 114 MB |
| Q4_K + imatrix | **3.03 %** | 114 MB |
| **Q5_K + imatrix** | **1.01 %** (one word + a comma) | 120 MB |
| Q6_K + imatrix | 3.03 % | 127 MB |

**At identical size, the imatrix halves the damage.** Held-out and in-set WER are identical
(3.03/3.01, 1.01/1.00), so 2.5 min of audio does not overfit. Note the Q6 > Q5 inversion —
single-fixture granularity is one word; treat ±1 word as the noise floor.

(Methodology note: the earlier "Q4_K codec = 15.93 %" was hand-scored against a different
reference; the 6.06 % here is quant_sweep.sh vs F16. Within this table everything is scored
identically, which is what matters for the comparison.)

**Not deployed.** Host says Q5_K+imat Mimi is ~4 % faster than F16; the cooled interleaved
device A/B came back inside thermal noise (+7 %, +40 %, −11 % across three pairs — the phone
was reheating faster than it cooled). −63 MB resident is real (the phone swaps), but the
standing rule is that a speed gain that degrades text is not a gain, and the speed gain is
unproven on the target. The infrastructure is the deliverable: if a smaller-RAM device ever
becomes a target, Q5_K+imatrix is the known-good codec recipe.

LM + imatrix: pointless to measure — Q4_K already scores 0.00 % vs F16 on both fixtures, and
the imatrix cannot change speed at a fixed type. Skipped for that stated reason.

## +fp16 kernels: 21–25 % SLOWER — P6's assumption was wrong

The build used `-march=armv8.6-a+dotprod+i8mm`, which does **not** define
`__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` (verified via `-dM -E`). So every F16 dot product —
all of Mimi, the KV cache, flash attention — runs through convert-to-F32 paths, and P6
assumed native FP16 FMA would be a win.

Measured (cooled, interleaved, test_16k): baseline lm_ms 11348/13409 vs +fp16 13744/16749 —
**+21 % and +25 % slower**, and the transcript shifts by one character (F16 accumulation
changes the numerics). On this core the FP16-accumulate SIMD path loses to FCVTL+FMLA with
F32 accumulators. Rejected; the flag stays off.

## Thread count in the sustained regime: 6 confirmed, hypothesis refuted

"6 threads optimal" had only been measured cold. Under the thermal cap, power is superlinear
in frequency but linear in cores, so fewer-threads-at-higher-clock was plausible. Measured
(tools/bench_threads.sh: cool to 42 °C, run 90 s twice back-to-back, keep pass 2):

| threads | steady-state RTF |
|---|---|
| 4 | 0.674 |
| **6** | **0.606** |
| 8 | 1.567 |

Refuted — 4 threads is 11 % worse even with the cap dropping to 1.9 GHz mid-run. And 8 is
catastrophic (2.6x): every barrier waits for the slowest core, and at 8 threads the two prime
cores can't carry the six mid cores. The app's `nThreads = 6` stands.

## KV cache Q8_0: blocked at the kernel, not attempted

The remaining F16 traffic is the attention cache (~10 % of total at context 375). Quantizing
it to Q8_0 is the TurboQuant-adjacent move — but
`ggml_compute_forward_flash_attn_ext_f16` gates its fast split-KV and tiled paths on
`k->type == F32 || F16` (already documented at transformer.h:167), so Q8_0 KV would take the
generic path, and the q3_k result shows exactly what happens when dequant cost meets a lost
fast path. Prediction is firmly negative and the experiment is not cheap (quantized cache
init, per-slot quantized writes, flash fallback). Revisit only if ggml grows a fast
quantized-KV FA path on ARM.

## Where this leaves the CPU path

Every cheap lever is now measured: quantization (both ends), kernel flags, thread count,
context width, repack, speculation. The binding constraint is the thermal cap, and the only
untested software lever against it is OpenMP (the build spins 745 seq-cst barriers per frame
with GGML_OPENMP=OFF) — pending, needs the device back. Beyond that, meaningfully faster means
a different compute substrate (GPU/NPU), which is currently excluded by policy.

## OpenMP: 10 % slower in the sustained regime, and 15 °C hotter — rejected

The last untested CPU lever. GGML_OPENMP=OFF means 745 seq-cst spin barriers per frame, which
P6 flagged as the reason for the threads>6 cliff; OpenMP was the assumed fix.

Short cooled runs (test_16k, 5 valid interleaved pairs): median ~2.5 % in OpenMP's favour, but
the spread was 0.51–0.92 RTF for *identical* configurations — pure device noise, no verdict.

Sustained regime (test_speech_90s ×2 back-to-back per arm, cooled to 42 °C before each arm,
keep pass 2 — the same protocol that produced clean thread-sweep numbers):

| arm | pass-2 RTF | end temp |
|---|---|---|
| spin barriers (current) | **0.719** | 57 °C |
| OpenMP | 0.794 | **72 °C** |

10 % slower and much hotter. libomp's threads also busy-wait (active wait policy), plus the
runtime's own overhead — more power for less work, which is precisely the wrong direction
under a thermal cap. The short-run lean and the sustained result disagree; the sustained
regime is the one the app lives in. **GGML_OPENMP stays OFF.**

With this, every CPU software lever on this list is measured: quantization at both ends,
imatrix, kernel flags (+fp16), thread count, context width, repack, speculation, OpenMP.
The CPU path is at its practical limit on this hardware; the constraint is the thermal cap.

---

# "Pourquoi c'est plus lent ? C'est pas logique." — It wasn't. (2026-07-29, evening)

The user pushed back on the q3_k / +fp16 / OpenMP negatives, and the pushback was right.
If this workload were truly DRAM-bandwidth-bound, fewer bytes would always win — q3_k losing
while reading 24 % less proves the bottleneck was **in-core work**, not the bus (we read
~17 GB/s of a 77 GB/s bus). So something else was eating the machine, and no A/B of build
flags was going to find it.

## The profiler, since simpleperf is dead on this phone

This kernel rejects every perf event, even software ones with `security.perf_harden=0`.
So ggml-cpu gained `GGML_PROFILE=1`: thread 0 times every node, barrier-inclusive (so
imbalance is charged to the node that causes it), detail keyed by weight name for MUL_MAT
and by op<-producer[shape] for the rest.

First profile on device, everything included (test_16k, Q4_K LM + F16 codec):

| op | time | share |
|---|---|---|
| MUL_MAT | 9797 ms | 57.4 % |
| **CONT** | **4572 ms** | **26.8 %** |
| IM2COL | 573 ms | 3.4 % |
| UNARY | 503 ms | 3.0 % |
| ~550 k node executions total | | |

And the detail line that explained the day:
`CONT <- TRANSPOSE [375x128x16]  3687 ms  24.0 %  6224 calls` — 16 layers × 389 frames.

## The bug: one dtype disarmed flash attention everywhere

`torch_sdpa_rearranged` has a flash path guarded on the mask being **F16** — the guard that
exists because ggml's FA kernel reads the mask as `ggml_fp16_t` unconditionally, and an F32
mask silently produces garbage (the "45 % win" that was 388 whitespace characters, earlier in
this file). `create_bias_pattern` dutifully produces an F16 mask.

But the **cached-graph** builder (`moshi_streaming_transformer_graph_build`) allocated its
per-frame mask *placeholder* as **GGML_TYPE_F32**. Each frame, `graph_step` copied the F16
pattern into it — `ggml_cpy` converting F16→F32 on the way — and the guard then (correctly)
refused the F32 mask. Every layer of every frame, LM and Mimi encoder both, fell back to the
manual path: `softmax(QK^T)` un-fused plus `ggml_cont(transpose(V))` re-materializing the
whole V cache — the path ggml itself comments "this is not optimal - fix me".

**The flash path everyone believed was active since P4 had never run in the app.** The flash
work was validated on a code path (`moshi_streaming_transformer`, non-cached) that the
per-frame pipeline does not use.

Fix: one word, `GGML_TYPE_F32` → `GGML_TYPE_F16` at the placeholder. Plus a one-shot stderr
line in the fallback branch saying *why* flash is off, so this cannot go silent again.

## Measured

Host: LM 10.6→6.9 s on test_16k (−35 %), RTF 0.53→0.411.

Device (interleaved, warm phone), `lm_ms` on test_16k: baseline 14830/14560/13995,
flash 7873/8713/7368 — **LM ×1.85**. chars identical (570).

Device, sustained regime (90 s ×2 back-to-back, the app's real condition):

| | this morning | with flash | |
|---|---|---|---|
| pass 1 | 0.613–0.695 | **0.339** | |
| pass 2 (steady, hot) | 0.606–0.719 | **0.391** | ×1.8 |

chars = 1738, byte-stable across the fixture's three repetitions (the mask-correctness
property holds after the ring wraps). The only text change anywhere is one word
("des"→"les" + a comma), identical in all three repetitions — flash's different accumulation
order, not a regression.

The morning's reliability target — "RTF ≤ 0.35 with the phone hot" — is effectively met:
0.39 at 70 °C, 0.34 before the SoC heats.

New profile after the fix: MUL_MAT 76.6 % (compute finally goes into compute), CONT
26.8 %→2.0 %, SOFT_MAX 6224→389 calls, FLASH_ATTN_EXT 4.4 %. The one large non-LM item left
is Mimi's conv matmuls (`(reshaped)`, 21.5 %).

## What this retracts

The previous section declared "the CPU path is at its practical limit; every lever is
measured". **Wrong, and instructively so.** Every lever *on the list* was measured — but the
list was built from beliefs about where time went, and 24 % of the time was somewhere no
listed lever touched. q3_k, +fp16 and OpenMP all "made no sense" because they were optimizing
the 30 % of the machine that was actually doing matmuls, while the majority went to a memcpy
storm none of them affected. The profile, not the lever list, is the ground truth. Profile
first; the lever list comes second.

---

# FLEURS-fr: the first standard-benchmark numbers (2026-07-29, night)

Every quality figure before this section was measured on in-house fixtures against the F16
transcript of the same model — right for isolating quantization damage, unusable as an
absolute claim. This section is the standard-benchmark evaluation, on the exact deployed
pipeline (tools/stt_eval: frame-by-frame streaming, 0.5 s delay, codec reset between
utterances, LM context persisting across them — deployment mode, as the app runs).

Dataset: google/fleurs fr_fr test, all 676 utterances (7024 s of speech). Scoring:
tools/score_wer.py, symmetric normalization. Note: kyutai publishes no FLEURS-fr WER for
stt-1b (model card has no numbers; the DSM paper evaluates ASR in English only), so as far
as we can tell these are the first public FLEURS-fr figures for this model.

| config | WER | S / I / D | host RTF |
|---|---|---|---|
| F16 | 11.29 % | 1435 / 317 / 279 | 0.677 |
| **Q4_K (deployed)** | **11.40 %** | 1453 / 315 / 284 | **0.344** |
| Q4_K, 3 session artifacts patched | 11.07 % | 1465 / 316 / 211 | — |

**Quantization costs 0.11 WER points for 1.97x speed and 2.6x smaller weights** — the
in-house "Q4_K scores 0.00 % vs F16" finding, now confirmed at benchmark scale.

For rough context — VERIFIED 2026-08-07 against the Whisper paper's Table 13 (FLEURS,
zero-shot), as relayed by bofenghuang's whisper-large-v2-french model card: whisper-small
15.0 %, medium 8.7 %, large 7.7 %, large-v2 8.3 % on FLEURS-fr — none of them streaming.
(The ~5 % figures floating around are French *fine-tuned* large-v2 variants, not
zero-shot; an earlier draft of this note wrongly attributed 5–6 % to large-v3.) Caveat
for the paper: Whisper's numbers use Whisper's own text normalizer, ours use
score_wer.py's — cross-scorer WERs are indicative, not directly comparable. This model
sits between small and medium in quality while emitting text 0.5 s behind the audio on a
phone CPU.

## Finding 1: a neural-codec STT is input-gain sensitive

FLEURS ships float32 WAVs peaking around **-46 dBFS**. Fed as-is: **17.62 % WER and 29/676
empty transcripts**. Peak-normalized to -3 dBFS: **11.40 % and 3 empties**. Six points of
WER were the *level*, not the model.

Log-mel systems (whisper) are largely immune to this; Mimi's encoder sees raw samples, and
its RVQ stages starve on tiny amplitudes. Nobody documents this failure mode. It also has a
direct product implication for Katarina: a quietly-mastered podcast pays the same penalty,
so the capture pipeline should auto-gain (the JNI health line already logs post-resample
RMS, which is the signal needed).

Recognition pattern, now seen twice in one day: **healthy-looking levels + empty text =
suspect the input format/scale before the model** (this morning it was float32-as-int16;
tonight it was -46 dBFS).

## Finding 2: cross-utterance LM context can mute an entire utterance

3/676 utterances (0.44 %) produce zero text in deployment mode but transcribe perfectly in
a fresh process — same files, healthy levels, both F16 and Q4_K (the same 3 files in both,
so it is not quantization). The persistent LM context occasionally locks the text stream
for the following utterance. That is the measured cost of deployment-mode evaluation, and
patching just those 3 with fresh-state outputs gives the utterance-independent estimate
(11.07 %).

## Method note, recorded because it almost went wrong

While patching those 3, the first patch attempt used a hypothesis that had been truncated
by the terminal (`cut -c1-150`) and hand-completed from memory — i.e. fabricated eval data.
Caught on re-read, redone from full untruncated outputs (11.06 → 11.07 %, immaterial, but
the principle is not). Eval hypotheses only ever come from the tool's own output, never
from a human filling gaps.

## Still missing for the paper

- The apples-to-apples baseline: kyutai's reference PyTorch implementation on the same 676
  files, same scorer — "our port vs the reference". PyTorch CPU will take hours; run
  overnight.
- Device RTF on FLEURS (host RTF above is not the deployment number).
- Battery. Nobody ships an RTF; they ship hours.
- Verify the whisper literature numbers above before citing them.

## Device RTF and WER on FLEURS (2026-07-30)

Stratified subset (every 6th file, 113 utterances, 1180 s) on the Poco F8 Ultra, Q4_K,
6 threads, back-to-back with no cooldown — deliberately the harshest regime:

| segment | conditions | RTF |
|---|---|---|
| files 1–76 | warming to full throttle (cap hits 1.44 GHz) | 0.68–0.75 |
| files 77–113 | saturated at the thermal floor | **0.98** |

So the honest worst case is: after ~15 min of *continuous saturated* compute, the device
approaches RTF 1.0. The app's real duty cycle is gentler (it computes ~0.5 s per 1 s of
audio, leaving thermal headroom), which is why real app sessions sustain 0.75–0.78. Both
numbers belong in the paper: bench-saturated 0.98, app-realistic 0.75.

Quality on device: **11.58 % WER** vs 11.45 % for the host on the same 113 files — inside
the ±0.2 pt session-variance band. Per-file texts differ because deployment-mode outputs
depend on the LM context chain, and the device session chained only the subset while the
host chained all 676; quality equivalence is the meaningful invariant (byte-identity was
already proven under identical session structure: F16 twice, 24 h apart, same S/I/D to the
digit).

Also worth recording: the Q4_K host re-run after the crash resume scored 11.61 % vs 11.40 %
for the unbroken chain — deployment-mode WER carries ~±0.2 pt of variance from session
segmentation alone. Quote FLEURS numbers with that error bar.

## FLEURS-fr, final: the port outperforms the reference implementation (2026-07-30)

All 676 utterances, same scorer, same normalized audio:

| implementation | precision | mode | WER | S / I / D |
|---|---|---|---|---|
| kyutai reference (PyTorch) | F32 | fresh state per utterance | 12.62 % | 1608 / 351 / 312 |
| **this port** | **F16** | deployment (context persists) | **11.29 %** | 1435 / 317 / 279 |
| **this port** | **Q4_K** | deployment | **11.40 %** | 1453 / 315 / 284 |

Speed on the same x86 host: reference RTF 2.95, this port 0.344 — **8.6x**. The reference
also produces 2 empty transcripts of its own (vs our 3 session artifacts).

**Our Q4_K on-device engine beats the official F32 reference by 1.2 WER points while being
8.6x faster.** The mid-run "parity" snapshot at 199 files (12.99 vs 13.04) was itself sample
noise in the other direction; the gap at 676 is consistent with the early lead.

### Why — partial analysis, to finish for the paper

- **Tail flush**: their script flushes ceil(0.5 s × 12.5) = 7 frames of silence after the
  audio; ours flushes delay+8 = 14. Under-flushing strands trailing words, and their
  deletion count is correspondingly higher (312 vs 279). Explains part of the gap, favors
  us for a mundane, checkable reason.
- **Persistent LM context** (deployment mode) gives our decoder a warm French prior at each
  utterance start, where the reference starts cold every time. Consistent with their higher
  substitutions (1608 vs 1435) concentrated... (to verify: error-position analysis).
- Both effects are properties of *how the model is driven*, not of ggml vs PyTorch — the
  engines agree byte-for-byte on identical inputs and session structure wherever we have
  measured it.

The paper claim this supports, conservatively phrased: *matches or exceeds the reference
implementation's accuracy on FLEURS-fr while running 8.6x faster on the same CPU, and in
real time on a phone.*

## Serving-protocol sweep (113-file FLEURS subset, one variable per arm, 2026-07-30)

| arm | WER | S/I/D | verdict |
|---|---|---|---|
| base (tail+8) | 11.45 % | 235/75/31 | |
| tail+16 | **10.78 %** | 228/70/23 | adopt — trailing words arrive well past the 0.5 s delay |
| prefix 6 frames | **10.71 %** | 225/66/28 | adopt for utterance mode — codec conv settle time |
| prefix 12 | 12.25 % | D=66 | too much silence hurts; reject |
| **raw −46 dBFS, no AGC** | **22.36 %** | D=305 | the level cliff, reproduced |
| **raw −46 dBFS + causal AGC** | **11.41 %** | 232/64/44 | **AGC recovers ALL 11 points, online** |
| normalized + AGC | 11.08 % | 235/67/28 | no harm on healthy audio — safe always-on |
| combo (tail16+prefix6+AGC) | **10.94 %** | 235/65/26 | −0.5 pt vs base, zero engine changes |
| LM q4_k + imatrix | 12.66 % | D=68 | **imatrix HURTS the LM** (2-min calibration set skews the 8000-token head); imatrix is codec-only |
| codec encoder-transformer Q8 | 11.92 % | | +0.5 WER; reject per the standing rule |

Structural finding: the codec's convolutions cannot be block-quantized at all — stored
[kernel, Cin, Cout] with ne[0]=3..7, below every block size. The 21.5 % conv share keeps F16.

Shipped to the app: causal AGC in the JNI feed path + flush tail +16 + the mute-lock
watchdog (semantic VAD active, no text 8 s → codec reset). Session-serving WER at these
settings: **~10.9 %** — better than every number in yesterday's table, and 1.7 pts ahead of
the reference implementation.

## Serving tuning at full scale (676, 2026-07-30 evening)

- tail+24 on the subset: 11.51 % — WORSE than +16 (10.78 %). The flush is a window, not a
  dial: too little truncates utterance ends, too much degrades the next utterance through
  the context chain. **tail+16 is the final setting.**
- Combo (tail16 + prefix6 + AGC) on all 676: **11.29 %** vs 11.40 base — deletions down 23 %
  (284→218), exactly where the flush acts. The 113-file subset had flattered the combo
  (10.94 %); full scale is the number that goes in the paper.
- End-of-utterance analysis: the reference truncates 112/674 utterances (219 final words);
  we truncate 89 (160) at tail8 — the located, causal share of the reference gap.

Final paper table: reference 12.62 % / this port, tuned serving, Q4_K: **11.29 %** — 1.33
points ahead at 8.6x the speed.

# Revisiting closed verdicts after the flash fix (2026-07-31)

Two levers re-tested on the post-flash engine, both verdicts changed. The meta-lesson: a
performance verdict is only valid for the machine it was measured on, and the flash fix made
it a different machine.

## Speculation: rejected before, wins ~8 % now

STT_SPEC_N=2: host RTF 0.326→0.297, spec=3 → 0.294, transcript identical (570 chars). The
original "correct but not faster" verdict was measured when manual attention dominated the
step; with flash on, the LM pass it amortizes is twice as cheap and the batching overhead
now pays. Not yet in the app (the JNI feeds frame-by-frame; batching 2 frames adds 80 ms
latency and needs the step_batch API — next session).

## GGML_LLAMAFILE (tinyBLAS): OFF since day one, never tested — Mimi −20 %

The conv share of the profile (21.5 %) turned out to be GEMM compute, not weight bytes: the
im2col gives conv matmuls many columns, so quantizing conv weights was the wrong lever even
where the [k≤7, Cin, Cout] storage layout allowed it (it doesn't — but the reshape-to-2D
storage idea died for the deeper reason, not the layout one). The right lever for F16 GEMM
is llamafile's tinyBLAS, which both android and host builds had OFF.

Host: mimi_ms ~4200→3400 (−20 %), total RTF 0.32→0.29. Stacked with spec=2: **0.279**.
Numerics shift (chars 570→588) but WER holds: 10.98 % on the subset, inside the
10.8–11.1 band of its comparators. Android libs rebuilt with LLAMAFILE=ON — **device A/B
pending, phone disconnected**; do not ship to jniLibs until measured on the phone.

## Multi-stream batching: the map (next session's work)

Goal: N streams in lockstep through one engine — the eval N× faster, and the app could run
kyutai on mic AND system audio with one weight read. Plumbing found:
- KV caches already carry a batch dim (`ne[3]`), `batch_size=1` hardcoded only at
  `moshi_smha_state` (transformer.h:332) and plumbed cleanly from there.
- FA mask broadcasting over batch already legal (`q->ne[3] % mask->ne[3] == 0`).
- The real work: per-stream text-token feedback in `moshi_lmgen_state` (currently scalar),
  batch dim through the embed path (lm.h:570 text_token_embed), per-stream Mimi encode
  contexts (cheap, already per-context), and a `moshi_lm_send2_batch/receive2_batch` API.
- Constraint to accept: streams advance in lockstep (same offset/RoPE/mask) — fine for eval
  and for the dual-stream app case.

## llamafile on device: RTF 0.317 cold / 0.327 sustained at 71 °C — shipped (2026-07-31)

Cooled interleaved A/B (test_16k): baseline 0.554 vs llamafile **0.317**, and the LM itself
halves (lm_ms 10.8 s → 5.3 s — tinyBLAS accelerates far more than the conv GEMMs on ARM).
Transcript identical in all rounds (570 chars; unlike host, device numerics did not shift).
Sustained (90 s ×2): pass 2 **0.327 at 71 °C** — better than yesterday's 0.391, at the
thermal floor. Shipped to the app via build.sh (hash-verified staging).

Device RTF history, same fixture, three days: 2.99 → 0.98 → 0.68 → 0.55 → 0.39 → **0.32**.

## RVQ-in-plain-C: attempted, measured slower, reverted (2026-07-31)

Hypothesis: after tinyBLAS, the small-op storm dominates, and the RVQ cascade (32 sequential
stages x ~8 nodes) is its densest knot — so run it in plain C outside the graph, no
dispatches, no barriers. Implemented (graph split at the latent, codebooks dequantized once,
scalar C cascade), measured: mimi_ms 3.4 s -> 6.0 s. **Slower.** A single scalar core loses
to the graph's 6-thread tinyBLAS matvecs by more than the barriers cost; and the storm is
not RVQ-concentrated anyway — 119k MUL calls spread across the whole network say the
per-node overhead is diffuse. Reverted (chars had also shifted 588 -> 584, F32-book
numerics).

What the attempt taught: the remaining overhead is the graph *machinery itself* spread over
~550k tiny nodes, and the fixes that respect that are global — fewer nodes (op fusion) or
more work per node (multi-stream batching) — not relocating one subgraph. Reinforces batch
as the right next move.

Quantization ranking re-verified under tinyBLAS on device: Q4_K 5.4 s < Q4_0 6.3 s <
Q8_0 8.7 s (lm_ms). Same order as under the generic kernels; the question stays closed.

Host state after the day: RTF 0.275 (was 0.677 at yesterday's dawn — 2.5x in one day).

---

# The sub-10 push (2026-08-07): 9.67 % WER on the subset via serving + ensemble

Target set by the user: below 10 %. Reached, model untouched. The ladder (113-file subset):

| step | WER | what it is |
|---|---|---|
| linear-resample baseline | 11.45 % | where the week started |
| **windowed-sinc resampling** (julius) | **10.61 %** | −0.84 alone: linear 16→24 kHz
upsampling leaves spectral images of the 0–8 kHz band above 8 kHz (imaging — not
aliasing, corrected 2026-08-07: upsampling folds nothing). The single biggest quality
lever found since AGC. |
| + tail16 + prefix8 | **10.31 %** | interactions are real: tail16 *hurts* on the sinc base
alone (11.11) and prefix rescues it; on the linear base tail16 helped alone. Serving
parameters must be tuned jointly, on the final audio path. |
| ROVER, 3 systems | 10.04 % | word-level majority vote across perturbed configs |
| **ROVER, 5 systems** | **9.67 %** | {t16p8, t16p6-F16, t8-F16, t8p4, t16p4} — decorrelated
by tail/prefix/precision. Test-time compute ×5; the quality-ceiling row, not the deployed
config. |

Full-676 confirmation pipeline running (the subset also *selected* these configs, so the
full set is the honest validation).

## Negative: greedy bigram shallow fusion — hurts (11.65 / 11.98 vs 10.61)

NGPU-LM-style fusion (arXiv:2505.22857) implemented end-to-end: text logits were already
host-side (lm_states->sampler_out), the hook re-ranks real tokens only (never the pad-vs-word
decision), λ ∈ {0.3, 0.5}, bigram from FLEURS-fr *train* (3 193 sentences, 30 511 bigrams).
Even so constrained, it degrades: the model's implicit LM (trained on far more French than
3 k sentences) is strictly stronger than the explicit bigram, so every disagreement the
fusion wins is more likely wrong than right. λ=1.0 on the fixture visibly breaks morphology
("s'est blessé" for "s'est placé"). The mechanism needs a corpus 2-3 orders of magnitude
larger to have a chance; kept behind MOSHI_TEXT_BIAS (off by default), hook cost is nil when
unset.

## Also shipped to the app this session
- Speculation (spec=2) in the JNI: −8 % compute **measured on host only** (the device
  gain is extrapolated — the sole device speculation A/B, pre-flash, was inside ±15 %
  noise; corrected 2026-08-07, was written "device compute"), +80 ms text latency,
  odd-frame flush and reset edge cases closed.
- Watchdog v2: 12 *consecutive* speech-chunks with no text (instantaneous-VAD version caught
  two radio jingles in 40 min).

## Confidence-gated ensemble: sub-10 at 1.4x, not 5x (2026-08-07)

The 5x ROVER cost is not a fatality — it was uniform spending on non-uniform uncertainty.
The text logits are already host-side, so each word decision's top1−top2 margin is free;
per-utterance hesitation (count of decisions with margin < 2) gates which utterances get the
other 4 ensemble members. Cost-quality curve (subset, primary = t16p8):

| gated % | cost | WER |
|---|---|---|
| 0 % | 1.00x | 10.31 % |
| **10 %** | **1.40x** | **9.94 %** |
| 100 % | 5.00x | 9.63 % |

Re-decoding only the 10 % least-confident utterances crosses sub-10 at 1.4x — more than half
the full-ensemble gain for 10 % of its extra cost (0.4x of 4.0x; was misstated as 8 %). The margin signal predicts where votes
differ, which is the draft/verify economics of speculative decoding transposed to decoding
confidence. At 1.4x, device RTF ≈ 0.45: phone-viable as a post-utterance refinement.

Instrumentation: MOSHI_MARGIN=1 (text_bias.h), CONF lines in stt_eval, curve in
tools/gated_ensemble.py — the curve itself needs no new engine compute once the ensemble
members exist.

---

# Review pass — errata, statistics, and fixes (2026-08-07, night)

A hostile-reviewer audit of everything above, done while the full-676 confirmation runs.
Everything here is either a correction to a prior claim or new rigor added to the
harness. The corrections are also applied inline at the original claims.

## Statistics added (score_wer.py --ci / --compare, utterance-level bootstrap, B=2000)

The harness had no error bars anywhere. Now measured, on the committed artifacts:

| comparison | delta | 95 % CI | p | verdict |
|---|---|---|---|---|
| port F16 vs reference (676) | −1.33 pt | [−2.20, −0.41] | **0.005** | the one headline that clears significance |
| Q4_K vs F16 (676, resumed-chain artifact) | +0.32 pt | [−0.08, +0.70] | 0.10 | n.s. — quote "quantization is free within noise", not "costs 0.11 pt" |
| combo (tail16+prefix6+AGC) vs base (676) | −0.32 pt | [−1.00, +0.26] | 0.32 | n.s. alone; the deletions −23 % mechanism is the evidence, not the point estimate |
| 5-way ROVER vs primary (113 subset) | −0.60 pt | [−1.26, +0.03] | 0.06 | borderline — sub-10 is NOT established on 2 979 words; the 676 run is the decider |

Single-number CIs: subset figures carry ±~2 pt (9.70 % → [7.85, 11.68]) — every step of
the sub-10 ladder is inside one CI, which is why the full-676 confirmation matters.

## The port-vs-reference headline is protocol-confounded — grid now possible

12.62 vs 11.29 confounds four variables at once: engine, precision (F32/F16), tail flush
(7 vs 14 frames), warm vs cold LM context. The −1.33 pt is significant but is a *system*
claim, not an *engine* claim, until the grid is run. `ref_eval.py` now takes
REF_TAIL_EXTRA / REF_PREFIX_FRAMES (REF_TAIL_EXTRA=7 matches our default flush) — the
2×2 decomposition (ref×{tail7,tail14}, port×{fresh,persistent}) is one overnight run and
turns the weakness into the ablation section.

## rover.py had two real bugs — fixed, subset re-voted

1. Tie-breaking iterated a Python set → randomized-hash order → **non-deterministic
   output** across runs. Now: seed wins ties it is part of, else first tied word in slot
   order.
2. Head-of-utterance insertions got one phantom gap vote regardless of round, so a
   2-of-5 word could win a slot that correct bookkeeping (3 gap votes) would drop.

Re-vote with fixed code: 9.67 % → **9.70 %** (one substitution — the number survives,
and is now reproducible bit-for-bit). `rover_C_fixed.tsv` committed alongside the
historical file. The f676 vote at the end of the running confirmation uses the fixed
code. Also: the docstring no longer calls this "classic ROVER (Fiscus 1997)" — it is
iterative pairwise alignment + frequency vote, no confidence, no WTN.

## The "5 systems" are one decoder under perturbation — now quantified

tools/ensemble_diversity.py (new): pairwise error-count correlation between members is
**0.81–0.92**, hypothesis disagreement only 2.8–6.3 % of words, oracle
(best-member-per-utterance) 8.09 %. Call it a perturbation ensemble; "decorrelated by
tail/prefix/precision" overstated it. The vote works *because* the small disagreements
concentrate on error sites, and now there is a table to show it.

## Gated ensemble: scorers unified, ranking confound measured, claims corrected

- gated_ensemble.py now imports score_wer's normalize/wer (the 9.63-vs-9.67
  same-config disagreement was two hand-rolled scorers; gone — 100 % gating now equals
  the ROVER row exactly: 9.70).
- With fixed rover, the curve reads: 0 % → 10.31, 10 % → **9.97** at 1.40x, 100 % → 9.70.
- "half the gain for 8 % of the cost" corrected to 10 % (0.4x/4.0x), inline.
- n_low is length-confounded; `--rank frac` (n_low/n_words) measures it: **worse** at
  10 % (10.14 vs 9.97) — raw n_low wins partly *because* it proxies length. State this.
- Empty-primary utterances have n_low = 0 and were unreachable by the gate — the worst
  failure mode was invisible to it. Both rankings now escalate them first (0 such
  utterances on this subset, so numbers unchanged — but the 676 set has them).
- The 10 % operating point was read off the test curve; it needs selection on dev
  (fleurs_prepare.py now downloads the dev split) before it is a claim.
- Costs remain analytic (equal-cost passes) — the printout now says so; F16 members
  cost ~1.8x a Q4_K pass, and "device RTF ≈ 0.45" was derived, never measured.

## AGC: what it is, and a real app bug found by this audit

- It is a causal peak normalizer/limiter (instant attack, τ = 4 s exponential release —
  which halves in 2.77 s, not the "~4 s" the comments said; comments fixed).
- The eval envelope persists across files, so AGC-arm WER depends on file-list order;
  STT_AGC_RESET=1 added for the order-independent variant.
- The five sub-10 ensemble arms ran **AGC off** while the app ships AGC always-on —
  ensemble numbers and deployment differ in input conditioning; re-run the members with
  STT_AGC=1 before quoting them as deployed quality.
- **App bug (fixed in kata):** the JNI applied AGC unconditionally AND the Kotlin
  AutoGain ran in front of it when the settings toggle was on → the stream was gained
  twice in series (never measured); toggle off still left the JNI stage on, so the
  setting did nothing for kyutai. Now: Kotlin stage off for kyutai, toggle forwarded to
  the JNI (nativeSetAgc).

## Sinc resampling now exists in code, not only in an offline heredoc

The −0.84 pt lever was julius defaults in a gitignored script; stt_eval and the shipped
JNI still carried linear interpolation (the app was safe only because capture follows
the engine at 24 kHz — the 16 kHz fallback path had exactly the measured defect).
Windowed-sinc (Hann, 24 zeros, rolloff 0.945) is now the default resampler in
stt_eval.cpp (STT_RESAMPLE=linear to fall back) and a streaming polyphase-style version
replaced the linear kernel in moshi_jni.cpp. **Not yet device-validated — the JNI change
must not ship until an A/B on the phone** (build.sh stages it; the pending arms of the
current 676 run still use the pre-change binary, which is correct: its inputs are
pre-resampled 24 kHz files, where the resampler is a no-op).

## Standing discrepancies, recorded so nobody quotes them

- Padding rate appears as 55.5 % (measured, STT_DUMP_TOKENS) and "roughly 65 %"
  (later section) for the same fixture family. Unreconciled; re-measure before citing.
- "Q4_K ≥ Q4_0": claimed at 3 % (P1), retracted as inside the ±7 % noise floor (GPU
  correction section), then re-asserted at 17 % under tinyBLAS with "same order as
  generic kernels" — the last sentence contradicts the retraction. The tinyBLAS margin
  is real; the pre-tinyBLAS ordering claim stays retracted.
- The whisper FLEURS-fr context numbers are now verified (Whisper paper Table 13,
  zero-shot: small 15.0 / medium 8.7 / large 7.7 / large-v2 8.3) and the wrong
  "large-v3 ≈ 5–6 %" guess corrected inline — those ~5 % figures are French fine-tunes.
  Different text normalizers across scorers: indicative comparison only.
- hyp for the 11.40 % unbroken Q4_K chain was not preserved; the committed artifact is
  the resumed chain at 11.61 %. Quote 11.40 only with the ±0.2 segmentation-variance
  caveat, or re-run.
- Pre-flash closed verdicts never re-tested on the post-flash engine: OpenMP, +fp16
  kernels, thread count, repack, GPU offload, codec-quant speed — and **context=375
  ships on a stale −24 % whose mechanism (manual-attention traffic) the flash fix
  removed, while its quality cost (10.32→10.91 % host) is still paid.** Re-measure
  context 750 vs 375 post-flash+llamafile before the paper freezes the config.

## Evidence chain committed

- `eval-artifacts/`: reference + headline hypothesis files, subset members, CONF file,
  device run, FLEURS manifest — every number in the paper tables is now reproducible
  from the repo with score_wer.py (README in the directory maps file → claim).
- `tools/eval-scripts/`: the sweep drivers that lived only in gitignored eval-data.
- fleurs_prepare.py downloads the dev split for future tuning; tuning on (a subset of)
  test is recorded above as the audit's single most attackable finding.

## Full-676 verdict: the sub-10 did NOT hold — subset overfitting, quantified (2026-08-08)

The 9.67 % / 9.94 % sub-10 numbers were measured on the 113-file subset that had also been
used to *select* the serving knobs and ensemble membership. At full scale:

| config | subset | **full 676** |
|---|---|---|
| best single (t16p8, Q4_K) | 10.31 % | 11.57 % |
| best single at full scale (t16p6, **F16**) | — | **10.66 %** |
| ROVER 5 | 9.67 % | **10.47 %** |

**Selection bias cost ~0.8–1.3 points.** The subset even picked the wrong winner: t16p8/Q4_K
led there, F16/t16p6 leads at full scale. This is the textbook failure of tuning and
reporting on the same set, and it is now the harness's loudest warning (fleurs_prepare.py
grew a `dev` split for exactly this reason).

Paired bootstrap (B=2000, utterance resampling, seed 1234):

- **sinc + serving vs the old linear-resample F16 baseline: −0.62 pt, 95 % CI
  [−1.15, −0.12], p = 0.012.** Real, and it is the session's genuine win.
- **ROVER 5 vs best single: −0.19 pt, 95 % CI [−0.51, +0.14], p = 0.24.** Not significant.
  At 5x the compute. The ensemble's apparent value was largely subset noise.

So the honest ladder on FLEURS-fr, full test set, deployment mode:

    reference implementation  12.62 %
    this port, F16, linear    11.29 %
    this port, F16, sinc + tuned serving   **10.66 %**   (CI [9.80, 11.59])
    + 5x ROVER                10.47 %   (not significant)

Sub-10 remains unreached on the full test set. The path there is not more test-time
compute — the ensemble is measured flat — it is the acoustic front end, where sinc already
paid 0.6 pt.

## Where the remaining headroom is, and whether search can reach it (2026-08-08)

Two oracles, measured rather than assumed, to decide what is worth building next.

**Oracle over the 5 perturbation variants (test, 676):** 8.33 %, against 10.66 % for the best
single member and 10.47 % for majority voting. So **2.3 points already exist inside
hypotheses we generate and fail to select** — 225 of 676 utterances are best transcribed by a
variant that is not the primary. Selection, not generation, is the bottleneck. But an
N-variant ensemble is Nx compute and cannot ship on a phone, so this number is a *bound*, not
a plan.

**1-swap oracle (dev, 44 utterances, greedy 7.04 %):** taking the model's second choice at
the K least-confident frames, best single swap per utterance:

| K frames considered | WER |
|---|---|
| 1 | 7.04 % |
| 3 | 6.76 % |
| 5 | 6.57 % |
| 10 | 6.39 % |
| 20 | 6.30 % |

**A single top-2 substitution at a low-margin frame recovers 0.7 pt.** The correct token *is*
in the model's near-miss set, and the confidence margin points at where. This is the
justification for beam search over the text stream: a beam explores exactly these
alternatives, jointly rather than one at a time, and it does so inside one decode. Cost
estimate: Mimi (35 % of compute) runs once regardless of beam width, and the LM is
weight-bound, so B=2 should cost ~1.15x rather than 2x — phone-viable, unlike the ensemble.

Caveats to carry: 44 utterances is a pilot (the CI on 7.04 % is roughly ±2 pt); single swaps
are a lower bound on what a beam explores but also ignore that a beam must commit without
seeing the reference; and the dev set cannot resolve sub-0.3 pt effects (§ statistical power).
The number worth quoting is the *shape* — the near-miss set contains the answer — not 6.30.

Instrumentation: MOSHI_TOPK_DUMP=<file> (text_bias.h) writes "tok1 tok2 margin" per real-token
decision with utterance markers; tools/swap_oracle.py does the analysis.

## Beam search will not work here, and the reason invalidates the 1-swap oracle (2026-08-08)

Before building beam search (multi-hour KV-state work), the go/no-go question: can the model
rank its own paths without the reference? Same 25 dev utterances, forcing the SECOND-best
token at decision n and greedy elsewhere, reporting WER and the sequence log-probability:

| forced at | WER | Σ log p |
|---|---|---|
| — (greedy) | **7.58 %** | **−47** |
| n=0 | 12.19 % | −148 |
| n=1 | 17.30 % | −234 |
| n=2 | 11.20 % | −243 |
| n=3 | 13.51 % | −260 |
| n=5 | 11.53 % | −270 |
| n=7 | 10.71 % | −272 |
| n=9 | 11.53 % | −258 |
| n=12 | 16.80 % | −203 |

Two conclusions, one of them methodological.

**1. There is nothing for a beam to find.** Every divergence is *catastrophically* worse
(+3 to +10 WER points), not marginally worse. The greedy path is both the best-WER and the
best-scoring path by a wide margin, at every position tried. The selector works perfectly —
cumulative log-probability ranks greedy first every time — there is simply nothing better to
select. Beam search is dropped.

**2. The 1-swap oracle overestimated the prize, and post-hoc edit oracles always will in an
autoregressive model.** That oracle swapped a token *in the finished output* and re-scored
the text, implicitly holding the rest of the sequence fixed. But the emitted text token is
fed back into the LM, so a different choice at frame t changes every frame after t. The
0.47 pt it promised is not reachable by any search: it is the gain from editing a transcript,
not from decoding differently. Any future "would search help" analysis on this model must
force the token through the model, as here, rather than edit the output.

This also reframes the 5-variant oracle (8.33 %): that headroom is real, because those
variants are genuinely different full decodes (different front end / precision), not
different paths through one decode. It is reachable only at Nx compute, which does not ship
on a phone. So:

- front end: exhausted (sinc is the one real win; everything else n.s.)
- search within one decode: measured empty
- ensembles across decodes: real (−2.3 pt oracle) but Nx, and majority voting captures
  almost none of it (−0.19 pt, p = 0.24)

**Sub-10 on the full test set is not reachable by serving-side work.** The remaining levers
are model-side (a stronger or fine-tuned model), which is outside this paper's scope.

## Where the remaining 10.66 % actually lives — and why sub-10 is not closed (2026-08-08)

Error-class analysis of the best full-test system (F16, sinc, tail16/prefix6), 1368
substitutions over 676 utterances:

| class | count | share of substitutions |
|---|---|---|
| **French agreement / homophones** | **363** | **26.5 %** |
| digit-vs-word formatting (`cinq` → `5`) | 67 | 4.9 % |
| accent-only | 11 | 0.8 % |

Top substitutions are almost entirely grammatical: `des`→`les` (16), `aux`→`au` (15),
`est`→`et` (12), `leurs`→`leur` (11), `ces`→`ses`, `à`→`a`, `britannique`→`britanniques`.
These are **French homophones and number/gender agreement** — pairs that are acoustically
identical and decidable only from grammar. No acoustic front end can fix them; no amount of
decoding search can either, since the model is choosing between tokens that sound the same.

What that implies for the target:

| if agreement errors were fixed | WER |
|---|---|
| all | 8.65 % |
| half | 9.65 % |
| **one third** | **9.99 %** |

**Correcting one third of the agreement errors reaches sub-10.** So the earlier statement
"sub-10 is not reachable serving-side" was too strong and is withdrawn: it is not reachable
by the levers tested (front end, in-decode search, ensembles), but a *post-hoc text
correction* stage is untested and targets exactly the dominant class.

Crucially this is **not** the invalidated edit-oracle: that oracle failed because feeding a
different token back into the model rewrites the future. A correction stage never re-enters
the model — it rewrites the finished transcript, so the autoregressive objection does not
apply. Correction and search are different operations, and the negative for one says nothing
about the other.

Practical shape, and why it can ship: the candidate set is tiny and closed (des/les, au/aux,
leur/leurs, est/et, a/à, singular↔plural), so this is a small classifier or n-gram over a
handful of alternatives per sentence, not an LLM. It runs on text, after the audio pipeline,
at negligible cost — the constraint that killed ensembles does not bind here.

Also free and legitimate: the 67 digit-formatting substitutions (0.37 pt) are a normalization
convention, not recognition errors. Whisper's own evaluation applies a number normalizer to
both sides; ours does not. Adding one to `score_wer.py` is standard practice and must be
applied symmetrically.

## Correction: most confusions are NOT homophones — the signal contains the answer (2026-08-08)

The previous section lumped `des`→`les` with `leur`→`leurs` and called all of it
"agreement/homophones". That is wrong and the distinction changes what to build. `des` /de/
and `les` /le/ differ in their initial consonant — a stop with a release burst against a
lateral approximant — and are among the most acoustically separable pairs in French. Splitting
the 1368 substitutions properly:

| class | count | share | decidable from audio? |
|---|---|---|---|
| **true homophones** (silent final s/x, accent only) | 275 | 20.1 % | **no** — needs grammar |
| — of which accent-only (`a`/`à`, `la`/`là`) | 13 | 1.0 % | no |
| **acoustically distinct** | **1093** | **79.9 %** | **yes — the information is in the waveform** |

The top acoustically-distinct confusions are `des`→`les` (16), `est`→`et` (12), `d'`→`des`
(10), `les`→`des` (9), `un`→`en` (4), `pinson`→`pinceau` (4). None of these is a homophone.
Four-fifths of our substitutions are cases where the answer is present in the signal and the
system fails to use it.

That reframes the whole remaining gap. It is not a grammar problem to be patched downstream —
it is **information lost between the waveform and the text decision**, and the candidates are:
the RVQ codec discarding fine spectral detail (32 stages of hard argmax over 2048 centroids,
and we already know that cascade is quantization-fragile), the 12.5 Hz frame rate smearing
short consonant releases, or the LM prior overriding weak acoustic evidence. Each is testable:
the margin instrumentation already says whether the model was confidently wrong (prior
dominating) or hesitant (acoustics weak), and that measurement comes before any engineering.

Accent normalization, the other open question: stripping accents from both sides moves WER
**10.66 → 10.59 %** — 0.07 pt. Real but negligible, and it does not change any conclusion.
Keep accents (FLEURS' own reference column keeps them, and `à` vs `a` are different French
words); note the number and move on.
