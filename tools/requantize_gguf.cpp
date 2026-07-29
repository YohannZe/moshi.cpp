// Requantize a moshi/mimi GGUF offline.
//
// WHY THIS EXISTS
// ---------------
// `moshi_lm_quantize(lm, "q4_k")` is a silent no-op when the weights come from a
// GGUF: both overloads of WeightLoader::fetch() in src/loader.h return
// get_tensor(name) immediately for the GGUF path and never consult dst_type or
// qtype. The published `stt-1b-en_fr-GGUF/model.gguf` is 102 BF16 tensors
// (989.2M params) + 33 F32, so every "-q q4_k" benchmark ever run against it
// actually measured BF16.
//
// That matters enormously on ARM: ggml_vec_dot_bf16 has AVX512BF16 / AVX512F /
// AVX2 / RISC-V / POWER9 paths and NO ARM path at all, so on a phone every MAC
// falls through to a scalar loop with two bf16->f32 conversions. BF16 is the
// single worst 16-bit format to ship for aarch64.
//
// This tool rewrites a GGUF with a per-tensor type policy so the engine can
// actually be measured, and so the phone gets kernels that exist.
//
// TYPE POLICY (see --help): matmul weights get the requested quant; norms stay
// F32 (they are ~0.1M params, quantizing them buys nothing and costs accuracy);
// embedding/codebook tables become F16 rather than quantized, because they are
// only ever read by ggml_get_rows one row at a time — the bandwidth is
// negligible but the precision is not. Nothing is ever emitted as BF16.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>

#include <ggml.h>
#include <gguf.h>

static void usage(const char* prog) {
    fprintf(stderr,
"usage: %s <in.gguf> <out.gguf> <quant> [--keep-f32-embeddings] [--quantize-all]\n"
"\n"
"  quant   q3_k | q4_0 | q4_k | iq4_xs | q5_k | q6_k | q8_0 | f16\n"
"\n"
"This model is bandwidth-bound on its own weights at batch 1 (Q8_0 predicted 1.89x the\n"
"bytes of Q4_K and measured 1.90x the time), so bits-per-weight is the dominant lever on\n"
"speed. Roughly: q3_k 3.4 bpw, iq4_xs 4.25, q4_k 4.5, q5_k 5.5, q6_k 6.6, q8_0 8.5.\n"
"\n"
"Default per-tensor policy:\n"
"  *_norm* / *norm.weight / 1-D tensors  -> F32   (tiny; precision matters)\n"
"  emb* / *_emb* / text_emb / codebook*  -> F16   (get_rows only; low traffic)\n"
"  everything else (matmul weights)      -> <quant>\n"
"  any row not a multiple of the block   -> F16   (with a warning)\n"
"\n"
"  --keep-f32-embeddings  leave embedding tables at F32 instead of F16\n"
"  --quantize-all         also quantize embedding tables to <quant>\n"
"\n"
"Never emits BF16: it has no ARM CPU kernel in ggml.\n", prog);
    exit(1);
}

static ggml_type parse_quant(const char* s) {
    if (!strcmp(s, "q3_k"))   return GGML_TYPE_Q3_K;
    // IQ4_XS is normally paired with an importance matrix. ggml_quantize_chunk accepts a
    // null imatrix and falls back to a plain search, which is what happens here — so judge
    // it on the measured word error, not on its reputation.
    if (!strcmp(s, "iq4_xs")) return GGML_TYPE_IQ4_XS;
    if (!strcmp(s, "q4_0")) return GGML_TYPE_Q4_0;
    if (!strcmp(s, "q4_k")) return GGML_TYPE_Q4_K;
    if (!strcmp(s, "q5_k")) return GGML_TYPE_Q5_K;
    if (!strcmp(s, "q6_k")) return GGML_TYPE_Q6_K;
    if (!strcmp(s, "q8_0")) return GGML_TYPE_Q8_0;
    if (!strcmp(s, "f16"))  return GGML_TYPE_F16;
    fprintf(stderr, "error: unknown quant '%s'\n", s);
    exit(1);
}

static bool name_contains(const std::string& n, const char* needle) {
    return n.find(needle) != std::string::npos;
}

// Norms and other 1-D parameters. Keeping these F32 is free (0.1M params total
// in the stt-1b model) and quantizing per-channel scales is actively harmful.
static bool is_norm_like(const std::string& n, const ggml_tensor* t) {
    if (ggml_n_dims(t) == 1) return true;
    return name_contains(n, "norm") || name_contains(n, "layer_scale")
        || name_contains(n, "alpha") || name_contains(n, "gamma")
        || name_contains(n, "beta")  || name_contains(n, ".bias");
}

// Lookup tables read by ggml_get_rows: one row per frame, so bandwidth is
// irrelevant and quantization error would be paid on every token.
static bool is_embedding_like(const std::string& n) {
    return name_contains(n, "emb")           // text_emb, emb.N (audio codebooks)
        || name_contains(n, "codebook")
        || name_contains(n, "_codes")
        || name_contains(n, "embedding");
}

// Convert any source dtype to F32 so ggml_quantize_chunk can consume it.
static bool to_f32(const ggml_tensor* t, std::vector<float>& out) {
    const int64_t n = ggml_nelements(t);
    out.resize(n);
    switch (t->type) {
        case GGML_TYPE_F32:
            memcpy(out.data(), t->data, n * sizeof(float));
            return true;
        case GGML_TYPE_F16:
            ggml_fp16_to_fp32_row((const ggml_fp16_t*)t->data, out.data(), n);
            return true;
        case GGML_TYPE_BF16:
            ggml_bf16_to_fp32_row((const ggml_bf16_t*)t->data, out.data(), n);
            return true;
        default: {
            // Already quantized: dequantize via the type traits.
            const auto* tt = ggml_get_type_traits(t->type);
            if (!tt || !tt->to_float) return false;
            tt->to_float(t->data, out.data(), n);
            return true;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4) usage(argv[0]);
    const char* in_path  = argv[1];
    const char* out_path = argv[2];
    const ggml_type quant = parse_quant(argv[3]);

    bool keep_f32_emb = false, quantize_all = false;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--keep-f32-embeddings")) keep_f32_emb = true;
        else if (!strcmp(argv[i], "--quantize-all"))   quantize_all = true;
        else usage(argv[0]);
    }

    ggml_context* ctx_in = nullptr;
    gguf_init_params p = { /*no_alloc*/ false, &ctx_in };
    gguf_context* gguf_in = gguf_init_from_file(in_path, p);
    if (!gguf_in) { fprintf(stderr, "error: failed to open %s\n", in_path); return 1; }

    const int64_t n_tensors = gguf_get_n_tensors(gguf_in);
    printf("%s: %lld tensors\n", in_path, (long long)n_tensors);

    // Carry over all metadata (arch, hyperparameters, tokenizer refs, ...).
    gguf_context* gguf_out = gguf_init_empty();
    gguf_set_kv(gguf_out, gguf_in);

    // Own the converted tensors: gguf_add_tensor keeps the pointer, so the data
    // must outlive the write.
    ggml_init_params op = { /*mem_size*/ (size_t)(n_tensors + 2) * ggml_tensor_overhead(),
                            /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    ggml_context* ctx_out = ggml_init(op);
    std::vector<std::vector<uint8_t>> storage(n_tensors);

    size_t bytes_in = 0, bytes_out = 0;
    int n_quant = 0, n_f16 = 0, n_f32 = 0, n_fallback = 0;
    std::vector<float> f32;

    for (int64_t i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_in, i);
        ggml_tensor* src = ggml_get_tensor(ctx_in, name);
        if (!src) { fprintf(stderr, "error: tensor %s missing from ctx\n", name); return 1; }
        const std::string n = name;

        // --- decide the target type ---
        ggml_type dst_type;
        const char* why;
        if (is_norm_like(n, src)) {
            dst_type = GGML_TYPE_F32;                     why = "norm/1d";
        } else if (is_embedding_like(n) && !quantize_all) {
            dst_type = keep_f32_emb ? GGML_TYPE_F32 : GGML_TYPE_F16;
            why = "lookup table";
        } else {
            dst_type = quant;                             why = "matmul";
        }

        // Quantized types need whole blocks per row.
        const int64_t blck = ggml_blck_size(dst_type);
        if (blck > 1 && src->ne[0] % blck != 0) {
            fprintf(stderr, "  warn: %s ne[0]=%lld not a multiple of %lld (%s block) "
                            "-> F16\n", name, (long long)src->ne[0], (long long)blck,
                            ggml_type_name(dst_type));
            dst_type = GGML_TYPE_F16;
            why = "block misfit";
            n_fallback++;
        }

        // Never emit BF16, whatever the source was: no ARM kernel exists.
        if (dst_type == GGML_TYPE_BF16) dst_type = GGML_TYPE_F16;

        // --- convert ---
        ggml_tensor* dst = ggml_new_tensor(ctx_out, dst_type, ggml_n_dims(src), src->ne);
        ggml_set_name(dst, name);
        const size_t nbytes = ggml_nbytes(dst);
        storage[i].resize(nbytes);
        dst->data = storage[i].data();

        if (dst_type == src->type) {
            memcpy(dst->data, src->data, nbytes);
        } else {
            if (!to_f32(src, f32)) {
                fprintf(stderr, "error: cannot convert %s from %s\n",
                        name, ggml_type_name(src->type));
                return 1;
            }
            if (dst_type == GGML_TYPE_F32) {
                memcpy(dst->data, f32.data(), nbytes);
            } else if (dst_type == GGML_TYPE_F16) {
                ggml_fp32_to_fp16_row(f32.data(), (ggml_fp16_t*)dst->data, ggml_nelements(src));
            } else {
                const int64_t n_per_row = src->ne[0];
                const int64_t nrows = ggml_nelements(src) / n_per_row;
                const size_t written = ggml_quantize_chunk(dst_type, f32.data(), dst->data,
                                                           0, nrows, n_per_row, nullptr);
                if (written != nbytes) {
                    fprintf(stderr, "error: %s quantize wrote %zu, expected %zu\n",
                            name, written, nbytes);
                    return 1;
                }
            }
        }

        gguf_add_tensor(gguf_out, dst);
        bytes_in  += ggml_nbytes(src);
        bytes_out += nbytes;
        if      (dst_type == GGML_TYPE_F32) n_f32++;
        else if (dst_type == GGML_TYPE_F16) n_f16++;
        else                                n_quant++;

        if (n_tensors <= 40 || i < 6 || i == n_tensors - 1) {
            printf("  %-44s %6s -> %-6s  %8.2f MB  (%s)\n", name,
                   ggml_type_name(src->type), ggml_type_name(dst_type),
                   nbytes / 1e6, why);
        }
    }

    printf("\n%lld tensors: %d %s, %d F16, %d F32", (long long)n_tensors,
           n_quant, ggml_type_name(quant), n_f16, n_f32);
    if (n_fallback) printf(" (%d fell back to F16 on block size)", n_fallback);
    printf("\n%.2f MB -> %.2f MB  (%.2fx smaller)\n",
           bytes_in / 1e6, bytes_out / 1e6, (double)bytes_in / (double)bytes_out);

    if (!gguf_write_to_file(gguf_out, out_path, /*only_meta*/ false)) {
        fprintf(stderr, "error: failed to write %s\n", out_path); return 1;
    }
    printf("wrote %s\n", out_path);

    gguf_free(gguf_out);
    gguf_free(gguf_in);
    ggml_free(ctx_out);
    ggml_free(ctx_in);
    return 0;
}
