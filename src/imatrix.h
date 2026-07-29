#pragma once
// Importance-matrix collection for quantization, à la llama.cpp's imatrix tool — which
// moshi.cpp never had, so every quantization so far minimized error ON THE WEIGHTS
// (‖W−Ŵ‖) when what matters is error on the OUTPUTS (‖(W−Ŵ)x‖ for the activations x the
// model actually sees). ggml_quantize_chunk already accepts per-column importance weights;
// nothing here produced them.
//
// The place this matters most in this project is the Mimi codec: it feeds a hard argmax
// cascade (32 residual stages over 2048 centroids), so quantization noise flips codes
// instead of being absorbed by a softmax — plain Q4_K on the codec measured 15.93 % word
// error. Importance weighting minimizes exactly the output perturbation that flips codes.
//
// Usage: MOSHI_IMATRIX=/path/out.imatrix, run any workload (stt_bench over the French
// fixtures), the file is written at process exit. Feed it to requantize_gguf --imatrix.
//
// Collection piggybacks on GraphContext::compute(): every graph in this codebase flows
// through it, and GraphContext allocates tensors with ggml_backend_alloc_ctx_tensors —
// permanent slots, not a shared arena — so activations are still readable after compute.
//
// Format (text, debuggable): one header line "moshi-imatrix v1", then per tensor:
//   <name> <ne0> <nchunk>
//   <ne0 doubles: mean of x² per column, accumulated over every token seen>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>

struct moshi_imatrix_entry {
    std::vector<double> sums;   // per-column running sum of x²
    long long           count = 0;  // tokens accumulated
};

struct moshi_imatrix_state {
    std::mutex mtx;
    std::map<std::string, moshi_imatrix_entry> entries;
    std::map<std::string, int> skipped;   // name -> reason logged once
    const char * out_path = nullptr;
    bool enabled = false;
    bool initialized = false;
};

inline moshi_imatrix_state & moshi_imatrix() {
    static moshi_imatrix_state s;
    return s;
}

inline void moshi_imatrix_save() {
    auto & s = moshi_imatrix();
    if (!s.enabled || !s.out_path) return;
    std::lock_guard<std::mutex> lock(s.mtx);
    FILE * f = fopen(s.out_path, "w");
    if (!f) { fprintf(stderr, "imatrix: cannot write %s\n", s.out_path); return; }
    fprintf(f, "moshi-imatrix v1\n");
    for (auto & [name, e] : s.entries) {
        if (!e.count) continue;
        fprintf(f, "%s %zu %lld\n", name.c_str(), e.sums.size(), e.count);
        for (size_t i = 0; i < e.sums.size(); i++)
            fprintf(f, "%.10g%c", e.sums[i] / (double)e.count,
                    (i + 1 == e.sums.size()) ? '\n' : ' ');
    }
    fclose(f);
    fprintf(stderr, "imatrix: wrote %zu tensors to %s\n", s.entries.size(), s.out_path);
}

inline void moshi_imatrix_init() {
    auto & s = moshi_imatrix();
    if (s.initialized) return;
    s.initialized = true;
    s.out_path = getenv("MOSHI_IMATRIX");
    // Guard on the value, not just presence: MOSHI_IMATRIX= (empty) must stay off.
    // getenv returns non-NULL for an empty value, and that exact mistake already produced
    // one fake A/B result in this project (the repack "14 % win" that was noise).
    s.enabled = s.out_path && s.out_path[0];
    if (s.enabled) {
        // Resume from an existing file so several stt_bench runs (one per WAV) accumulate
        // into one imatrix instead of each overwriting the last.
        FILE * f = fopen(s.out_path, "r");
        if (f) {
            char header[64];
            if (fgets(header, sizeof(header), f) &&
                !strncmp(header, "moshi-imatrix v1", 16)) {
                char name[512]; size_t ne0; long long count;
                while (fscanf(f, "%511s %zu %lld", name, &ne0, &count) == 3) {
                    auto & e = s.entries[name];
                    e.sums.assign(ne0, 0.0);
                    e.count = count;
                    for (size_t i = 0; i < ne0; i++) {
                        double mean;
                        if (fscanf(f, "%lf", &mean) != 1) { e.count = 0; break; }
                        e.sums[i] = mean * (double)count;   // stored as mean; keep sums
                    }
                }
                fprintf(stderr, "imatrix: resuming from %s (%zu tensors)\n",
                        s.out_path, s.entries.size());
            }
            fclose(f);
        }
        fprintf(stderr, "imatrix: collecting to %s\n", s.out_path);
        atexit(moshi_imatrix_save);
    }
}

// Accumulate activation statistics from one computed graph.
//
// For C = mul_mat(W, X) with W [ne00, ne01] and X [ne10=ne00, tokens...], the importance of
// W's column i is E[X(i,·)²]. Only weight-like src0 qualify: a leaf (op NONE) with a name —
// KV-cache reads are views and RoPE'd Q/K are ops, so they filter out naturally.
inline void moshi_imatrix_collect(struct ggml_cgraph * gf) {
    auto & s = moshi_imatrix();
    if (!s.initialized) moshi_imatrix_init();
    if (!s.enabled || !gf) return;

    std::lock_guard<std::mutex> lock(s.mtx);
    std::vector<float> buf;
    for (int n = 0; n < ggml_graph_n_nodes(gf); n++) {
        ggml_tensor * node = ggml_graph_node(gf, n);
        if (node->op != GGML_OP_MUL_MAT) continue;
        ggml_tensor * w = node->src[0];
        ggml_tensor * x = node->src[1];
        if (!w || !x) continue;
        if (w->op != GGML_OP_NONE || !w->name[0]) continue;      // weights only
        if (x->type != GGML_TYPE_F32) {
            if (!s.skipped.count(w->name)) {
                s.skipped[w->name] = 1;
                fprintf(stderr, "imatrix: skip %s (activations are %s, not F32)\n",
                        w->name, ggml_type_name(x->type));
            }
            continue;
        }
        if (!ggml_is_contiguous(x)) {
            if (!s.skipped.count(w->name)) {
                s.skipped[w->name] = 1;
                fprintf(stderr, "imatrix: skip %s (non-contiguous activations)\n", w->name);
            }
            continue;
        }

        const int64_t ne0    = x->ne[0];
        const int64_t tokens = ggml_nelements(x) / ne0;
        auto & e = s.entries[w->name];
        if (e.sums.empty()) e.sums.assign((size_t)ne0, 0.0);
        if ((int64_t)e.sums.size() != ne0) continue;   // same name, different shape: bail

        buf.resize((size_t)ggml_nelements(x));
        ggml_backend_tensor_get(x, buf.data(), 0, ggml_nbytes(x));
        for (int64_t t = 0; t < tokens; t++) {
            const float * row = buf.data() + t * ne0;
            for (int64_t i = 0; i < ne0; i++)
                e.sums[(size_t)i] += (double)row[i] * (double)row[i];
        }
        e.count += tokens;
    }
}

// ---- reader side (requantize_gguf) ------------------------------------------------------

inline std::map<std::string, std::vector<float>> moshi_imatrix_load(const char * path) {
    std::map<std::string, std::vector<float>> out;
    FILE * f = fopen(path, "r");
    if (!f) { fprintf(stderr, "imatrix: cannot read %s\n", path); return out; }
    char header[64];
    if (!fgets(header, sizeof(header), f) || strncmp(header, "moshi-imatrix v1", 16)) {
        fprintf(stderr, "imatrix: %s is not a moshi-imatrix v1 file\n", path);
        fclose(f);
        return out;
    }
    char name[512];
    size_t ne0; long long count;
    while (fscanf(f, "%511s %zu %lld", name, &ne0, &count) == 3) {
        std::vector<float> v(ne0);
        for (size_t i = 0; i < ne0; i++) {
            double d;
            if (fscanf(f, "%lf", &d) != 1) { fclose(f); return out; }
            v[i] = (float)d;
        }
        out[name] = std::move(v);
    }
    fclose(f);
    return out;
}
