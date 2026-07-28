// Does batching amortize weight loads on this CPU?
//
// The load-bearing assumption behind text-stream speculation (BENCH.md) is that a matmul
// with 2 columns of activations costs about the same as one with 1, because the workload is
// bandwidth-bound on weights and the weights are read once either way. If instead batch 2
// costs ~2x batch 1, the whole idea is worthless and there is no point building the
// speculation machinery.
//
// This measures exactly that, on the real weight shapes of kyutai stt-1b's LM
// (dim 2048, hidden 8448 = 2048 * 4.125), in isolation from moshi.cpp.
//
// usage: bench_batch [type] [threads] [iters]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct Shape { const char * name; int64_t k, n; };

int main(int argc, char** argv) {
    const char * tname = argc > 1 ? argv[1] : "q4_K";
    const int threads  = argc > 2 ? atoi(argv[2]) : 6;
    const int iters    = argc > 3 ? atoi(argv[3]) : 200;

    ggml_type wtype = GGML_TYPE_Q4_K;
    if (!strcmp(tname, "q4_0")) wtype = GGML_TYPE_Q4_0;
    else if (!strcmp(tname, "q8_0")) wtype = GGML_TYPE_Q8_0;
    else if (!strcmp(tname, "f16")) wtype = GGML_TYPE_F16;

    // One transformer layer of kyutai stt-1b, by weight shape.
    const std::vector<Shape> shapes = {
        { "self_attn.in_proj  [2048x6144]", 2048, 6144 },  // q,k,v fused
        { "self_attn.out_proj [2048x2048]", 2048, 2048 },
        { "gating.linear_in   [2048x8448]", 2048, 8448 },
        { "gating.linear_out  [4224x2048]", 4224, 2048 },
    };

    // BENCH_BACKEND=gpu picks the first non-CPU device, to compare achievable matmul
    // throughput between the CPU and the Adreno.
    ggml_backend_load_all();
    ggml_backend_t backend = nullptr;
    const char* want = getenv("BENCH_BACKEND");
    if (want && !strcmp(want, "gpu")) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            auto dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) continue;
            backend = ggml_backend_dev_init(dev, nullptr);
            if (backend) { printf("backend: %s\n", ggml_backend_dev_name(dev)); break; }
        }
    }
    if (!backend) backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) { fprintf(stderr, "no backend\n"); return 1; }
    {
        auto dev = ggml_backend_get_device(backend);
        auto reg = ggml_backend_dev_backend_reg(dev);
        auto set_n = (ggml_backend_set_n_threads_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_n) set_n(backend, threads);
    }

    printf("type=%s threads=%d iters=%d\n\n", ggml_type_name(wtype), threads, iters);
    printf("%-34s %10s %10s %10s %8s\n",
           "shape", "batch1 ms", "batch2 ms", "batch4 ms", "b2/b1");

    double tot1 = 0, tot2 = 0, tot4 = 0;
    for (const auto & s : shapes) {
        double ms[3] = {0, 0, 0};
        const int batches[3] = { 1, 2, 4 };
        for (int bi = 0; bi < 3; bi++) {
            const int B = batches[bi];
            ggml_init_params ip = { (size_t)16*1024*1024, nullptr, /*no_alloc*/ true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * w = ggml_new_tensor_2d(ctx, wtype, s.k, s.n);
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k, B);
            ggml_tensor * y = ggml_mul_mat(ctx, w, x);
            auto buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            std::vector<uint8_t> wdata(ggml_nbytes(w), 0x11);
            std::vector<float> xdata(ggml_nelements(x), 0.01f);
            ggml_backend_tensor_set(w, wdata.data(), 0, wdata.size());
            ggml_backend_tensor_set(x, xdata.data(), 0, xdata.size() * 4);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, y);

            ggml_backend_graph_compute(backend, gf);   // warm
            const double t0 = now_ms();
            for (int i = 0; i < iters; i++) ggml_backend_graph_compute(backend, gf);
            ms[bi] = (now_ms() - t0) / iters;
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
        printf("%-34s %10.3f %10.3f %10.3f %8.2f\n",
               s.name, ms[0], ms[1], ms[2], ms[1] / ms[0]);
        tot1 += ms[0]; tot2 += ms[1]; tot4 += ms[2];
    }

    printf("\n%-34s %10.3f %10.3f %10.3f %8.2f\n", "TOTAL (one layer)", tot1, tot2, tot4, tot2 / tot1);
    printf("\nA batch2/batch1 ratio near 1.0 means weight loads dominate and batching is\n"
           "nearly free — speculation is worth ~28%%. Near 2.0 means compute dominates and\n"
           "the idea is dead. Per-frame cost for 2 frames: %.3f ms batched vs %.3f ms not,\n"
           "i.e. %+.1f%%.\n",
           tot2, 2 * tot1, 100.0 * (tot2 - 2 * tot1) / (2 * tot1));

    ggml_backend_free(backend);
    return 0;
}
