// Streaming STT benchmark for kyutai stt-1b-en_fr via moshi.cpp.
//
// Deliberately depends on NOTHING but libmoshi + ggml — no SDL, no FFmpeg.
// That is the point: it proves the model path is portable to Android/JNI, and it
// doubles as the prototype of the JNI feed loop.
//
// Mirrors voxtral.cpp/tools/test_stream.cpp so results are directly comparable:
// same fixtures, same RESULT line, same RTF definition (compute_ms / audio_ms).
//
// usage: stt_bench <model_dir> <audio.wav> [threads] [model_gguf]
//   model_dir   dir holding config.json (mimi + tokenizer resolved relative to it)
//   audio.wav   16-bit PCM mono WAV, any sample rate (resampled to 24 kHz)
//   STT_SPEC_N=k  speculate k frames per LM pass (1 = off). See BENCH.md.
//   model_gguf  weights file inside model_dir, overriding config.json's
//               moshi_name. Use this to A/B quantizations produced by
//               tools/requantize_gguf.
//
// NOTE: there is deliberately no `quant` option. moshi_lm_quantize() is a silent
// no-op for GGUF inputs (src/loader.h fetch() returns get_tensor() without ever
// consulting dst_type/qtype), so passing "-q q4_k" against a GGUF measures the
// file's existing types and nothing else. Quantize offline instead.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <moshi/moshi.h>

static double elapsed_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Minimal 16-bit PCM WAV reader (same logic as voxtral's test_stream).
static bool load_wav(const char* path, std::vector<float>& out, int& sample_rate) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    char riff[4]; size_t r = fread(riff, 1, 4, f); (void)r;
    if (memcmp(riff, "RIFF", 4) != 0) { fclose(f); return false; }
    uint32_t file_size; r = fread(&file_size, 4, 1, f);
    char wave[4]; r = fread(wave, 1, 4, f);
    while (1) {
        char id[4]; if (fread(id, 1, 4, f) != 4) break;
        uint32_t size; r = fread(&size, 4, 1, f);
        if (memcmp(id, "fmt ", 4) == 0) {
            uint16_t fmt, channels; uint32_t sr, bps; uint16_t align, bits;
            r = fread(&fmt, 2, 1, f); r = fread(&channels, 2, 1, f);
            r = fread(&sr, 4, 1, f); r = fread(&bps, 4, 1, f);
            r = fread(&align, 2, 1, f); r = fread(&bits, 2, 1, f);
            sample_rate = sr;
            if (size > 16) fseek(f, size - 16, SEEK_CUR);
        } else if (memcmp(id, "data", 4) == 0) {
            int n_samples = size / 2;
            std::vector<int16_t> raw(n_samples);
            r = fread(raw.data(), 2, n_samples, f);
            out.resize(n_samples);
            for (int i = 0; i < n_samples; i++) out[i] = raw[i] / 32768.0f;
            break;
        } else { fseek(f, size, SEEK_CUR); }
    }
    fclose(f);
    return !out.empty();
}

// Linear resample to the codec's rate. Our fixtures are 16 kHz (what the app
// captures) and Mimi wants 24 kHz. NOTE: upsampling adds no content above 8 kHz,
// so this is a domain shift vs the 24 kHz audio the model was trained on. A fair
// quality verdict needs a natively-24 kHz fixture; see the baseline notes.
static void resample_linear(const std::vector<float>& in, int in_rate,
                            std::vector<float>& out, int out_rate) {
    if (in_rate == out_rate) { out = in; return; }
    const double ratio = (double)in_rate / (double)out_rate;
    const size_t n_out = (size_t)((double)in.size() / ratio);
    out.resize(n_out);
    for (size_t i = 0; i < n_out; i++) {
        const double pos = i * ratio;
        const size_t i0 = (size_t)pos;
        const size_t i1 = std::min(i0 + 1, in.size() - 1);
        const double frac = pos - (double)i0;
        out[i] = (float)((1.0 - frac) * in[i0] + frac * in[i1]);
    }
}

// SentencePiece pieces encode spaces as U+2581 (0xE2 0x96 0x81); same unpacking
// as moshi-stt.cpp.
static std::string detok(const std::string& piece) {
    std::string s;
    for (size_t i = 0; i < piece.size(); i++) {
        if ((signed char)piece[i] == -30) { s += ' '; i += 2; continue; }
        s += piece[i];
    }
    return s;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model_dir> <audio.wav> [threads] [model_gguf]\n", argv[0]);
        return 1;
    }
    std::string model_dir = argv[1];
    const char* audio_path = argv[2];
    const int threads = argc > 3 ? atoi(argv[3]) : 6;
    const char* model_override = argc > 4 ? argv[4] : nullptr;
    if (!model_dir.empty() && model_dir.back() != '/') model_dir += '/';

    std::vector<float> audio_in; int in_rate = 0;
    if (!load_wav(audio_path, audio_in, in_rate)) {
        fprintf(stderr, "failed to load %s\n", audio_path); return 1;
    }
    const double audio_sec = (double)audio_in.size() / in_rate;
    printf("audio: %.1fs at %dHz\n", audio_sec, in_rate);

    moshi_config_t cfg;
    const std::string cfg_path = model_dir + "config.json";
    if (moshi_get_config(&cfg, cfg_path.c_str()) != 0) {
        fprintf(stderr, "failed to read %s\n", cfg_path.c_str()); return 1;
    }
    printf("config: dim=%lld layers=%lld n_q=%lld dep_q=%lld context=%lld delay=%.2fs\n",
           (long long)cfg.dim, (long long)cfg.num_layers, (long long)cfg.n_q,
           (long long)cfg.dep_q, (long long)cfg.context,
           cfg.stt_config.audio_delay_seconds);

    // Backend. STT_BACKEND=gpu offloads to the first non-CPU device (Adreno via OpenCL on
    // this phone); anything else stays on CPU. moshi keeps a CPU backend alongside for ops
    // the accelerator does not implement.
    ggml_backend_load_all();
    const char* want = getenv("STT_BACKEND");
    const bool want_gpu = want && !strcmp(want, "gpu");

    ggml_backend* backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend_cpu) { fprintf(stderr, "no cpu backend\n"); return 1; }
    ggml_backend* backend = backend_cpu;

    if (want_gpu) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            auto dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) continue;
            auto b = ggml_backend_dev_init(dev, nullptr);
            if (b) { backend = b; printf("backend: %s\n", ggml_backend_dev_name(dev)); break; }
        }
        if (backend == backend_cpu)
            fprintf(stderr, "warning: no non-CPU backend found, staying on CPU\n");
    }
    if (backend == backend_cpu) printf("backend: CPU\n");
    {
        auto dev = ggml_backend_get_device(backend_cpu);
        auto reg = ggml_backend_dev_backend_reg(dev);
        auto set_n_threads = (ggml_backend_set_n_threads_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_n_threads) set_n_threads(backend_cpu, threads);
    }

    moshi_context_t* moshi = moshi_alloc(backend, backend_cpu);

    auto t0 = std::chrono::steady_clock::now();
    const std::string lm_path = model_dir + (model_override ? model_override : cfg.moshi_name);
    moshi_lm_t* lm = moshi_lm_from_files(moshi, &cfg, lm_path.c_str());
    if (!lm) { fprintf(stderr, "failed to open %s\n", lm_path.c_str()); return 1; }
    moshi_lm_gen_t* gen = moshi_lm_generator(lm);

    const std::string tok_path  = model_dir + cfg.tokenizer_name;
    const std::string mimi_path = model_dir + cfg.mimi_name;
    tokenizer_t* tok = tokenizer_alloc(tok_path.c_str());
    mimi_codec_t* codec = mimi_alloc(moshi, mimi_path.c_str(), (int)cfg.n_q);
    if (!tok || !codec) { fprintf(stderr, "failed to load tokenizer/codec\n"); return 1; }

    if (moshi_lm_load(lm) != 0) { fprintf(stderr, "failed to load weights\n"); return 1; }
    const double load_ms = elapsed_ms(t0);
    printf("load: %.0f ms  (%s)\n", load_ms, lm_path.c_str());

    const float frame_rate = mimi_frame_rate(codec);
    const int frame_size = mimi_frame_size(codec);
    const int codec_rate = (int)(frame_rate * frame_size + 0.5f);
    printf("codec: %.2f frames/s, frame_size=%d -> %d Hz\n",
           frame_rate, frame_size, codec_rate);

    std::vector<float> audio;
    resample_linear(audio_in, in_rate, audio, codec_rate);
    if (in_rate != codec_rate) {
        printf("resampled %d -> %d Hz (%zu -> %zu samples)\n",
               in_rate, codec_rate, audio_in.size(), audio.size());
    }

    mimi_encode_context_t* enc = mimi_encode_alloc_context(codec);
    // STT_RESET_FIRST reproduces what the app does: KyutaiLocalClient.streamReset() is called
    // before the first frame. The bench never did, which is exactly how a reset regression
    // could pass every bench run and still break the app.
    if (getenv("STT_RESET_FIRST")) { printf("reset before first frame\n"); mimi_encode_reset(enc); }
    moshi_lm_start(moshi, gen, cfg.lm_gen_config.temp, cfg.lm_gen_config.temp_text);

    // Feed frame by frame — exactly what the JNI layer would do.
    // Tail padding flushes the 0.5 s text delay.
    const int tail_frames = (int)(cfg.stt_config.audio_delay_seconds * frame_rate) + 8;
    const int n_frames = (int)(audio.size() / frame_size);
    std::vector<int16_t> tokens(cfg.n_q);
    std::vector<float> silence(frame_size, 0.0f);
    std::string full_text;
    int text_tokens = 0, vad_hits = 0;

    printf("\n=== Streaming (%d frames + %d tail, %d threads) ===\n",
           n_frames, tail_frames, threads);

    const int spec_n = getenv("STT_SPEC_N") ? std::max(1, atoi(getenv("STT_SPEC_N"))) : 1;
    if (spec_n > 1) printf("speculation: %d frames per LM pass\n", spec_n);

    double compute_ms = 0, mimi_ms = 0, lm_ms = 0;
    double worst_frame_ms = 0;
    long lm_passes = 0, spec_accepted = 0;
    // Audio tokens for frames encoded but not yet consumed by the LM.
    std::vector<std::vector<int16_t>> pending_frames;

    // STT_SESSIONS replays the audio N times through ONE context, with exactly the sequence
    // the app performs between two capture sessions: the tail silence (KyutaiLocalClient
    // .streamFlush) followed by a codec-only reset (.streamReset, which deliberately keeps
    // the LM context). Chasing sessions that produced zero text needs this, because a
    // single-session bench cannot see a state problem that only appears on session 2.
    const int sessions = getenv("STT_SESSIONS") ? std::max(1, atoi(getenv("STT_SESSIONS"))) : 1;
    std::vector<size_t> session_chars;

    for (int s = 0; s < sessions; s++) {
    if (s > 0) {
        mimi_encode_reset(enc);
        printf("\n--- session %d: codec reset, LM context kept (as the app does)\n", s + 1);
    }
    const size_t chars_at_session_start = full_text.size();

    for (int i = 0; i < n_frames + tail_frames; i++) {
        float* frame = (i < n_frames) ? (audio.data() + (size_t)i * frame_size)
                                      : silence.data();
        auto tf = std::chrono::steady_clock::now();
        mimi_encode_send(enc, frame);
        mimi_encode_receive(enc, tokens.data());
        const double mimi_frame_ms = elapsed_ms(tf);
        mimi_ms += mimi_frame_ms;

        int text_token = 0; float vad = 0;
        double lm_frame_ms = 0;

        if (spec_n > 1) {
            pending_frames.push_back(tokens);
            const bool last = (i == n_frames + tail_frames - 1);
            if ((int)pending_frames.size() < spec_n && !last) {
                compute_ms += mimi_frame_ms;
                continue;   // accumulate until we have a full speculation window
            }
            auto tl = std::chrono::steady_clock::now();
            while (!pending_frames.empty()) {
                std::vector<int> toks; std::vector<float> vs;
                const int acc = moshi_lm_step_batch(gen, pending_frames, toks, vs);
                lm_passes++; spec_accepted += acc;
                for (size_t k = 0; k < toks.size(); k++) {
                    if (vs[k] > 0.5f) vad_hits++;
                    if (toks[k] != 0 && toks[k] != 3) {
                        full_text += detok(tokenizer_id_to_piece(tok, toks[k]));
                        text_tokens++;
                    }
                }
                pending_frames.erase(pending_frames.begin(), pending_frames.begin() + acc);
            }
            lm_frame_ms = elapsed_ms(tl);
            lm_ms += lm_frame_ms;
            compute_ms += mimi_frame_ms + lm_frame_ms;
            worst_frame_ms = std::max(worst_frame_ms, mimi_frame_ms + lm_frame_ms);
            continue;
        }

        auto tl = std::chrono::steady_clock::now();
        moshi_lm_send2(gen, tokens);
        moshi_lm_receive2(gen, text_token, vad);
        lm_frame_ms = elapsed_ms(tl);
        lm_ms   += lm_frame_ms;
        const double frame_ms = mimi_frame_ms + lm_frame_ms;
        compute_ms += frame_ms;
        worst_frame_ms = std::max(worst_frame_ms, frame_ms);
        lm_passes++; spec_accepted++;

        if (vad > 0.5f) vad_hits++;
        // Dump the text-stream pattern: 0/3 are padding, anything else is a real token.
        // Used to measure how speculatable the text stream is — see BENCH.md.
        if (getenv("STT_DUMP_TOKENS"))
            fprintf(stderr, "TOK %d %d\n", i, (text_token != 0 && text_token != 3) ? 1 : 0);
        // 0 = pad, 3 = existing_text_padding_id
        if (text_token != 0 && text_token != 3) {
            full_text += detok(tokenizer_id_to_piece(tok, text_token));
            text_tokens++;
        }
        // Progress once per second of audio, with a rolling RTF.
        if ((i + 1) % (int)frame_rate == 0) {
            printf("  %5.1fs  rtf=%.3f  chars=%zu\n",
                   (i + 1) / frame_rate, compute_ms / ((i + 1) / frame_rate * 1000.0),
                   full_text.size());
        }
    }
    session_chars.push_back(full_text.size() - chars_at_session_start);
    }

    // Every duration below is over the audio actually processed, i.e. sessions x the file.
    const double audio_total_sec = audio_sec * sessions;
    const double frame_ms_budget = 1000.0 / frame_rate;
    printf("\n=== Results ===\n");
    if (sessions > 1) {
        printf("chars per session:");
        for (size_t s = 0; s < session_chars.size(); s++)
            printf(" %zu", session_chars[s]);
        printf("   (identical audio each time -- any drop is a state bug)\n");
    }
    printf("frames: %d (+%d tail)\n", n_frames, tail_frames);
    printf("text tokens: %d, vad frames >0.5: %d\n", text_tokens, vad_hits);
    if (spec_n > 1)
        printf("speculation: %ld LM passes for %ld frames = %.2f frames/pass "
               "(ideal %d)\n", lm_passes, spec_accepted,
               (double)spec_accepted / lm_passes, spec_n);
    printf("compute: %.0f ms for %.0f ms audio = %.3fx RT\n",
           compute_ms, audio_total_sec * 1000.0, compute_ms / (audio_total_sec * 1000.0));
    const int total_frames = n_frames + tail_frames;
    printf("per-frame: %.2f ms avg (budget %.2f ms), worst %.1f ms\n",
           compute_ms / total_frames, frame_ms_budget, worst_frame_ms);
    printf("  mimi encoder: %.2f ms/frame (%.0f%% of compute)\n",
           mimi_ms / total_frames, 100.0 * mimi_ms / compute_ms);
    printf("  lm (1B)     : %.2f ms/frame (%.0f%% of compute)\n",
           lm_ms / total_frames, 100.0 * lm_ms / compute_ms);
    printf("full text: %s\n", full_text.c_str());
    printf("RESULT engine=kyutai-stt-1b audio=%s threads=%d weights=%s "
           "spec_n=%d load_ms=%.0f compute_ms=%.0f mimi_ms=%.0f lm_ms=%.0f "
           "audio_ms=%.0f rtf=%.3f chars=%zu\n",
           audio_path, threads, model_override ? model_override : cfg.moshi_name.c_str(),
           spec_n, load_ms, compute_ms, mimi_ms, lm_ms, audio_total_sec * 1000.0,
           compute_ms / (audio_total_sec * 1000.0), full_text.size());
    return 0;
}
