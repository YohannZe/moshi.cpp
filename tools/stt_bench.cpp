// Streaming STT benchmark for kyutai stt-1b-en_fr via moshi.cpp.
//
// Deliberately depends on NOTHING but libmoshi + ggml — no SDL, no FFmpeg.
// That is the point: it proves the model path is portable to Android/JNI, and it
// doubles as the prototype of the JNI feed loop.
//
// Mirrors voxtral.cpp/tools/test_stream.cpp so results are directly comparable:
// same fixtures, same RESULT line, same RTF definition (compute_ms / audio_ms).
//
// usage: stt_bench <model_dir> <audio.wav> [threads] [quant]
//   model_dir  dir holding config.json (mimi + tokenizer resolved relative to it)
//   audio.wav  16-bit PCM mono WAV, any sample rate (resampled to 24 kHz)
//   quant      q8_0 | q4_k | q4_0  (omit for the model's native precision)

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
        fprintf(stderr, "usage: %s <model_dir> <audio.wav> [threads] [quant]\n", argv[0]);
        return 1;
    }
    std::string model_dir = argv[1];
    const char* audio_path = argv[2];
    const int threads = argc > 3 ? atoi(argv[3]) : 6;
    const char* quant  = argc > 4 ? argv[4] : nullptr;
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

    // Backend: CPU only, to match the voxtral baseline.
    ggml_backend_load_all();
    ggml_backend* backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) { fprintf(stderr, "no cpu backend\n"); return 1; }
    {
        auto dev = ggml_backend_get_device(backend);
        auto reg = ggml_backend_dev_backend_reg(dev);
        auto set_n_threads = (ggml_backend_set_n_threads_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_n_threads) set_n_threads(backend, threads);
    }

    moshi_context_t* moshi = moshi_alloc(backend, backend);

    auto t0 = std::chrono::steady_clock::now();
    const std::string lm_path  = model_dir + cfg.moshi_name;
    moshi_lm_t* lm = moshi_lm_from_files(moshi, &cfg, lm_path.c_str());
    if (!lm) { fprintf(stderr, "failed to open %s\n", lm_path.c_str()); return 1; }
    if (quant && !moshi_lm_quantize(lm, quant)) {
        fprintf(stderr, "unknown quant %s\n", quant); return 1;
    }
    moshi_lm_gen_t* gen = moshi_lm_generator(lm);

    const std::string tok_path  = model_dir + cfg.tokenizer_name;
    const std::string mimi_path = model_dir + cfg.mimi_name;
    tokenizer_t* tok = tokenizer_alloc(tok_path.c_str());
    mimi_codec_t* codec = mimi_alloc(moshi, mimi_path.c_str(), (int)cfg.n_q);
    if (!tok || !codec) { fprintf(stderr, "failed to load tokenizer/codec\n"); return 1; }

    if (moshi_lm_load(lm) != 0) { fprintf(stderr, "failed to load weights\n"); return 1; }
    const double load_ms = elapsed_ms(t0);
    printf("load: %.0f ms%s\n", load_ms, quant ? " (incl. quantize)" : "");

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

    double compute_ms = 0, mimi_ms = 0, lm_ms = 0;
    double worst_frame_ms = 0;
    for (int i = 0; i < n_frames + tail_frames; i++) {
        float* frame = (i < n_frames) ? (audio.data() + (size_t)i * frame_size)
                                      : silence.data();
        auto tf = std::chrono::steady_clock::now();
        mimi_encode_send(enc, frame);
        mimi_encode_receive(enc, tokens.data());
        const double mimi_frame_ms = elapsed_ms(tf);
        auto tl = std::chrono::steady_clock::now();
        moshi_lm_send2(gen, tokens);
        int text_token = 0; float vad = 0;
        moshi_lm_receive2(gen, text_token, vad);
        const double lm_frame_ms = elapsed_ms(tl);
        const double frame_ms = mimi_frame_ms + lm_frame_ms;
        mimi_ms += mimi_frame_ms;
        lm_ms   += lm_frame_ms;
        compute_ms += frame_ms;
        worst_frame_ms = std::max(worst_frame_ms, frame_ms);

        if (vad > 0.5f) vad_hits++;
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

    const double frame_ms_budget = 1000.0 / frame_rate;
    printf("\n=== Results ===\n");
    printf("frames: %d (+%d tail)\n", n_frames, tail_frames);
    printf("text tokens: %d, vad frames >0.5: %d\n", text_tokens, vad_hits);
    printf("compute: %.0f ms for %.0f ms audio = %.3fx RT\n",
           compute_ms, audio_sec * 1000.0, compute_ms / (audio_sec * 1000.0));
    const int total_frames = n_frames + tail_frames;
    printf("per-frame: %.2f ms avg (budget %.2f ms), worst %.1f ms\n",
           compute_ms / total_frames, frame_ms_budget, worst_frame_ms);
    printf("  mimi encoder: %.2f ms/frame (%.0f%% of compute)\n",
           mimi_ms / total_frames, 100.0 * mimi_ms / compute_ms);
    printf("  lm (1B)     : %.2f ms/frame (%.0f%% of compute)\n",
           lm_ms / total_frames, 100.0 * lm_ms / compute_ms);
    printf("full text: %s\n", full_text.c_str());
    printf("RESULT engine=kyutai-stt-1b audio=%s threads=%d quant=%s "
           "load_ms=%.0f compute_ms=%.0f mimi_ms=%.0f lm_ms=%.0f "
           "audio_ms=%.0f rtf=%.3f chars=%zu\n",
           audio_path, threads, quant ? quant : "native",
           load_ms, compute_ms, mimi_ms, lm_ms, audio_sec * 1000.0,
           compute_ms / (audio_sec * 1000.0), full_text.size());
    return 0;
}
