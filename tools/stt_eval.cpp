// Batch transcription for WER evaluation (FLEURS etc.) — loads the model ONCE and streams
// every WAV in a list through it, printing one "HYP\t<basename>\t<text>" line per file.
//
// Exists because stt_bench does one file per process: 676 FLEURS utterances would pay 676
// model loads. This tool reuses stt_bench's exact feed loop (frame-by-frame, tail-silence
// flush), so the WER it produces is the WER of the deployed pipeline, not of a friendlier
// batch mode.
//
// Session semantics, stated because a reviewer will ask: the Mimi codec state is reset
// between utterances (mimi_encode_reset — fresh conv rings), but the LM keeps its rolling
// context window across files, exactly as the Katarina app keeps it across capture
// sessions. This is deliberate deployment-mode evaluation; utterance-independent evaluation
// would need one LM context per file, and moshi_lm_start is not re-entrant.
//
// usage: stt_eval <model_dir> <list.txt> [threads] [model_gguf]
//   list.txt   one absolute WAV path per line (16-bit PCM mono, any rate)
//   STT_MIMI   overrides the codec file, as in stt_bench
//
// Serving-protocol knobs (the WER-relevant layer -- these, not the engine, are where our
// port beats the reference implementation, so they deserve to be sweepable):
//   STT_TAIL_EXTRA=n   silence frames flushed beyond the theoretical delay (default 8).
//                      The reference script flushes ceil(delay*fps)=7 total and strands
//                      trailing words; we default to delay+8=14.
//   STT_PREFIX=n       silence frames fed before each utterance (default 0).
//   STT_AGC=1          causal automatic gain control on the input, per frame: tracks a
//                      decaying peak envelope and scales toward -3 dBFS. Exists because a
//                      neural-codec STT is input-gain sensitive (FLEURS at -46 dBFS: 17.6%
//                      WER raw vs 11.4% normalized) and a real capture pipeline cannot
//                      normalize offline. Same algorithm as the app's JNI layer.
//   STT_RESAMPLE=linear  fall back to 2-point linear interpolation for rate conversion.
//                      Default is windowed-sinc (Hann, 24 zero-crossings, rolloff 0.945 —
//                      julius' defaults, which measured -0.84 WER pt vs linear on the
//                      FLEURS 113 subset). Linear interpolation's sinc^2 response leaves
//                      spectral IMAGES of the input band above the source Nyquist
//                      (imaging, not aliasing: 16->24 kHz is upsampling), energy Mimi
//                      never saw in training.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <moshi/moshi.h>
#include "../src/moshi/models/text_bias.h"

static bool load_wav(const char* path, std::vector<float>& out, int& sample_rate) {
    out.clear();
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


// Windowed-sinc resampling (Hann window, ZEROS zero-crossings per side, rolloff on the
// lower Nyquist). Offline form — the whole file is available; the app's streaming path
// needs the same kernel in polyphase form. Matches julius.resample_frac's design
// parameters, which produced the sinc arm of the 2026-08 sweep.
static void resample_sinc(const std::vector<float>& in, int in_rate,
                          std::vector<float>& out, int out_rate) {
    if (in_rate == out_rate) { out = in; return; }
    // Sweepable: 16->24 kHz upsampling leaves spectral images above the source Nyquist that
    // Mimi never saw in training, and where/how hard we cut them is a free quality knob.
    const int    ZEROS   = getenv("STT_RS_ZEROS")   ? atoi(getenv("STT_RS_ZEROS")) : 24;
    const double ROLLOFF = getenv("STT_RS_ROLLOFF") ? atof(getenv("STT_RS_ROLLOFF")) : 0.945;
    const double ratio = (double)in_rate / (double)out_rate;   // input samples per output
    // cutoff in cycles per INPUT sample: half the lower of the two Nyquists, scaled back
    const double c = 0.5 * ROLLOFF * std::min(1.0, (double)out_rate / (double)in_rate);
    const double W = ZEROS / (2.0 * c);                        // kernel half-width, input samples
    const size_t n_out = (size_t)((double)in.size() / ratio);
    out.assign(n_out, 0.0f);
    for (size_t i = 0; i < n_out; i++) {
        const double t = i * ratio;
        const long k0 = std::max(0L, (long)std::ceil(t - W));
        const long k1 = std::min((long)in.size() - 1, (long)std::floor(t + W));
        double acc = 0.0;
        for (long k = k0; k <= k1; k++) {
            const double u = t - (double)k;
            const double x = 2.0 * c * u;
            const double s = (x == 0.0) ? 1.0 : std::sin(M_PI * x) / (M_PI * x);
            const double w = 0.5 * (1.0 + std::cos(M_PI * u / W));   // Hann over |u|<=W
            acc += in[(size_t)k] * 2.0 * c * s * w;
        }
        out[i] = (float)acc;
    }
}

static void resample(const std::vector<float>& in, int in_rate,
                     std::vector<float>& out, int out_rate) {
    const char* mode = getenv("STT_RESAMPLE");
    if (mode && strcmp(mode, "linear") == 0) resample_linear(in, in_rate, out, out_rate);
    else                                     resample_sinc(in, in_rate, out, out_rate);
}

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
        fprintf(stderr, "usage: %s <model_dir> <list.txt> [threads] [model_gguf]\n", argv[0]);
        return 1;
    }
    std::string model_dir = argv[1];
    const char* list_path = argv[2];
    const int threads = argc > 3 ? atoi(argv[3]) : 6;
    const char* model_override = argc > 4 ? argv[4] : nullptr;
    if (!model_dir.empty() && model_dir.back() != '/') model_dir += '/';

    std::vector<std::string> files;
    {
        FILE* f = fopen(list_path, "r");
        if (!f) { fprintf(stderr, "cannot read %s\n", list_path); return 1; }
        char line[4096];
        while (fgets(line, sizeof(line), f)) {
            std::string s(line);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            if (!s.empty()) files.push_back(s);
        }
        fclose(f);
    }
    fprintf(stderr, "%zu files\n", files.size());

    moshi_config_t cfg;
    const std::string cfg_path = model_dir + "config.json";
    if (moshi_get_config(&cfg, cfg_path.c_str()) != 0) {
        fprintf(stderr, "failed to read %s\n", cfg_path.c_str()); return 1;
    }

    ggml_backend_load_all();
    ggml_backend* backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    {
        auto dev = ggml_backend_get_device(backend);
        auto reg = ggml_backend_dev_backend_reg(dev);
        auto set_n_threads = (ggml_backend_set_n_threads_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_n_threads) set_n_threads(backend, threads);
    }

    moshi_context_t* moshi = moshi_alloc(backend, backend);
    const std::string lm_path = model_dir + (model_override ? model_override : cfg.moshi_name);
    moshi_lm_t* lm = moshi_lm_from_files(moshi, &cfg, lm_path.c_str());
    if (!lm) { fprintf(stderr, "failed to open %s\n", lm_path.c_str()); return 1; }
    moshi_lm_gen_t* gen = moshi_lm_generator(lm);

    const char* mimi_override = getenv("STT_MIMI");
    const std::string tok_path  = model_dir + cfg.tokenizer_name;
    const std::string mimi_path = model_dir + (mimi_override ? mimi_override : cfg.mimi_name);
    tokenizer_t* tok = tokenizer_alloc(tok_path.c_str());
    mimi_codec_t* codec = mimi_alloc(moshi, mimi_path.c_str(), (int)cfg.n_q);
    if (!tok || !codec) { fprintf(stderr, "failed to load tokenizer/codec\n"); return 1; }
    // from_files only OPENS the GGUF; this loads the weights. Omitting it leaves every
    // tensor null and the first graph build segfaults in ggml_get_rows.
    if (moshi_lm_load(lm) != 0) { fprintf(stderr, "failed to load weights\n"); return 1; }

    const float frame_rate = mimi_frame_rate(codec);
    const int   frame_size = mimi_frame_size(codec);
    const int   codec_rate = (int)(frame_rate * frame_size + 0.5f);
    const int   tail_extra = getenv("STT_TAIL_EXTRA") ? atoi(getenv("STT_TAIL_EXTRA")) : 8;
    const int   tail_frames = (int)(cfg.stt_config.audio_delay_seconds * frame_rate) + tail_extra;
    const int   prefix_frames = getenv("STT_PREFIX") ? atoi(getenv("STT_PREFIX")) : 0;
    const bool  agc_on = getenv("STT_AGC") && getenv("STT_AGC")[0] == '1';
    const float hpf_hz  = getenv("STT_HPF")     ? (float)atof(getenv("STT_HPF")) : 0.f;
    const float preemph = getenv("STT_PREEMPH") ? (float)atof(getenv("STT_PREEMPH")) : 0.f;
    fprintf(stderr, "front-end: resample=%s zeros=%s rolloff=%s hpf=%.0fHz preemph=%.2f\n",
            getenv("STT_RESAMPLE") ? getenv("STT_RESAMPLE") : "sinc",
            getenv("STT_RS_ZEROS") ? getenv("STT_RS_ZEROS") : "24",
            getenv("STT_RS_ROLLOFF") ? getenv("STT_RS_ROLLOFF") : "0.945",
            hpf_hz, preemph);
    fprintf(stderr, "serving: tail=%d (delay+%d) prefix=%d agc=%d\n",
            tail_frames, tail_extra, prefix_frames, agc_on);

    mimi_encode_context_t* enc = mimi_encode_alloc_context(codec);
    moshi_lm_start(moshi, gen, cfg.lm_gen_config.temp, cfg.lm_gen_config.temp_text);

    std::vector<int16_t> tokens(cfg.n_q);
    std::vector<float> silence(frame_size, 0.0f);
    double total_audio_s = 0, total_compute_ms = 0;

    for (size_t fi = 0; fi < files.size(); fi++) {
        std::vector<float> audio_in, audio;
        int in_rate = 0;
        if (!load_wav(files[fi].c_str(), audio_in, in_rate)) {
            fprintf(stderr, "SKIP %s (load failed)\n", files[fi].c_str());
            continue;
        }
        resample(audio_in, in_rate, audio, codec_rate);

        // Front-end shaping, after resampling so coefficients mean the same thing whatever
        // the source rate.
        if (hpf_hz > 0.f) {   // one-pole high-pass: DC/rumble costs the codec real codes
            const float a = expf(-2.f * (float)M_PI * hpf_hz / (float)codec_rate);
            float px = 0.f, py = 0.f;
            for (size_t i = 0; i < audio.size(); i++) {
                const float x = audio[i];
                py = a * (py + x - px); px = x; audio[i] = py;
            }
        }
        if (preemph != 0.f) { // y[n]=x[n]-k*x[n-1]: tilts energy toward the band top, which
            float prev = 0.f; // is exactly where a 16 kHz source is empty vs 24 kHz training
            for (size_t i = 0; i < audio.size(); i++) {
                const float x = audio[i];
                audio[i] = x - preemph * prev; prev = x;
            }
        }
        total_audio_s += (double)audio_in.size() / in_rate;

        mimi_encode_reset(enc);   // fresh conv state per utterance; LM context persists
        moshi_margin_reset();

        // Causal AGC — strictly speaking a causal peak normalizer / limiter: one-pole peak
        // envelope with instantaneous attack (a loud sample raises the envelope
        // immediately) and slow exponential release (tau = 4 s, i.e. ~2.77 s to halve),
        // gain aimed at -3 dBFS peak, capped at 40x so digital silence is not amplified
        // into hiss. Causal by construction: the gain applied to a sample depends only on
        // past samples, so the identical code can run in the live capture path. Note the
        // envelope persists across files (like a live stream), which makes AGC-arm WER
        // depend on file-list order; STT_AGC_RESET=1 resets it per utterance.
        if (agc_on) {
            static float env = 0.f;                  // persists across files, like a live stream
            if (getenv("STT_AGC_RESET")) env = 0.f;
            const float target = 0.7f, gmax = 40.f;
            const float release = expf(-1.f / (4.f * codec_rate));  // per-sample decay
            for (size_t i = 0; i < audio.size(); i++) {
                const float a = fabsf(audio[i]);
                env = a > env ? a : env * release;
                const float g = env > 1e-4f ? (target / env < gmax ? target / env : gmax)
                                            : gmax;
                audio[i] *= g;
            }
        }

        const int n_frames = (int)(audio.size() / frame_size);
        std::string text;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = -prefix_frames; i < n_frames + tail_frames; i++) {
            float* frame = (i >= 0 && i < n_frames)
                               ? (audio.data() + (size_t)i * frame_size)
                               : silence.data();
            mimi_encode_send(enc, frame);
            mimi_encode_receive(enc, tokens.data());
            moshi_lm_send2(gen, tokens);
            int text_token = 0; float vad = 0;
            moshi_lm_receive2(gen, text_token, vad);
            if (text_token != 0 && text_token != 3)
                text += detok(tokenizer_id_to_piece(tok, text_token));
        }
        total_compute_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

        const char* base = strrchr(files[fi].c_str(), '/');
        base = base ? base + 1 : files[fi].c_str();
        printf("HYP\t%s\t%s\n", base, text.c_str());
        if (getenv("MOSHI_MARGIN")) {
            float mn, mean; int n, n_low;
            moshi_margin_stats(&mn, &mean, &n, &n_low);
            printf("CONF\t%s\t%.3f\t%.3f\t%d\t%d\n", base, mn, mean, n, n_low);
        }
        if (getenv("MOSHI_TOPK_DUMP")) {   // utterance boundary marker for the dump file
            if (FILE * tf = moshi_topk_file()) fprintf(tf, "# %s\n", base);
        }
        fflush(stdout);
        if ((fi + 1) % 25 == 0)
            fprintf(stderr, "  %zu/%zu  rtf=%.3f\n", fi + 1, files.size(),
                    total_compute_ms / (total_audio_s * 1000.0));
    }

    fprintf(stderr, "RESULT eval files=%zu audio_s=%.0f compute_ms=%.0f rtf=%.3f\n",
            files.size(), total_audio_s, total_compute_ms,
            total_compute_ms / (total_audio_s * 1000.0));
    return 0;
}
