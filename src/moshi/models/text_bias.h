#pragma once
// Greedy-mode shallow fusion with a token bigram LM (NGPU-LM style, arXiv:2505.22857 —
// greedy fusion recovers most of the greedy-vs-beam gap without a beam).
//
// Constraint specific to delayed-streams STT: most frames emit padding (id<=3), and the
// bigram knows nothing about frame timing — naive fusion over the full vocabulary would
// re-rank the pad-vs-word decision and hallucinate words into silence. So the hook only
// re-ranks WHICH word, never WHETHER a word: if the acoustic argmax is padding, it stands;
// if it is a real token, the fused score (logit + lambda * logp_bigram(t | prev_word))
// re-picks among real tokens only. This targets substitutions — the dominant error class —
// and cannot change segmentation.
//
// Enabled by MOSHI_TEXT_BIAS=<table file> (+ MOSHI_TEXT_BIAS_LAMBDA, default 0.5).
// Table lines: "<prev> <next> <logprob>", prev=-2 rows are the unigram backoff.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <unordered_map>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>

struct moshi_text_bias_t {
    bool  tried = false, on = false;
    float lambda = 0.5f;
    float floor = -12.f;                                  // unseen-token logprob
    std::unordered_map<long long, float> bi;              // key = prev * 65536 + next
    std::unordered_map<int, float> uni;
    int prev = -1;                                        // last real token; -1 = start
    std::vector<float> logits;
};

inline moshi_text_bias_t & moshi_text_bias() {
    static moshi_text_bias_t s;
    return s;
}

inline void moshi_text_bias_init() {
    auto & s = moshi_text_bias();
    s.tried = true;
    const char * path = getenv("MOSHI_TEXT_BIAS");
    if (!path || !path[0]) return;
    FILE * f = fopen(path, "r");
    if (!f) { fprintf(stderr, "text-bias: cannot read %s\n", path); return; }
    long long p; int t; float lp;
    size_t n = 0;
    while (fscanf(f, "%lld %d %f", &p, &t, &lp) == 3) {
        if (p == -2) s.uni[t] = lp;
        else         s.bi[p * 65536 + t] = lp;
        n++;
    }
    fclose(f);
    if (getenv("MOSHI_TEXT_BIAS_LAMBDA")) s.lambda = atof(getenv("MOSHI_TEXT_BIAS_LAMBDA"));
    s.on = true;
    fprintf(stderr, "text-bias: %zu entries, lambda=%.2f\n", n, s.lambda);
}

inline float moshi_text_bias_lp(moshi_text_bias_t & s, int prev, int t) {
    if (prev >= 0) {
        auto it = s.bi.find((long long)prev * 65536 + t);
        if (it != s.bi.end()) return it->second;
    }
    auto iu = s.uni.find(t);
    // backoff to unigram with a mild penalty; unseen everywhere gets the floor
    return iu != s.uni.end() ? iu->second - 1.f : s.floor;
}

// ---- Confidence instrumentation (MOSHI_MARGIN=1) ----------------------------------------
//
// The top1-top2 logit margin at each real-token decision, accumulated per utterance. This is
// the gating signal for adaptive test-time compute: ensemble votes only ever differ where
// the model hesitated, so extra compute (ensemble members, lookahead branches) should be
// spent only on low-margin decisions. Same idea as speculative decoding's cheap-draft /
// verify split, applied to decoding confidence.
struct moshi_margin_t {
    bool tried = false, on = false;
    float min_margin = 1e9f;
    double sum = 0.0;
    int n = 0, n_low = 0;          // n_low: decisions with margin < 2.0
    std::vector<float> logits;
};
inline moshi_margin_t & moshi_margin() { static moshi_margin_t s; return s; }
extern "C" inline void moshi_margin_reset() {
    auto & m = moshi_margin();
    m.min_margin = 1e9f; m.sum = 0.0; m.n = 0; m.n_low = 0;
}
extern "C" inline void moshi_margin_stats(float * mn, float * mean, int * n, int * n_low) {
    auto & m = moshi_margin();
    *mn = m.n ? m.min_margin : 1e9f;
    *mean = m.n ? (float)(m.sum / m.n) : 0.f;
    *n = m.n; *n_low = m.n_low;
}
// MOSHI_TOPK_DUMP=<file>: per real-token decision, "tok1 tok2 margin". Feeds the 1-swap
// oracle analysis that decides whether a beam is worth building: if the reference word is
// reachable by taking top-2 at a low-margin frame, beam search can capture it inside a
// single decode (batch dim, ~1.15x compute) instead of N full passes (Nx).
inline FILE * moshi_topk_file() {
    static FILE * f = nullptr;
    static bool tried = false;
    if (!tried) {
        tried = true;
        const char * p = getenv("MOSHI_TOPK_DUMP");
        if (p && p[0]) f = fopen(p, "w");
    }
    return f;
}

// Sequence score and forced divergence, for the beam-search feasibility question.
//
// A beam is only useful if the model can rank its own paths without the reference. So:
// MOSHI_FORCE_ALT_AT=<n> takes the SECOND-best token at the n-th real-token decision (and
// greedy everywhere else), and MOSHI_LOGPROB accumulates sum log softmax(chosen) over the
// utterance. Running the same audio with several forced n and comparing (WER, logprob) says
// whether cumulative logprob is a usable selector — which is exactly the decision a beam
// makes at every frame. If it is not, a beam cannot work here and there is no point
// building one.
inline int & moshi_force_alt_at() { static int v = -2; if (v == -2) {
    const char * e = getenv("MOSHI_FORCE_ALT_AT"); v = (e && e[0]) ? atoi(e) : -1; } return v; }
inline double & moshi_seq_logprob() { static double v = 0.0; return v; }
inline int & moshi_decision_index() { static int v = 0; return v; }
extern "C" inline void moshi_seq_reset() { moshi_seq_logprob() = 0.0; moshi_decision_index() = 0; }
extern "C" inline double moshi_seq_score() { return moshi_seq_logprob(); }

// ── Hotword / contextual biasing (LOGIC-style logit boosting, training-free) ─────────────
//
// A prefix state machine over per-utterance bias words. At each real-token decision, the
// first token of every bias word gets +b_start and the continuation token of every
// partially-matched word gets +b_cont before the argmax; the chosen token then advances or
// kills the partial matches. PAD(0)/WORD(3) hold state (the DSM text stream may interleave
// them inside a word's frames). This targets the one residual error class no serving knob
// reached — proper nouns — and in the app the rolling transcript is a free bias source.
// Off unless bias words are set. Distinct from the (measured, negative) bigram fusion:
// that shifted EVERY decision through a global prior; this touches only an explicit,
// short list of continuations.
struct moshi_bias_state_t {
    std::vector<std::vector<int>> words;
    std::vector<std::pair<int,int>> active;   // (word index, next position)
    float b_start = 2.0f, b_cont = 4.0f;
    bool on = false;
};
inline moshi_bias_state_t & moshi_bias() { static moshi_bias_state_t s; return s; }
extern "C" inline void moshi_bias_clear() {
    auto & s = moshi_bias(); s.words.clear(); s.active.clear(); s.on = false;
}
extern "C" inline void moshi_bias_add(const int * t, int n) {
    if (n <= 0) return;
    auto & s = moshi_bias(); s.words.emplace_back(t, t + n); s.on = true;
}
extern "C" inline void moshi_bias_boosts(float bs, float bc) {
    moshi_bias().b_start = bs; moshi_bias().b_cont = bc;
}

inline int moshi_hotword_pick(ggml_tensor * logits_t, int tok) {
    auto & s = moshi_bias();
    if (!s.on) return tok;
    const int64_t V = ggml_nelements(logits_t);
    static std::vector<float> lg, delta;
    lg.resize((size_t)V);
    ggml_backend_tensor_get(logits_t, lg.data(), 0, (size_t)V * sizeof(float));
    delta.assign((size_t)V, 0.f);
    for (auto & a : s.active) {
        const int t = s.words[a.first][a.second];
        if (delta[(size_t)t] < s.b_cont) delta[(size_t)t] = s.b_cont;
    }
    for (auto & w : s.words)
        if (delta[(size_t)w[0]] < s.b_start) delta[(size_t)w[0]] = s.b_start;
    int best = tok; float bestv = -INFINITY;
    for (int64_t t = 0; t < V; t++) {
        const float v = lg[(size_t)t] + (t > 3 ? delta[(size_t)t] : 0.f);
        if (v > bestv) { bestv = v; best = (int)t; }
    }
    if (best > 3) {                                     // real token: advance/kill matches
        std::vector<std::pair<int,int>> next;
        for (auto & a : s.active)
            if (s.words[a.first][a.second] == best &&
                a.second + 1 < (int)s.words[a.first].size())
                next.push_back({a.first, a.second + 1});
        for (size_t w = 0; w < s.words.size(); w++)
            if (s.words[w][0] == best && s.words[w].size() > 1)
                next.push_back({(int)w, 1});
        s.active.swap(next);
    }                                                   // PAD/WORD: hold state
    return best;
}

inline void moshi_margin_note(ggml_tensor * logits_t, int argmax_tok) {
    auto & m = moshi_margin();
    if (!m.tried) {
        m.tried = true;
        const char * e = getenv("MOSHI_MARGIN");
        m.on = e && e[0] == '1';
    }
    if (!m.on || argmax_tok <= 3) return;   // confidence of WORD decisions only
    const int64_t V = ggml_nelements(logits_t);
    m.logits.resize((size_t)V);
    ggml_backend_tensor_get(logits_t, m.logits.data(), 0, (size_t)V * sizeof(float));
    float top1 = -INFINITY, top2 = -INFINITY;
    for (int64_t t = 0; t < V; t++) {
        const float v = m.logits[(size_t)t];
        if (v > top1) { top2 = top1; top1 = v; }
        else if (v > top2) top2 = v;
    }
    const float margin = top1 - top2;
    if (margin < m.min_margin) m.min_margin = margin;
    m.sum += margin; m.n++;
    if (margin < 2.0f) m.n_low++;

    if (FILE * f = moshi_topk_file()) {
        int t2 = -1;
        float best2 = -INFINITY;
        for (int64_t t = 0; t < V; t++) {
            if ((int)t == argmax_tok) continue;
            if (m.logits[(size_t)t] > best2) { best2 = m.logits[(size_t)t]; t2 = (int)t; }
        }
        fprintf(f, "%d %d %.3f\n", argmax_tok, t2, margin);
        fflush(f);
    }
}

// Chooses the emitted token and accumulates its log-probability. Returns possibly the
// second-best when MOSHI_FORCE_ALT_AT names this decision.
inline int moshi_seq_pick(ggml_tensor * logits_t, int tok) {
    if (tok <= 3) return tok;                       // padding: not a decision
    const int idx = moshi_decision_index()++;
    const int64_t V = ggml_nelements(logits_t);
    static std::vector<float> lg;
    lg.resize((size_t)V);
    ggml_backend_tensor_get(logits_t, lg.data(), 0, (size_t)V * sizeof(float));
    int chosen = tok;
    if (idx == moshi_force_alt_at()) {              // force the runner-up here
        float best2 = -INFINITY; int t2 = tok;
        for (int64_t t = 0; t < V; t++)
            if ((int)t != tok && lg[(size_t)t] > best2) { best2 = lg[(size_t)t]; t2 = (int)t; }
        chosen = t2;
    }
    // log softmax of the chosen token, in a numerically safe form
    float mx = -INFINITY;
    for (int64_t t = 0; t < V; t++) mx = std::max(mx, lg[(size_t)t]);
    double z = 0.0;
    for (int64_t t = 0; t < V; t++) z += std::exp((double)lg[(size_t)t] - mx);
    moshi_seq_logprob() += (double)lg[(size_t)chosen] - mx - std::log(z);
    return chosen;
}

// Drop-in for moshi_argmax_host on the text logits tensor.
inline int moshi_text_bias_pick(ggml_tensor * logits_t, int plain_argmax) {
    auto & s = moshi_text_bias();
    if (!s.tried) moshi_text_bias_init();
    if (!s.on) return plain_argmax;
    // Padding decision is the acoustic model's alone.
    if (plain_argmax <= 3) return plain_argmax;

    const int64_t V = ggml_nelements(logits_t);
    s.logits.resize((size_t)V);
    ggml_backend_tensor_get(logits_t, s.logits.data(), 0, (size_t)V * sizeof(float));

    int best = plain_argmax;
    float best_s = -INFINITY;
    for (int t = 4; t < (int)V; t++) {
        const float sc = s.logits[t] + s.lambda * moshi_text_bias_lp(s, s.prev, t);
        if (sc > best_s) { best_s = sc; best = t; }
    }
    s.prev = best;
    return best;
}
