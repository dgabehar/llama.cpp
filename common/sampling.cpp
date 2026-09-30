#include "sampling.h"

#include "common.h"
#include "fit.h"
#include "log.h"
#include "reasoning-budget.h"

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <map>
#include <cstring>
#include <unordered_map>
#include <vector>

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};


// Empty-reply guard: masks the EOG tokens until the reply holds something besides whitespace and the
// `inert` tokens (reasoning tags). K2-Horizon spells a Kimi-K2-style tool-call opener
// (`<|tool_calls_section_begin|>` / `<|tool_call_begin|>`) with ordinary pieces (they are not vocab tokens)
// and sometimes stops right after it, which the client takes for a finished turn, so EOG stays masked inside an
// unfinished Kimi section/call too. When the request offers tools, `hold` (tag pairs: the native section
// open/close, then the inner call/arg pairs) arms a stronger rule: from the moment either the native opener or a
// Kimi opener is seen, EOG is masked until the native section closed and every inner tag is balanced (after a
// Kimi opener the model then usually rewrites the call in its own format, which the parser understands), with
// a hard cap of k_hold_cap generated tokens so a model that never closes cannot run
// to max_tokens. It sits beside the grammar and the reasoning budget, not in the chain:
// it must only see generated tokens (the chain also accepts the prompt), and it is cloned/copied with the
// sampler, so speculative-decoding rollbacks restore its state too.
//
// k_hold_cap: real opencode tool-call arguments (9.6k calls, the k2-evidence session DB) have p50 157 / p95 1635 /
// p99 3500 characters, about 1200 tokens at p99; twice that. Tripping the cap only restores the old behaviour
// (EOG allowed), it never truncates a call.
static const int k_hold_cap = 2048;

// A Kimi-form call is short (harness, 117 complete Kimi calls: median 42, longest 115 tokens; native switch after the opener: p90 118, max 255 tokens) and the model
// either rewrites it natively soon or is looping (live 2026-09-29: ~1500 tokens of the opener repeated), so the
// Kimi hold gets its own small cap, and a repeated unfinished opener releases it at once.
static const int k_kimi_hold_cap = 192;

// A reasoning line repeated this many times in an open block is a loop: force the close tag.
static const int k_rep_lines = 3;

struct common_empty_reply_guard {
    const llama_vocab *      vocab;
    std::vector<llama_token> eog;
    std::vector<llama_token> inert;
    // tag pairs { section open, section close, then inner open/close pairs (call, arg key, arg value, ...) };
    // empty = not armed
    std::vector<std::string> hold;
    bool                     seen_output = false;
    // unconsumed tail of the generated text (a marker may span pieces) and the opener state
    std::string              tail;
    bool                     in_section = false;
    bool                     in_call    = false;
    bool                     held       = false;  // armed: a native opener was seen, the call was not completed
    bool                     kimi_held  = false;  // a Kimi opener was seen (not looping): hold, capped by k_kimi_hold_cap
    bool                     kimi_loop  = false;  // the Kimi opener repeated unfinished: no more Kimi hold
    int                      n_kimi     = 0;      // generated tokens since the Kimi hold started
    int                      n_held     = 0;      // generated tokens since the hold started
    bool                     n_open     = false;  // the native section opener itself was seen during this hold
    bool                     n_closed   = false;  // ... and its close
    std::vector<int>         depth;               // open inner tags per inner pair (only counted inside a section)
    // armed: token ids of every tag except the section opener; masked while no native section is open (a call
    // that skips the opener, or a stray close, has no valid continuation; seen live as "</ifm|tool_calls>" first)
    std::vector<llama_token> tag_ids;
    // the generation starts inside an open reasoning block (tools offered): EOG stays masked until an inert close
    // tag or a call opener, capped by k_hold_cap; r_open is the live state, open_initial what reset restores
    bool                     open_initial = false;
    bool                     r_open       = false;
    llama_token              close_id     = LLAMA_TOKEN_NULL; // the block's close tag: an EOG inside the block becomes this
    int                      n_reason     = 0;
    // repetition breaker for the open block: a line (16+ chars) written k_rep_lines times forces the close tag
    std::string              line;
    std::map<std::string, int> seen_lines;
    bool                     force_close  = false;
    // per token id: 1 = EOG, 2 = tag (built at init)
    std::vector<uint8_t>     flags;
};

static const char * const k_kimi_markers[] = {
    "<|tool_calls_section_begin|>", "<|tool_calls_section_end|>", "<|tool_call_begin|>", "<|tool_call_end|>",
};

static void common_empty_reply_guard_release(common_empty_reply_guard * ctx) {
    ctx->held = ctx->n_open = ctx->n_closed = ctx->kimi_held = ctx->kimi_loop = false;
    ctx->n_held = ctx->n_kimi = 0;
    ctx->depth.assign(ctx->depth.size(), 0);
}

// consume the tool-call markers found in `tail`, keeping only what may still be the start of one
static void common_empty_reply_guard_scan(common_empty_reply_guard * ctx) {
    std::vector<std::string> markers(k_kimi_markers, k_kimi_markers + 4);
    markers.insert(markers.end(), ctx->hold.begin(), ctx->hold.end()); // 4 + 2i: open, 5 + 2i: close of pair i
    const bool armed = !ctx->hold.empty();
    size_t keep = 0;
    for (const auto & m : markers) {
        keep = std::max(keep, m.size() - 1);
    }
    for (;;) {
        size_t best  = std::string::npos;
        int    which = -1;
        for (int i = 0; i < (int) markers.size(); ++i) {
            const size_t pos = ctx->tail.find(markers[i]);
            if (pos != std::string::npos && pos < best) {
                best  = pos;
                which = i;
            }
        }
        if (which < 0) {
            break;
        }
        switch (which) {
            case 0:
            case 2: {
                // the opener again while the first is unfinished: the model is looping, let it end
                ctx->r_open = false; // a call was started: the call holds take over
                const bool repeat = which == 0 ? ctx->in_section : ctx->in_call;
                (which == 0 ? ctx->in_section : ctx->in_call) = true;
                if (repeat) {
                    ctx->kimi_held = false;
                    ctx->kimi_loop = true;
                    LOG_DBG("%s: empty-reply guard: Kimi opener repeated, releasing EOG\n", __func__);
                } else if (!ctx->kimi_loop && !ctx->kimi_held && !ctx->n_open) {
                    ctx->kimi_held = true;
                    ctx->n_kimi    = 0;
                }
                break;
            }
            case 1: ctx->in_section = false; ctx->in_call = false; if (!armed) { ctx->kimi_held = false; } break;
            case 3: ctx->in_call    = false; if (!armed && !ctx->in_section) { ctx->kimi_held = false; } break;
            default: {
                const int  pair = (which - 4) / 2;
                const bool open = (which - 4) % 2 == 0;
                if (pair == 0 && open) {
                    if (!ctx->n_open) {
                        ctx->n_held = 0;
                    }
                    ctx->r_open = false;
                    ctx->held = ctx->n_open = true;
                    ctx->n_closed = false;
                    ctx->kimi_held = ctx->kimi_loop = false; // the model switched to its own format
                } else if (pair == 0) {
                    // a close without its opener (seen live after a Kimi opener) closes nothing
                    ctx->n_closed = ctx->n_open;
                } else if (ctx->n_open) {
                    int & d = ctx->depth[pair - 1];
                    d = open ? d + 1 : std::max(0, d - 1);
                }
                // released once the section is closed and every inner tag is balanced
                if (ctx->n_open && ctx->n_closed &&
                    std::all_of(ctx->depth.begin(), ctx->depth.end(), [](int d) { return d == 0; })) {
                    common_empty_reply_guard_release(ctx);
                }
            }
        }
        ctx->tail.erase(0, best + markers[which].size());
    }
    if (ctx->tail.size() > keep) {
        ctx->tail.erase(0, ctx->tail.size() - keep);
    }
}

// true while EOG must stay masked (before the caller's "last candidate" check)
static bool common_empty_reply_guard_holds(const common_empty_reply_guard * ctx);
static bool common_empty_reply_guard_holds(const common_empty_reply_guard * ctx) {
    if (!ctx->seen_output) {
        return true;
    }
    return (ctx->r_open && ctx->n_reason < k_hold_cap) || (ctx->held && ctx->n_held < k_hold_cap) || (ctx->kimi_held && ctx->n_kimi < k_kimi_hold_cap);
}

static const char * common_empty_reply_guard_name(const struct llama_sampler * /*smpl*/) {
    return "empty-reply-guard";
}

static void common_empty_reply_guard_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_empty_reply_guard *) smpl->ctx;
    if (llama_vocab_is_eog(ctx->vocab, token)) {
        return;
    }
    if (std::find(ctx->inert.begin(), ctx->inert.end(), token) != ctx->inert.end()) {
        if (ctx->open_initial || ctx->r_open) {
            const std::string tag = common_token_to_piece(ctx->vocab, token, true);
            ctx->r_open = tag.compare(0, 2, "</") != 0; // a close tag ends the block, an open tag (re)opens one
            ctx->n_reason = 0;
            ctx->force_close = false;
            ctx->line.clear();
            ctx->seen_lines.clear();
        }
        return;
    }
    if (ctx->r_open && ctx->close_id != LLAMA_TOKEN_NULL) {
        // the model writes its next step in the reasoning, then repeats it to max_tokens ("Let me check ..." x 100)
        ctx->line += common_token_to_piece(ctx->vocab, token, false);
        for (size_t nl; (nl = ctx->line.find('\n')) != std::string::npos; ctx->line.erase(0, nl + 1)) {
            std::string l = ctx->line.substr(0, nl);
            l.erase(0, l.find_first_not_of(" \t\r"));
            if (l.size() >= 16 && ++ctx->seen_lines[l] >= k_rep_lines) {
                ctx->force_close = true;
                LOG_DBG("%s: empty-reply guard: a reasoning line repeated %d times, forcing the close tag\n", __func__, k_rep_lines);
            }
        }
    }
    if (ctx->r_open && ++ctx->n_reason == k_hold_cap) {
        LOG_DBG("%s: empty-reply guard: %d tokens inside an unclosed reasoning block, releasing EOG\n", __func__, k_hold_cap);
    }
    const std::string piece = common_token_to_piece(ctx->vocab, token, false);
    // the native markers are special tokens, which only render with special=true
    ctx->tail += ctx->hold.empty() ? piece : common_token_to_piece(ctx->vocab, token, true);
    if (ctx->kimi_held && ++ctx->n_kimi == k_kimi_hold_cap) {
        LOG_DBG("%s: empty-reply guard: %d tokens after a Kimi opener, releasing EOG\n", __func__, k_kimi_hold_cap);
    }
    if (ctx->held && ++ctx->n_held == k_hold_cap) {
        LOG_DBG("%s: empty-reply guard: %d tokens inside an open tool call, releasing EOG\n", __func__, k_hold_cap);
    }
    common_empty_reply_guard_scan(ctx);
    if (ctx->seen_output) {
        return;
    }
    // a control token that renders as nothing is not an answer either, but is not a whitespace token
    // the model can legitimately chain, so treat anything else than plain whitespace as output
    const bool blank = piece.find_first_not_of(" \t\r\n") == std::string::npos && !piece.empty();
    if (!blank) {
        ctx->seen_output = true;
    }
}

static void common_empty_reply_guard_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    const auto * ctx = (const common_empty_reply_guard *) smpl->ctx;
    const bool mask_eog  = common_empty_reply_guard_holds(ctx);
    const bool mask_tags = !ctx->tag_ids.empty() && !ctx->n_open;
    if (ctx->r_open && ctx->close_id != LLAMA_TOKEN_NULL && (ctx->force_close || ctx->n_reason < k_hold_cap)) {
        const auto find = [&](llama_token id) -> llama_token_data * {
            if (id >= 0 && (size_t) id < cur_p->size && cur_p->data[id].id == id) {
                return &cur_p->data[id];
            }
            for (size_t i = 0; i < cur_p->size; ++i) {
                if (cur_p->data[i].id == id) { return &cur_p->data[i]; }
            }
            return nullptr;
        };
        if (ctx->force_close) {
            if (auto * c = find(ctx->close_id)) {
                for (size_t i = 0; i < cur_p->size; ++i) {
                    cur_p->data[i].logit = -INFINITY;
                }
                c->logit = 0.0f;
                return;
            }
        }
        // The model wants to end the turn inside an unclosed reasoning block: turn that into closing the block (masking
        // EOG alone pushed it into a repetition loop instead). The close tag takes the best EOG logit.
        float best = -INFINITY;
        for (llama_token id : ctx->eog) {
            if (const auto * d = find(id)) { best = std::max(best, d->logit); }
        }
        if (auto * c = find(ctx->close_id)) {
            c->logit = std::max(c->logit, best);
        }
    }
    if (!mask_eog && !mask_tags) {
        return;
    }
    // O(1) membership from a per-vocab flag table (built at init); was a std::find per candidate (~1.5 ms/token at 250k)
    const uint8_t flag = (mask_eog ? 1 : 0) | (mask_tags ? 2 : 0);
    const auto masked = [&](llama_token id) {
        return id >= 0 && (size_t) id < ctx->flags.size() && (ctx->flags[id] & flag);
    };
    // never mask the last candidate standing: stop at the first other finite candidate
    bool has_other = false;
    for (size_t i = 0; i < cur_p->size && !has_other; ++i) {
        has_other = cur_p->data[i].logit != -INFINITY && !masked(cur_p->data[i].id);
    }
    if (!has_other) {
        return;
    }
    // full-vocab candidates in id order (the usual case): jump straight to the few masked ids
    const auto direct = [&](const std::vector<llama_token> & ids, bool on) {
        return !on || std::all_of(ids.begin(), ids.end(), [&](llama_token id) {
            return id >= 0 && (size_t) id < cur_p->size && cur_p->data[id].id == id;
        });
    };
    if (direct(ctx->eog, mask_eog) && direct(ctx->tag_ids, mask_tags)) {
        for (llama_token id : ctx->eog)     { if (mask_eog)  { cur_p->data[id].logit = -INFINITY; } }
        for (llama_token id : ctx->tag_ids) { if (mask_tags) { cur_p->data[id].logit = -INFINITY; } }
        return;
    }
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (masked(cur_p->data[i].id)) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
}

static void common_empty_reply_guard_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_empty_reply_guard *) smpl->ctx;
    ctx->seen_output = false;
    ctx->tail.clear();
    ctx->in_section = ctx->in_call = false;
    ctx->r_open = ctx->open_initial;
    ctx->n_reason = 0;
    ctx->force_close = false;
    ctx->line.clear();
    ctx->seen_lines.clear();
    common_empty_reply_guard_release(ctx);
}

static struct llama_sampler * common_empty_reply_guard_clone(const struct llama_sampler * smpl);

static void common_empty_reply_guard_free(struct llama_sampler * smpl) {
    delete (common_empty_reply_guard *) smpl->ctx;
}

static struct llama_sampler_i common_empty_reply_guard_i = {
    /* .name              = */ common_empty_reply_guard_name,
    /* .accept            = */ common_empty_reply_guard_accept,
    /* .apply             = */ common_empty_reply_guard_apply,
    /* .reset             = */ common_empty_reply_guard_reset,
    /* .clone             = */ common_empty_reply_guard_clone,
    /* .free              = */ common_empty_reply_guard_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

static struct llama_sampler * common_empty_reply_guard_clone(const struct llama_sampler * smpl) {
    return llama_sampler_init(&common_empty_reply_guard_i,
            new common_empty_reply_guard(*(const common_empty_reply_guard *) smpl->ctx));
}

struct llama_sampler * common_sampler_init_empty_reply_guard(const struct llama_vocab * vocab, const std::vector<llama_token> & inert, const std::vector<std::string> & hold, const std::string & reasoning_close) {
    auto * ctx = new common_empty_reply_guard { vocab, {}, inert, hold.size() >= 2 && hold.size() % 2 == 0 ? hold : std::vector<std::string>() };
    ctx->open_initial = ctx->r_open = !reasoning_close.empty() && !ctx->hold.empty();
    if (ctx->open_initial) {
        const auto toks = common_tokenize(vocab, reasoning_close, false, true);
        ctx->close_id = toks.size() == 1 ? toks[0] : LLAMA_TOKEN_NULL;
    }
    ctx->depth.assign(ctx->hold.size() / 2 > 0 ? ctx->hold.size() / 2 - 1 : 0, 0);
    // the tags are single vocab tokens in K2; a tag that is not (other vocab) is simply not masked
    for (size_t i = 1; i < ctx->hold.size(); ++i) {
        const auto toks = common_tokenize(vocab, ctx->hold[i], false, true);
        if (toks.size() == 1) {
            ctx->tag_ids.push_back(toks[0]);
        }
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (llama_token id = 0; id < n_vocab; ++id) {
        if (llama_vocab_is_eog(vocab, id)) {
            ctx->eog.push_back(id);
        }
    }
    ctx->flags.assign(n_vocab, 0);
    for (llama_token id : ctx->eog)     { ctx->flags[id] |= 1; }
    for (llama_token id : ctx->tag_ids) { if (id >= 0 && id < n_vocab) { ctx->flags[id] |= 2; } }
    return llama_sampler_init(&common_empty_reply_guard_i, ctx);
}

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * eguard; // empty-reply guard, see above
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    void reset() {
        prev.clear();

        llama_sampler_reset(chain);
    }

    void set_logits(struct llama_context * ctx, int idx) {
        const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (sampled_probs) {
            const uint32_t sampled_probs_count = llama_get_sampled_probs_count_ith(ctx, idx);
            cur.resize(sampled_probs_count);
            for (uint32_t i = 0; i < sampled_probs_count; ++i) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
            }
        } else if (sampled_logits) {
            const uint32_t sampled_logits_count = llama_get_sampled_logits_count_ith(ctx, idx);
            cur.resize(sampled_logits_count);
            for (uint32_t i = 0; i < sampled_logits_count; i++) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
            }
        } else {
            const auto * logits = llama_get_logits_ith(ctx, idx);
            GGML_ASSERT(logits != nullptr);
            cur.resize(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }

    common_time_meas tm() {
        return common_time_meas(t_total_us, params.no_perf);
    }

    mutable int64_t t_total_us = 0;
};

std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau, adaptive_target, adaptive_decay);

    return std::string(result);
}

struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params) {
    if (!std::isfinite(params.penalty_repeat) ||
        params.penalty_repeat <= 0.0f ||
        !std::isfinite(1.0f/params.penalty_repeat)) {
        throw std::invalid_argument("penalty_repeat must be finite and greater than 0");
    }
    if (!std::isfinite(params.penalty_freq)) {
        throw std::invalid_argument("penalty_freq must be finite");
    }
    if (!std::isfinite(params.penalty_present)) {
        throw std::invalid_argument("penalty_present must be finite");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * eguard  = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparams);

    std::vector<llama_sampler *> samplers;

    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
#else
        GGML_ABORT("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    trigger_patterns.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
             if (params.grammar_lazy) {
                 grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                         trigger_patterns_c.data(), trigger_patterns_c.size(),
                         trigger_tokens.data(), trigger_tokens.size());
             } else {
                 grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
             }
        }
    }
    if (!grmr && !grammar_str.empty()) {
        throw std::runtime_error("failed to parse grammar");
    }

    // Compute prefill tokens from the generation prompt
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && std::isspace(piece[0]) && !std::isspace(params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to exclude
                continue;
            }
            LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
            prefill_tokens.push_back(tokens[i]);
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        try {
            for (const auto & token : prefill_tokens) {
                llama_sampler_accept(grmr, token);
                LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
            }
        } catch (std::exception &e) {
            LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
            throw e;
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0 || params.reasoning_control)) {
        rbudget = common_reasoning_budget_init(
            vocab,
            {params.reasoning_budget_start},
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(rbudget, token);
            LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
        }
    }

    // logit bias: user biases + model suppress tokens (-INFINITY)
    {
        std::vector<llama_logit_bias> merged = params.logit_bias;

        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        for (int32_t i = 0; i < n_suppress; ++i) {
            merged.push_back({ suppress[i], -INFINITY });
        }

        if (!merged.empty()) {
            samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), merged.size(), merged.data()));
        }
    }

    if (params.no_empty_reply) {
        eguard = common_sampler_init_empty_reply_guard(vocab, params.no_empty_reply_inert, params.no_empty_reply_hold, params.no_empty_reply_open);
    }

    if (params.mirostat == 0) {

        bool use_adaptive_p = false; // see below

        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }
                        samplers.push_back(llama_sampler_init_dry(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    samplers.push_back(llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
                    // a single token, so we will add `dist` at the end of the chain by default,
                    // unless the user specifically included `adaptive-p`. we set this flag here
                    // so we know to add the sampler at the very end.
                    use_adaptive_p = true;
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            // default: sample from distribution
            samplers.push_back(llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    if (grmr && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);

        params.backend_sampling = false;
    }

    if (rbudget && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with reasoning budget, disabling\n", __func__);

        params.backend_sampling = false;
    }

    if (eguard && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with the empty-reply guard, disabling\n", __func__);

        params.backend_sampling = false;
    }

    auto * result = new common_sampler {
        /* .params  = */ params,
        /* .grmr    = */ grmr,
        /* .rbudget = */ rbudget,
        /* .eguard  = */ eguard,
        /* .chain   = */ chain,
        /* .prev    = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur     = */ {},
        /* .cur_p   = */ {},
    };

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->eguard);
    llama_sampler_free(gsmpl->chain);

    delete gsmpl;
}

static bool grammar_should_apply(struct common_sampler * gsmpl) {
    if (!gsmpl->grmr) {
        return false;
    }
    if (!gsmpl->rbudget) {
        return true;
    }
    if (gsmpl->params.grammar_lazy) {
        // if grammar is lazy, only apply when reasoning budget is not active
        const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated) {
    if (!gsmpl) {
        return;
    }

    const auto tm = gsmpl->tm();

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    const auto accept_grammar = is_generated && grammar_should_apply(gsmpl);

    if (gsmpl->rbudget && is_generated) {
        llama_sampler_accept(gsmpl->rbudget, token);

        // if done, replay end sequence which may contain a grammar trigger
        const bool is_done = common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_DONE;
        if (gsmpl->grmr && !accept_grammar && is_done) {
            const llama_tokens * end_seq = common_reasoning_budget_get_end_match(gsmpl->rbudget);
            if (end_seq) {
                for (const llama_token end_token : *end_seq) {
                    llama_sampler_accept(gsmpl->grmr, end_token);
                }
            }
        }
    }

    if (gsmpl->grmr && accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    if (gsmpl->eguard && is_generated) {
        llama_sampler_accept(gsmpl->eguard, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    gsmpl->reset();
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    return new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .eguard  = */ llama_sampler_clone(gsmpl->eguard),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ gsmpl->cur,
        /* .cur_p   = */ gsmpl->cur_p,
    };
}

void common_sampler_copy(const common_sampler * src, common_sampler * dst) {
    if (!src || !dst || src == dst) {
        return;
    }

    GGML_ASSERT((src->grmr == nullptr) == (dst->grmr == nullptr));
    GGML_ASSERT((src->rbudget == nullptr) == (dst->rbudget == nullptr));
    GGML_ASSERT((src->eguard == nullptr) == (dst->eguard == nullptr));

    llama_sampler_copy(src->grmr,    dst->grmr);
    llama_sampler_copy(src->rbudget, dst->rbudget);
    llama_sampler_copy(src->eguard,  dst->eguard);
    llama_sampler_copy(src->chain,   dst->chain);

    dst->params     = src->params;
    dst->prev       = src->prev;
    dst->cur        = src->cur;
    dst->cur_p      = src->cur_p;
    dst->cur_p.data = src->cur_p.data ? dst->cur.data() : nullptr; // re-point to dst's buffer
    dst->t_total_us = src->t_total_us;
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    const double t_sampling_ms = gsmpl ? 1e-3*gsmpl->t_total_us : 0;

    llama_perf_sampler_data data_smpl;
    llama_perf_context_data data_ctx;

    memset(&data_smpl, 0, sizeof(data_smpl));
    memset(&data_ctx,  0, sizeof(data_ctx));

    if (gsmpl) {
        auto & data = data_smpl;

        data = llama_perf_sampler(gsmpl->chain);

        // note: the sampling time includes the samplers time + extra time spent in common/sampling
        LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, t_sampling_ms);
        LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, data.t_sample_ms, data.n_sample);
    }

    if (ctx) {
        auto & data = data_ctx;

        data = llama_perf_context(ctx);

        const double t_end_ms = 1e-3 * ggml_time_us();

        const double t_total_ms = t_end_ms - data.t_start_ms;
        const double t_unacc_ms = t_total_ms - (t_sampling_ms + data.t_p_eval_ms + data.t_eval_ms);
        const double t_unacc_pc = 100.0 * t_unacc_ms /  t_total_ms;

        LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
        LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
        LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
        LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
        LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n", __func__, t_unacc_ms, t_unacc_pc);
        LOG_INF("%s:    graphs reused = %10d\n", __func__, data.n_reused);

        common_memory_breakdown_print(ctx);
    }
}

struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return nullptr;
    }

    return gsmpl->chain;
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    llama_synchronize(ctx);

    // start measuring sampling time after the llama_context synchronization in order to not measure any ongoing async operations
    const auto tm = gsmpl->tm();

    llama_token id = LLAMA_TOKEN_NULL;

    auto & grmr  = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    gsmpl->set_logits(ctx, idx);

    // Check if a backend sampler has already sampled a token in which case we
    // return that token id directly.
    {
        id = llama_get_sampled_token_ith(ctx, idx);

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, id);

            GGML_ASSERT(!gsmpl->grmr    && "using grammar in combination with backend sampling is not supported");
            GGML_ASSERT(!gsmpl->rbudget && "using reasoning budget in combination with backend sampling is not supported");
            GGML_ASSERT(!gsmpl->eguard  && "using the empty-reply guard in combination with backend sampling is not supported");

            for (size_t i = 0; i < cur_p.size; ++i) {
                if (cur_p.data[i].id == id) {
                    cur_p.selected = i;
                    break;
                }
            }

            return id;
        }
    }

    // apply reasoning budget first
    llama_sampler_apply(rbudget, &cur_p);
    llama_sampler_apply(gsmpl->eguard, &cur_p);

    if (grammar_first && grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }

    // check if it the sampled token fits the grammar (grammar-based rejection sampling)
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    llama_sampler_apply(rbudget,  &cur_p);
    llama_sampler_apply(gsmpl->eguard, &cur_p);

    if (grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr,  &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    id = cur_p.data[cur_p.selected].id;

    return id;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        if (draft[i] != id) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    return common_reasoning_budget_force(gsmpl->rbudget);
}

// helpers

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    const auto tm = gsmpl->tm();

    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
        });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ");
        result += std::string(llama_sampler_name(smpl)) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return 's';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'e';
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return 'a';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return "top_n_sigma";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return "adaptive_p";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names) {
    // sampler names can be written multiple ways; generate aliases from canonical names
    static const auto sampler_name_map = []{
        // canonical sampler name mapping
        std::unordered_map<std::string, common_sampler_type> canonical_name_map {
            { "dry",         COMMON_SAMPLER_TYPE_DRY         },
            { "top_k",       COMMON_SAMPLER_TYPE_TOP_K       },
            { "top_p",       COMMON_SAMPLER_TYPE_TOP_P       },
            { "top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
            { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P   },
            { "min_p",       COMMON_SAMPLER_TYPE_MIN_P       },
            { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
            { "xtc",         COMMON_SAMPLER_TYPE_XTC         },
            { "infill",      COMMON_SAMPLER_TYPE_INFILL      },
            { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES   },
            { "adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P  }
        };
        std::unordered_map<std::string, common_sampler_type> alias_name_map;
        for (const auto & entry : canonical_name_map) {
            const std::string & canonical = entry.first;
            if (canonical.find('_') == std::string::npos) {
                continue;
            }
            // kebab-case: "top-k", "min-p", etc.
            {
                std::string kebab_case = canonical;
                std::replace(kebab_case.begin(), kebab_case.end(), '_', '-');
                alias_name_map.insert({kebab_case, entry.second});
            }
            // no dash: "topk", "minp", etc.
            {
                std::string no_dash = canonical;
                no_dash.erase(std::remove(no_dash.begin(), no_dash.end(), '_'), no_dash.end());
                alias_name_map.insert({no_dash, entry.second});
            }
        }
        // misc. aliases
        alias_name_map.insert({"nucleus", COMMON_SAMPLER_TYPE_TOP_P});
        alias_name_map.insert({"temp",    COMMON_SAMPLER_TYPE_TEMPERATURE});
        alias_name_map.insert({"typ",     COMMON_SAMPLER_TYPE_TYPICAL_P});
        // include aliases + canonical names in the complete mapping
        alias_name_map.merge(canonical_name_map);
        return alias_name_map;
    }();

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        auto sampler = sampler_name_map.find(name_lower);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name_lower.c_str());
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_N_SIGMA), COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_PENALTIES),   COMMON_SAMPLER_TYPE_PENALTIES },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_ADAPTIVE_P),  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, c);
        }
    }

    return samplers;
}
