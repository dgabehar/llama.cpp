// Empty-reply guard (common/sampling.cpp): EOG stays masked before any output and while the text is inside
// an unfinished Kimi-K2-style tool-call opener. The pieces below are the real K2-Horizon-7B output that ended
// an opencode turn (2026-09-28): "\n<|tool_calls_section_begin|><|tool_call_begin|>" then EOG. Those markers
// are not vocab tokens of that model, so a vocab that does not have them either (gpt-2) tokenizes them the same way.

#include <algorithm>
#include <chrono>
#include <random>
#include "llama.h"
#include "common.h"
#include "sampling.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static llama_token g_eog = LLAMA_TOKEN_NULL;
static llama_token g_other = LLAMA_TOKEN_NULL;

static void feed(llama_sampler * g, const llama_vocab * vocab, const std::string & text) {
    for (llama_token t : common_tokenize(vocab, text, false, false)) {
        llama_sampler_accept(g, t);
    }
}

// true when the guard leaves EOG selectable next to an ordinary candidate
static bool eog_allowed(llama_sampler * g) {
    llama_token_data d[2] = { { g_eog, 1.0f, 0.0f }, { g_other, 0.5f, 0.0f } };
    llama_token_data_array p = { d, 2, -1, false };
    llama_sampler_apply(g, &p);
    return !std::isinf(d[0].logit);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s vocab.gguf\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    for (llama_token t = 0; t < llama_vocab_n_tokens(vocab); ++t) {
        if (llama_vocab_is_eog(vocab, t)) { g_eog = t; break; }
    }
    g_other = common_tokenize(vocab, "a", false, false)[0];
    CHECK(g_eog != LLAMA_TOKEN_NULL);

    llama_sampler * g = common_sampler_init_empty_reply_guard(vocab, {});

    // before any output: masked (the original empty-reply case)
    CHECK(!eog_allowed(g));
    feed(g, vocab, "\n");
    CHECK(!eog_allowed(g));

    // the incident: the Kimi opener alone counts as output, but the turn must not end on it
    feed(g, vocab, "<|tool_calls_section_begin|>");
    CHECK(!eog_allowed(g));
    feed(g, vocab, "<|tool_call_begin|>");
    CHECK(!eog_allowed(g));

    // a clone made mid-call keeps the state (speculative rollback), a reset drops it
    llama_sampler * c = llama_sampler_clone(g);
    CHECK(!eog_allowed(c));
    feed(g, vocab, "functions.bash:0<|tool_call_argument_begin|>{\"command\":\"ls\"}<|tool_call_end|>");
    CHECK(!eog_allowed(g));                 // still inside the section
    CHECK(!eog_allowed(c));                 // the clone did not see the tail
    feed(g, vocab, "<|tool_calls_section_end|>");
    CHECK(eog_allowed(g));
    llama_sampler_reset(c);
    CHECK(!eog_allowed(c));                 // reset: nothing generated yet
    llama_sampler_free(c);
    llama_sampler_free(g);

    // a bare call without a section is closed by its own end marker
    g = common_sampler_init_empty_reply_guard(vocab, {});
    feed(g, vocab, "ok <|tool_call_begin|>x");
    CHECK(!eog_allowed(g));
    feed(g, vocab, "<|tool_call_end|>");
    CHECK(eog_allowed(g));
    llama_sampler_free(g);

    // ordinary text and lookalikes never mask
    g = common_sampler_init_empty_reply_guard(vocab, {});
    feed(g, vocab, "Hello, the tool_call_begin marker is <|tool_call| but not closed as such.");
    CHECK(eog_allowed(g));
    llama_sampler_free(g);

    // never mask the last candidate standing
    g = common_sampler_init_empty_reply_guard(vocab, {});
    feed(g, vocab, "<|tool_call_begin|>");
    llama_token_data d[1] = { { g_eog, 1.0f, 0.0f } };
    llama_token_data_array p = { d, 1, -1, false };
    llama_sampler_apply(g, &p);
    CHECK(!std::isinf(d[0].logit));
    llama_sampler_free(g);

    // ---- armed (tools offered): hold from a native or Kimi opener until the native close, capped
    const std::vector<std::string> hold = { "<ifm|tool_calls>", "</ifm|tool_calls>" };
    const int cap = 2048; // k_hold_cap in sampling.cpp

    // no tools: the native markers mean nothing
    g = common_sampler_init_empty_reply_guard(vocab, {}, {});
    feed(g, vocab, "ok <ifm|tool_calls>\n<ifm|tool_call>bash");
    CHECK(eog_allowed(g));
    llama_sampler_free(g);

    // a native call: open holds, close releases; ordinary text never holds
    g = common_sampler_init_empty_reply_guard(vocab, {}, hold);
    feed(g, vocab, "Hello there.");
    CHECK(eog_allowed(g));
    feed(g, vocab, "<ifm|tool_calls>\n<ifm|tool_call>bash\n<ifm|arg_key>command</ifm|arg_key>");
    CHECK(!eog_allowed(g));
    llama_sampler * c2 = llama_sampler_clone(g);
    feed(g, vocab, "\n<ifm|arg_value>ls</ifm|arg_value>\n</ifm|tool_call>\n");
    CHECK(!eog_allowed(g));                 // a call closed, the section is not
    feed(g, vocab, "</ifm|tool_calls>");
    CHECK(eog_allowed(g));
    CHECK(!eog_allowed(c2));                // clone kept the hold, did not see the close
    llama_sampler_reset(c2);
    feed(c2, vocab, "x");
    CHECK(eog_allowed(c2));                 // reset dropped it
    llama_sampler_free(c2);
    // re-arms on the next call
    feed(g, vocab, "<ifm|tool_calls>");
    CHECK(!eog_allowed(g));
    llama_sampler_free(g);

    // the incident: the Kimi opener (plus a whole Kimi call and its section end) holds until the NATIVE close
    g = common_sampler_init_empty_reply_guard(vocab, {}, hold);
    feed(g, vocab, "\n<|tool_calls_section_begin|><|tool_call_begin|>");
    CHECK(!eog_allowed(g));
    feed(g, vocab, "functions.bash:0<|tool_call_argument_begin|>{\"command\":\"ls\"}<|tool_call_end|><|tool_calls_section_end|>");
    CHECK(!eog_allowed(g));                 // the Kimi form is not a call the parser understands
    feed(g, vocab, "</ifm|tool_calls>");    // a bare native close (seen live) closes nothing
    CHECK(!eog_allowed(g));
    feed(g, vocab, "<ifm|tool_calls>\n<ifm|tool_call>bash\n</ifm|tool_call>\n</ifm|tool_calls>");
    CHECK(eog_allowed(g));
    llama_sampler_free(g);

    // cap: a model that never closes is released after `cap` generated tokens
    g = common_sampler_init_empty_reply_guard(vocab, {}, hold);
    feed(g, vocab, "<ifm|tool_calls>");
    for (int i = 0; i < cap - 1; ++i) {
        llama_sampler_accept(g, g_other);
    }
    CHECK(!eog_allowed(g));
    llama_sampler_accept(g, g_other);
    CHECK(eog_allowed(g));
    feed(g, vocab, "</ifm|tool_calls><ifm|tool_calls>"); // a later call holds again, with a fresh counter
    CHECK(!eog_allowed(g));
    llama_sampler_free(g);

    // open reasoning block (prompt ends inside it, tools offered): EOG masked until the block closes or a call opens.
    // Stand-ins for the reasoning tags: "<" opens, "</" closes (the guard tells them apart by the "</" prefix).
    {
        const auto t_open  = common_tokenize(vocab, "<", false, false);
        const auto t_close = common_tokenize(vocab, "</", false, false);
        if (t_open.size() == 1 && t_close.size() == 1) {
            const std::vector<llama_token> inert = { t_open[0], t_close[0] };
            // the incident: reasoning text, no close, then EOG
            g = common_sampler_init_empty_reply_guard(vocab, inert, hold, true);
            CHECK(!eog_allowed(g));
            feed(g, vocab, "Next I will run the tests");
            CHECK(!eog_allowed(g));                 // unclosed reasoning: text does not release
            llama_sampler * c3 = llama_sampler_clone(g);
            llama_sampler_accept(g, t_close[0]);
            CHECK(eog_allowed(g));                  // closed
            CHECK(!eog_allowed(c3));                // the clone kept the open block
            llama_sampler_reset(c3);
            CHECK(!eog_allowed(c3));
            llama_sampler_free(c3);
            llama_sampler_free(g);
            // cap: released after k_hold_cap tokens
            g = common_sampler_init_empty_reply_guard(vocab, inert, hold, true);
            for (int i = 0; i < cap - 1; ++i) { llama_sampler_accept(g, g_other); }
            CHECK(!eog_allowed(g));
            llama_sampler_accept(g, g_other);
            CHECK(eog_allowed(g));
            llama_sampler_free(g);
            // a native call opener inside the block hands over to the call hold
            g = common_sampler_init_empty_reply_guard(vocab, {}, hold, true); // (the stand-in tags would eat parts of the markers)
            feed(g, vocab, "x<ifm|tool_calls>");
            CHECK(!eog_allowed(g));
            feed(g, vocab, "</ifm|tool_calls>");
            CHECK(eog_allowed(g));
            llama_sampler_free(g);
            // no tools (no hold) or prompt not in reasoning: never held
            g = common_sampler_init_empty_reply_guard(vocab, inert, {}, true);
            feed(g, vocab, "text");
            CHECK(eog_allowed(g));
            llama_sampler_free(g);
            g = common_sampler_init_empty_reply_guard(vocab, inert, hold, false);
            feed(g, vocab, "text");
            CHECK(eog_allowed(g));
            llama_sampler_free(g);
        } else {
            fprintf(stderr, "skip open-reasoning tests: stand-in tags are not single tokens\n");
        }
    }

    // every tag that opens must close: the full native tag set (xml, xml_typed, json call formats)
    const std::vector<std::string> full = {
        "<ifm|tool_calls>", "</ifm|tool_calls>", "<ifm|tool_call>", "</ifm|tool_call>", "<ifm|arg_key>", "</ifm|arg_key>",
        "<ifm|arg_type>", "</ifm|arg_type>", "<ifm|arg_value>", "</ifm|arg_value>" };
    // section closed with an arg_value still open: keep holding, release when it balances
    g = common_sampler_init_empty_reply_guard(vocab, {}, full);
    feed(g, vocab, "<ifm|tool_calls>\n<ifm|tool_call>bash\n<ifm|arg_key>command</ifm|arg_key>\n<ifm|arg_value>ls");
    CHECK(!eog_allowed(g));
    feed(g, vocab, "\n</ifm|tool_call>\n</ifm|tool_calls>");
    CHECK(!eog_allowed(g));                 // section closed, arg_value still open
    feed(g, vocab, "</ifm|arg_value>");
    CHECK(eog_allowed(g));                  // balanced now
    llama_sampler_free(g);
    // a fully balanced xml_typed call releases at the section close, not before
    g = common_sampler_init_empty_reply_guard(vocab, {}, full);
    feed(g, vocab, "<ifm|tool_calls>\n<ifm|tool_call>bash\n<ifm|arg_key>a</ifm|arg_key>\n<ifm|arg_type>string</ifm|arg_type>\n"
                   "<ifm|arg_value>x</ifm|arg_value>\n</ifm|tool_call>\n");
    CHECK(!eog_allowed(g));
    feed(g, vocab, "</ifm|tool_calls>");
    CHECK(eog_allowed(g));
    llama_sampler_free(g);
    // an unclosed arg_type inside a closed section holds too
    g = common_sampler_init_empty_reply_guard(vocab, {}, full);
    feed(g, vocab, "<ifm|tool_calls><ifm|tool_call>bash<ifm|arg_type>string</ifm|tool_call></ifm|tool_calls>");
    CHECK(!eog_allowed(g));
    llama_sampler_free(g);
    // json format: <ifm|tool_call>{...}</ifm|tool_call>
    g = common_sampler_init_empty_reply_guard(vocab, {}, full);
    feed(g, vocab, "<ifm|tool_calls>\n<ifm|tool_call>{\"name\":\"bash\",\"arguments\":{\"command\":\"ls\"}}");
    CHECK(!eog_allowed(g));
    feed(g, vocab, "</ifm|tool_call>\n");
    CHECK(!eog_allowed(g));
    feed(g, vocab, "</ifm|tool_calls>");
    CHECK(eog_allowed(g));
    llama_sampler_free(g);
    // a stray inner close never goes negative, inner tags outside a section are ignored
    g = common_sampler_init_empty_reply_guard(vocab, {}, full);
    feed(g, vocab, "</ifm|arg_value><ifm|arg_value>x <ifm|tool_calls></ifm|tool_calls>");
    CHECK(eog_allowed(g));
    llama_sampler_free(g);
    // the cap releases an unbalanced call as well
    g = common_sampler_init_empty_reply_guard(vocab, {}, full);
    feed(g, vocab, "<ifm|tool_calls><ifm|tool_call>bash<ifm|arg_value>");
    for (int i = 0; i < cap; ++i) {
        llama_sampler_accept(g, g_other);
    }
    CHECK(eog_allowed(g));
    llama_sampler_free(g);

    // ---- live loop 2026-09-29 (opencode session ses_f16214d2..., ~1500 tokens): the opener repeated with blank
    // lines between the repeats, no call ever started. Real text, trimmed. The repeat releases EOG at once.
    const std::string loop1 = "\n<|tool_calls_section_begin|><|tool_call_begin|>\n\n\n\n\n\n\n\n";
    const std::string loop2 = "<|tool_calls_section_begin|><|tool_call_begin|>\n\n\n\n\n\n\n<|tool_calls_section_begin|><|tool_call_begin|>\n\n";
    for (const std::vector<std::string> & h : { hold, full, std::vector<std::string>() }) {
        g = common_sampler_init_empty_reply_guard(vocab, {}, h);
        feed(g, vocab, loop1);
        CHECK(!eog_allowed(g));             // the first opener still holds
        feed(g, vocab, loop2);
        CHECK(eog_allowed(g));              // repeated with no end marker: looping, release
        feed(g, vocab, "<|tool_calls_section_begin|><|tool_call_begin|>");
        CHECK(eog_allowed(g));              // and it stays released
        llama_sampler_free(g);
    }
    // a repeated bare call opener is a loop too; a call after a finished one is not
    g = common_sampler_init_empty_reply_guard(vocab, {}, hold);
    feed(g, vocab, "<|tool_call_begin|>x<|tool_call_end|><|tool_call_begin|>y");
    CHECK(!eog_allowed(g));
    feed(g, vocab, "<|tool_call_begin|>");
    CHECK(eog_allowed(g));
    llama_sampler_free(g);

    // the Kimi hold has its own small cap (k_kimi_hold_cap), the native hold keeps 2048
    const int kcap = 192;
    g = common_sampler_init_empty_reply_guard(vocab, {}, hold);
    feed(g, vocab, "<|tool_calls_section_begin|>"); // the cap counts the tokens after the opener marker
    for (int i = 0; i < kcap - 1; ++i) {
        llama_sampler_accept(g, g_other);
    }
    CHECK(!eog_allowed(g));
    llama_sampler_accept(g, g_other);
    CHECK(eog_allowed(g));
    llama_sampler_free(g);
    // ... and switching to the native format after the Kimi opener gives the native cap
    g = common_sampler_init_empty_reply_guard(vocab, {}, hold);
    feed(g, vocab, "<|tool_calls_section_begin|><|tool_call_begin|>");
    for (int i = 0; i < 100; ++i) {
        llama_sampler_accept(g, g_other);
    }
    feed(g, vocab, "<ifm|tool_calls>");
    for (int i = 0; i < cap - 2; ++i) {     // far past the Kimi cap
        llama_sampler_accept(g, g_other);
    }
    CHECK(!eog_allowed(g));
    llama_sampler_free(g);

    // never mask the last candidate standing
    g = common_sampler_init_empty_reply_guard(vocab, {}, hold);
    feed(g, vocab, "<ifm|tool_calls>");
    llama_token_data d2[1] = { { g_eog, 1.0f, 0.0f } };
    llama_token_data_array p2 = { d2, 1, -1, false };
    llama_sampler_apply(g, &p2);
    CHECK(!std::isinf(d2[0].logit));
    llama_sampler_free(g);

    // ---- structural tags are masked while no native section is open (needs a vocab where the tags are single
    // tokens; K2's are, here llama-3 special tokens stand in): section = start_header/end_header, inner = begin_of_text/eot
    if (argc > 2) {
        llama_model * m2 = llama_model_load_from_file(argv[2], mparams);
        CHECK(m2 != nullptr);
        if (m2) {
            const llama_vocab * v2 = llama_model_get_vocab(m2);
            const std::vector<std::string> tags = { "<|start_header_id|>", "<|end_header_id|>", "<|begin_of_text|>", "<|reserved_special_token_5|>" };
            std::vector<llama_token> id;
            for (const auto & t : tags) {
                auto tk = common_tokenize(v2, t, false, true);
                CHECK(tk.size() == 1);
                id.push_back(tk[0]);
            }
            const llama_token ord = common_tokenize(v2, "a", false, false)[0];
            // which of { section open, section close, inner open, inner close, ordinary } survive
            auto survivors = [&](llama_sampler * g2) {
                llama_token_data d[5] = { { id[0], 1.f, 0.f }, { id[1], 1.f, 0.f }, { id[2], 1.f, 0.f }, { id[3], 1.f, 0.f }, { ord, 1.f, 0.f } };
                llama_token_data_array p = { d, 5, -1, false };
                llama_sampler_apply(g2, &p);
                std::string r;
                for (auto & x : d) { r += std::isinf(x.logit) ? '0' : '1'; }
                return r;
            };
            llama_sampler * g2 = common_sampler_init_empty_reply_guard(v2, {}, tags);
            llama_sampler_accept(g2, ord);                      // some output, no section yet
            CHECK(survivors(g2) == "10001");                    // only the opener and ordinary text
            llama_sampler_accept(g2, id[0]);
            CHECK(survivors(g2) == "11111");                    // inside the section everything is allowed
            llama_sampler_accept(g2, id[2]);
            llama_sampler_accept(g2, id[3]);
            llama_sampler_accept(g2, id[1]);                    // balanced and closed: outside again
            CHECK(survivors(g2) == "10001");
            llama_sampler * g3 = common_sampler_init_empty_reply_guard(v2, {}, {}); // not armed: nothing masked
            llama_sampler_accept(g3, ord);
            CHECK(survivors(g3) == "11111");
            llama_sampler_free(g3);
            // never mask the last candidate standing
            llama_token_data d1[1] = { { id[1], 1.f, 0.f } };
            llama_token_data_array p1 = { d1, 1, -1, false };
            llama_sampler_apply(g2, &p1);
            CHECK(!std::isinf(d1[0].logit));
            // the masking must equal the reference std::find rule for any candidate layout (identity, shuffled, sparse,
            // sorted, with -inf, ids past the vocab), and its per-token cost is timed against that old rule
            {
                const int n_v = llama_vocab_n_tokens(v2);
                std::vector<llama_token> eogs;
                for (llama_token t = 0; t < n_v; ++t) { if (llama_vocab_is_eog(v2, t)) { eogs.push_back(t); } }
                const std::vector<llama_token> tag_v = { id[1], id[2], id[3] };
                auto ref = [&](std::vector<llama_token_data> & c, bool m_eog, bool m_tag) {
                    auto masked = [&](llama_token t) {
                        return (m_eog && std::find(eogs.begin(), eogs.end(), t) != eogs.end()) ||
                               (m_tag && std::find(tag_v.begin(), tag_v.end(), t) != tag_v.end());
                    };
                    size_t n_other = 0;
                    for (auto & x : c) { if (!masked(x.id) && x.logit != -INFINITY) { n_other++; } }
                    if (n_other == 0) { return; }
                    for (auto & x : c) { if (masked(x.id)) { x.logit = -INFINITY; } }
                };
                std::mt19937 rng(42);
                for (int iter = 0; iter < 200; ++iter) {
                    const bool armed_open = iter % 2;            // tags masked only while no native section is open
                    const bool seen       = iter % 4 < 2;        // EOG masked only before any output
                    std::vector<llama_token_data> c;
                    const int kind = iter % 5;
                    for (llama_token t = 0; t < n_v; ++t) {
                        if (kind == 3 && rng() % 3) { continue; } // sparse
                        c.push_back({ t, (rng() % 7 == 0) ? -INFINITY : (float) (rng() % 100) / 10.f, 0.f });
                    }
                    if (kind == 1 || kind == 4) { std::shuffle(c.begin(), c.end(), rng); }
                    if (kind == 2) { std::sort(c.begin(), c.end(), [](auto & a, auto & b) { return a.logit > b.logit; }); }
                    if (kind == 4) { c.push_back({ n_v + 5, 1.f, 0.f }); c.push_back({ -1, 1.f, 0.f }); }
                    if (iter % 11 == 0) { for (auto & x : c) { if (x.id != id[1] && x.id != eogs[0]) { x.logit = -INFINITY; } } } // last candidates
                    llama_sampler * gg = common_sampler_init_empty_reply_guard(v2, {}, tags);
                    if (!seen) { llama_sampler_accept(gg, ord); }
                    if (armed_open) { llama_sampler_accept(gg, id[0]); }
                    std::vector<llama_token_data> exp = c, got = c;
                    ref(exp, seen || armed_open, !armed_open); // an open native section holds EOG too
                    llama_token_data_array pp = { got.data(), got.size(), -1, false };
                    llama_sampler_apply(gg, &pp);
                    bool same = true;
                    for (size_t i = 0; i < exp.size(); ++i) { same = same && exp[i].logit == got[i].logit; }
                    CHECK(same);
                    llama_sampler_free(gg);
                }
                // timing on a 250k candidate array (identity order, as the sampler sees it), no output yet, tags masked
                const int N = 250000;
                std::vector<llama_token_data> big(N);
                for (int i = 0; i < N; ++i) { big[i] = { i, (float) (i % 97), 0.f }; }
                llama_sampler * gg = common_sampler_init_empty_reply_guard(v2, {}, tags);
                using clk = std::chrono::steady_clock;
                const int reps = 200;
                double t_new = 0, t_old = 0;
                for (int r = 0; r < reps; ++r) {
                    std::vector<llama_token_data> a = big, b = big;
                    llama_token_data_array pa = { a.data(), a.size(), -1, false };
                    auto t0 = clk::now(); llama_sampler_apply(gg, &pa); auto t1 = clk::now();
                    ref(b, true, true);   auto t2 = clk::now();
                    t_new += std::chrono::duration<double, std::milli>(t1 - t0).count();
                    t_old += std::chrono::duration<double, std::milli>(t2 - t1).count();
                }
                printf("guard apply over %d candidates: new %.4f ms/token, old std::find rule %.4f ms/token\n", N, t_new / reps, t_old / reps);
                llama_sampler_free(gg);
            }
            llama_sampler_free(g2);
            llama_model_free(m2);
        }
    }

    llama_model_free(model);
    llama_backend_free();
    if (n_fail) {
        fprintf(stderr, "%d failure(s)\n", n_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
