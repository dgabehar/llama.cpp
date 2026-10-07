// Empty-reply guard (common/sampling.cpp): EOG stays masked before any output and while the text is inside
// an unfinished Kimi-K2-style tool-call opener. The pieces below are the real K2-Horizon-7B output that ended
// an opencode turn (2026-09-28): "\n<|tool_calls_section_begin|><|tool_call_begin|>" then EOG. Those markers
// are not vocab tokens of that model, so a vocab that does not have them either (gpt-2) tokenizes them the same way.

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

    llama_model_free(model);
    llama_backend_free();
    if (n_fail) {
        fprintf(stderr, "%d failure(s)\n", n_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
