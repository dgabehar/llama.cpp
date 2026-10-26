// common_sampler_copy must restore a lazy grammar exactly: whether it is still waiting for its
// trigger, and a partial trigger held in its buffer. The server's speculative verify path relies on
// this when it restores a checkpoint after a partial draft acceptance (--spec-grammar).
//
// usage: test-sampler-copy-grammar <vocab.gguf>

#include "common.h"
#include "sampling.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

static const llama_vocab * g_vocab;

// accept the tokens of `text` as generated tokens; false if the grammar rejected one
static bool accept(common_sampler * s, const std::string & text) {
    try {
        for (llama_token t : common_tokenize(g_vocab, text, false, false)) {
            common_sampler_accept(s, t, true);
        }
    } catch (const std::exception &) {
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <vocab.gguf>\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    CHECK(model);
    g_vocab = llama_model_get_vocab(model);

    common_params_sampling params;
    params.grammar          = common_grammar(COMMON_GRAMMAR_TYPE_TOOL_CALLS, "root ::= \"<tool_call>\" [a-z]+");
    params.grammar_lazy     = true;
    params.grammar_triggers = { { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<tool_call>" } };

    // a lazy grammar brings the reasoning-budget sampler along; the grammar stays off while thinking
    params.reasoning_budget_start = common_tokenize(g_vocab, "<think>", false, false);
    params.reasoning_budget_end   = { common_tokenize(g_vocab, "</think>", false, false) };

    // saved while thinking: a tool call opened after the rollback point must not leak into the restored state
    {
        common_sampler * a = common_sampler_init(model, params);
        CHECK(accept(a, "<think>let me"));
        common_sampler * at = common_sampler_clone(a);
        // tokenized apart so the end tag matches the reasoning-budget end sequence
        CHECK(accept(a, " see"));
        CHECK(accept(a, "</think>"));
        CHECK(accept(a, "<tool_call>ab"));
        {
            common_sampler * t = common_sampler_clone(a);
            CHECK(!accept(t, "9")); // done thinking and triggered
            common_sampler_free(t);
        }
        common_sampler_copy(at, a);
        CHECK(accept(a, " <tool_call>9 9")); // still thinking, so no grammar
        CHECK(accept(a, "</think>"));
        CHECK(accept(a, "ok <tool_call>z"));
        CHECK(!accept(a, "9"));
        common_sampler_free(at);
        common_sampler_free(a);
    }

    // free text, then half of the trigger
    common_sampler * s = common_sampler_init(model, params);
    CHECK(s);
    CHECK(accept(s, "sure, 123 <tool"));

    common_sampler * save = common_sampler_clone(s);

    // run ahead past the trigger, as a verify step would, then roll back
    CHECK(accept(s, "_call>abc"));
    {
        common_sampler * t = common_sampler_clone(s);
        CHECK(!accept(t, "9")); // triggered: digits are out
        common_sampler_free(t);
    }
    common_sampler_copy(save, s);

    // restored: the trigger has not fired, and the buffered "<tool" is still there,
    // so finishing the trigger fires it and the grammar then applies
    {
        common_sampler * t = common_sampler_clone(s);
        CHECK(accept(t, "9")); // still free text
        common_sampler_free(t);
    }
    CHECK(accept(s, "_call>xyz"));
    CHECK(!accept(s, "9"));

    // copy into a sampler that was never on this path (the slot's sampler after a reset)
    {
        common_sampler * fresh = common_sampler_init(model, params);
        common_sampler_copy(save, fresh);
        CHECK(accept(fresh, "_call>q"));
        CHECK(!accept(fresh, "7"));
        common_sampler_free(fresh);
    }

    common_sampler_free(save);
    common_sampler_free(s);
    llama_model_free(model);
    llama_backend_free();

    printf("OK\n");
    return 0;
}
