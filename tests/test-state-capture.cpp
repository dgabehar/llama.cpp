// Tests llama_state_seq_capture_*: the state captured inside one llama_decode() must be identical to the
// state read between llama_decode() calls that end at the same positions.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <vector>

struct seg {
    llama_seq_id seq_id;
    llama_pos    p0; // first position
    llama_pos    p1; // one past the last position
};

static llama_token tok(llama_seq_id s, llama_pos p) {
    return (llama_token) (1 + (p*7 + s*13) % 97);
}

static bool decode_segs(llama_context * ctx, const std::vector<seg> & segs) {
    int32_t n = 0;
    for (const auto & sg : segs) {
        n += sg.p1 - sg.p0;
    }
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (const auto & sg : segs) {
        for (llama_pos p = sg.p0; p < sg.p1; ++p) {
            common_batch_add(batch, tok(sg.seq_id, p), p, { sg.seq_id }, p + 1 == sg.p1);
        }
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static std::vector<uint8_t> get_state(llama_context * ctx, llama_seq_id s, llama_state_seq_flags flags) {
    std::vector<uint8_t> data(llama_state_seq_get_size_ext(ctx, s, flags));
    data.resize(llama_state_seq_get_data_ext(ctx, data.data(), data.size(), s, flags));
    return data;
}

static std::vector<uint8_t> get_capture(llama_context * ctx, llama_seq_id s, llama_pos pos, llama_pos * pos_max = nullptr) {
    std::vector<uint8_t> data(llama_state_seq_capture_get(ctx, s, pos, nullptr, 0, nullptr, nullptr));
    llama_pos pmin = -1;
    llama_pos pmax = -1;
    if (!data.empty()) {
        data.resize(llama_state_seq_capture_get(ctx, s, pos, data.data(), data.size(), &pmin, &pmax));
    }
    if (pos_max) {
        *pos_max = pmax;
    }
    return data;
}

struct cap_point {
    llama_seq_id          seq_id;
    llama_pos             pos;
    llama_state_seq_flags flags;
};

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { n_fail++; fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

// decode the segments of every call in one batch with captures at the ends of all calls but the last,
// and compare with a reference that decodes call by call and reads the state in between
static void run_case(const char * name, llama_model * model, llama_context_params cparams,
                     const std::vector<std::vector<seg>> & calls, const std::vector<cap_point> & caps) {
    fprintf(stderr, "%s: %s\n", __func__, name);

    llama_context * ctx_ref = llama_init_from_model(model, cparams);
    llama_context * ctx_cap = llama_init_from_model(model, cparams);
    if (!ctx_ref || !ctx_cap) {
        CHECK(false, "failed to create contexts");
        return;
    }

    // reference: one decode per call, state read after each call
    std::vector<std::vector<uint8_t>> ref(caps.size());
    for (const auto & call : calls) {
        CHECK(decode_segs(ctx_ref, call), "reference decode failed");
        for (size_t i = 0; i < caps.size(); ++i) {
            for (const auto & sg : call) {
                if (sg.seq_id == caps[i].seq_id && sg.p1 - 1 == caps[i].pos) {
                    ref[i] = get_state(ctx_ref, caps[i].seq_id, caps[i].flags);
                }
            }
        }
    }

    // captured: all calls merged into one batch, per seq in position order
    std::vector<seg> merged;
    for (const auto & call : calls) {
        for (const auto & sg : call) {
            bool found = false;
            for (auto & m : merged) {
                if (m.seq_id == sg.seq_id && m.p1 == sg.p0) {
                    m.p1  = sg.p1;
                    found = true;
                }
            }
            if (!found) {
                merged.push_back(sg);
            }
        }
    }

    for (const auto & c : caps) {
        CHECK(llama_state_seq_capture_add(ctx_cap, c.seq_id, c.pos, c.flags), "capture_add failed for seq %d pos %d", c.seq_id, c.pos);
    }

    // a capture that is not reached must not be reported as taken
    CHECK(llama_state_seq_capture_add(ctx_cap, 0, 100000, 0), "capture_add failed for an unreached pos");

    CHECK(decode_segs(ctx_cap, merged), "capture decode failed");

    for (size_t i = 0; i < caps.size(); ++i) {
        llama_pos pmax = -1;
        const auto got = get_capture(ctx_cap, caps[i].seq_id, caps[i].pos, &pmax);
        CHECK(!got.empty(), "capture seq %d pos %d not taken", caps[i].seq_id, caps[i].pos);
        CHECK(pmax == caps[i].pos, "capture seq %d pos %d has pos_max %d", caps[i].seq_id, caps[i].pos, pmax);
        CHECK(got.size() == ref[i].size() && memcmp(got.data(), ref[i].data(), got.size()) == 0,
              "capture seq %d pos %d flags %u differs from the reference (%zu vs %zu bytes)",
              caps[i].seq_id, caps[i].pos, caps[i].flags, got.size(), ref[i].size());
    }
    CHECK(get_capture(ctx_cap, 0, 100000).empty(), "unreached capture reported as taken");

    // the live states at the end must match too
    for (const auto & sg : merged) {
        const auto a = get_state(ctx_ref, sg.seq_id, 0);
        const auto b = get_state(ctx_cap, sg.seq_id, 0);
        CHECK(a.size() == b.size() && memcmp(a.data(), b.data(), a.size()) == 0, "final state of seq %d differs", sg.seq_id);
    }

    // decoding again after a full removal retakes the captures (as chained MTP heads do)
    llama_state_seq_capture_clear(ctx_cap);
    for (const auto & c : caps) {
        llama_state_seq_capture_add(ctx_cap, c.seq_id, c.pos, c.flags);
    }
    for (int pass = 0; pass < 2; ++pass) {
        for (const auto & sg : merged) {
            llama_memory_seq_rm(llama_get_memory(ctx_cap), sg.seq_id, -1, -1);
        }
        CHECK(decode_segs(ctx_cap, merged), "capture re-decode failed");
    }
    for (size_t i = 0; i < caps.size(); ++i) {
        const auto got = get_capture(ctx_cap, caps[i].seq_id, caps[i].pos);
        CHECK(got.size() == ref[i].size() && memcmp(got.data(), ref[i].data(), got.size()) == 0,
              "retaken capture seq %d pos %d differs from the reference", caps[i].seq_id, caps[i].pos);
    }

    // clear drops everything
    llama_state_seq_capture_clear(ctx_cap);
    CHECK(get_capture(ctx_cap, caps[0].seq_id, caps[0].pos).empty(), "capture survived clear");

    llama_free(ctx_ref);
    llama_free(ctx_cap);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to load model\n", __func__);
        return 1;
    }

    const llama_state_seq_flags partial = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;

    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx      = 512;
    cparams.n_batch    = 256;
    cparams.n_ubatch   = 16;
    cparams.n_seq_max  = 2;
    cparams.kv_unified = false;
    cparams.n_rs_seq   = 0;

    // one seq, checkpoints like the server's (partial state), plus a full state capture
    run_case("single seq", model, cparams,
        { { { 0, 0, 21 } }, { { 0, 21, 51 } }, { { 0, 51, 55 } }, { { 0, 55, 70 } } },
        { { 0, 20, partial }, { 0, 50, partial }, { 0, 54, 0 } });

    // two seqs in one batch with the same cut, then a cut in one seq only
    run_case("two seqs", model, cparams,
        { { { 0, 0, 21 }, { 1, 0, 21 } }, { { 0, 21, 38 }, { 1, 21, 38 } }, { { 0, 38, 45 }, { 1, 38, 45 } } },
        { { 0, 20, partial }, { 1, 20, partial }, { 1, 37, partial } });

    // quantized KV cache: the full state has block-quantized ranges
    {
        auto cq = cparams;
        cq.type_k = GGML_TYPE_Q8_0;
        cq.type_v = GGML_TYPE_Q8_0;
        cq.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        run_case("q8_0 kv, full state", model, cq,
            { { { 0, 0, 33 } }, { { 0, 33, 40 } } },
            { { 0, 32, 0 } });
    }

    // on-device flags are not supported
    {
        llama_context * ctx = llama_init_from_model(model, cparams);
        CHECK(!llama_state_seq_capture_add(ctx, 0, 1, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE), "capture_add accepted ON_DEVICE");
        llama_free(ctx);
    }

    llama_model_free(model);

    if (n_fail == 0) {
        fprintf(stderr, "test-state-capture: OK\n");
    }
    return n_fail == 0 ? 0 : 1;
}
