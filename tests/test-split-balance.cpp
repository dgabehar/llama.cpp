// Tests the --split-balance cost model and optimizer with synthetic device profiles.

#include "split-balance.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { n_fail++; fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

// shaped like Qwen3.8-27B Q4_K: 64 layers, ~250 MB and ~0.85 GFLOP per layer per token
static common_split_model_shape qwen27b() {
    common_split_model_shape s;
    s.n_layer      = 64;
    s.layer_bytes  = 250e6;
    s.layer_flops  = 0.85e9;
    s.act_bytes    = 5120 * 4;
    s.output_bytes = 400e6;
    return s;
}

// fast local GPU (~190 GB/s, ~15 TFLOPS) and a remote one ~2.6x slower to decode, ~3.7x slower to prefill
static common_split_device_perf fast_gpu() {
    common_split_device_perf p;
    p.s_decode_byte  = 1.0 / 190e9;
    p.s_prefill_flop = 1.0 / 15e12;
    return p;
}

static common_split_device_perf slow_remote() {
    common_split_device_perf p;
    p.s_decode_byte  = 2.6 / 190e9;
    p.s_prefill_flop = 3.7 / 15e12;
    p.t_hop          = 150e-6;
    p.s_hop_byte     = 1.0 / 2e9;
    return p;
}

static std::vector<uint32_t> brute_force(common_split_balance_mode mode, const common_split_model_shape & shape,
                                         const std::vector<common_split_device_perf> & perf,
                                         const std::vector<uint32_t> & caps, const common_split_workload & wl) {
    // reference for 3 devices
    std::vector<uint32_t> best;
    double best_c = INFINITY;
    for (uint32_t a = 0; a <= shape.n_layer; a++) {
        for (uint32_t b = 0; a + b <= shape.n_layer; b++) {
            std::vector<uint32_t> n = { a, b, shape.n_layer - a - b };
            if (n[0] > caps[0] || n[1] > caps[1] || n[2] > caps[2]) {
                continue;
            }
            const double td = common_split_predict_decode(shape, perf, n);
            const double tp = common_split_predict_prefill(shape, perf, n, wl);
            const double c = mode == COMMON_SPLIT_BALANCE_DECODE ? td : mode == COMMON_SPLIT_BALANCE_PREFILL ? tp : tp + wl.n_gen * td;
            if (c < best_c * (1 - 1e-9)) {
                best_c = c;
                best = n;
            }
        }
    }
    return best;
}

int main() {
    const auto shape = qwen27b();
    // device order as llama.cpp lists them: RPC first, then the local GPU
    const std::vector<common_split_device_perf> perf = { slow_remote(), fast_gpu() };
    common_split_workload wl;
    wl.n_prompt = 8192;
    wl.n_gen    = 0;
    wl.n_ubatch = 512;
    wl.n_batch  = 8192;

    // memory mode leaves the split to llama.cpp
    CHECK(common_split_balance_optimize(COMMON_SPLIT_BALANCE_MEMORY, shape, perf, { 64, 64 }, wl).empty(), "memory mode must not choose a split");

    // decode: everything on the fast device when it fits
    {
        auto n = common_split_balance_optimize(COMMON_SPLIT_BALANCE_DECODE, shape, perf, { 64, 64 }, wl);
        CHECK(n.size() == 2 && n[0] == 0 && n[1] == 64, "decode should avoid the slow device, got %u/%u", n[0], n[1]);
    }
    // decode: fast device capped -> only the overflow goes to the slow one
    {
        auto n = common_split_balance_optimize(COMMON_SPLIT_BALANCE_DECODE, shape, perf, { 64, 50 }, wl);
        CHECK(n.size() == 2 && n[0] == 14 && n[1] == 50, "decode should fill the fast device to its cap, got %u/%u", n[0], n[1]);
    }
    // prefill: stages balanced, so the slow device gets ~1/(1+3.7) of the layers
    {
        auto n = common_split_balance_optimize(COMMON_SPLIT_BALANCE_PREFILL, shape, perf, { 64, 64 }, wl);
        CHECK(n.size() == 2 && n[0] >= 10 && n[0] <= 15, "prefill should give the slow device ~13 layers, got %u/%u", n[0], n[1]);
        // and it must predict faster prompt processing than either device alone
        const double t_split = common_split_predict_prefill(shape, perf, n, wl);
        const double t_fast  = common_split_predict_prefill(shape, perf, { 0, 64 }, wl);
        CHECK(t_split < t_fast, "balanced prefill should beat the fast device alone (%.3f vs %.3f s)", t_split, t_fast);
    }
    // every llama_decode() call drains the pipeline, so a smaller n_batch predicts less overlap
    {
        const std::vector<uint32_t> n = { 13, 51 };
        common_split_workload small = wl;
        small.n_batch = 2048;
        const double t_big   = common_split_predict_prefill(shape, perf, n, wl);
        const double t_small = common_split_predict_prefill(shape, perf, n, small);
        CHECK(t_small > t_big, "n_batch 2048 should predict slower prefill than 8192 (%.3f vs %.3f s)", t_small, t_big);
        // one device has nothing to overlap, so n_batch doesn't matter there
        const double t1_big   = common_split_predict_prefill(shape, perf, { 0, 64 }, wl);
        const double t1_small = common_split_predict_prefill(shape, perf, { 0, 64 }, small);
        CHECK(std::fabs(t1_big - t1_small) < 1e-9, "single device prefill should not depend on n_batch");
    }

    // auto: decode-heavy request behaves like decode, prompt-heavy like prefill
    {
        common_split_workload chat = wl;
        chat.n_prompt = 512;
        chat.n_gen    = 1024;
        auto n = common_split_balance_optimize(COMMON_SPLIT_BALANCE_AUTO, shape, perf, { 64, 64 }, chat);
        CHECK(n.size() == 2 && n[0] == 0, "decode-heavy auto should keep layers off the slow device, got %u/%u", n[0], n[1]);

        common_split_workload rag = wl;
        rag.n_prompt = 32768;
        rag.n_gen    = 16;
        auto m = common_split_balance_optimize(COMMON_SPLIT_BALANCE_AUTO, shape, perf, { 64, 64 }, rag);
        CHECK(m.size() == 2 && m[0] > 0, "prompt-heavy auto should use the slow device, got %u/%u", m[0], m[1]);
    }
    // infeasible caps -> no split
    CHECK(common_split_balance_optimize(COMMON_SPLIT_BALANCE_AUTO, shape, perf, { 20, 20 }, wl).empty(), "caps below n_layer must fail");

    // 3 devices: optimizer matches brute force in every mode, with and without caps
    {
        std::vector<common_split_device_perf> p3 = { slow_remote(), slow_remote(), fast_gpu() };
        p3[1].s_decode_byte *= 0.7;
        p3[1].s_prefill_flop *= 0.5;
        for (auto mode : { COMMON_SPLIT_BALANCE_DECODE, COMMON_SPLIT_BALANCE_PREFILL, COMMON_SPLIT_BALANCE_AUTO }) {
            for (const auto & caps : std::vector<std::vector<uint32_t>>{ { 64, 64, 64 }, { 30, 20, 30 } }) {
                common_split_workload w3 = wl;
                w3.n_gen = 64;
                auto got = common_split_balance_optimize(mode, shape, p3, caps, w3);
                auto ref = brute_force(mode, shape, p3, caps, w3);
                const double c_got = common_split_predict_prefill(shape, p3, got, w3) + (mode == COMMON_SPLIT_BALANCE_PREFILL ? 0 : 1) * w3.n_gen * common_split_predict_decode(shape, p3, got);
                const double c_ref = common_split_predict_prefill(shape, p3, ref, w3) + (mode == COMMON_SPLIT_BALANCE_PREFILL ? 0 : 1) * w3.n_gen * common_split_predict_decode(shape, p3, ref);
                if (mode == COMMON_SPLIT_BALANCE_DECODE) {
                    CHECK(std::fabs(common_split_predict_decode(shape, p3, got) - common_split_predict_decode(shape, p3, ref)) < 1e-12,
                          "decode mode not optimal for 3 devices");
                } else {
                    CHECK(c_got <= c_ref * (1 + 1e-9), "%s mode not optimal for 3 devices (%.6f vs %.6f)", common_split_balance_mode_name(mode), c_got, c_ref);
                }
                CHECK(got.size() == 3 && got[0] <= caps[0] && got[1] <= caps[1] && got[2] <= caps[2], "caps violated");
            }
        }
    }

    if (n_fail == 0) {
        printf("test-split-balance: OK\n");
    }
    return n_fail == 0 ? 0 : 1;
}
