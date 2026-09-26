#include "split-balance.h"
#include "build-info.h"

#include "common.h"
#include "fit.h"
#include "json.h"
#include "log.h"
#include "gguf.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <regex>
#include <map>
#include <sstream>
#include <thread>

const char * common_split_balance_mode_name(common_split_balance_mode mode) {
    switch (mode) {
        case COMMON_SPLIT_BALANCE_MEMORY:  return "memory";
        case COMMON_SPLIT_BALANCE_DECODE:  return "decode";
        case COMMON_SPLIT_BALANCE_PREFILL: return "prefill";
        case COMMON_SPLIT_BALANCE_AUTO:    return "auto";
    }
    return "?";
}

// Calibration times a layer's weight matmuls and attention; norms, RoPE, SSM scans and the host side
// of each graph are not timed. These factors cover them (fitted against measured llama-server runs).
static constexpr double k_prefill_overhead = 1.0;
static constexpr double k_decode_overhead  = 1.0;

double common_split_predict_decode(
        const common_split_model_shape & shape,
        const std::vector<common_split_device_perf> & perf,
        const std::vector<uint32_t> & n_layers) {
    // the output layer always stays on the last device, so that device is always part of the pipeline
    const int last = (int) n_layers.size() - 1;
    double t = 0.0;
    for (size_t i = 0; i < n_layers.size(); i++) {
        if (n_layers[i] == 0 && (int) i != last) {
            continue;
        }
        const auto & p = perf[i];
        double t_dev = n_layers[i] * shape.layer_bytes * p.s_decode_byte;
        if ((int) i == last) {
            t_dev += shape.output_bytes * (p.s_output_byte > 0.0 ? p.s_output_byte : p.s_decode_byte);
        }
        // a remote stage costs a round trip per token: activations in, activations out
        t += t_dev * k_decode_overhead + 2.0 * (p.t_hop + shape.act_bytes * p.s_hop_byte);
    }
    return t;
}

double common_split_predict_prefill(
        const common_split_model_shape & shape,
        const std::vector<common_split_device_perf> & perf,
        const std::vector<uint32_t> & n_layers,
        const common_split_workload & wl) {
    const uint32_t n_prompt = std::max<uint32_t>(1, wl.n_prompt);
    const uint32_t ub = std::max<uint32_t>(1, std::min(wl.n_ubatch, n_prompt));
    // the prompt is decoded n_batch tokens at a time and each call drains the pipeline
    const uint32_t nb = std::max<uint32_t>(ub, wl.n_batch > 0 ? std::min(wl.n_batch, n_prompt) : n_prompt);

    // ubatches are pipelined across the stages: the slowest stage sets the pace, and the
    // pipeline takes (n_stages - 1) extra stage times to fill
    double t_stage_max = 0.0;
    double t_stage_sum = 0.0;
    int    n_stages    = 0;
    for (size_t i = 0; i < n_layers.size(); i++) {
        if (n_layers[i] == 0 && i + 1 != n_layers.size()) {
            continue;
        }
        const auto & p = perf[i];
        const double t = n_layers[i] * shape.layer_flops * ub * p.s_prefill_flop * k_prefill_overhead
                       + 2.0 * (p.t_hop + ub * shape.act_bytes * p.s_hop_byte);
        t_stage_max = std::max(t_stage_max, t);
        t_stage_sum += t;
        n_stages++;
    }
    if (n_stages == 0) {
        return 0.0;
    }
    // per call: with a single ubatch nothing overlaps; the fill term covers that exactly
    // (the last, shorter ubatch of a call is counted as a full one)
    // the stages don't overlap perfectly (host-side copies between them, submission stalls): measured on
    // an 8060S + 780M (RPC) split, about 80% of the shorter stages' time is hidden
    constexpr double overlap = 0.8;
    const double t_steady = t_stage_max + (1.0 - overlap) * (t_stage_sum - t_stage_max);
    auto t_call = [&](uint32_t n_tokens) {
        const uint32_t n_ub = (n_tokens + ub - 1) / ub;
        return (n_ub - 1) * t_steady + t_stage_sum;
    };
    const uint32_t n_full = n_prompt / nb;
    const uint32_t n_rest = n_prompt % nb;
    return n_full * t_call(nb) + (n_rest > 0 ? t_call(n_rest) : 0.0);
}

struct split_cost {
    double primary   = std::numeric_limits<double>::infinity();
    double secondary = std::numeric_limits<double>::infinity();
};

std::vector<uint32_t> common_split_balance_optimize(
        common_split_balance_mode mode,
        const common_split_model_shape & shape,
        const std::vector<common_split_device_perf> & perf,
        const std::vector<uint32_t> & max_layers,
        const common_split_workload & wl) {
    const size_t nd = perf.size();
    if (mode == COMMON_SPLIT_BALANCE_MEMORY || nd == 0 || max_layers.size() != nd || shape.n_layer == 0) {
        return {};
    }
    uint64_t cap_sum = 0;
    for (uint32_t c : max_layers) {
        cap_sum += c;
    }
    if (cap_sum < shape.n_layer) {
        return {};
    }

    auto cost = [&](const std::vector<uint32_t> & n) {
        const double td = common_split_predict_decode(shape, perf, n);
        const double tp = common_split_predict_prefill(shape, perf, n, wl);
        split_cost c;
        switch (mode) {
            case COMMON_SPLIT_BALANCE_DECODE:  c.primary = td; c.secondary = tp; break;
            case COMMON_SPLIT_BALANCE_PREFILL: c.primary = tp; c.secondary = td; break;
            default:                           c.primary = tp + wl.n_gen * td; c.secondary = td; break;
        }
        return c;
    };
    auto better = [](const split_cost & a, const split_cost & b) {
        const double eps = 1e-9 * std::max(1.0, std::fabs(b.primary));
        if (a.primary < b.primary - eps) {
            return true;
        }
        return a.primary <= b.primary + eps && a.secondary < b.secondary;
    };

    std::vector<uint32_t> best;
    split_cost best_cost;
    std::vector<uint32_t> cur(nd, 0);

    // exhaustive over all compositions of n_layer within the caps; this is small for the
    // device counts a layer split is used with (65 candidates for 2 devices, ~2k for 3)
    std::function<void(size_t, uint32_t)> search = [&](size_t i, uint32_t left) {
        if (i + 1 == nd) {
            if (left > max_layers[i]) {
                return;
            }
            cur[i] = left;
            const split_cost c = cost(cur);
            if (best.empty() || better(c, best_cost)) {
                best = cur;
                best_cost = c;
            }
            return;
        }
        // the remaining devices must be able to take what this one leaves
        uint64_t rest_cap = 0;
        for (size_t j = i + 1; j < nd; j++) {
            rest_cap += max_layers[j];
        }
        const uint32_t lo = rest_cap >= left ? 0 : (uint32_t) (left - rest_cap);
        const uint32_t hi = std::min(left, max_layers[i]);
        for (uint32_t n = lo; n <= hi; n++) {
            cur[i] = n;
            search(i + 1, left - n);
        }
    };

    uint64_t n_candidates = 1;
    for (size_t i = 1; i < nd; i++) {
        n_candidates *= (uint64_t) shape.n_layer + 1;
    }
    if (n_candidates <= 4'000'000) {
        search(0, shape.n_layer);
        return best;
    }

    // many devices: start from a speed-proportional split and move single layers
    // between devices while that improves the cost
    std::vector<uint32_t> n(nd, 0);
    {
        double inv_sum = 0.0;
        for (const auto & p : perf) {
            inv_sum += 1.0 / std::max(1e-18, p.s_decode_byte);
        }
        uint32_t assigned = 0;
        for (size_t i = 0; i < nd; i++) {
            n[i] = std::min<uint32_t>(max_layers[i], (uint32_t) (shape.n_layer * (1.0 / std::max(1e-18, perf[i].s_decode_byte)) / inv_sum));
            assigned += n[i];
        }
        for (size_t i = 0; assigned < shape.n_layer; i = (i + 1) % nd) {
            if (n[i] < max_layers[i]) {
                n[i]++;
                assigned++;
            }
        }
    }
    split_cost c_n = cost(n);
    for (bool improved = true; improved; ) {
        improved = false;
        for (size_t from = 0; from < nd; from++) {
            for (size_t to = 0; to < nd; to++) {
                if (from == to || n[from] == 0 || n[to] >= max_layers[to]) {
                    continue;
                }
                n[from]--; n[to]++;
                const split_cost c = cost(n);
                if (better(c, c_n)) {
                    c_n = c;
                    improved = true;
                } else {
                    n[from]++; n[to]--;
                }
            }
        }
    }
    return n;
}

//
// calibration
//

static bool dev_is_remote(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    return reg != nullptr && strcmp(ggml_backend_reg_name(reg), "RPC") == 0;
}

static std::string cache_key(ggml_backend_dev_t dev, const common_split_calib_model & m, const common_split_workload & wl,
                             bool with_build = true) {
    size_t free = 0;
    size_t total = 0;
    ggml_backend_dev_memory(dev, &free, &total);
    // the model part of the key: every timed layer kind with its shapes and types
    std::ostringstream ms;
    for (const auto & l : m.layers) {
        ms << l.share << '*' << l.byte_scale << ':' << l.n_head << '/' << l.n_head_kv << '/' << l.head_dim_k << '/' << l.head_dim_v
           << "|gdn" << l.gdn_state << '/' << l.gdn_h_k << '/' << l.gdn_h_v << '/' << l.conv_k << '/' << l.conv_ch;
        for (const auto & w : l.weights) {
            ms << ',' << w.ne[0] << 'x' << w.ne[1] << 'x' << w.ne[2] << ggml_type_name(w.type);
        }
        ms << ';';
    }
    ms << "nodes" << (int) m.nodes_tg_layer << '/' << (int) m.nodes_pp_layer;
    ms << "moe" << m.n_expert_used << "of" << m.n_expert << "|out" << m.output.ne[0] << 'x' << m.output.ne[1]
       << ggml_type_name(m.output.type);
    std::ostringstream ss;
    // the build is part of the key: kernels change between builds (the remote worker's build is not visible
    // from here, driver/kernel upgrades neither, hence also the age limit on entries)
    ss << ggml_backend_dev_name(dev) << "|" << ggml_backend_dev_description(dev) << "|" << (total >> 20) << "MiB|layers"
       << std::hex << std::hash<std::string>{}(ms.str()) << std::dec << "|ub" << wl.n_ubatch << "|p" << wl.n_prompt
       << "g" << wl.n_gen;
    if (with_build) {
        ss << "|" << llama_commit();
    }
    return ss.str();
}

// fills a quantized tensor with valid data by quantizing a few random rows and repeating them
static void fill_weight(ggml_tensor * t) {
    const int64_t n_per_row = t->ne[0];
    const int64_t n_rows    = ggml_nrows(t);
    const int64_t n_src     = std::min<int64_t>(16, n_rows);
    std::vector<float> src(n_per_row * n_src);
    uint32_t state = 12345;
    for (auto & v : src) {
        state = state * 1664525u + 1013904223u;
        v = ((state >> 8) & 0xffff) / 65536.0f - 0.5f;
    }
    const size_t row_size = ggml_row_size(t->type, n_per_row);
    std::vector<uint8_t> rows(row_size * n_src);
    ggml_quantize_chunk(t->type, src.data(), rows.data(), 0, n_src, n_per_row, nullptr);
    std::vector<uint8_t> data(ggml_nbytes(t));
    for (int64_t r = 0; r < n_rows; r++) {
        memcpy(data.data() + r * row_size, rows.data() + (r % n_src) * row_size, row_size);
    }
    ggml_backend_tensor_set(t, data.data(), 0, data.size());
}

static void fill_f32(ggml_tensor * t) {
    std::vector<float> data(ggml_nelements(t));
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = (float) (i % 17) * 0.05f - 0.4f;
    }
    ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
}

// Reading a value back is what makes this wait for the result: for a remote (RPC) device,
// graph_compute has no reply and synchronize only waits for the request to be sent.
static void compute_and_wait(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    float v;
    ggml_backend_tensor_get(ggml_graph_node(gf, -1), &v, 0, sizeof(v));
}

static double time_graph(ggml_backend_t backend, ggml_cgraph * gf, int reps, bool warmup = true) {
    if (warmup) {
        compute_and_wait(backend, gf); // compiles pipelines
    }
    std::vector<double> t;
    for (int r = 0; r < reps; r++) {
        const int64_t t0 = ggml_time_us();
        compute_and_wait(backend, gf);
        t.push_back((ggml_time_us() - t0) * 1e-6);
    }
    std::nth_element(t.begin(), t.begin() + t.size() / 2, t.end());
    return t[t.size() / 2];
}

// fills an expert-id tensor [n_used, n_tokens]: each token goes to n_used distinct experts, rotating
// through all n_e of them so every expert gets an equal share
static void fill_ids(ggml_tensor * ids, int n_e, int offset) {
    const int64_t n_used = ids->ne[0];
    std::vector<int32_t> data(ggml_nelements(ids));
    for (int64_t t = 0; t < ids->ne[1]; t++) {
        for (int64_t j = 0; j < n_used; j++) {
            data[t * n_used + j] = (int32_t) ((offset + t * n_used + j) % n_e);
        }
    }
    ggml_backend_tensor_set(ids, data.data(), 0, ggml_nbytes(ids));
}

// graphs timed for one layer kind: a single token through every weight (and the attention of one
// query over the reference context), and one ubatch through them, experts in a graph of their own
struct calib_layer_graphs {
    ggml_cgraph * decode      = nullptr;
    ggml_cgraph * prefill     = nullptr;
    ggml_cgraph * prefill_exp = nullptr; // null without experts
    double        share       = 0.0;
    double        byte_scale  = 1.0;
};

// Adds `out` to the graph through a one-element copy: the op's own output then has a consumer, so
// the graph allocator can reuse its memory for the next op instead of keeping every output alive.
static void add_timed_op(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * out) {
    ggml_build_forward_expand(gf, ggml_cont(ctx, ggml_view_1d(ctx, out, 1, 0)));
}

static bool calibrate_device(ggml_backend_dev_t dev, const common_split_calib_model & m, const common_split_workload & wl,
                             int64_t n_embd, common_split_device_perf & out) {
    ggml_backend_ptr backend { ggml_backend_dev_init(dev, nullptr) };
    if (!backend) {
        return false;
    }
    const bool remote = dev_is_remote(dev);

    const int64_t ub     = std::max<uint32_t>(1, std::min(wl.n_ubatch, std::max<uint32_t>(1, wl.n_prompt)));
    // attention runs over the context seen by the reference request: on average half the prompt while it
    // is processed, the prompt plus half the generated tokens while decoding
    const int64_t n_kv_pp = GGML_PAD(std::max<int64_t>(256, wl.n_prompt / 2 + ub / 2), 256);
    const int64_t n_kv_tg = GGML_PAD(std::max<int64_t>(256, (int64_t) wl.n_prompt + wl.n_gen / 2), 256);

    const bool    moe    = m.n_expert > 0 && m.n_expert_used > 0 && m.n_expert_used <= m.n_expert;
    const int64_t n_used = moe ? m.n_expert_used : 1;

    // one layer of every kind is uploaded; a stack of experts larger than the budget is timed with fewer
    // experts and a proportionally smaller batch, so every expert still gets the model's number of rows
    size_t free = 0;
    size_t total = 0;
    ggml_backend_dev_memory(dev, &free, &total);
    const size_t budget = std::min<size_t>(size_t(1536) << 20, std::max<size_t>(size_t(256) << 20, free / 4));
    size_t bytes_all = 0;
    size_t bytes_exp = 0;
    for (const auto & l : m.layers) {
        for (const auto & w : l.weights) {
            const size_t b = ggml_row_size(w.type, w.ne[0]) * w.ne[1] * w.ne[2];
            bytes_all += b;
            bytes_exp += w.ne[2] > 1 ? b : 0;
        }
    }
    int64_t n_e = moe ? m.n_expert : 1;
    if (moe && bytes_all > budget && bytes_exp > 0) {
        const double keep = std::max(0.0, (double) budget - (double) (bytes_all - bytes_exp)) / (double) bytes_exp;
        n_e = std::clamp<int64_t>((int64_t) (m.n_expert * keep), std::min<int64_t>(m.n_expert, std::max<int64_t>(2 * n_used, 8)), m.n_expert);
    }
    const double  exp_scale = (double) n_e / std::max<uint32_t>(1, m.n_expert);
    const int64_t ub_exp    = std::max<int64_t>(1, (int64_t) std::llround(ub * exp_scale));

    // what one node costs to submit and run: a chain of scales over one token's activations (dispatch bound)
    // and over a ubatch of them (memory bound). Long enough that a backend which submits one command buffer
    // per node (e.g. Vulkan with GGML_VK_MAX_NODES_PER_SUBMIT=1) pays its real per-submit cost here too, not
    // just the cheap, unsaturated cost a short chain would see. The decode graphs below repeat a layer's own
    // ops to the same length, for the same reason (see n_rep below).
    constexpr int n_chain = 256;

    size_t n_tensors = 16;
    for (const auto & l : m.layers) {
        n_tensors += 6 * l.weights.size() + 32;
    }

    // A layer's decode ops (its weight matmuls, attention and GDN) are timed together as one graph call.
    // On their own they are far shorter than n_chain, so under per-node submits this graph never reaches
    // the same submit-queue depth a real, many-layer decode graph reaches -- it pays the cheap, unsaturated
    // per-submit cost instead of the real one, the same way the untimed ops would without n_chain above.
    // Repeat the ops until the graph is long enough, and divide the measured time by the repeat count.
    // Prefill is not repeated: its nodes are already much heavier (a whole ubatch each), so the same fixed
    // per-submit cost is a far smaller share of it.
    std::vector<int> n_rep(m.layers.size(), 1);
    std::vector<int> n1_decode(m.layers.size(), 0);
    std::vector<int> n1_prefill(m.layers.size(), 0);
    std::vector<int> n1_prefill_exp(m.layers.size(), 0);
    for (size_t li = 0; li < m.layers.size(); li++) {
        const auto & l = m.layers[li];
        int n_exp = 0, n_nonexp = 0;
        for (const auto & w : l.weights) {
            (w.ne[2] > 1 ? n_exp : n_nonexp)++;
        }
        const int n_gdn  = l.gdn_state > 0 ? 2 : 0;
        const int n_attn = l.n_head > 0 ? 1 : 0;
        n1_decode[li]      = n_exp + n_nonexp + n_gdn + n_attn;
        n1_prefill[li]     = n_nonexp + n_gdn + n_attn;
        n1_prefill_exp[li] = n_exp;
        n_rep[li] = n1_decode[li] > 0 ? std::max(1, (n_chain + n1_decode[li] - 1) / n1_decode[li]) : 1;
    }
    size_t n_rep_tensors = 0;
    for (size_t li = 0; li < m.layers.size(); li++) {
        n_rep_tensors += (size_t) n_rep[li] * n1_decode[li] * 3; // the op itself plus add_timed_op's view and cont
    }

    ggml_init_params wparams = {
        /* .mem_size   = */ ggml_tensor_overhead() * n_tensors,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr wctx { ggml_init(wparams) };          // weights and inputs, allocated once
    ggml_init_params gparams = {
        // 2 * n_chain: the untimed-op probe below builds two chains of n_chain scale nodes in this same context
        /* .mem_size   = */ ggml_tensor_overhead() * (n_tensors * 4 + 2 * n_chain + n_rep_tensors) + (3 * m.layers.size() + 4) * ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr gctx { ggml_init(gparams) };          // ops, placed by a graph allocator per graph

    // inputs, one per distinct input width and token count
    std::map<std::pair<int64_t, int64_t>, ggml_tensor *> inputs;
    auto input = [&](int64_t ne0, int64_t n_tok, bool exp) {
        const auto key = std::make_pair(ne0, exp ? -n_tok : n_tok);
        auto it = inputs.find(key);
        if (it != inputs.end()) {
            return it->second;
        }
        ggml_tensor * t = exp ? ggml_new_tensor_3d(wctx.get(), GGML_TYPE_F32, ne0, 1, n_tok)
                              : ggml_new_tensor_2d(wctx.get(), GGML_TYPE_F32, ne0, n_tok);
        inputs[key] = t;
        return t;
    };
    ggml_tensor * ids_tg = moe ? ggml_new_tensor_2d(wctx.get(), GGML_TYPE_I32, n_used, 1)      : nullptr;
    ggml_tensor * ids_pp = moe ? ggml_new_tensor_2d(wctx.get(), GGML_TYPE_I32, n_used, ub_exp) : nullptr;
    ggml_tensor * x_hop  = ggml_new_tensor_2d(wctx.get(), GGML_TYPE_F32, n_embd, ub);
    ggml_tensor * out_w  = nullptr;
    if (m.output.ne[1] > 0) {
        const ggml_type ot = ggml_quantize_requires_imatrix(m.output.type) ? GGML_TYPE_Q4_K : m.output.type;
        const int64_t rows_budget = std::max<int64_t>(256, (int64_t) ((size_t(128) << 20) / ggml_row_size(ot, m.output.ne[0])));
        out_w = ggml_new_tensor_2d(wctx.get(), ot, m.output.ne[0], std::min<int64_t>(m.output.ne[1], rows_budget));
        ggml_tensor * x_out = ggml_new_tensor_2d(wctx.get(), GGML_TYPE_F32, m.output.ne[0], 1);
        ggml_set_name(x_out, "calib_out_x");
    }

    struct attn_io { ggml_tensor * q_tg, * q_pp, * k_tg, * v_tg, * k_pp, * v_pp; };
    // gated delta net inputs for 1 and ub tokens: q, k, v, g, beta, the state, and the conv input and kernel
    struct gdn_io { ggml_tensor * q[2], * k[2], * v[2], * g[2], * b[2], * sx[2], * s, * c; };
    std::vector<gdn_io> gdn(m.layers.size(), gdn_io{});
    std::vector<std::vector<ggml_tensor *>> weights(m.layers.size());
    std::vector<attn_io> attn(m.layers.size(), attn_io{});
    for (size_t li = 0; li < m.layers.size(); li++) {
        const auto & l = m.layers[li];
        for (const auto & w : l.weights) {
            const bool exp = w.ne[2] > 1;
            weights[li].push_back(exp ? ggml_new_tensor_3d(wctx.get(), w.type, w.ne[0], w.ne[1], std::min<int64_t>(w.ne[2], n_e))
                                      : ggml_new_tensor_2d(wctx.get(), w.type, w.ne[0], w.ne[1]));
            input(w.ne[0], 1, exp);
            input(w.ne[0], exp ? ub_exp : ub, exp);
        }
        if (l.gdn_state > 0) {
            auto & d = gdn[li];
            const int64_t S = l.gdn_state;
            for (int j = 0; j < 2; j++) {
                const int64_t T = j == 0 ? 1 : ub;
                d.q[j]  = ggml_new_tensor_4d(wctx.get(), GGML_TYPE_F32, S, l.gdn_h_k, T, 1);
                d.k[j]  = ggml_new_tensor_4d(wctx.get(), GGML_TYPE_F32, S, l.gdn_h_k, T, 1);
                d.v[j]  = ggml_new_tensor_4d(wctx.get(), GGML_TYPE_F32, S, l.gdn_h_v, T, 1);
                d.g[j]  = ggml_new_tensor_4d(wctx.get(), GGML_TYPE_F32, 1, l.gdn_h_v, T, 1);
                d.b[j]  = ggml_new_tensor_4d(wctx.get(), GGML_TYPE_F32, 1, l.gdn_h_v, T, 1);
                d.sx[j] = ggml_new_tensor_3d(wctx.get(), GGML_TYPE_F32, l.conv_k - 1 + T, l.conv_ch, 1);
            }
            d.s = ggml_new_tensor_4d(wctx.get(), GGML_TYPE_F32, S, S, l.gdn_h_v, 1);
            d.c = ggml_new_tensor_2d(wctx.get(), GGML_TYPE_F32, l.conv_k, l.conv_ch);
        }
        if (l.n_head > 0) {
            auto & a = attn[li];
            a.q_tg = ggml_new_tensor_3d(wctx.get(), GGML_TYPE_F32, l.head_dim_k, 1,  l.n_head);
            a.q_pp = ggml_new_tensor_3d(wctx.get(), GGML_TYPE_F32, l.head_dim_k, ub, l.n_head);
            a.k_tg = ggml_new_tensor_3d(wctx.get(), GGML_TYPE_F16, l.head_dim_k, n_kv_tg, l.n_head_kv);
            a.v_tg = ggml_new_tensor_3d(wctx.get(), GGML_TYPE_F16, l.head_dim_v, n_kv_tg, l.n_head_kv);
            a.k_pp = ggml_view_3d(wctx.get(), a.k_tg, l.head_dim_k, n_kv_pp, l.n_head_kv, a.k_tg->nb[1], a.k_tg->nb[2], 0);
            a.v_pp = ggml_view_3d(wctx.get(), a.v_tg, l.head_dim_v, n_kv_pp, l.n_head_kv, a.v_tg->nb[1], a.v_tg->nb[2], 0);
        }
    }

    // the graphs
    std::vector<calib_layer_graphs> graphs(m.layers.size());
    bool attn_ok = true;
    for (size_t li = 0; li < m.layers.size(); li++) {
        const auto & l = m.layers[li];
        auto & g = graphs[li];
        g.share      = l.share;
        g.byte_scale = l.byte_scale;
        g.decode  = ggml_new_graph(gctx.get());
        g.prefill = ggml_new_graph(gctx.get());
        bool has_exp = false;
        for (size_t wi = 0; wi < l.weights.size(); wi++) {
            has_exp = has_exp || l.weights[wi].ne[2] > 1;
        }
        if (has_exp) {
            g.prefill_exp = ggml_new_graph(gctx.get());
        }
        // decode ops repeat n_rep[li] times (see n_rep's declaration above); prefill/prefill_exp ops are
        // only built on the first pass
        bool layer_attn_ok = true;
        for (int rep = 0; rep < n_rep[li]; rep++) {
            for (size_t wi = 0; wi < l.weights.size(); wi++) {
                ggml_tensor * w  = weights[li][wi];
                const bool   exp = l.weights[wi].ne[2] > 1;
                if (exp) {
                    add_timed_op(gctx.get(), g.decode, ggml_mul_mat_id(gctx.get(), w, input(w->ne[0], 1, true), ids_tg));
                    if (rep == 0) {
                        add_timed_op(gctx.get(), g.prefill_exp, ggml_mul_mat_id(gctx.get(), w, input(w->ne[0], ub_exp, true), ids_pp));
                    }
                } else {
                    add_timed_op(gctx.get(), g.decode, ggml_mul_mat(gctx.get(), w, input(w->ne[0], 1, false)));
                    if (rep == 0) {
                        add_timed_op(gctx.get(), g.prefill, ggml_mul_mat(gctx.get(), w, input(w->ne[0], ub, false)));
                    }
                }
            }
            if (l.gdn_state > 0) {
                const auto & d = gdn[li];
                add_timed_op(gctx.get(), g.decode, ggml_ssm_conv(gctx.get(), d.sx[0], d.c));
                add_timed_op(gctx.get(), g.decode, ggml_gated_delta_net(gctx.get(), d.q[0], d.k[0], d.v[0], d.g[0], d.b[0], d.s, 1));
                if (rep == 0) {
                    add_timed_op(gctx.get(), g.prefill, ggml_ssm_conv(gctx.get(), d.sx[1], d.c));
                    add_timed_op(gctx.get(), g.prefill, ggml_gated_delta_net(gctx.get(), d.q[1], d.k[1], d.v[1], d.g[1], d.b[1], d.s, 1));
                }
            }
            if (l.n_head > 0 && layer_attn_ok) {
                const auto & a = attn[li];
                const float scale = 1.0f / std::sqrt((float) l.head_dim_k);
                ggml_tensor * fa_tg = ggml_flash_attn_ext(gctx.get(), a.q_tg, a.k_tg, a.v_tg, nullptr, scale, 0.0f, 0.0f);
                if (rep == 0) {
                    ggml_tensor * fa_pp = ggml_flash_attn_ext(gctx.get(), a.q_pp, a.k_pp, a.v_pp, nullptr, scale, 0.0f, 0.0f);
                    layer_attn_ok = ggml_backend_supports_op(backend.get(), fa_tg) && ggml_backend_supports_op(backend.get(), fa_pp);
                    if (layer_attn_ok) {
                        add_timed_op(gctx.get(), g.decode,  fa_tg);
                        add_timed_op(gctx.get(), g.prefill, fa_pp);
                    }
                } else {
                    add_timed_op(gctx.get(), g.decode, fa_tg);
                }
            }
        }
        attn_ok = attn_ok && layer_attn_ok;
        for (ggml_cgraph * gf : { g.decode, g.prefill, g.prefill_exp }) {
            if (gf == nullptr) {
                continue;
            }
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                if (!ggml_backend_supports_op(backend.get(), ggml_graph_node(gf, i))) {
                    LOG_WRN("%s: %s cannot run %s, not calibrating it\n", __func__, ggml_backend_dev_name(dev),
                            ggml_op_desc(ggml_graph_node(gf, i)));
                    return false;
                }
            }
        }
    }
    if (!attn_ok) {
        LOG_WRN("%s: %s cannot run the attention op of the calibration, timing the matmuls only\n", __func__,
                ggml_backend_dev_name(dev));
    }

    ggml_backend_buffer_ptr wbuf { ggml_backend_alloc_ctx_tensors(wctx.get(), backend.get()) };
    if (!wbuf) {
        return false;
    }
    for (size_t li = 0; li < m.layers.size(); li++) {
        for (ggml_tensor * w : weights[li]) {
            fill_weight(w);
        }
        const auto & d = gdn[li];
        if (d.s != nullptr) {
            for (int j = 0; j < 2; j++) {
                for (ggml_tensor * t : { d.q[j], d.k[j], d.v[j], d.g[j], d.b[j], d.sx[j] }) {
                    fill_f32(t);
                }
            }
            fill_f32(d.s);
            fill_f32(d.c);
        }
        const auto & a = attn[li];
        if (a.k_tg != nullptr) {
            // f16 weights fill: fill_weight quantizes to the tensor's type
            fill_weight(a.k_tg);
            fill_weight(a.v_tg);
            fill_f32(a.q_tg);
            fill_f32(a.q_pp);
        }
    }
    for (auto & [key, t] : inputs) {
        fill_f32(t);
    }
    if (moe) {
        // the decode token and the prompt tokens spread over all timed experts, as uniform routing would
        fill_ids(ids_tg, (int) n_e, 0);
        fill_ids(ids_pp, (int) n_e, 0);
    }
    fill_f32(x_hop);
    if (out_w != nullptr) {
        fill_weight(out_w);
        fill_f32(ggml_get_tensor(wctx.get(), "calib_out_x"));
    }

    // round trip: a small write and a read back
    out.t_hop      = 0.0;
    out.s_hop_byte = 0.0;
    if (remote) {
        float v = 1.0f;
        std::vector<double> rt;
        for (int i = 0; i < 20; i++) {
            const int64_t t0 = ggml_time_us();
            ggml_backend_tensor_set(x_hop, &v, 0, sizeof(v));
            ggml_backend_tensor_get(x_hop, &v, 0, sizeof(v));
            rt.push_back((ggml_time_us() - t0) * 1e-6);
        }
        std::nth_element(rt.begin(), rt.begin() + rt.size() / 2, rt.end());
        out.t_hop = rt[rt.size() / 2] / 2.0;

        // bandwidth: one ubatch of activations
        std::vector<uint8_t> buf(ggml_nbytes(x_hop));
        std::vector<double> bw;
        for (int i = 0; i < 3; i++) {
            const int64_t t0 = ggml_time_us();
            ggml_backend_tensor_set(x_hop, buf.data(), 0, buf.size());
            bw.push_back((ggml_time_us() - t0) * 1e-6);
        }
        std::nth_element(bw.begin(), bw.begin() + bw.size() / 2, bw.end());
        out.s_hop_byte = std::max(0.0, bw[bw.size() / 2] - out.t_hop) / (double) buf.size();
    }

    // every graph gets its own allocation of the op outputs; they are timed one after another
    // kind: 0 decode, 1 prefill, 2 prefill of the experts
    struct timed { ggml_cgraph * gf; ggml_gallocr_t alloc; double best; double share; int kind; double byte_scale = 1.0; int n_rep = 1; };
    std::vector<timed> all;
    for (size_t li = 0; li < graphs.size(); li++) {
        auto & g = graphs[li];
        int kind = 0;
        for (ggml_cgraph * gf : { g.decode, g.prefill, g.prefill_exp }) {
            if (gf != nullptr && ggml_graph_n_nodes(gf) > 0) {
                const int rep = kind == 0 ? n_rep[li] : 1; // only decode graphs are repeated, see n_rep above
                all.push_back({ gf, nullptr, std::numeric_limits<double>::infinity(), g.share, kind, g.byte_scale, rep });
            }
            kind++;
        }
    }
    // the output layer: memory bound like decode, but a single wide matrix (often another weight type); a slice
    // of it keeps the upload small
    if (m.output.ne[1] > 0) {
        ggml_cgraph * g_out = ggml_new_graph(gctx.get());
        ggml_tensor * x_out = ggml_get_tensor(wctx.get(), "calib_out_x");
        add_timed_op(gctx.get(), g_out, ggml_mul_mat(gctx.get(), out_w, x_out));
        if (ggml_backend_supports_op(backend.get(), ggml_graph_node(g_out, 0))) {
            all.push_back({ g_out, nullptr, std::numeric_limits<double>::infinity(), 0.0, 3 });
        }
    }

    // the untimed-op probe: a chain of scales over one token's activations (dispatch bound) and over a
    // ubatch of them (memory bound) -- see n_chain's declaration above for why it needs to be this long
    for (int j = 0; j < 2; j++) {
        ggml_cgraph * g_ops = ggml_new_graph(gctx.get());
        ggml_tensor * y = j == 0 ? ggml_view_1d(gctx.get(), x_hop, n_embd, 0) : x_hop;
        for (int i = 0; i < n_chain; i++) {
            y = ggml_scale(gctx.get(), y, 1.0f);
        }
        ggml_build_forward_expand(g_ops, y);
        all.push_back({ g_ops, nullptr, std::numeric_limits<double>::infinity(), 0.0, j == 0 ? 4 : 5 });
    }

    // the fixed cost of running a graph and reading a value back (submit, fence, and for a remote device the
    // round trip): a real model pays it once per token or ubatch, not per layer, so it comes off every timing
    ggml_cgraph * g_sync = ggml_new_graph(gctx.get());
    ggml_build_forward_expand(g_sync, ggml_cont(gctx.get(), ggml_view_1d(gctx.get(), x_hop, 1, 0)));
    all.push_back({ g_sync, nullptr, std::numeric_limits<double>::infinity(), 0.0, -1 });

    struct gallocr_guard {
        std::vector<timed> & v;
        ~gallocr_guard() { for (auto & t : v) { if (t.alloc) { ggml_gallocr_free(t.alloc); } } }
    } guard { all };
    for (auto & t : all) {
        t.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get()));
        if (!ggml_gallocr_alloc_graph(t.alloc, t.gf)) {
            return false;
        }
    }

    // Other load on the node (compiles, other models) only ever slows a timing down, so each graph keeps its
    // fastest round. Rounds repeat until the last one agrees with the best within 10%, at most 4.
    // one untimed pass first: an idle GPU needs a moment to raise its clocks. It also sizes the repetitions:
    // a graph that runs for tens of ms is timed once per round (a slow device spent most of its calibration
    // repeating its long prefill graphs), a short one takes the median of 3.
    std::vector<int> reps(all.size());
    for (size_t i = 0; i < all.size(); i++) {
        compute_and_wait(backend.get(), all[i].gf);
        const int64_t t0 = ggml_time_us();
        compute_and_wait(backend.get(), all[i].gf);
        reps[i] = ggml_time_us() - t0 > 50000 ? 1 : 3;
    }
    double worst_spread = 1.0;
    std::vector<double> first(all.size());
    int n_rounds = 0;
    for (int round = 0; round < 4; round++) {
        n_rounds++;
        bool settled = round > 0;
        for (size_t i = 0; i < all.size(); i++) {
            const double t = time_graph(backend.get(), all[i].gf, reps[i], false);
            if (round == 0) {
                first[i] = t;
            }
            settled = settled && t <= 1.1 * all[i].best;
            all[i].best = std::min(all[i].best, t);
            if (all[i].kind >= 0) {
                worst_spread = std::max(worst_spread, first[i] / all[i].best);
            }
        }
        if (settled) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    for (const auto & t : all) {
        LOG_DBG("%s: %s: graph kind %d, %d nodes: %.3f ms\n", __func__, ggml_backend_dev_name(dev), t.kind,
                ggml_graph_n_nodes(t.gf), t.best * 1e3);
    }
    // a spread inside one run is mostly clock ramp-up; a load that lasts the whole run slows every round
    // alike, which is what the comparison with the device's reference calibration catches
    LOG_DBG("%s: %s: %d rounds, first round %.2fx its best\n", __func__, ggml_backend_dev_name(dev), n_rounds, worst_spread);

    // per mean layer, turned into the rates the cost model multiplies with layer_bytes / layer_flops
    double t_sync = 0.0;
    for (const auto & t : all) {
        if (t.kind == -1) {
            t_sync = t.best;
        }
    }
    double t_decode  = 0.0;
    double t_prefill = 0.0;
    double t_op_tg   = 0.0;
    double t_op_pp   = 0.0;
    for (auto & t : all) {
        // decode graphs ran n_rep copies of the layer's ops back to back (see n_rep above); the fixed
        // per-call sync cost is paid once for the whole call, so it comes off before dividing back down
        t.best = std::max(t.best - t_sync, 0.1 * t.best) / t.n_rep;
        switch (t.kind) {
            case 3: out.s_output_byte = t.best / (double) ggml_nbytes(out_w); break;
            case 4: t_op_tg = t.best / n_chain; break;
            case 5: t_op_pp = t.best / n_chain; break;
            case 0: t_decode  += t.share * t.best * t.byte_scale;             break;
            case 1: t_prefill += t.share * t.best * t.byte_scale;             break;
            case 2: t_prefill += t.share * t.best * t.byte_scale / exp_scale; break;
        }
    }
    // the layer's other ops: its real node count minus the ops timed above. Computed from n1_decode/
    // n1_prefill/n1_prefill_exp (the single-pass op counts) rather than by counting the (now repeated)
    // decode graphs' nodes directly.
    double timed_tg = 0.0;
    double timed_pp = 0.0;
    for (size_t li = 0; li < m.layers.size(); li++) {
        timed_tg += m.layers[li].share * n1_decode[li];
        timed_pp += m.layers[li].share * (n1_prefill[li] + n1_prefill_exp[li]);
    }
    if (m.nodes_tg_layer > 0.0) {
        t_decode  += std::max(0.0, m.nodes_tg_layer - timed_tg) * t_op_tg;
    }
    if (m.nodes_pp_layer > 0.0) {
        t_prefill += std::max(0.0, m.nodes_pp_layer - timed_pp) * t_op_pp;
    }

    out.s_decode_byte  = t_decode  / std::max(1.0, m.layer_bytes);
    out.s_prefill_flop = t_prefill / std::max(1.0, m.layer_flops * (double) ub);
    return true;
}

// A short decode timing (a fraction of the full calibration) used to notice that the device behind
// a cached entry has changed or was measured under load. For RPC devices the cache key only knows
// the endpoint, so a worker with other hardware, driver or build behind the same address would
// otherwise reuse stale speeds.
static bool quick_check(ggml_backend_dev_t dev, int64_t n_embd, int64_t n_ff, ggml_type wtype, double & s_byte) {
    ggml_backend_ptr backend { ggml_backend_dev_init(dev, nullptr) };
    if (!backend) {
        return false;
    }
    const int n_decode = 8;
    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * (n_decode + 4) + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx { ggml_init(params) };
    // big enough to stream from memory rather than a cache (~32 MiB), small enough to upload quickly
    const int64_t rows_32m = (int64_t) ((32u << 20) / ggml_row_size(wtype, n_embd));
    const int64_t n_rows   = std::min<int64_t>(3 * n_ff, std::max<int64_t>(n_ff / 8, rows_32m));
    ggml_tensor * w  = ggml_new_tensor_2d(ctx.get(), wtype, n_embd, n_rows);
    ggml_tensor * x1 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, 1);
    ggml_cgraph * gd = ggml_new_graph(ctx.get());
    for (int i = 0; i < n_decode; i++) {
        ggml_build_forward_expand(gd, ggml_mul_mat(ctx.get(), w, x1));
    }
    if (!ggml_backend_supports_op(backend.get(), ggml_graph_node(gd, 0))) {
        return false;
    }
    ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()) };
    if (!buf) {
        return false;
    }
    fill_weight(w);
    fill_f32(x1);
    // best of a few spaced tries: a busy moment on the node must not look like different hardware
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 3; i++) {
        if (i > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }
        best = std::min(best, time_graph(backend.get(), gd, 7));
    }
    s_byte = best / (double) (n_decode * ggml_nbytes(w));
    return true;
}

static bool perf_valid(const common_split_device_perf & p) {
    auto pos = [](double v) { return std::isfinite(v) && v > 0.0; };
    auto nonneg = [](double v) { return std::isfinite(v) && v >= 0.0; };
    return pos(p.s_decode_byte) && pos(p.s_prefill_flop) && nonneg(p.t_hop) && nonneg(p.s_hop_byte) && nonneg(p.s_output_byte);
}

// the widest weight of the timed layers, used for the quick re-check of cached entries
static bool check_shape(const common_split_calib_model & m, int64_t & n_embd, int64_t & n_rows, ggml_type & type) {
    size_t best = 0;
    for (const auto & l : m.layers) {
        for (const auto & w : l.weights) {
            const size_t b = ggml_row_size(w.type, w.ne[0]) * w.ne[1];
            if (b > best) {
                best   = b;
                n_embd = w.ne[0];
                n_rows = w.ne[1];
                type   = w.type;
            }
        }
    }
    return best > 0;
}

// weight types that can't be quantized without an importance matrix are timed as a similar type
static common_split_calib_model timing_types(common_split_calib_model m) {
    for (auto & l : m.layers) {
        for (auto & w : l.weights) {
            if (ggml_quantize_requires_imatrix(w.type)) {
                w.type = GGML_TYPE_Q4_K;
            }
        }
    }
    return m;
}

std::vector<common_split_device_perf> common_split_balance_calibrate(
        const std::vector<ggml_backend_dev_t> & devs,
        const common_split_calib_model & model_in,
        const common_split_workload & wl,
        const std::string & cache_path, bool force) {
    const common_split_calib_model model = timing_types(model_in);
    int64_t   n_embd = 0;
    int64_t   n_ff   = 0;
    ggml_type wtype  = GGML_TYPE_COUNT;
    if (model.layers.empty() || !check_shape(model, n_embd, n_ff, wtype)) {
        return {};
    }
    const int64_t n_embd_act = model.n_embd > 0 ? model.n_embd : n_embd;

    auto read_cache = [&]() {
        common_json c = common_json::object();
        if (!cache_path.empty()) {
            std::ifstream f(cache_path);
            if (f) {
                std::stringstream ss;
                ss << f.rdbuf();
                c = common_json::parse_no_throw(ss.str());
                if (!c.is_object()) {
                    c = common_json::object();
                }
            }
        }
        return c;
    };
    // read even when forced: force only skips the entries of the devices calibrated now, the file keeps the rest
    common_json cache = read_cache();
    common_json written = common_json::object(); // this run's entries, merged into the file as it is at write time

    // measurements older than this are taken again (driver, firmware or worker updates, thermal changes)
    constexpr int64_t max_age_s = 7 * 24 * 3600;
    const int64_t now_s = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    // An entry's perf, if it is valid and fresh. Every entry is re-checked with a quick timing (s_now, the
    // same memory-bound matmul as its s_check_byte): an RPC key cannot see the hardware behind the endpoint,
    // and the entry may have been measured while something else loaded the device.
    auto read_entry = [&](const common_json & e, common_split_device_perf & p, double & s_check) {
        const int64_t t_meas = e.value("measured_at", (int64_t) 0);
        const int64_t ttl    = e.value("ttl", max_age_s);
        if (now_s - t_meas > ttl || t_meas > now_s) {
            return false;
        }
        p.s_decode_byte  = e.at("s_decode_byte").get<double>();
        p.s_prefill_flop = e.at("s_prefill_flop").get<double>();
        p.t_hop          = e.at("t_hop").get<double>();
        p.s_hop_byte     = e.at("s_hop_byte").get<double>();
        p.s_output_byte  = e.value("s_output_byte", 0.0);
        s_check          = e.at("s_check_byte").get<double>();
        // both time memory-bound matmuls, so they agree within a factor; an entry where they are far apart
        // was not written by one calibration run. Wide: a mixed-type layer vs one matmul type can differ by
        // 5x on CPU workers (MXFP4).
        const double self_ratio = p.s_decode_byte / s_check;
        return perf_valid(p) && s_check > 0.0 && self_ratio <= 16.0 && self_ratio >= 1.0 / 16.0;
    };
    auto entry_json = [&](const common_split_device_perf & p, double s_check, int64_t ttl) {
        return common_json::object({
            {"s_check_byte",   s_check},
            {"s_decode_byte",  p.s_decode_byte},
            {"s_prefill_flop", p.s_prefill_flop},
            {"t_hop",          p.t_hop},
            {"s_hop_byte",     p.s_hop_byte},
            {"s_output_byte",  p.s_output_byte},
            {"measured_at",    now_s},
            {"ttl",            ttl},
        });
    };
    // A node that is busy for a whole calibration measures slow in every round. The reference entry (per
    // device and model, not per build) keeps its last calibration that did not look busy: a new one clearly
    // slower than it is taken as busy and only kept for an hour, and after a build change a device whose
    // quick check still matches its reference reuses it instead of calibrating again.
    constexpr double  busy_ratio = 1.25;
    constexpr int64_t busy_ttl_s = 3600;

    std::vector<common_split_device_perf> perf(devs.size());
    bool dirty = false;
    for (size_t i = 0; i < devs.size(); i++) {
        const std::string key     = cache_key(devs[i], model, wl);
        const std::string ref_key = "ref|" + cache_key(devs[i], model, wl, false);
        const char * name = ggml_backend_dev_name(devs[i]);

        common_split_device_perf ref;
        double ref_check = 0.0;
        bool   has_ref   = false;
        if (cache.contains(ref_key)) {
            try {
                has_ref = read_entry(cache.at(ref_key), ref, ref_check);
            } catch (const std::exception &) {
                has_ref = false;
            }
            if (!has_ref) {
                LOG_INF("%s: the reference calibration of %s is expired or unusable, not using it\n", __func__, name);
            }
        }

        double s_now = 0.0;
        bool   have_now = false;
        auto check_now = [&]() {
            if (!have_now) {
                have_now = quick_check(devs[i], n_embd, n_ff, wtype, s_now);
            }
            return have_now;
        };

        if (!force && cache.contains(key)) {
            try {
                double s_check = 0.0;
                if (!read_entry(cache.at(key), perf[i], s_check)) {
                    throw std::runtime_error("expired or unusable");
                }
                if (!check_now()) {
                    throw std::runtime_error("the quick check failed");
                }
                const double ratio = s_now / s_check;
                // slower: 1.5x is wide enough for noise and a moment of load, a different worker is typically
                // off by 2x or more; faster: the entry was measured while the device was busy
                if (ratio > 1.5 || ratio < 1.0 / busy_ratio) {
                    LOG_INF("%s: %s runs at %.2fx its cached speed, calibrating again\n", __func__, name, 1.0 / ratio);
                    throw std::runtime_error("");
                }
                continue;
            } catch (const std::exception & e) {
                // measure again
                if (e.what()[0] != '\0') {
                    LOG_INF("%s: cached calibration of %s: %s, calibrating again\n", __func__, name, e.what());
                }
            }
        }

        if (has_ref && !force && check_now()) {
            const double ratio = s_now / ref_check;
            if (ratio <= 1.1 && ratio >= 1.0 / 1.1) {
                LOG_INF("%s: %s matches its reference calibration on the quick check, reusing it\n", __func__, name);
                perf[i] = ref;
                written[key] = entry_json(ref, ref_check, max_age_s);
                dirty = true;
                continue;
            }
        }
        const int64_t t0 = ggml_time_us();
        if (!calibrate_device(devs[i], model, wl, n_embd_act, perf[i])) {
            LOG_WRN("%s: could not calibrate %s, not balancing by speed\n", __func__, ggml_backend_dev_name(devs[i]));
            return {};
        }
        if (!perf_valid(perf[i])) {
            LOG_WRN("%s: calibration of %s gave invalid timings, not balancing by speed\n", __func__, ggml_backend_dev_name(devs[i]));
            return {};
        }
        double s_check = 0.0;
        if (!quick_check(devs[i], n_embd, n_ff, wtype, s_check)) {
            s_check = 0.0;
        }
        LOG_INF("%s: calibrated %s in %.1f s\n", __func__, name, (ggml_time_us() - t0) * 1e-6);
        int64_t ttl = max_age_s;
        if (has_ref) {
            const double slow = std::max(perf[i].s_decode_byte / ref.s_decode_byte, perf[i].s_prefill_flop / ref.s_prefill_flop);
            if (slow > busy_ratio) {
                LOG_WRN("%s: %s measured %.2fx slower than its reference calibration, the node is probably busy; "
                        "using this measurement for now and measuring again on the next start after an hour\n",
                        __func__, name, slow);
                ttl = busy_ttl_s;
            }
        }
        written[key] = entry_json(perf[i], s_check, ttl);
        if (ttl == max_age_s && s_check > 0.0) {
            written[ref_key] = entry_json(perf[i], s_check, max_age_s);
        }
        dirty = true;
    }

    if (dirty && !cache_path.empty()) {
        // another server on this node may have written the file since it was read: merge into its current state
        cache = read_cache();
        for (auto it = written.begin(); it != written.end(); ++it) {
            cache[it.key()] = it.value();
        }
        // drop entries of other builds, expired ones and malformed ones, so the file does not grow with every
        // build and a bad entry can't break every later start
        const std::string build = std::string("|") + llama_commit();
        std::vector<std::string> drop;
        for (auto it = cache.begin(); it != cache.end(); ++it) {
            const std::string & k = it.key();
            const bool is_ref     = k.rfind("ref|", 0) == 0;
            const bool same_build = k.size() >= build.size() && k.compare(k.size() - build.size(), build.size(), build) == 0;
            int64_t t_meas = 0;
            int64_t ttl    = max_age_s;
            bool    valid  = it.value().is_object();
            if (valid) {
                try {
                    t_meas = it.value().value("measured_at", (int64_t) 0);
                    ttl    = it.value().value("ttl", max_age_s);
                } catch (const std::exception &) {
                    valid = false;
                }
            }
            if (!valid || (!same_build && !is_ref) || now_s - t_meas > ttl) {
                drop.push_back(k);
            }
        }
        for (const auto & k : drop) {
            cache.erase(k);
        }
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(cache_path).parent_path(), ec);
        // per process: two servers on one node (sharing a cache dir) must not write the same temp file
        const std::string tmp = cache_path + ".tmp." + std::to_string(std::random_device{}());
        std::ofstream f(tmp);
        bool ok = false;
        if (f) {
            f << cache.dump(2);
            f.close();
            ok = !f.fail();
            if (ok) {
                std::filesystem::rename(tmp, cache_path, ec);
                ok = !ec;
            }
        }
        if (!ok) {
            LOG_WRN("%s: could not write the calibration cache %s\n", __func__, cache_path.c_str());
        }
    }
    return perf;
}

//
// applying it at model load
//

struct model_layer_info {
    common_split_model_shape shape;
    int64_t   n_embd = 0;
    int64_t   n_ff   = 0;
    uint32_t  n_expert      = 0; // MoE: n_ff is the expert FFN size
    uint32_t  n_expert_used = 0;
    ggml_type wtype  = GGML_TYPE_COUNT;
    bool      overridden = false; // a tensor buffer override matches a tensor of this model
    common_split_calib_model calib;
};

// layer sizes, compute and the dominant FFN matmul shape from the GGUF metadata
static bool read_model_layers(const char * path, uint32_t n_layer, const llama_model_tensor_buft_override * overrides,
                              model_layer_info & info) {
    // same matching as the model loader; -ncmoe on a dense model adds patterns that match nothing
    std::vector<std::regex> ov;
    for (const auto * o = overrides; o != nullptr && o->pattern != nullptr; o++) {
        ov.emplace_back(o->pattern);
    }

    ggml_context * meta = nullptr;
    gguf_init_params gp = { /* .no_alloc = */ true, /* .ctx = */ &meta };
    gguf_context * gctx = gguf_init_from_file(path, gp);
    if (gctx == nullptr) {
        return false;
    }
    ggml_context_ptr meta_ptr { meta };
    gguf_context_ptr gguf_ptr { gctx }; // the layer-kind pass below still reads per-layer keys

    std::string arch;
    {
        const int64_t kid = gguf_find_key(gctx, "general.architecture");
        if (kid >= 0) {
            arch = gguf_get_val_str(gctx, kid);
        }
    }
    auto get_u32 = [&](const std::string & key, uint32_t def) {
        const int64_t kid = gguf_find_key(gctx, (arch + "." + key).c_str());
        if (kid < 0) {
            return def;
        }
        switch (gguf_get_kv_type(gctx, kid)) {
            case GGUF_TYPE_UINT32: return gguf_get_val_u32(gctx, kid);
            case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(gctx, kid);
            default:               return def;
        }
    };
    // a per-layer value: either one number for all layers or an array with one per layer
    auto get_u32_layer = [&](const std::string & key, uint32_t il, uint32_t def) {
        const int64_t kid = gguf_find_key(gctx, (arch + "." + key).c_str());
        if (kid < 0) {
            return def;
        }
        if (gguf_get_kv_type(gctx, kid) == GGUF_TYPE_ARRAY) {
            if (il >= gguf_get_arr_n(gctx, kid)) {
                return def;
            }
            const void * data = gguf_get_arr_data(gctx, kid);
            switch (gguf_get_arr_type(gctx, kid)) {
                case GGUF_TYPE_UINT32: return ((const uint32_t *) data)[il];
                case GGUF_TYPE_INT32:  return (uint32_t) ((const int32_t *) data)[il];
                default:               return def;
            }
        }
        return get_u32(key, def);
    };
    const uint32_t n_expert      = get_u32("expert_count", 0);
    const uint32_t n_expert_used = get_u32("expert_used_count", 0);
    // a token only reads (and multiplies with) the experts it is routed to
    const double expert_frac = n_expert > 0 && n_expert_used > 0 ? (double) n_expert_used / n_expert : 1.0;

    double layer_bytes = 0.0;
    double layer_flops = 0.0;
    std::map<ggml_type, double> type_bytes;
    double output_bytes = 0.0;
    double tok_embd_bytes = 0.0;
    bool   has_output = false;

    int64_t dense_embd = 0;
    int64_t dense_ff   = 0;

    // per layer: the weight matrices (name after "blk.N.", shape, type) and whether it has a recurrent block
    struct layer_weights {
        std::vector<std::pair<std::string, common_split_calib_weight>> w;
        bool attn = false;
        bool ssm  = false;
        bool gdn  = false; // gated delta net gating tensors
        int64_t conv_k  = 0;
        int64_t conv_ch = 0;
    };
    std::vector<layer_weights> per_layer(n_layer);

    const int64_t n_tensors = gguf_get_n_tensors(gctx);
    for (int64_t i = 0; i < n_tensors; i++) {
        const std::string name = gguf_get_tensor_name(gctx, i);
        const ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
        if (t == nullptr) {
            continue;
        }
        for (const auto & re : ov) {
            if (!info.overridden && std::regex_search(name, re)) {
                info.overridden = true;
            }
        }
        const bool   is_expert = name.find("_exps") != std::string::npos;
        const double frac      = is_expert ? expert_frac : 1.0;
        if (name.rfind("blk.", 0) == 0) {
            int il = -1;
            if (sscanf(name.c_str(), "blk.%d.", &il) != 1 || il < 0 || (uint32_t) il >= n_layer) {
                continue; // e.g. nextn/MTP layers that are not part of the repeating split
            }
            {
                const std::string suffix = name.substr(name.find('.', 4) + 1);
                auto & lw = per_layer[il];
                if (suffix.rfind("ssm_", 0) == 0) {
                    lw.ssm = true;
                }
                if (suffix.rfind("ssm_beta.", 0) == 0 || suffix.rfind("ssm_alpha.", 0) == 0 || suffix.rfind("ssm_ba.", 0) == 0) {
                    lw.gdn = true;
                }
                if (suffix.rfind("ssm_conv1d.", 0) == 0) {
                    lw.conv_k  = t->ne[0];
                    lw.conv_ch = t->ne[1];
                }
                if (suffix.rfind("attn_q.", 0) == 0 || suffix.rfind("attn_k.", 0) == 0 || suffix.rfind("attn_qkv.", 0) == 0) {
                    lw.attn = true;
                }
                // matrices a token is multiplied with; conv kernels and other small ones are left to the overhead
                if (t->ne[1] > 1 && t->ne[0] >= 64 && ggml_n_dims(t) <= 3) {
                    common_split_calib_weight cw;
                    cw.ne[0] = t->ne[0];
                    cw.ne[1] = t->ne[1];
                    cw.ne[2] = t->ne[2];
                    cw.type  = t->type;
                    lw.w.emplace_back(suffix, cw);
                }
            }
            layer_bytes += ggml_nbytes(t) * frac;
            if (t->ne[1] > 1) {
                layer_flops += 2.0 * ggml_nelements(t) * frac;
            }
            if (t->ne[1] > 1) {
                type_bytes[t->type] += ggml_nbytes(t) * frac;
            }
            // the matmul shape to time: the experts' for a MoE model (they hold most of its weights)
            const bool up_exps = name.find(".ffn_up_exps.weight") != std::string::npos ||
                                 name.find(".ffn_gate_up_exps.weight") != std::string::npos;
            const bool up      = name.find(".ffn_up.weight") != std::string::npos;
            if (up && dense_ff == 0) {
                dense_embd = t->ne[0];
                dense_ff   = t->ne[1];
            }
            if (up_exps && n_expert > 0 && info.n_expert == 0) {
                info.n_embd        = t->ne[0];
                info.n_ff          = t->ne[1];
                info.n_expert      = n_expert;
                info.n_expert_used = n_expert_used;
            }
        } else if (name == "output.weight" || name == "token_embd.weight") {
            common_split_calib_weight cw;
            cw.ne[0] = t->ne[0];
            cw.ne[1] = t->ne[1];
            cw.type  = t->type;
            if (name == "output.weight") {
                output_bytes = ggml_nbytes(t);
                has_output = true;
                info.calib.output = cw;
            } else {
                tok_embd_bytes = ggml_nbytes(t);
                if (!has_output) {
                    info.calib.output = cw; // tied embeddings, replaced if an output.weight follows
                }
            }
        }
    }

    if (info.n_ff == 0 && dense_ff > 0) {
        info.n_embd = dense_embd;
        info.n_ff   = dense_ff;
    }

    // time the type that holds most of the weight bytes (mixed-quant files differ per tensor)
    double best_bytes = 0.0;
    for (const auto & [type, bytes] : type_bytes) {
        if (bytes > best_bytes) {
            best_bytes = bytes;
            info.wtype = type;
        }
    }

    if (info.wtype == GGML_TYPE_COUNT || info.n_ff == 0 || n_layer == 0) {
        return false;
    }
    info.shape.n_layer      = n_layer;
    info.shape.layer_bytes  = layer_bytes / n_layer;
    info.shape.layer_flops  = layer_flops / n_layer;
    info.shape.act_bytes    = (double) info.n_embd * sizeof(float);
    info.shape.output_bytes = has_output ? output_bytes : tok_embd_bytes; // tied embeddings

    // the layer kinds to time: layers with the same weights (names, shapes) and attention shape are one kind.
    // Mixed quants (e.g. Unsloth UD) vary the types per layer -- 53 type layouts over Qwen3.8's 3 kinds of
    // layer, 100 s of calibration on a 780M -- so each kind is timed on its most common type layout and its
    // decode time scaled by the kind's mean weight bytes.
    {
        const uint32_t n_embd_model = get_u32("embedding_length", (uint32_t) info.n_embd);
        struct kind_acc {
            common_split_calib_layer l;
            std::map<std::string, std::pair<int, std::vector<common_split_calib_weight>>> layouts; // types -> count, weights
            double bytes = 0.0;
            int    n     = 0;
        };
        std::vector<kind_acc> acc;
        std::map<std::string, size_t> kinds;
        auto & cm = info.calib;
        auto w_bytes = [](const std::vector<common_split_calib_weight> & ws) {
            double b = 0.0;
            for (const auto & w : ws) {
                b += (double) ggml_row_size(w.type, w.ne[0]) * w.ne[1] * w.ne[2];
            }
            return b;
        };
        for (uint32_t il = 0; il < n_layer; il++) {
            auto & lw = per_layer[il];
            std::sort(lw.w.begin(), lw.w.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
            common_split_calib_layer l;
            const uint32_t n_head    = get_u32_layer("attention.head_count",    il, 0);
            const uint32_t n_head_kv = get_u32_layer("attention.head_count_kv", il, n_head);
            if (lw.attn && !lw.ssm && n_head > 0 && n_head_kv > 0) {
                l.n_head     = n_head;
                l.n_head_kv  = n_head_kv;
                l.head_dim_k = get_u32("attention.key_length",   n_embd_model / n_head);
                l.head_dim_v = get_u32("attention.value_length", l.head_dim_k);
            }
            if (lw.gdn && lw.conv_k > 1) {
                l.gdn_state = get_u32("ssm.state_size",    0);
                l.gdn_h_k   = get_u32("ssm.group_count",   0);
                l.gdn_h_v   = get_u32("ssm.time_step_rank", 0);
                l.conv_k    = (uint32_t) lw.conv_k;
                l.conv_ch   = (uint32_t) lw.conv_ch;
                if (l.gdn_state == 0 || l.gdn_h_k == 0 || l.gdn_h_v == 0 || l.gdn_h_v % l.gdn_h_k != 0) {
                    l.gdn_state = l.gdn_h_k = l.gdn_h_v = l.conv_k = l.conv_ch = 0;
                }
            }
            std::ostringstream sig;
            sig << l.n_head << '/' << l.n_head_kv << '/' << l.head_dim_k << '/' << l.head_dim_v << "|gdn" << l.gdn_state
                << '/' << l.gdn_h_k << '/' << l.gdn_h_v << '/' << l.conv_k << '/' << l.conv_ch;
            std::ostringstream types;
            std::vector<common_split_calib_weight> ws;
            for (const auto & [suffix, w] : lw.w) {
                sig << ';' << suffix << ':' << w.ne[0] << 'x' << w.ne[1] << 'x' << w.ne[2];
                types << ggml_type_name(w.type) << ',';
                ws.push_back(w);
            }
            auto it = kinds.find(sig.str());
            if (it == kinds.end()) {
                it = kinds.emplace(sig.str(), acc.size()).first;
                acc.push_back({ l, {}, 0.0, 0 });
            }
            auto & a = acc[it->second];
            auto & layout = a.layouts[types.str()];
            layout.first++;
            layout.second = ws;
            a.bytes += w_bytes(ws);
            a.n++;
        }
        for (auto & a : acc) {
            const std::vector<common_split_calib_weight> * rep = nullptr;
            int rep_n = 0;
            for (const auto & [types, cw] : a.layouts) {
                if (cw.first > rep_n) {
                    rep_n = cw.first;
                    rep   = &cw.second;
                }
            }
            a.l.weights    = *rep;
            a.l.share      = (double) a.n / n_layer;
            const double b = w_bytes(a.l.weights);
            a.l.byte_scale = b > 0.0 ? (a.bytes / a.n) / b : 1.0;
            cm.layers.push_back(std::move(a.l));
        }
        cm.n_expert      = n_expert;
        cm.n_expert_used = n_expert_used;
        cm.n_embd        = n_embd_model > 0 ? n_embd_model : info.n_embd;
        cm.layer_bytes   = info.shape.layer_bytes;
        cm.layer_flops   = info.shape.layer_flops;
    }
    return true;
}

// layer counts -> tensor_split in the form common_fit_params uses: the output layer counts
// as one extra layer on the last device
static void set_split(float * tensor_split, const std::vector<uint32_t> & n) {
    for (size_t i = 0; i < n.size(); i++) {
        tensor_split[i] = (float) n[i] + (i + 1 == n.size() ? 1.0f : 0.0f);
    }
}

void common_split_balance_apply(
        common_params & params,
        llama_model_params & mparams,
        const llama_context_params & cparams,
        const common_fit_extra_model * extra) {
    const auto mode = (common_split_balance_mode) params.split_balance;
    if (mode == COMMON_SPLIT_BALANCE_MEMORY) {
        return;
    }
    if (mparams.split_mode != LLAMA_SPLIT_MODE_LAYER) {
        return;
    }

    const int64_t t_start = ggml_time_us();
    const ggml_log_level lvl = GGML_LOG_LEVEL_ERROR;

    // probe: an even split, measured with the model's real buffers
    std::vector<ggml_backend_dev_t> devs;
    uint32_t n_layer = 0;
    float probe_split[128] = { 0 };
    llama_model_params mp = mparams;
    common_device_memory_data_vec mem;
    try {
        mem = common_get_device_memory_data_with_extra(params.model.path.c_str(), &mp, &cparams, extra, devs, n_layer, lvl);
    } catch (const std::exception & e) {
        LOG_WRN("%s: could not measure device memory (%s), keeping the memory split\n", __func__, e.what());
        return;
    }
    const size_t nd = devs.size();
    if (nd < 2) {
        return;
    }
    if (mparams.n_gpu_layers >= 0 && (uint32_t) mparams.n_gpu_layers < n_layer + 1) {
        LOG_INF("%s: model is only partially offloaded, keeping the memory split\n", __func__);
        return;
    }

    model_layer_info info;
    if (!read_model_layers(params.model.path.c_str(), n_layer, mparams.tensor_buft_overrides, info)) {
        LOG_WRN("%s: could not read the layer shapes, keeping the memory split\n", __func__);
        return;
    }
    {
        int32_t nodes_pp = -1;
        int32_t nodes_tg = -1;
        common_fit_last_graph_nodes(nodes_pp, nodes_tg);
        if (nodes_pp > 0 && nodes_tg > 0 && n_layer > 0) {
            info.calib.nodes_pp_layer = (double) nodes_pp / n_layer;
            info.calib.nodes_tg_layer = (double) nodes_tg / n_layer;
        }
    }
    if (info.overridden) {
        LOG_INF("%s: tensor buffer overrides move weights of this model off the split, keeping the memory split\n", __func__);
        return;
    }

    std::vector<uint32_t> probe(nd, n_layer / nd);
    probe.back() += n_layer - (n_layer / nd) * nd;
    set_split(probe_split, probe);
    mp.tensor_split = probe_split;
    try {
        mem = common_get_device_memory_data_with_extra(params.model.path.c_str(), &mp, &cparams, extra, devs, n_layer, lvl);
    } catch (const std::exception & e) {
        LOG_WRN("%s: could not measure device memory (%s), keeping the memory split\n", __func__, e.what());
        return;
    }

    // linear memory model per device: fixed (compute buffers, output layer) + per layer (weights, KV/state)
    std::vector<double> fixed(nd), per_layer(nd), target(nd);
    for (size_t i = 0; i < nd; i++) {
        const bool   last  = i + 1 == nd;
        const double out   = last ? info.shape.output_bytes : 0.0;
        const double layer = std::max(0.0, (double) mem[i].model + mem[i].context - out);
        per_layer[i] = probe[i] > 0 ? layer / probe[i] : info.shape.layer_bytes;
        fixed[i]     = (double) mem[i].compute + out;
        target[i]    = (double) mem[i].free - (double) params.fit_params_target[i];
    }
    auto caps_from_targets = [&]() {
        std::vector<uint32_t> caps(nd);
        for (size_t i = 0; i < nd; i++) {
            const double room = target[i] - fixed[i];
            caps[i] = room <= 0.0 ? 0 : (uint32_t) std::min<double>(n_layer, std::floor(room / std::max(1.0, per_layer[i])));
        }
        return caps;
    };

    const std::string cache_path = fs_get_cache_directory() + "split-balance.json";
    common_split_workload wl;
    wl.n_prompt = params.split_workload_prompt;
    wl.n_gen    = params.split_workload_gen;
    wl.n_ubatch = cparams.n_ubatch;
    wl.n_batch  = cparams.n_batch;

    const auto perf = common_split_balance_calibrate(devs, info.calib, wl, cache_path, params.split_calibrate_force);
    if (perf.size() != nd) {
        return;
    }

    // optimize, then check the choice against real buffer sizes; shrink the cap of any device
    // that would overflow and try again
    std::vector<uint32_t> n;
    for (int attempt = 0; attempt < 5; attempt++) {
        const auto caps = caps_from_targets();
        n = common_split_balance_optimize(mode, info.shape, perf, caps, wl);
        if (n.empty()) {
            LOG_WRN("%s: no split fits the free memory of the devices, keeping the memory split\n", __func__);
            return;
        }
        set_split(probe_split, n);
        try {
            mem = common_get_device_memory_data_with_extra(params.model.path.c_str(), &mp, &cparams, extra, devs, n_layer, lvl);
        } catch (const std::exception & e) {
            LOG_WRN("%s: could not measure device memory (%s), keeping the memory split\n", __func__, e.what());
            return;
        }
        bool fits = true;
        for (size_t i = 0; i < nd; i++) {
            const double used = (double) mem[i].model + mem[i].context + mem[i].compute;
            if (used > target[i]) {
                fits = false;
                // fewer layers here next time: pretend the device is smaller by the overflow
                target[i] -= used - target[i] + per_layer[i];
            }
        }
        if (fits) {
            break;
        }
        if (attempt == 4) {
            LOG_WRN("%s: could not find a split that fits, keeping the memory split\n", __func__);
            return;
        }
    }

    // report: what was measured, what was chosen, and what the memory split would have done
    std::vector<uint32_t> n_mem(nd, 0);
    {
        double free_sum = 0.0;
        for (size_t i = 0; i < nd; i++) {
            free_sum += (double) mem[i].free;
        }
        uint32_t assigned = 0;
        for (size_t i = 0; i + 1 < nd; i++) {
            n_mem[i] = (uint32_t) std::lround(n_layer * (double) mem[i].free / free_sum);
            assigned += n_mem[i];
        }
        n_mem.back() = n_layer > assigned ? n_layer - assigned : 0;
    }
    LOG_INF("%s: device calibration (%s):\n", __func__, ggml_type_name(info.wtype));
    for (size_t i = 0; i < nd; i++) {
        LOG_INF("%s:   %-24s decode %7.1f GB/s  prefill %7.2f TFLOPS  hop %6.3f ms  layers %3u (memory split: %u)\n", __func__,
                ggml_backend_dev_name(devs[i]), 1e-9 / perf[i].s_decode_byte, 1e-12 / perf[i].s_prefill_flop,
                perf[i].t_hop * 1e3, n[i], n_mem[i]);
    }
    auto tps = [&](const std::vector<uint32_t> & x, double & pp, double & tg) {
        tg = 1.0 / common_split_predict_decode(info.shape, perf, x);
        pp = wl.n_prompt / common_split_predict_prefill(info.shape, perf, x, wl);
    };
    double pp_new, tg_new, pp_mem, tg_mem;
    tps(n, pp_new, tg_new);
    tps(n_mem, pp_mem, tg_mem);
    LOG_INF("%s: --split-balance %s: predicted pp%u %.0f t/s, tg %.1f t/s (memory split: %.0f t/s, %.1f t/s), took %.1f s\n",
            __func__, common_split_balance_mode_name(mode), wl.n_prompt, pp_new, tg_new, pp_mem, tg_mem,
            (ggml_time_us() - t_start) * 1e-6);

    set_split(params.tensor_split, n);
    mparams.tensor_split = params.tensor_split;
}
