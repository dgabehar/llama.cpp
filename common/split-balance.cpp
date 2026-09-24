#include "split-balance.h"

#include "common.h"
#include "fit.h"
#include "json.h"
#include "log.h"
#include "gguf.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <sstream>

const char * common_split_balance_mode_name(common_split_balance_mode mode) {
    switch (mode) {
        case COMMON_SPLIT_BALANCE_MEMORY:  return "memory";
        case COMMON_SPLIT_BALANCE_DECODE:  return "decode";
        case COMMON_SPLIT_BALANCE_PREFILL: return "prefill";
        case COMMON_SPLIT_BALANCE_AUTO:    return "auto";
    }
    return "?";
}

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
        double bytes = n_layers[i] * shape.layer_bytes;
        if ((int) i == last) {
            bytes += shape.output_bytes;
        }
        // a remote stage costs a round trip per token: activations in, activations out
        t += bytes * p.s_decode_byte + 2.0 * (p.t_hop + shape.act_bytes * p.s_hop_byte);
    }
    return t;
}

double common_split_predict_prefill(
        const common_split_model_shape & shape,
        const std::vector<common_split_device_perf> & perf,
        const std::vector<uint32_t> & n_layers,
        const common_split_workload & wl) {
    const uint32_t ub = std::max<uint32_t>(1, std::min(wl.n_ubatch, std::max<uint32_t>(1, wl.n_prompt)));
    const uint32_t n_ub = (std::max<uint32_t>(1, wl.n_prompt) + ub - 1) / ub;

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
        const double t = n_layers[i] * shape.layer_flops * ub * p.s_prefill_flop
                       + 2.0 * (p.t_hop + ub * shape.act_bytes * p.s_hop_byte);
        t_stage_max = std::max(t_stage_max, t);
        t_stage_sum += t;
        n_stages++;
    }
    if (n_stages == 0) {
        return 0.0;
    }
    // with a single ubatch nothing overlaps; the fill term covers that exactly
    return (n_ub - 1) * t_stage_max + t_stage_sum;
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

static std::string cache_key(ggml_backend_dev_t dev, int64_t n_embd, int64_t n_ff, ggml_type wtype, uint32_t n_ubatch) {
    size_t free = 0;
    size_t total = 0;
    ggml_backend_dev_memory(dev, &free, &total);
    std::ostringstream ss;
    ss << ggml_backend_dev_name(dev) << "|" << ggml_backend_dev_description(dev) << "|" << (total >> 20) << "MiB|"
       << n_embd << "x" << n_ff << "|" << ggml_type_name(wtype) << "|ub" << n_ubatch;
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

static double time_graph(ggml_backend_t backend, ggml_cgraph * gf, int reps) {
    compute_and_wait(backend, gf); // warmup, compiles pipelines
    std::vector<double> t;
    for (int r = 0; r < reps; r++) {
        const int64_t t0 = ggml_time_us();
        compute_and_wait(backend, gf);
        t.push_back((ggml_time_us() - t0) * 1e-6);
    }
    std::nth_element(t.begin(), t.begin() + t.size() / 2, t.end());
    return t[t.size() / 2];
}

static bool calibrate_device(ggml_backend_dev_t dev, int64_t n_embd, int64_t n_ff, ggml_type wtype, uint32_t n_ubatch,
                             common_split_device_perf & out) {
    ggml_backend_ptr backend { ggml_backend_dev_init(dev, nullptr) };
    if (!backend) {
        return false;
    }

    // several distinct weights so the decode timing streams from memory instead of a cache
    const int n_w = 3;
    const int n_decode  = 12; // single-token matmuls per timed graph
    const int n_prefill = 3;  // n_ubatch-token matmuls per timed graph

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * (n_w + 2 + n_decode + n_prefill + 8) + 2 * ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx { ggml_init(params) };
    ggml_tensor * w[n_w];
    for (auto & wi : w) {
        wi = ggml_new_tensor_2d(ctx.get(), wtype, n_embd, n_ff);
    }
    ggml_tensor * x1 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, 1);
    ggml_tensor * xb = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, n_ubatch);

    ggml_cgraph * gd = ggml_new_graph(ctx.get());
    for (int i = 0; i < n_decode; i++) {
        ggml_build_forward_expand(gd, ggml_mul_mat(ctx.get(), w[i % n_w], x1));
    }
    ggml_cgraph * gp = ggml_new_graph(ctx.get());
    for (int i = 0; i < n_prefill; i++) {
        ggml_build_forward_expand(gp, ggml_mul_mat(ctx.get(), w[i % n_w], xb));
    }
    if (!ggml_backend_supports_op(backend.get(), ggml_graph_node(gd, 0)) ||
        !ggml_backend_supports_op(backend.get(), ggml_graph_node(gp, 0))) {
        return false;
    }

    // weights, inputs and every matmul result in one buffer; both graphs stay allocated
    ggml_backend_buffer_ptr wbuf { ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()) };
    if (!wbuf) {
        return false;
    }
    for (auto * wi : w) {
        fill_weight(wi);
    }
    fill_f32(x1);
    fill_f32(xb);

    // decode: bytes streamed per second
    const double t_decode = time_graph(backend.get(), gd, 5);
    out.s_decode_byte = t_decode / (double) (n_decode * ggml_nbytes(w[0]));

    const double t_prefill = time_graph(backend.get(), gp, 3);
    out.s_prefill_flop = t_prefill / (n_prefill * 2.0 * n_embd * n_ff * n_ubatch);

    out.t_hop      = 0.0;
    out.s_hop_byte = 0.0;
    if (dev_is_remote(dev)) {
        // round trip: small write + read back
        float v = 1.0f;
        std::vector<double> rt;
        for (int i = 0; i < 20; i++) {
            const int64_t t0 = ggml_time_us();
            ggml_backend_tensor_set(x1, &v, 0, sizeof(v));
            ggml_backend_tensor_get(x1, &v, 0, sizeof(v));
            rt.push_back((ggml_time_us() - t0) * 1e-6);
        }
        std::nth_element(rt.begin(), rt.begin() + rt.size() / 2, rt.end());
        out.t_hop = rt[rt.size() / 2] / 2.0;

        // bandwidth: one ubatch of activations
        std::vector<uint8_t> buf(ggml_nbytes(xb));
        std::vector<double> bw;
        for (int i = 0; i < 3; i++) {
            const int64_t t0 = ggml_time_us();
            ggml_backend_tensor_set(xb, buf.data(), 0, buf.size());
            bw.push_back((ggml_time_us() - t0) * 1e-6);
        }
        std::nth_element(bw.begin(), bw.begin() + bw.size() / 2, bw.end());
        out.s_hop_byte = std::max(0.0, bw[bw.size() / 2] - out.t_hop) / (double) buf.size();
    }
    return true;
}

std::vector<common_split_device_perf> common_split_balance_calibrate(
        const std::vector<ggml_backend_dev_t> & devs,
        int64_t n_embd, int64_t n_ff, ggml_type wtype, uint32_t n_ubatch,
        const std::string & cache_path, bool force) {
    if (ggml_quantize_requires_imatrix(wtype)) {
        wtype = GGML_TYPE_Q4_K; // timing only needs a similar type, not the exact one
    }

    common_json cache = common_json::object();
    if (!cache_path.empty() && !force) {
        std::ifstream f(cache_path);
        if (f) {
            std::stringstream ss;
            ss << f.rdbuf();
            cache = common_json::parse_no_throw(ss.str());
            if (!cache.is_object()) {
                cache = common_json::object();
            }
        }
    }

    std::vector<common_split_device_perf> perf(devs.size());
    bool dirty = false;
    for (size_t i = 0; i < devs.size(); i++) {
        const std::string key = cache_key(devs[i], n_embd, n_ff, wtype, n_ubatch);
        if (cache.contains(key)) {
            try {
                const common_json & e = cache.at(key);
                perf[i].s_decode_byte  = e.at("s_decode_byte").get<double>();
                perf[i].s_prefill_flop = e.at("s_prefill_flop").get<double>();
                perf[i].t_hop          = e.at("t_hop").get<double>();
                perf[i].s_hop_byte     = e.at("s_hop_byte").get<double>();
                continue;
            } catch (const std::exception &) {
                // fall through and measure again
            }
        }
        const int64_t t0 = ggml_time_us();
        if (!calibrate_device(devs[i], n_embd, n_ff, wtype, n_ubatch, perf[i])) {
            LOG_WRN("%s: could not calibrate %s, not balancing by speed\n", __func__, ggml_backend_dev_name(devs[i]));
            return {};
        }
        LOG_INF("%s: calibrated %s in %.1f s\n", __func__, ggml_backend_dev_name(devs[i]), (ggml_time_us() - t0) * 1e-6);
        cache[key] = common_json::object({
            {"s_decode_byte",  perf[i].s_decode_byte},
            {"s_prefill_flop", perf[i].s_prefill_flop},
            {"t_hop",          perf[i].t_hop},
            {"s_hop_byte",     perf[i].s_hop_byte},
        });
        dirty = true;
    }

    if (dirty && !cache_path.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(cache_path).parent_path(), ec);
        const std::string tmp = cache_path + ".tmp";
        std::ofstream f(tmp);
        if (f) {
            f << cache.dump(2);
            f.close();
            std::filesystem::rename(tmp, cache_path, ec);
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
    ggml_type wtype  = GGML_TYPE_COUNT;
};

// layer sizes, compute and the dominant FFN matmul shape from the GGUF metadata
static bool read_model_layers(const char * path, uint32_t n_layer, model_layer_info & info) {
    ggml_context * meta = nullptr;
    gguf_init_params gp = { /* .no_alloc = */ true, /* .ctx = */ &meta };
    gguf_context * gctx = gguf_init_from_file(path, gp);
    if (gctx == nullptr) {
        return false;
    }
    ggml_context_ptr meta_ptr { meta };

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
    const uint32_t n_expert      = get_u32("expert_count", 0);
    const uint32_t n_expert_used = get_u32("expert_used_count", 0);
    // a token only reads (and multiplies with) the experts it is routed to
    const double expert_frac = n_expert > 0 && n_expert_used > 0 ? (double) n_expert_used / n_expert : 1.0;

    double layer_bytes = 0.0;
    double layer_flops = 0.0;
    double output_bytes = 0.0;
    double tok_embd_bytes = 0.0;
    bool   has_output = false;

    const int64_t n_tensors = gguf_get_n_tensors(gctx);
    for (int64_t i = 0; i < n_tensors; i++) {
        const std::string name = gguf_get_tensor_name(gctx, i);
        const ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
        if (t == nullptr) {
            continue;
        }
        const bool   is_expert = name.find("_exps") != std::string::npos;
        const double frac      = is_expert ? expert_frac : 1.0;
        if (name.rfind("blk.", 0) == 0) {
            int il = -1;
            if (sscanf(name.c_str(), "blk.%d.", &il) != 1 || il < 0 || (uint32_t) il >= n_layer) {
                continue; // e.g. nextn/MTP layers that are not part of the repeating split
            }
            layer_bytes += ggml_nbytes(t) * frac;
            if (t->ne[1] > 1) {
                layer_flops += 2.0 * ggml_nelements(t) * frac;
            }
            if (info.wtype == GGML_TYPE_COUNT && (name.find(".ffn_up.weight") != std::string::npos || name.find(".ffn_up_exps.weight") != std::string::npos)) {
                info.n_embd = t->ne[0];
                info.n_ff   = t->ne[1];
                info.wtype  = t->type;
            }
        } else if (name == "output.weight") {
            output_bytes = ggml_nbytes(t);
            has_output = true;
        } else if (name == "token_embd.weight") {
            tok_embd_bytes = ggml_nbytes(t);
        }
    }
    gguf_free(gctx);

    if (info.wtype == GGML_TYPE_COUNT || n_layer == 0) {
        return false;
    }
    info.shape.n_layer      = n_layer;
    info.shape.layer_bytes  = layer_bytes / n_layer;
    info.shape.layer_flops  = layer_flops / n_layer;
    info.shape.act_bytes    = (double) info.n_embd * sizeof(float);
    info.shape.output_bytes = has_output ? output_bytes : tok_embd_bytes; // tied embeddings
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
    if (mparams.tensor_buft_overrides && mparams.tensor_buft_overrides->pattern) {
        LOG_INF("%s: tensor overrides are in use (partial offload), keeping the memory split\n", __func__);
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
    if (!read_model_layers(params.model.path.c_str(), n_layer, info)) {
        LOG_WRN("%s: could not read the layer shapes, keeping the memory split\n", __func__);
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
    const auto perf = common_split_balance_calibrate(devs, info.n_embd, info.n_ff, info.wtype,
                                                      cparams.n_ubatch, cache_path, params.split_calibrate_force);
    if (perf.size() != nd) {
        return;
    }

    common_split_workload wl;
    wl.n_prompt = params.split_workload_prompt;
    wl.n_gen    = params.split_workload_gen;
    wl.n_ubatch = cparams.n_ubatch;

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
