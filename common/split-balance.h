#pragma once

// Speed-aware layer split across devices (--split-balance).
//
// A layer split runs the devices one after another for every token, so the
// best split depends on the workload:
//   - decode (one token at a time) only gets slower with every layer placed on
//     a slower device, so the fastest devices should be filled first
//   - prefill pipelines ubatches across the devices, so the stages should take
//     about the same time, i.e. layers proportional to device speed
// The default split (proportional to free memory) is neither, and on
// heterogeneous devices it is the slowest of the three.

#include "ggml-backend.h"
#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

enum common_split_balance_mode {
    COMMON_SPLIT_BALANCE_MEMORY  = 0, // split by free memory (llama.cpp default)
    COMMON_SPLIT_BALANCE_DECODE  = 1, // minimize per-token decode time
    COMMON_SPLIT_BALANCE_PREFILL = 2, // minimize pipelined prompt processing time
    COMMON_SPLIT_BALANCE_AUTO    = 3, // minimize predicted time of a reference request
};

// measured speed of one device, see common_split_balance_calibrate
struct common_split_device_perf {
    double s_decode_byte   = 0.0; // seconds per weight byte in a single-token matmul (memory bound)
    double s_prefill_flop  = 0.0; // seconds per flop in an n_ubatch-token matmul (compute bound)
    double t_hop           = 0.0; // fixed seconds per hand-off into/out of this device (0 = local)
    double s_hop_byte      = 0.0; // seconds per activation byte moved into/out of this device
    double s_output_byte   = 0.0; // seconds per output-layer weight byte per token (0 = s_decode_byte)
};

// what the cost model needs to know about the model
struct common_split_model_shape {
    uint32_t n_layer       = 0;   // repeating layers
    double   layer_bytes   = 0.0; // mean weight bytes of one repeating layer
    double   layer_flops   = 0.0; // mean flops per token of one repeating layer
    double   act_bytes     = 0.0; // bytes per token crossing a device boundary (n_embd floats)
    double   output_bytes  = 0.0; // weight bytes of the output layer (always on the last device)
};

struct common_split_workload {
    uint32_t n_prompt = 4096;
    uint32_t n_gen    = 256;
    uint32_t n_ubatch = 512;
    uint32_t n_batch  = 2048; // the pipeline drains at the end of every llama_decode() call
};

// predicted seconds per decoded token / for processing n_prompt tokens, for
// n_layers[i] repeating layers on device i (the output layer is always on the
// last device)
double common_split_predict_decode(
        const common_split_model_shape & shape,
        const std::vector<common_split_device_perf> & perf,
        const std::vector<uint32_t> & n_layers);

double common_split_predict_prefill(
        const common_split_model_shape & shape,
        const std::vector<common_split_device_perf> & perf,
        const std::vector<uint32_t> & n_layers,
        const common_split_workload & wl);

// Best number of repeating layers per device for the mode, with at most
// max_layers[i] on device i. Returns an empty vector for
// COMMON_SPLIT_BALANCE_MEMORY or when the caps can't hold the model.
std::vector<uint32_t> common_split_balance_optimize(
        common_split_balance_mode mode,
        const common_split_model_shape & shape,
        const std::vector<common_split_device_perf> & perf,
        const std::vector<uint32_t> & max_layers,
        const common_split_workload & wl);

// one weight matrix of a layer; ne[2] > 1 is a stack of experts (multiplied with mul_mat_id)
struct common_split_calib_weight {
    int64_t   ne[3] = { 0, 0, 1 };
    ggml_type type  = GGML_TYPE_F16;
};

// one kind of repeating layer (hybrid models have several) and the share of the layers like it
struct common_split_calib_layer {
    std::vector<common_split_calib_weight> weights;
    double   share      = 1.0;
    uint32_t n_head     = 0; // 0: no attention op in this layer
    uint32_t n_head_kv  = 0;
    uint32_t head_dim_k = 0;
    uint32_t head_dim_v = 0;
    // gated delta net (qwen35/qwen3next linear attention): conv + recurrence; 0 = none
    uint32_t gdn_state  = 0; // S: head size of the recurrent state
    uint32_t gdn_h_k    = 0;
    uint32_t gdn_h_v    = 0;
    uint32_t conv_k     = 0; // conv kernel width
    uint32_t conv_ch    = 0; // conv channels
};

// what the calibration times: the model's own layers with its real shapes, types and expert count
struct common_split_calib_model {
    std::vector<common_split_calib_layer> layers;
    uint32_t n_expert      = 0;
    uint32_t n_expert_used = 0;
    int64_t  n_embd        = 0;   // width of the activations handed between devices
    common_split_calib_weight output; // the output layer (token_embd when tied); ne[1] == 0: none
    // compute nodes per repeating layer in the model's real graphs (0 = unknown): the calibration times the
    // heavy ops of a layer, the rest (norms, RoPE, elementwise, copies) is charged per node
    double nodes_tg_layer = 0.0;
    double nodes_pp_layer = 0.0;
    double   layer_bytes   = 0.0; // as in common_split_model_shape, to turn layer times into rates
    double   layer_flops   = 0.0;
};

// Times each device on one layer of every kind the model has (real weight shapes and types,
// flash attention at the workload's context), weighted by how many layers of each kind there
// are, and expresses the result as the per-byte / per-flop rates the cost model uses. Remote
// (RPC) devices are timed over their normal protocol, so nothing changes on the worker. Each
// timing is repeated and the fastest kept, since other load on a node only slows it down.
// Results are cached in cache_path (empty = no cache) keyed by device, model layers and build.
std::vector<common_split_device_perf> common_split_balance_calibrate(
        const std::vector<ggml_backend_dev_t> & devs,
        const common_split_calib_model & model,
        const common_split_workload & wl,
        const std::string & cache_path, bool force);

const char * common_split_balance_mode_name(common_split_balance_mode mode);

struct common_params;
struct common_fit_extra_model;

// Chooses the layer split for params.split_balance and writes it to params.tensor_split and
// mparams (in the same form common_fit_params uses). Does nothing for the memory mode, a split
// given by the user, a single device, or a partial offload.
void common_split_balance_apply(
        common_params & params,
        llama_model_params & mparams,
        const llama_context_params & cparams,
        const common_fit_extra_model * extra);
