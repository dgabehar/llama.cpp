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

// Times each device with synthetic matmuls shaped like the model's layers
// (n_embd x n_ff weights of type wtype). Remote (RPC) devices are timed over
// their normal protocol, so nothing changes on the worker. Results are cached
// in cache_path (empty = no cache) keyed by device, shape and type.
std::vector<common_split_device_perf> common_split_balance_calibrate(
        const std::vector<ggml_backend_dev_t> & devs,
        int64_t n_embd, int64_t n_ff, ggml_type wtype, uint32_t n_ubatch,
        const std::string & cache_path, bool force,
        uint32_t n_expert = 0, uint32_t n_expert_used = 0); // MoE: time the routed expert matmuls

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
