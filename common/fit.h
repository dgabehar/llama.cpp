#pragma once

#include "ggml.h"
#include "llama.h"

#include <vector>

enum common_params_fit_status {
    COMMON_PARAMS_FIT_STATUS_SUCCESS = 0, // found allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_FAILURE = 1, // could not find allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_ERROR   = 2, // a hard error occurred, e.g. because no model could be found at the specified path
};

// a second model that shares the devices of the main model, e.g. a draft model
//   - its context follows the context of the main model, so its memory is measured again whenever that context changes
//   - shares_model tells the fit that the weights are already counted in the main model, as for an MTP context
struct common_fit_extra_model {
    const char * path_model;
    llama_model_params * mparams;
    llama_context_params * cparams;
    bool shares_model;
};

// fits mparams and cparams to free device memory (assumes system memory is unlimited)
//   - returns true if the parameters could be successfully modified to fit device memory
//   - this function is NOT thread safe because it modifies the global llama logger state
//   - only parameters that have the same value as in llama_default_model_params are modified
//     with the exception of the context size which is modified if and only if equal to 0
common_params_fit_status common_fit_params(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams,
                              float * tensor_split,          // writable buffer for tensor split, needs at least llama_max_devices elements
   llama_model_tensor_buft_override * tensor_buft_overrides, // writable buffer for overrides, needs at least llama_max_tensor_buft_overrides elements
                             size_t * margins,               // margins of memory to leave per device in bytes
                           uint32_t   n_ctx_min,             // minimum context size to set when trying to reduce memory use
      const common_fit_extra_model * extra,                  // model to fit alongside the main one, nullptr if there is none
                     ggml_log_level   log_level);            // minimum log level to print during fitting, lower levels go to debug log

// print estimated memory to stdout
void common_fit_print(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams);

void common_memory_breakdown_print(const llama_context * ctx);

struct common_device_memory_data {
    int64_t total;
    int64_t free;
    size_t  model;
    size_t  context;
    size_t  compute;
};

using common_device_memory_data_vec = std::vector<common_device_memory_data>;

// compute nodes (views/reshapes excluded) of the prompt and single-token graphs reserved by the most recent
// memory probe of the main model, -1 if none ran
void common_fit_last_graph_nodes(int32_t & pp, int32_t & tg);

// Same as common_get_device_memory_data, plus the memory of an extra model (draft/MTP) added to the
// main model's devices the way common_fit_params accounts for it. extra may be nullptr.
common_device_memory_data_vec common_get_device_memory_data_with_extra(
                         const char * path_model,
                 const llama_model_params * mparams,
               const llama_context_params * cparams,
       const common_fit_extra_model * extra,
      std::vector<ggml_backend_dev_t> & devs,
                           uint32_t & hp_ngl,
                     ggml_log_level   log_level);

// Clamp a candidate context size so its projected KV cache memory use on one device -- linearly
// extrapolated from bytes_per_ctx, a real measurement taken at a small, always-safe-to-build
// context -- does not exceed that device's free memory minus a margin and whatever else it is
// already using (fixed_use, e.g. model weights and compute buffers). Used to size --fit's
// auto-context probe from measured memory up front, instead of only reacting to an allocation
// failure: a per-op size ceiling is not the only way an oversized probe can go wrong, and on a
// UMA device individual KV cache tensors can pass such a check while still not fitting in real
// memory once summed across layers.
// Returns n_ctx_max unchanged if bytes_per_ctx <= 0 (this device holds none of the KV cache), and
// never returns less than n_ctx_min_total.
uint32_t common_fit_clamp_ctx_to_free_memory(
                           uint32_t   n_ctx_max,
                           uint32_t   n_ctx_min_total,
                            int64_t   dev_free,
                            int64_t   fixed_use,
                            int64_t   margin,
                            int64_t   bytes_per_ctx);

// An integrated GPU's own reported free memory is carved from the same physical RAM the host
// reports free, and can ignore what every other process on the host is currently holding (a known
// gap on AMD APUs, and by the same GTT-pool mechanism potentially any other iGPU). Returns
// dev_free unchanged if is_igpu is false, otherwise the smaller of dev_free and host_free.
int64_t common_fit_cap_igpu_free(
                            int64_t   dev_free,
                            int64_t   host_free,
                               bool   is_igpu);

// Load a model + context with no_alloc and return the per-device memory breakdown.
common_device_memory_data_vec common_get_device_memory_data(
                         const char * path_model,
           const llama_model_params * mparams,
         const llama_context_params * cparams,
    std::vector<ggml_backend_dev_t> & devs,
                           uint32_t & hp_ngl,
                           uint32_t & hp_n_ctx_train,
                           uint32_t & hp_n_expert,
                     ggml_log_level   log_level);
