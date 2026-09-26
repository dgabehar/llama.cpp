// Tests common_fit_clamp_ctx_to_free_memory(), the pure sizing computation --fit's auto-context
// probe uses to clamp its starting size from measured free memory, instead of only reacting to
// an allocation failure. See common/fit.cpp and common/fit.h for the real usage.

#include "fit.h"

#include <cstdio>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { n_fail++; fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

int main() {
    constexpr int64_t GiB = 1024 * 1024 * 1024;

    // plenty of free memory -- the full requested context already fits, no clamp:
    {
        const uint32_t n_ctx = common_fit_clamp_ctx_to_free_memory(
            /*n_ctx_max=*/ 1'000'000, /*n_ctx_min_total=*/ 4096,
            /*dev_free=*/ 32 * GiB, /*fixed_use=*/ 4 * GiB, /*margin=*/ 1 * GiB,
            /*bytes_per_ctx=*/ 1024);
        CHECK(n_ctx == 1'000'000, "expected no clamp, got %u", n_ctx);
    }

    // free memory can only back a fraction of the requested context -- clamp to what fits:
    {
        // budget = 24 - 4 - 1 = 19 GiB; at 128 KiB/ctx that's exactly 155648 units of context
        const int64_t bytes_per_ctx = 128 * 1024;
        const uint32_t n_ctx = common_fit_clamp_ctx_to_free_memory(
            /*n_ctx_max=*/ 2'097'152, /*n_ctx_min_total=*/ 4096,
            /*dev_free=*/ 24 * GiB, /*fixed_use=*/ 4 * GiB, /*margin=*/ 1 * GiB,
            bytes_per_ctx);
        const int64_t budget = 19 * GiB;
        const uint32_t expected = (uint32_t) (budget / bytes_per_ctx);
        CHECK(n_ctx == expected, "expected %u, got %u", expected, n_ctx);
        CHECK(n_ctx < 2'097'152, "clamp should have reduced n_ctx_max, got %u", n_ctx);
    }

    // fixed use (model + compute) plus margin already exceeds free memory on their own --
    // there is no room for any context beyond the floor:
    {
        const uint32_t n_ctx = common_fit_clamp_ctx_to_free_memory(
            /*n_ctx_max=*/ 1'000'000, /*n_ctx_min_total=*/ 4096,
            /*dev_free=*/ 4 * GiB, /*fixed_use=*/ 5 * GiB, /*margin=*/ 1 * GiB,
            /*bytes_per_ctx=*/ 1024);
        CHECK(n_ctx == 4096, "expected the floor of 4096, got %u", n_ctx);
    }

    // this device holds none of the KV cache (e.g. every layer lives elsewhere) -- nothing to
    // clamp, the requested size passes through unchanged regardless of free memory:
    {
        const uint32_t n_ctx = common_fit_clamp_ctx_to_free_memory(
            /*n_ctx_max=*/ 2'097'152, /*n_ctx_min_total=*/ 4096,
            /*dev_free=*/ 1 * GiB, /*fixed_use=*/ 0, /*margin=*/ 0,
            /*bytes_per_ctx=*/ 0);
        CHECK(n_ctx == 2'097'152, "expected passthrough, got %u", n_ctx);
    }

    // the clamp never returns less than n_ctx_min_total, even with zero free memory:
    {
        const uint32_t n_ctx = common_fit_clamp_ctx_to_free_memory(
            /*n_ctx_max=*/ 1'000'000, /*n_ctx_min_total=*/ 4096,
            /*dev_free=*/ 0, /*fixed_use=*/ 0, /*margin=*/ 0,
            /*bytes_per_ctx=*/ 1024);
        CHECK(n_ctx == 4096, "expected the floor of 4096, got %u", n_ctx);
    }

    // the computed size rounds down (floor division), never over budget, as long as the result
    // stays above the n_ctx_min_total floor:
    {
        // budget = 5000*1024 + 100 bytes at 1024 bytes/ctx -> floor(5000 + 100/1024) = 5000
        const uint32_t n_ctx = common_fit_clamp_ctx_to_free_memory(
            /*n_ctx_max=*/ 1'000'000, /*n_ctx_min_total=*/ 4096,
            /*dev_free=*/ 4 * GiB + 5000 * 1024 + 100, /*fixed_use=*/ 4 * GiB, /*margin=*/ 0,
            /*bytes_per_ctx=*/ 1024);
        CHECK(n_ctx == 5000, "expected floor-divided 5000, got %u", n_ctx);
    }

    if (n_fail == 0) {
        printf("all tests passed\n");
    } else {
        printf("%d test(s) failed\n", n_fail);
    }
    return n_fail == 0 ? 0 : 1;
}
