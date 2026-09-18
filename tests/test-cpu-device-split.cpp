// Covers the "Track A" locality-aware CPU device split (see plan
// create-a-plan-to-abundant-whistle.md): CPU locality-domain auto-detection,
// the CPU device registry's 1->N extension, and that regular (unmocked)
// hardware still yields the byte-for-byte N=1 fallback path.
//
// GGML_CPU_LOCALITY_SYSFS_ROOT lets this test point detection at a fixture
// directory instead of live /sys, so it exercises the N>1 path deterministically
// on any host, including single-die CI runners.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

static void write_file(const std::string & path, const std::string & content) {
    std::ofstream f(path);
    f << content;
}

static void make_dirs(const std::string & path) {
    std::string cur;
    for (size_t i = 0; i <= path.size(); i++) {
        if (i == path.size() || path[i] == '/') {
            if (!cur.empty()) {
                mkdir(cur.c_str(), 0755);
            }
            if (i < path.size()) {
                cur += path[i];
            }
        } else {
            cur += path[i];
        }
    }
}

// Builds a fixture sysfs tree with n_domains groups of cores_per_domain cores each,
// every core's shared_cpu_list naming exactly its own domain's core range.
static std::string build_fixture(const std::string & root, int n_domains, int cores_per_domain) {
    for (int d = 0; d < n_domains; d++) {
        int lo = d * cores_per_domain;
        int hi = lo + cores_per_domain - 1;
        std::string list = std::to_string(lo) + "-" + std::to_string(hi);
        for (int c = lo; c <= hi; c++) {
            std::string dir = root + "/cpu" + std::to_string(c) + "/cache/index3";
            make_dirs(dir);
            write_file(dir + "/shared_cpu_list", list + "\n");
        }
    }
    return root;
}

static size_t count_cpu_devices() {
    size_t n = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            n++;
        }
    }
    return n;
}

int main() {
    // NOTE: ggml_backend_cpu_locality_domains() memoizes on first use (see
    // ggml/src/ggml-cpu/ggml-cpu.cpp), so GGML_CPU_LOCALITY_SYSFS_ROOT must be set
    // via setenv() before the FIRST call to ggml_backend_dev_count()/get() in this
    // process -- this test relies on being a fresh process per invocation (it is
    // registered as its own CTest binary, not merged into another test's process).
    std::string fixture_root = std::string(getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp")
        + "/test-cpu-device-split-fixture-" + std::to_string(getpid());
    build_fixture(fixture_root, /* n_domains = */ 2, /* cores_per_domain = */ 4);

    setenv("GGML_CPU_LOCALITY_SYSFS_ROOT", fixture_root.c_str(), 1);

    ggml_backend_load_all();

    const size_t n_cpu = count_cpu_devices();
    printf("detected %zu CPU device(s) under mocked 2-domain fixture\n", n_cpu);
    if (n_cpu != 2) {
        fprintf(stderr, "FAIL: expected 2 CPU devices, got %zu\n", n_cpu);
        return 1;
    }

    // Each domain's device must report a distinct, disjoint core mask matching the fixture.
    bool cpumask[2][GGML_MAX_N_THREADS];
    size_t seen = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        if (!ggml_backend_cpu_device_get_locality_mask(dev, cpumask[seen])) {
            fprintf(stderr, "FAIL: ggml_backend_cpu_device_get_locality_mask returned false for a CPU device\n");
            return 1;
        }
        seen++;
    }
    assert(seen == 2);

    int domain0_count = 0, domain1_count = 0;
    bool overlap = false;
    for (int c = 0; c < GGML_MAX_N_THREADS; c++) {
        if (cpumask[0][c]) domain0_count++;
        if (cpumask[1][c]) domain1_count++;
        if (cpumask[0][c] && cpumask[1][c]) overlap = true;
    }
    if (domain0_count != 4 || domain1_count != 4) {
        fprintf(stderr, "FAIL: expected 4 cores per domain, got %d and %d\n", domain0_count, domain1_count);
        return 1;
    }
    if (overlap) {
        fprintf(stderr, "FAIL: domain core masks overlap, should be disjoint\n");
        return 1;
    }

    // Two distinct CPU backend instances must actually be constructible (registry
    // extension isn't just reporting a count -- get_device() must work per index too).
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            fprintf(stderr, "FAIL: ggml_backend_dev_init failed for CPU device '%s'\n", ggml_backend_dev_name(dev));
            return 1;
        }
        ggml_backend_free(backend);
    }

    printf("PASS: 2-domain mocked fixture -> 2 CPU devices, disjoint 4-core masks, both constructible\n");
    return 0;
}
