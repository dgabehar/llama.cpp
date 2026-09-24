// Client side of test-rpc-server-multiclient.sh: each invocation is one rpc-server
// client (its own process, so its own connection). Most modes speak the wire
// protocol directly so the driver can hold connections open, send malformed
// requests and close sockets at awkward moments; "compute" uses the real
// ggml RPC backend.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// wire layout of the protocol structs (ggml/src/ggml-rpc/ggml-rpc.cpp), packed
#pragma pack(push, 1)
struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char     name[GGML_MAX_NAME];
    int32_t  use_count;
};

struct rpc_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};
#pragma pack(pop)

enum : uint8_t {
    CMD_ALLOC_BUFFER      = 0,
    CMD_BUFFER_GET_BASE   = 3,
    CMD_SET_TENSOR        = 6,
    CMD_GET_DEVICE_MEMORY = 11,
    CMD_HELLO             = 14,
};

static constexpr size_t CONN_CAPS_SIZE = 24;

static int g_timeout_ms = 5000;

static bool parse_endpoint(const std::string & ep, std::string & host, int & port) {
    size_t pos = ep.rfind(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = ep.substr(0, pos);
    port = std::atoi(ep.c_str() + pos + 1);
    return port > 0;
}

static int connect_to(const std::string & ep) {
    std::string host;
    int port;
    if (!parse_endpoint(ep, host, port)) {
        return -1;
    }
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1 ||
        connect(fd, (sockaddr *) &addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static bool send_all(int fd, const void * data, size_t size) {
    const uint8_t * p = (const uint8_t *) data;
    while (size > 0) {
        ssize_t n = send(fd, p, size, MSG_NOSIGNAL);
        if (n <= 0) {
            return false;
        }
        p += n;
        size -= n;
    }
    return true;
}

// false on EOF, error or timeout
static bool recv_all(int fd, void * data, size_t size) {
    uint8_t * p = (uint8_t *) data;
    while (size > 0) {
        pollfd pfd = { fd, POLLIN, 0 };
        if (poll(&pfd, 1, g_timeout_ms) <= 0) {
            return false;
        }
        ssize_t n = recv(fd, p, size, 0);
        if (n <= 0) {
            return false;
        }
        p += n;
        size -= n;
    }
    return true;
}

static bool send_cmd(int fd, uint8_t cmd, const void * input, uint64_t size) {
    return send_all(fd, &cmd, 1) && send_all(fd, &size, sizeof(size)) && (size == 0 || send_all(fd, input, size));
}

static bool recv_rsp(int fd, void * output, uint64_t size) {
    uint64_t out_size;
    return recv_all(fd, &out_size, sizeof(out_size)) && out_size == size && recv_all(fd, output, size);
}

static bool hello(int fd) {
    uint8_t req[CONN_CAPS_SIZE] = {};
    uint8_t rsp[4 + CONN_CAPS_SIZE];
    if (!send_cmd(fd, CMD_HELLO, req, sizeof(req)) || !recv_rsp(fd, rsp, sizeof(rsp))) {
        return false;
    }
    return rsp[0] == RPC_PROTO_MAJOR_VERSION;
}

static bool device_memory(int fd, uint64_t & free_mem, uint64_t & total_mem) {
    uint32_t dev = 0;
    uint64_t rsp[2];
    if (!send_cmd(fd, CMD_GET_DEVICE_MEMORY, &dev, sizeof(dev)) || !recv_rsp(fd, rsp, sizeof(rsp))) {
        return false;
    }
    free_mem  = rsp[0];
    total_mem = rsp[1];
    return true;
}

static bool alloc_buffer(int fd, uint64_t size, uint64_t & remote_ptr, uint64_t & base) {
    rpc_msg_alloc_buffer_req req = { 0, size };
    uint64_t rsp[2];
    if (!send_cmd(fd, CMD_ALLOC_BUFFER, &req, sizeof(req)) || !recv_rsp(fd, rsp, sizeof(rsp)) || rsp[0] == 0) {
        return false;
    }
    remote_ptr = rsp[0];
    return send_cmd(fd, CMD_BUFFER_GET_BASE, &remote_ptr, sizeof(remote_ptr)) && recv_rsp(fd, &base, sizeof(base));
}

// true if the peer closed the connection (EOF/reset) within the timeout
static bool wait_closed(int fd) {
    pollfd pfd = { fd, POLLIN, 0 };
    if (poll(&pfd, 1, g_timeout_ms) <= 0) {
        return false;
    }
    uint8_t b;
    return recv(fd, &b, 1, 0) <= 0;
}

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// hello EP: HELLO must be answered within the timeout
static int mode_hello(const std::string & ep) {
    double t0 = now_s();
    int fd = connect_to(ep);
    bool ok = fd >= 0 && hello(fd);
    printf("hello: %s in %.3fs\n", ok ? "ok" : "FAILED", now_s() - t0);
    if (fd >= 0) {
        close(fd);
    }
    return ok ? 0 : 1;
}

// probe-hello EP N: N times, connect+close (a TCP health probe) and immediately
// connect again and HELLO; the probe must not hold a slot against the client
static int mode_probe_hello(const std::string & ep, int n) {
    for (int i = 0; i < n; i++) {
        int probe = connect_to(ep);
        if (probe < 0) {
            printf("probe-hello: probe connect FAILED\n");
            return 1;
        }
        close(probe);
        int fd = connect_to(ep);
        bool ok = fd >= 0 && hello(fd);
        if (fd >= 0) {
            close(fd);
        }
        if (!ok) {
            printf("probe-hello: client rejected right after probe %d\n", i);
            return 1;
        }
    }
    printf("probe-hello: %d ok\n", n);
    return 0;
}

// rejected EP: the server must accept and then immediately close the connection
static int mode_rejected(const std::string & ep) {
    int fd = connect_to(ep);
    if (fd < 0) {
        printf("rejected: connect failed\n");
        return 1;
    }
    bool closed = wait_closed(fd);
    close(fd);
    printf("rejected: %s\n", closed ? "connection closed by server" : "FAILED, connection still open");
    return closed ? 0 : 1;
}

// hold EP SECONDS: HELLO + allocate, stay connected, then check the connection still works
static int mode_hold(const std::string & ep, double seconds) {
    int fd = connect_to(ep);
    uint64_t remote_ptr, base, free_mem, total_mem;
    if (fd < 0 || !hello(fd) || !alloc_buffer(fd, 1 << 20, remote_ptr, base)) {
        printf("hold: setup FAILED\n");
        return 1;
    }
    printf("hold: connected\n");
    fflush(stdout);
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    bool ok = device_memory(fd, free_mem, total_mem) && total_mem > 0;
    close(fd);
    printf("hold: %s after %.1fs\n", ok ? "still works" : "FAILED", seconds);
    return ok ? 0 : 1;
}

// malformed EP: SET_TENSOR with a data pointer outside its buffer must close this
// connection (not abort the server)
static int mode_malformed(const std::string & ep) {
    int fd = connect_to(ep);
    uint64_t remote_ptr, base;
    if (fd < 0 || !hello(fd) || !alloc_buffer(fd, 4096, remote_ptr, base)) {
        printf("malformed: setup FAILED\n");
        return 1;
    }
    rpc_tensor t = {};
    t.id     = 1;
    t.type   = GGML_TYPE_F32;
    t.buffer = remote_ptr;
    t.ne[0] = 16; t.ne[1] = 1; t.ne[2] = 1; t.ne[3] = 1;
    t.nb[0] = 4;  t.nb[1] = 64; t.nb[2] = 64; t.nb[3] = 64;
    t.data   = base + (1 << 20); // far past the 4 KiB buffer
    std::vector<uint8_t> msg(sizeof(t) + 1 + 8 + 64, 0);
    memcpy(msg.data(), &t, sizeof(t));
    bool sent = send_cmd(fd, CMD_SET_TENSOR, msg.data(), msg.size());
    bool closed = sent && wait_closed(fd);
    close(fd);
    printf("malformed: %s\n", closed ? "connection closed by server" : "FAILED, connection not closed");
    return closed ? 0 : 1;
}

// fin-close EP: send HELLO in two parts and close right after the second, so the
// server reads a complete request from a peer that has already sent FIN. Its
// reply is then answered with RST and the next write fails with EPIPE, which
// raises SIGPIPE unless the server suppresses it.
static int mode_fin_close(const std::string & ep) {
    int fd = connect_to(ep);
    if (fd < 0) {
        return 1;
    }
    uint8_t  cmd  = CMD_HELLO;
    uint64_t size = CONN_CAPS_SIZE;
    uint8_t  req[CONN_CAPS_SIZE] = {};
    bool ok = send_all(fd, &cmd, 1) && send_all(fd, &size, sizeof(size));
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // server now blocked reading the body
    ok = ok && send_all(fd, req, sizeof(req));
    close(fd);
    printf("fin-close: %s\n", ok ? "sent" : "send FAILED");
    return ok ? 0 : 1;
}

// flood EP N MS: open N connections without speaking, hold them MS ms, close them
static int mode_flood(const std::string & ep, int n, int ms) {
    std::vector<int> fds;
    for (int i = 0; i < n; i++) {
        int fd = connect_to(ep);
        if (fd >= 0) {
            fds.push_back(fd);
        }
    }
    printf("flood: %zu of %d connections open\n", fds.size(), n);
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    for (int fd : fds) {
        close(fd);
    }
    return 0;
}

// compute EP ITERS SEED: y = W x on the RPC backend, checked against the CPU backend
static int mode_compute(const std::string & ep, int iters, int seed) {
    ggml_backend_load_all();
    ggml_backend_t rpc = ggml_backend_rpc_init(ep.c_str(), 0);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (rpc == nullptr || cpu == nullptr) {
        printf("compute: backend init FAILED\n");
        return 1;
    }
    const int n = 256;
    ggml_init_params params = {
        /* .mem_size   = */ 3 * ggml_tensor_overhead() + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    std::vector<float> w(n * n), x(n), y_ref(n), y(n);
    srand(seed);
    for (auto & v : w) { v = (float) rand() / RAND_MAX - 0.5f; }

    int failures = 0;
    for (int it = 0; it < iters; it++) {
        for (auto & v : x) { v = (float) rand() / RAND_MAX - 0.5f; }
        for (int pass = 0; pass < 2; pass++) {
            ggml_backend_t backend = pass == 0 ? cpu : rpc;
            ggml_context * ctx = ggml_init(params);
            ggml_tensor * tw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
            ggml_tensor * tx = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            ggml_tensor * ty = ggml_mul_mat(ctx, tw, tx);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, ty);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            ggml_backend_tensor_set(tw, w.data(), 0, ggml_nbytes(tw));
            ggml_backend_tensor_set(tx, x.data(), 0, ggml_nbytes(tx));
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                failures++;
            }
            ggml_backend_tensor_get(ty, pass == 0 ? y_ref.data() : y.data(), 0, ggml_nbytes(ty));
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
        for (int i = 0; i < n; i++) {
            if (std::fabs(y[i] - y_ref[i]) > 1e-3f) {
                failures++;
                break;
            }
        }
    }
    ggml_backend_free(rpc);
    ggml_backend_free(cpu);
    printf("compute(seed %d): %d/%d iterations %s\n", seed, iters - failures, iters, failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODE ENDPOINT [ARGS...]\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    const std::string ep   = argv[2];
    auto arg = [&](int i, double def) { return argc > i ? std::atof(argv[i]) : def; };

    if (mode == "hello")          { g_timeout_ms = (int) arg(3, 1000); return mode_hello(ep); }
    if (mode == "probe-hello")    { g_timeout_ms = 2000; return mode_probe_hello(ep, (int) arg(3, 20)); }
    if (mode == "rejected")       { return mode_rejected(ep); }
    if (mode == "hold")           { return mode_hold(ep, arg(3, 2.0)); }
    if (mode == "malformed")      { return mode_malformed(ep); }
    if (mode == "fin-close")      { return mode_fin_close(ep); }
    if (mode == "flood")          { return mode_flood(ep, (int) arg(3, 100), (int) arg(4, 1000)); }
    if (mode == "compute")        { return mode_compute(ep, (int) arg(3, 20), (int) arg(4, 1)); }
    fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    return 2;
}
