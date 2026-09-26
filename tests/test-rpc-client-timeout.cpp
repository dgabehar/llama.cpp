// Regression test for the RPC *client* hang fixed on 2026-09-26.
//
// Root cause: rpc_dispatcher (the RPC client used by a llama-server master
// to talk to each --rpc worker) set no receive timeout and no TCP keepalive
// on its connections -- socket_t::connect() leaves both unset, and every
// dispatcher call blocks on std::future::wait() with no deadline. A worker
// that completes the HELLO handshake and then goes silently unresponsive
// (accepts a request but never answers it, without closing the socket --
// e.g. a CPU rpc-server straining under an oversized/lopsided layer share
// from the --split-balance memory-split fallback, or any other cause of a
// peer that's alive but not making progress) wedges the calling thread in
// rpc_dispatcher::send() forever. This is the mechanism behind the
// K2-Horizon-7B / two-CPU-rpc-server hang reported in QA round 8 (R1):
// "no split fits the free memory of the devices, keeping the memory split"
// followed by a hang in rpc_dispatcher::send()/future.wait() inside
// ggml_backend_rpc_buffer_set_tensor, needing an external `timeout` to kill.
//
// This test reproduces "worker alive, socket open, never answers" without
// needing a real second rpc-server or the ability to SIGSTOP a process:
// "server" mode is a minimal hand-rolled stub that completes the real HELLO
// handshake (so the real client backend connects successfully), then reads
// the header of the one request that follows and goes silent forever.
// "client" mode makes one real ggml_backend_rpc_get_device_memory() call
// against that stub. With GGML_RPC_CLIENT_TIMEOUT_SEC set short by the
// driver script, the fixed client must abort (via its own recv() timing
// out) well inside a bounded window instead of hanging -- the driver
// script asserts on that bound, and ctest's own TIMEOUT property is a
// second, coarser safety net if a regression reintroduces the old
// unbounded wait.

#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// Wire layout, mirroring ggml/src/ggml-rpc/ggml-rpc.cpp (packed).
#pragma pack(push, 1)
struct rpc_msg_hello_req {
    uint8_t conn_caps[24];
};
struct rpc_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[24];
};
#pragma pack(pop)

static constexpr uint8_t RPC_CMD_HELLO = 14;

static bool read_exact(int fd, void * buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, (char *) buf + got, n - got);
        if (r <= 0) {
            return false;
        }
        got += (size_t) r;
    }
    return true;
}

static bool write_exact(int fd, const void * buf, size_t n) {
    size_t sent = 0;
    while (sent < n) {
        ssize_t w = write(fd, (const char *) buf + sent, n - sent);
        if (w <= 0) {
            return false;
        }
        sent += (size_t) w;
    }
    return true;
}

// A minimal RPC "server" that completes the handshake correctly, then goes
// silent forever on the first real request instead of answering it.
static int mode_stall_server(int port) {
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) {
        perror("socket");
        return 1;
    }
    int on = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons((uint16_t) port);
    addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(srv, (sockaddr *) &addr, sizeof(addr)) < 0) {
        perror("bind");
        return 1;
    }
    if (listen(srv, 1) < 0) {
        perror("listen");
        return 1;
    }
    printf("stall-server: listening on port %d\n", port);
    fflush(stdout);

    int fd = accept(srv, nullptr, nullptr);
    if (fd < 0) {
        perror("accept");
        return 1;
    }

    // ---- HELLO request: | cmd(1) | size(8) | rpc_msg_hello_req |
    uint8_t cmd = 0;
    if (!read_exact(fd, &cmd, 1) || cmd != RPC_CMD_HELLO) {
        fprintf(stderr, "stall-server: expected HELLO, got cmd=%u\n", cmd);
        return 1;
    }
    uint64_t req_size = 0;
    if (!read_exact(fd, &req_size, sizeof(req_size)) || req_size != sizeof(rpc_msg_hello_req)) {
        fprintf(stderr, "stall-server: bad HELLO request size %" PRIu64 "\n", req_size);
        return 1;
    }
    rpc_msg_hello_req req{};
    if (!read_exact(fd, &req, sizeof(req))) {
        fprintf(stderr, "stall-server: failed to read HELLO body\n");
        return 1;
    }

    // ---- HELLO response: | size(8) | rpc_msg_hello_rsp | -- answer with a
    // real, matching protocol version and all-zero conn_caps (no RDMA), so
    // the real client backend's negotiate_hello() accepts it and proceeds.
    rpc_msg_hello_rsp rsp{};
    rsp.major = RPC_PROTO_MAJOR_VERSION;
    rsp.minor = RPC_PROTO_MINOR_VERSION;
    rsp.patch = RPC_PROTO_PATCH_VERSION;
    uint64_t rsp_size = sizeof(rsp);
    if (!write_exact(fd, &rsp_size, sizeof(rsp_size)) || !write_exact(fd, &rsp, sizeof(rsp))) {
        fprintf(stderr, "stall-server: failed to send HELLO response\n");
        return 1;
    }
    printf("stall-server: handshake complete, now going silent on the next request\n");
    fflush(stdout);

    // ---- Next request: read its header (cmd + size), then go silent
    // forever -- never answer, never close the socket. This is "worker
    // process alive, connection open, just never responds."
    if (!read_exact(fd, &cmd, 1)) {
        fprintf(stderr, "stall-server: peer closed before sending a second request\n");
        return 1;
    }
    if (!read_exact(fd, &req_size, sizeof(req_size))) {
        return 1;
    }
    std::vector<uint8_t> body(req_size);
    if (req_size > 0 && !read_exact(fd, body.data(), body.size())) {
        return 1;
    }
    printf("stall-server: received cmd=%u, going silent\n", cmd);
    fflush(stdout);
    // Sleep well past any sane test timeout. The client's own recv timeout
    // (this test's whole point) is what must end this, not us.
    std::this_thread::sleep_for(std::chrono::hours(1));
    return 0;
}

// One real RPC client call against the stub above. Expected to fail (abort)
// once its receive timeout fires, not hang -- the driver script measures
// how long this process actually took to exit.
static int mode_client(const std::string & endpoint) {
    ggml_backend_load_all();
    size_t free_mem  = 0;
    size_t total_mem = 0;
    // GGML_RPC_CLIENT_TIMEOUT_SEC is expected to be set (short) by the
    // driver script before this process starts.
    ggml_backend_rpc_get_device_memory(endpoint.c_str(), 0, &free_mem, &total_mem);
    // Only reached if the stub answered, which it never does by design --
    // if we get here the test setup is broken, not the fix.
    fprintf(stderr, "client: unexpectedly got a response (free=%zu total=%zu)\n", free_mem, total_mem);
    return 1;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s server PORT | %s client ENDPOINT\n", argv[0], argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    if (mode == "server") {
        return mode_stall_server(std::atoi(argv[2]));
    }
    if (mode == "client") {
        return mode_client(argv[2]);
    }
    fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    return 2;
}
