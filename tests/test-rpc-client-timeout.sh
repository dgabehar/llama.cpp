#!/usr/bin/env bash
# Driver for test-rpc-client-timeout.cpp -- see that file's header comment
# for what this proves. Starts the stub "server" (completes the HELLO
# handshake, then goes silent forever on the next request), then runs the
# real RPC client against it with a short GGML_RPC_CLIENT_TIMEOUT_SEC and
# asserts it fails within a bounded window instead of hanging.
set -uo pipefail

bin=$1
port=$((40000 + $$ % 10000))
endpoint="127.0.0.1:${port}"
test_dir=$(mktemp -d)

cleanup() {
    kill "${srv_pid:-}" 2>/dev/null || true
    wait "${srv_pid:-}" 2>/dev/null || true
    rm -rf "$test_dir"
}
trap cleanup EXIT

# Do NOT probe readiness with a bare TCP connect+close (e.g. bash's
# /dev/tcp): the stub server below calls accept() exactly once and expects
# that one connection to be the real client's HELLO handshake, not a probe.
# A connect-then-immediately-close probe would consume that single accept()
# with a connection that never sends anything, and the stub would exit
# treating it as a malformed handshake -- wait on its own log line instead.
wait_for_log() {
    local pattern=$1
    for _ in {1..600}; do
        if grep -q "$pattern" "$test_dir/server.log" 2>/dev/null; then
            return 0
        fi
        sleep 0.05
    done
    return 1
}

"$bin" server "$port" >"$test_dir/server.log" 2>&1 &
srv_pid=$!
if ! wait_for_log "listening on port"; then
    echo "stub server never came up"
    cat "$test_dir/server.log"
    exit 1
fi

# Short client-side timeout so a correct fix finishes this test in seconds;
# a regression back to the old unbounded wait hangs here instead, and is
# caught by this test's own ctest TIMEOUT property.
export GGML_RPC_CLIENT_TIMEOUT_SEC=2

t0=$(date +%s.%N)
"$bin" client "$endpoint" >"$test_dir/client.log" 2>"$test_dir/client.err"
rc=$?
t1=$(date +%s.%N)
elapsed=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.1f", b - a }')

echo "client exit=$rc elapsed=${elapsed}s"
cat "$test_dir/client.log"
cat "$test_dir/client.err"

if [ "$rc" -eq 0 ]; then
    echo "FAIL: client exited 0 -- it should have failed once the stub went silent" >&2
    exit 1
fi

# Bounded: the fixed client must fail once its ~2s receive timeout fires,
# not instantly (that would mean it failed for some unrelated reason before
# ever reaching the stalled request) and not after a long wait (that would
# mean the timeout never took effect and it's approaching an unbounded
# hang -- this script's caller, ctest, has its own hard TIMEOUT as the
# final backstop for that case).
awk_cmp() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a < b) }'; }
if awk_cmp "$elapsed" "1.0"; then
    echo "FAIL: client failed after only ${elapsed}s, before the ~2s timeout could have fired -- test setup is broken" >&2
    exit 1
fi
if ! awk_cmp "$elapsed" "30.0"; then
    echo "FAIL: client took ${elapsed}s -- the receive timeout does not appear to be bounding the wait" >&2
    exit 1
fi

echo "OK: client failed cleanly after ${elapsed}s (bounded by GGML_RPC_CLIENT_TIMEOUT_SEC=2), not a hang"
exit 0
