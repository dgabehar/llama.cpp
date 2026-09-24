#!/usr/bin/env bash
# rpc-server must serve several clients at once, and no single client (slow,
# vanished, malformed, flooding) may starve or take down the others.
set -euo pipefail

server=$1
client=$2
port=$((30000 + $$ % 10000))
ep="127.0.0.1:${port}"
test_dir=$(mktemp -d)
pid=""

cleanup() {
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    wait 2>/dev/null || true
    rm -rf "$test_dir"
}
trap cleanup EXIT

fail() {
    echo "FAIL: $*"
    echo "--- server log ---"
    cat "$test_dir/server.log" || true
    exit 1
}

wait_for_port() {
    for _ in {1..600}; do
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            return 0
        fi
        sleep 0.05
    done
    return 1
}

start_server() {
    "$server" --device CPU --host 127.0.0.1 --port "$port" "$@" >"$test_dir/server.log" 2>&1 &
    pid=$!
    wait_for_port || fail "server did not start"
}

stop_server() {
    kill -0 "$pid" 2>/dev/null || fail "server died"
    kill "$pid"
    wait "$pid" 2>/dev/null || true
    pid=""
}

alive() {
    kill -0 "$pid" 2>/dev/null || fail "server died ($1)"
}

echo "== concurrent clients, per-connection backends"
start_server --max-clients 4
"$client" hold "$ep" 3 >"$test_dir/hold.log" &
hold_pid=$!
sleep 0.5
# a second client is served while the first holds its connection
"$client" hello "$ep" 1000 || fail "second client starved while first connected"
# probe-style connect/close must not disturb the live session
for _ in 1 2 3; do (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null || fail "probe connect"; done
# a malformed request closes only its own connection
"$client" malformed "$ep" || fail "malformed request not rejected"
alive "malformed request"
# a client that closes with replies in flight must not SIGPIPE the server
for _ in 1 2 3; do
    "$client" fin-close "$ep" >/dev/null || fail "server gone after a client closed early (SIGPIPE?)"
    sleep 0.1
done
sleep 0.3
alive "write to closed client"
# concurrent compute from two clients gives correct results
"$client" compute "$ep" 20 1 & c1=$!
"$client" compute "$ep" 20 2 & c2=$!
wait "$c1" || fail "concurrent compute (client 1)"
wait "$c2" || fail "concurrent compute (client 2)"
wait "$hold_pid" || { cat "$test_dir/hold.log"; fail "held client broken by other clients"; }
# every connection above was reaped
"$client" hello "$ep" 1000 || fail "hello after load"
sleep 0.2
last=$(grep "Accepted client" "$test_dir/server.log" | tail -1)
[[ "$last" == *", 1 active" ]] || fail "connections not reaped: $last"
stop_server

echo "== --serialize-compute"
start_server --serialize-compute
"$client" compute "$ep" 10 3 & c1=$!
"$client" compute "$ep" 10 4 & c2=$!
wait "$c1" || fail "serialized compute (client 1)"
wait "$c2" || fail "serialized compute (client 2)"
stop_server

echo "== client cap"
start_server --max-clients 2
"$client" hold "$ep" 2 >/dev/null & h1=$!
"$client" hold "$ep" 2 >/dev/null & h2=$!
sleep 0.5
"$client" rejected "$ep" || fail "client over the cap was not rejected"
wait "$h1" || fail "held client 1 broken by rejected client"
wait "$h2" || fail "held client 2 broken by rejected client"
sleep 0.2
"$client" hello "$ep" 1000 || fail "slot not freed after clients left"
grep -q "Rejected client" "$test_dir/server.log" || fail "no rejection logged"
stop_server

echo "== accept errors (fd exhaustion) are not fatal"
(
    ulimit -n 48
    exec "$server" --device CPU --host 127.0.0.1 --port "$port" --max-clients 0 >"$test_dir/server.log" 2>&1
) &
pid=$!
wait_for_port || fail "server did not start"
"$client" flood "$ep" 100 1500
sleep 0.5
alive "fd exhaustion"
grep -q "accept failed" "$test_dir/server.log" || fail "fd exhaustion was not reached"
"$client" hello "$ep" 2000 || fail "server stopped serving after accept errors"
stop_server

echo "OK"
