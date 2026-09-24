# Fleet-specific notes (dgabehar/llama.cpp fork)

This file exists only on this fork (`dgabehar/llama.cpp`) -- it does not
exist upstream, so it carries zero rebase-conflict risk, unlike `AGENTS.md`
and `CLAUDE.md` which are shared with `ggml-org/llama.cpp` and get replayed
through on every rebase. Put fleet-specific operational knowledge here, not
in those two files.

## What `fleet-patches` is

The `fleet-patches` branch = current `upstream/master` + this fork's own
patch commits on top. It's what `home-infrastructure`'s
`deployments/llamacpp-rpc/image/Dockerfile` pins its `LLAMACPP_REF` build arg
against (a specific commit SHA, never the floating branch name). See that
repo's `deployments/llamacpp-rpc/AGENTS.md` for the image-build side of this
workflow (CI-verifies a `LLAMACPP_REF` bump, don't pre-build locally).

`.github/workflows/sync-with-upstream.yml` (this fork's own CI, patch 0005 in
`fleet-patches/`) rebases this branch onto `upstream/master` weekly and
on-dispatch, force-pushing a clean rebase or opening a tracking issue on
conflict.

## Fleet patch maintenance (added 2026-09-15)

Every fork-exclusive commit on `fleet-patches` is also mirrored to a
`git format-patch` file in `fleet-patches/*.patch` (a real directory in this
repo, same name as the branch -- don't confuse the two). The git branch is
still the source of truth; the files exist so a patch survives without a
live clone of the branch, and so a rebase conflict has something static to
work from instead of only `git show` on a branch mid-rebase.

**After adding, amending, or squashing any fork-exclusive commit**, run
`fleet-patches/regenerate.sh` and commit the result in the same PR/commit as
the code change -- the mirror is not automatically kept in sync.

**Adding a new patch**: commit it same as any of the existing ones (see
`git log --oneline origin/fleet-patches --not upstream/master` for the
current set) -- own commit, real commit message with root cause +
reproduction + fix rationale (see any existing patch for the house style),
`Assisted-by: Claude Sonnet` not `Co-Authored-By:` per `AGENTS.md`'s hard
rule for this repo. Regenerate the mirror afterward.

**No `git push`/PR without Doug's explicit go-ahead each time** -- this
applies to this fork too, not just upstream contributions (`AGENTS.md`'s
push/PR restriction is written for upstream, but this fork follows the same
discipline by standing convention).

## QA requirement for every fork-exclusive change (added 2026-09-18)

**With every change to this fork (a new patch, an amend, a `LLAMACPP_REF`
bump into `home-infrastructure`), Murat (the `bmad-tea` test-architect
persona) must update the e2e test plan
(`home-infrastructure/docs/architecture/litellm-fleet-e2e-test-plan.md`)
and Dawn (the `bmad-dawn-qa-executor` persona) must execute that test plan
AND perform exploratory/adversarial testing beyond its written coverage.**
Standing requirement, not a one-off -- this is the loop that caught every
real regression/gap this fleet found this session (the system-message
hook's `role=="developer"` gap, the `n_parallel` cross-node replication
mismatch, the slot-action queue-timeout gap), each time on a change that
looked complete from build+unit-test success alone. Applies regardless of
how small the change looks or how confident the build/local-test result
is -- a green `tests/test-chat.cpp`/`test_slot_save.py` run is necessary,
not sufficient, evidence for a fleet-facing change.

## CPU locality-domain split ("Track A", added 2026-09-18)

New `--cpu-split auto|off` flag (default `auto`). Extends the CPU backend
device registry from a hardcoded 1 device to N, one per detected "locality
domain" -- a group of cores sharing an L3 cache slice (proxy for a CCD/die
boundary on multi-chiplet CPUs like AMD Strix Halo/gabesrv10's two CCDs).
Detection parses `/sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list`
(`GGML_CPU_LOCALITY_SYSFS_ROOT` overrides the root for testing) and collapses
to the existing single-device behavior on any monolithic die (every other
current fleet node) or any `/sys` read failure -- `auto` is a verified no-op
there, safe as the fleet-wide default. On a real multi-domain box, startup
calibration (`common_cpu_split_calibrate()`, reusing
`tools/tuning/bench.h`'s benchmark primitives) times a representative FFN
matmul per domain and weights each domain's share of CPU-resident layers
(`src/llama-model.cpp`'s `get_layer_buft_list`/`dev_layer`) proportionally,
the same `splits[]`/`upper_bound` mechanism already used for multi-GPU
`--tensor-split`. Device count is locked to real detected domains only --
no override to split finer than that (a single sequential generation's
per-token layer dependency means there's no known benefit to splitting a
shared-L3 domain further). See
`~/.claude/plans/create-a-plan-to-abundant-whistle.md` for the full design
and the "Strix Split" research artifact
(https://claude.ai/artifact/Mht44KMMnoWH8DbnVvUUYk) for the prior-art
investigation (RPC-over-loopback already works but is the wrong tool; this
is the maintainers' own previously-proposed real fix, upstream discussions
#12303 and #19102).

**Known caveat, not yet resolved**: on a real multi-domain box, a layer
whose CPU-resident tensors land on a non-domain-0 device
(e.g. `CPU1`) can trigger the existing `resolve_fused_ops`
(`src/llama-context.cpp`) device-mismatch fallback for Flash Attention --
`ggml_backend_sched_get_tensor_backend()` resolves that layer's FA fused-op
tensor to a different device than `model.dev_layer(il)` reports for the
layer as a whole, and the existing safety net gracefully disables FA for
that context rather than crashing (confirmed live on a mocked 2-domain
fixture: `resolve_fused_ops: layer 3 is assigned to device CPU1 but Flash
Attention is assigned to device CPU`). This is a real, if silent,
performance regression (FA off) on any node where this feature is both
enabled (`auto`) and actually engages a >1-domain split, not a correctness
bug -- the existing fallback is doing exactly what it's designed to do.

**Decision (2026-09-18, Doug): ship as-is.** Don't block the gabesrv10
rollout fixing this preemptively -- the whole point of Track A is measuring
instead of assuming, so let the real gabesrv10 A/B (`--cpu-split auto` vs
`off`, per the plan's verification section) show whether the split's net
gain still wins with FA off on cross-domain layers, before spending effort
on a fix that might not be needed.

**TODO: investigate co-locating a layer's fused ops with its own CPU
tensors (a `dev_layer`-level change to `resolve_fused_ops` or the per-layer
split logic) if we run into issues** -- i.e. if the gabesrv10 benchmark
shows the FA-off cost is eating into or exceeding the split's own gain.
Don't implement this speculatively; it's a real design decision, not
obvious from the code alone. Flag this caveat to Murat/Dawn explicitly when
testing this on gabesrv10 -- the A/B tok/s comparison is what determines
whether the TODO is needed at all.

## rpc-server multi-client (added 2026-09-23)

Upstream `ggml-rpc-server` served one connection at a time
(`while (true) { accept(); rpc_serve_client(); }` with `listen(fd, 1)`), and a
llama-server master holds its RPC socket for the model's whole lifetime. So
once a master connected, everything else sat in the kernel backlog: a second
master, `--list-devices` (about 127s of SYN retries, then `GGML_ABORT "Failed
to connect"`), and k8s `tcpSocket` probes. That's the root cause of the
Aug-2026 worker liveness-kill problem (the chart's `failureThreshold: 45`
workaround) and of the 2026-09-23 gabesrv10->gabesrv06 Thunderbolt test
failures. Upstream issue ggml-org/llama.cpp#28908.

Patches 0057-0061 and 0063 (see `fleet-patches/README.md`):

- **Thread per connection** (0057-0058, upstream PR #28916 cherry-picks),
  then 0059 replaces #28916's detached threads and global compute mutex
  with a tracked connection list (joined, reaped, logged with id + peer),
  `listen(fd, SOMAXCONN)`, and **per-connection backend instances**: each
  connection gets its own `ggml_backend_dev_init()` backend, so clients
  compute concurrently with no server-side lock. The device-level
  queue/submit locking in the backends (e.g. Vulkan's `device->mutex`,
  `device_submit_mutex`) handles the sharing, same as several llama
  contexts on one GPU.
- **New rpc-server flags:** `--max-clients N` (default 8, 0 = unlimited;
  over the cap the socket is accepted and closed immediately),
  `--keepalive N` (TCP keepalive idle seconds, default 30, so a vanished
  master's buffers are freed in about 60s instead of never), and
  `--serialize-compute` (fallback: one shared backend per device plus a
  per-device mutex, in case a backend proves unsafe with several
  instances).
- **One client can't kill the others** (0060): `MSG_NOSIGNAL` on send (a
  client that FIN-closed before its reply killed the server with SIGPIPE,
  exit 141), transient `accept()` errors (fd exhaustion etc.) no longer end
  the accept loop, and out-of-bounds tensors and failed graph computes now
  close only that connection instead of `GGML_ASSERT`-aborting.
  Tensor-cache writes are atomic (tmp + rename).
- **Test:** `test-rpc-server-multiclient` (0061, label `main`, UNIX only).
  Every case in it fails against the pre-fix server.

No wire-protocol change; old clients work unchanged. Still open,
client-side: the coordinator aborts when a worker dies (client half of
upstream PR #26724), and `get_dispatcher` holds its global mutex during a
blocking `connect`.

**Live results (2026-09-23, gabesrv10 master -> gabesrv06 worker over
Thunderbolt, hostNetwork raw pods, test image
`llamacpp-rpc-server:commit-2120ac537...@sha256:2e6b361d...`):**

- `--list-devices` from a second process while a master is connected:
  0.24s (was a ~127s hang, then `GGML_ABORT "Failed to connect"`).
- Two masters on one worker (Qwen3.8-27B + K2-Horizon-7B), concurrent
  chat requests: correct answers, no worker errors. Per-connection backend
  instances cost no measurable extra device memory (~28 MiB difference vs
  `--serialize-compute`).
- Dead master (flow black-holed with iptables, then SIGKILL, so no
  FIN/RST reaches the worker): dropped by keepalive after ~36s, and worker
  free memory returned to the expected level.
- Probe racing a client at the cap got the client rejected; fixed in
  patch 0063 (reap grace at the cap) with a regression test.
- Throughput, Qwen3.8-27B Q4_K_XL, ~1.9k-token prompt + 256 tokens,
  median of 3:

  | setup | pp t/s | tg t/s | load |
  |---|---|---|---|
  | gabesrv10 alone | 283 | 11.6 | |
  | split over Thunderbolt | 139 | 7.0 | 25s |
  | split over LAN (192.168.1.x) | 124 | 6.9 | 64s |
  | TB split, 2 masters concurrent, per-connection backends | 114 | 6.3 | |
  | TB split, 2 masters concurrent, `--serialize-compute` | 112 | 6.6 | |

  So: a split is much slower than one node for a model that fits on one
  node. Only use RPC for models that don't fit. Thunderbolt mostly speeds
  up loading. Steady-state speed is close to LAN, because a layer split
  only sends activations per token. With two masters the GPU is the
  bottleneck, so per-connection backends vs `--serialize-compute` is a
  wash on throughput. Per-connection backends stay the default for
  isolation, not speed.

## Known-fragile areas (real bugs found here, not upstream-tracked until filed)

- **draft-mtp + `--parallel>1` + split (non-unified) KV cache on
  hybrid/linear-attention architectures** (Qwen3.5/Qwen3.6/Qwen3.8's
  `qwen35`/`qwen3_5` family) is a genuinely fragile combination this fleet
  runs in production. Two real bugs found and patched here (`a1e979568`,
  `b74b2e31c`), both around checkpoint-based slot save/restore interacting
  with the draft model's own KV state. A third symptom, root-caused and
  FIXED (2026-09-15): MTP speculative decoding, combined with
  grammar-constrained (tool-call) decoding, corrupted tool-call output by
  occasionally re-emitting structural template tags (`<parameter=...>`,
  `</function>`, `<tool_call>`) as literal string content -- matches the
  broad shape of the still-open upstream tracking issue
  [ggml-org/llama.cpp#23577](https://github.com/ggml-org/llama.cpp/issues/23577)
  and the closed, "expected behavior"
  [ggml-org/llama.cpp#23335](https://github.com/ggml-org/llama.cpp/issues/23335)
  ("different kernels for different batch sizes"). Live logit-margin tracing
  on gabesrv06 pinned the mechanism precisely: near-tied greedy picks (margins
  as low as ~0.03) at the qwen3-coder template's structurally-ambiguous
  decision points, consistent with batch-size numerical noise, not a
  checkpoint-restore/grammar-desync bug (ruled out via direct tracing --
  zero restore events fired during a traced corrupted run). Fixed with two
  layered patches:
  - **Primary fix** (`d4836a23c`, `server: disable speculative decoding for
    grammar-constrained requests`): `get_n_draft_max()` returns 0 whenever
    the request's sampling params carry an active grammar (tool-calling,
    JSON-schema output). Confirmed correctly scoped -- plain chat requests
    still draft normally (`draft_n > 0`).
  - **Defense-in-depth** (`92a035776`, `qwen3-coder: bound xml-arg-string
    with until_one_of, not a bare until`): hardens the tool-call parser's
    string-argument grammar rule to exclude all structural tags from the
    matched span, not just the single expected terminator. Tested alone and
    found insufficient by itself (converts silent corruption into silent
    failure -- no tool call produced at all) -- kept as a second layer
    behind the primary fix, not a replacement for it.
  Both verified live on gabesrv06 (4/4 clean tool calls) before merging to
  `fleet-patches`.

- **CPU locality-domain split ("Track A") caused a real production outage
  on first rollout (2026-09-18) -- crash-looped BOTH gabesrv06 and
  gabesrv10 on model load, `GGML_ASSERT(ggml_backend_supports_buft(
  backends[b], sched->bufts[b]))` in `ggml_backend_sched_new`, even though
  gabesrv06 is a monolithic-die node where the feature should have been a
  complete no-op (1 detected domain, identical to pre-feature behavior).
  Root cause, FIXED (`5f25d2b3e`, `fix: don't reject a GPU's host buffer
  type in ggml_backend_cpu_device_supports_buft`): a device-identity guard
  added for this feature (`buft->device != nullptr && buft->device != dev`
  -> reject) was meant to stop domain N's private `layer_buft` from being
  accepted by domain M's CPU backend, but `llama-context.cpp`'s
  pre-existing, unrelated optimization -- substituting a GPU's own host
  buffer type for the CPU backend's buft whenever `model.devices` is
  non-empty, for fast CPU<->GPU transfer -- also produces a buft with a
  non-null `->device` (the GPU itself). The guard rejected that too, on
  every GPU-offload node regardless of CPU domain count. Fix narrows the
  rejection to only a buft whose device is itself a *CPU*-type device
  different from `dev`. **Process gap that let this reach production**: a
  first fix attempt (the `GGML_BACKEND_DL` link error, `33581fbd8`) was
  verified with a full CI-flag build + `--help` + a mocked CTest, but never
  a real model load -- rolled straight to gabesrv06/gabesrv10 and crashed
  both immediately. Rolled back via `helm rollback` within minutes (no
  extended outage), fixed, and re-verified with a real GGUF, `-ngl 999`
  GPU offload, `--cpu-split auto` explicitly enabled, and a real
  `/v1/chat/completions` round-trip before the next deploy attempt. **Any
  future change to this feature's `ggml_backend_cpu_device_supports_buft`/
  `ggml_backend_cpu_device_get_buffer_type` logic needs a real model-load
  smoke test with GPU offload before being considered verified** -- a
  passing build or a synthetic unit test does not exercise the real
  `llama_context::sched_reserve()` path this bug lived in.

- **`tests/test-chat.cpp`'s `test_template_output_peg_parsers()` crashed
  (uncaught `std::terminate`) on `models/templates/Qwen3-Coder.jinja`'s
  `html` tool.** Found 2026-09-17 while fixing the server-side "now finding
  less tool calls" invariant crash (see `35e651927`, `server: end
  generation cleanly on a tool-call-diff inconsistency`); that fix alone
  didn't cover this test since it calls `compute_diffs()` directly, not
  through the server's `update_chat_msg()`. **FIXED** (see the commit
  titled `peg-parser: don't prematurely close an until_one_of span on an
  ambiguous partial delimiter match`) -- this genuinely was a fixable
  PEG-parser bug, not an inherent property of incremental parsing. Root
  cause, traced with `--template Qwen3-Coder --detailed`: the html markup
  value's content starts with a literal `<`, which is also the first
  character of several of `xml-arg-string`'s `until_one_of` terminators
  (`<tool_call>`, `<function=`, `<parameter=`). At the exact byte-offset
  where only that lone `<` has streamed in so far, `common_trie::check_at`
  correctly reports `PARTIAL_MATCH` (ambiguous -- could still complete into
  a real terminator, or turn out to be ordinary content), but
  `common_peg_until_parser`'s executor (`common/peg-parser.cpp`) treated
  `PARTIAL_MATCH` the same as `COMPLETE_MATCH` and returned `SUCCESS` with
  an empty captured span right there. That made the immediately-following
  `arg_close` literal (`\n</parameter>\n`) get tested against the real next
  characters (`html>...`) and fail outright (not "need more input"), which
  collapsed the whole in-progress tool call from the AST instead of just
  waiting for one more byte to disambiguate. Fixed by making the
  `PARTIAL_MATCH` branch return `NEED_MORE_INPUT` (matching the existing
  ran-off-the-end-of-input fallback a few lines below) whenever
  `ctx.is_lenient()`, so `SEQUENCE`/`REPEAT` correctly stay pending instead
  of hard-failing. Confirmed via the same before/after
  build-and-run-test-chat method as `35e651927`.
