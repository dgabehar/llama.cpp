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

## Speed-aware split and pipeline fixes (added 2026-09-24)

Why a gabesrv10 (8060S) + gabesrv06 (780M, RPC over Thunderbolt) split of
Qwen3.8-27B ran at about half of gabesrv10 alone: **not RPC.** A decoded
token needs one round trip and ~40 KB of inputs. A layer split runs the
devices one after another, and the default split (proportional to free
memory) put 23 of 64 layers on a GPU that is ~2.6x slower per layer for
decode and ~3.7x for prefill. Prompt processing can overlap the devices
across ubatches, but four things stopped that overlap:

1. **`GGML_VK_MAX_NODES_PER_SUBMIT=1`** (fleet env). Every node became a
   job in amdgpu's 32-job ring queue (`sched_jobs`), so `vkQueueSubmit`
   blocked and `graph_compute` turned synchronous.
2. **Vulkan sized submits from the previous graph's flops.** The first
   prompt ubatch after a 1-token decode went out node by node, which is
   the same blocking as item 1.
3. **Too many flop-sized submits.** ~110 per 512-token ubatch on the 8060S,
   so the queue filled anyway.
4. **Scheduler reallocs and checkpoint breaks.** Scheduler reallocs synced
   all backends when the ubatch shape changed. llama-server also split
   every prompt at its context checkpoints, and each break drained the
   pipeline.

Patches (see `fleet-patches/README.md`):

- **vulkan: size submits from the current graph** fixes item 2.
- **llama: restore the worst-case sched plan before large ubatches** fixes
  the reallocs.
- **vulkan: keep a graph's flop-sized submits below the job queue** (at
  most 12) fixes item 3. The weak-AMD timeout cap stays as it was.
- **llama, server: capture context checkpoints inside a batch.** New
  experimental `llama_state_seq_capture_add/get/clear`: the ubatch is cut at
  the checkpoint token, and the state is copied on the device in stream
  order. The server no longer splits the prompt for checkpoints.
  - Saves are byte-identical to the old behavior when ubatch boundaries
    match.
  - `LLAMA_SERVER_CKPT_CAPTURE=0` restores the old behavior.
  - eagle3 still splits.
- **common: `--split-balance auto|decode|prefill|memory`**, default `auto`.
  - At startup it times each device on the model's own layers (RPC devices
    over the normal protocol, so the worker needs no change; see
    "Per-layer calibration" below) and caches the result in
    `~/.cache/llama.cpp/split-balance.json`, keyed by device, model shape
    and build and re-measured after 7 days (`--split-calibrate force` to
    redo it now).
  - It picks layer counts that minimize the predicted time of a
    `--split-workload P:G` request (default 4096:256), capped by the fit
    memory projection.
  - `memory` is the old behavior, for when the model only fits split.
    An explicit `-ts` always wins.

**Results (Qwen3.8-27B Q4_K_XL, llama-server, 1.9K / 8.1K-token prompts +
256 generated tokens):**

| config | layers on 06 | pp 1.9K | pp 8.1K | tg |
|---|---|---|---|---|
| before: memory split, old binary | 23 | 164* | | 7.2 |
| `--split-balance memory` (new binary) | 23 | 175 | 185 | 6.8 |
| `auto` (default, 4096:256) | 0 | 309 | 294 | 12.0 |
| `auto`, `--split-workload 32768:16 -b 8192` (570db1ff4 only) | 13 | 261 | 312 | 7.8 |
| fit-limited (`-fitt 1024,100000`), `memory` | 23 | 172 | 184 | 6.8 |
| fit-limited, `auto` | 21 | 172 | 199 | 7.0 |

\* at 20/80, the earlier measurement. The same split with the capture and
scheduler fixes: 262.

The 32768:16 row is from 570db1ff4. From d7cd53de7 on (dominant-type and
n_batch-drain cost model), `auto` keeps that workload on gabesrv10 too.

**Before/after, measured by QA (2026-09-24, image d963996 vs b5f1d1bd,
llama-server, median of 3, drift 3%):**

| config | pp 1.9K | pp 8.1K | tg |
|---|---|---|---|
| production today: old image, memory split (23 on 06), submit env on | 138.6 | 141.5 | 6.7 |
| new image, `memory` split, capture on | 180.0 | 186.5 | 7.1 |
| old image, 20/80 | 170 | 183 | 7.9 |
| new image, 20/80 | 265 | 305 | 8.4 |
| new image, `auto` (0 on 06) | 352 | 335 | 12.2 |
| gabesrv10 alone, old image | 284 | 317 | |
| gabesrv10 alone, new image | 347 | 330 | |

**Using it in the chart:**

- A master that reaches a worker over the Thunderbolt link must run with
  `hostNetwork`. From the pod network, packets forwarded from `tb-*`
  (MTU 65520, GRO) into the Calico veth (MTU 1450) are fragmented and dropped
  (16% retransmits, pp 2.7 t/s).
- `-ncmoe`/`-ot` patterns that match no tensor of the model (the chart's
  `nCpuMoe` default on a dense model) don't disable balancing. Patterns that
  do match keep the memory split.

**In-batch checkpoint capture is for multi-device models only.** On one
device it cost 12-15% prompt speed. The worst case was the gabesrv10 router
(Qwen3.8-27B, MTP draft, `--parallel 3`, `-b/-ub 1024`), where QA measured
0.85x on 1.5K-token prompts. llama-server now captures only when the model
spans several non-CPU devices (`llama_model_n_devices_used`).
`LLAMA_SERVER_CKPT_CAPTURE=1` forces it on anyway, and `=0` turns it off.

A capture is read after the next `llama_decode()` has been submitted, and it
waits on an event instead of synchronizing the context. Slot saves stay
byte-identical to the split path.

**Direct I/O (`--load-mode dio`)** now goes through a bounce buffer where the
destination refuses DMA (EFAULT on amdgpu pinned memory). Before, the whole
file fell back to buffered reads and filled the page cache.

**MoE calibration:** MoE models are timed with `mul_mat_id` over their expert
shape. gpt-oss-20b on a 3080 Ti:

| | pp | tg |
|---|---|---|
| predicted | 1535 | 98.1 |
| measured | 1796 | 99.6 |

The dense matmul used before was off by about 2x.

**Per-layer calibration (added 2026-09-25):** the calibration times every
repeating layer kind the model has (hybrid models have several), with the
real weight shapes and types, the real expert count, flash attention at the
workload's context, and GDN (`ssm_conv` + `gated_delta_net`). It also times
a slice of the output layer, which decode runs once per token on the last
device. The layer's other ops (norms, RoPE, elementwise) are charged per
node: the fit probe records the real graph's compute node count
(`llama_graph_n_compute_nodes`) and a chain of small ops measures their cost
on each device. The fixed cost of a graph run plus readback is measured and
subtracted.

Layers are grouped by structure, not by weight types. Mixed quants vary the
types per layer: Qwen3.8 UD-Q4_K_XL has 53 type layouts over 3 structures,
which took 118 s to calibrate on a 780M worker. Each structure is timed on
its most common type layout, with decode *and prefill* scaled by the
structure's mean bytes (added 2026-09-25, pp-predict fix: prefill was left
unscaled at first, so a structure whose untimed types were the heavier ones
had its prefill rate under-measured -- both the memory-bound decode matmul
and the dequant work a prefill matmul pays scale with weight size, not just
element count). Graphs over 50 ms are timed once per round.

**Busy nodes:** after a warm-up pass each timing runs up to 4 rounds and
keeps the fastest. A load that lasts the whole calibration slows every round
alike, so the check is against the cache instead:

- `ref|<device, model, workload>`: the device's last calibration that didn't
  look busy. It is not keyed by build, so it survives image bumps.
- A new calibration more than 1.25x slower than the reference logs "probably
  busy" and is cached for 1 hour.
- On a cache hit, a device more than 1.25x faster than cached is calibrated
  again. So is one more than 1.5x slower, as before.
- After a build change, a device whose quick check is within 10% of its
  reference reuses it: about 2 s instead of a full calibration.
- `--split-calibrate force` re-measures without dropping other cache entries.

The cache is `split-balance.json` under `LLAMA_CACHE` (default
`~/.cache/llama.cpp`). The home-infrastructure chart mounts it on a node-local
hostPath (`master.calibrationCache`); in a bare container it's lost on every
restart.

Round 5 of QA found a use-after-free in `read_model_layers`: the GGUF context
was freed before the per-layer key reads, and 10 of 17 fleet starts crashed.
An ASan build with two CPU `rpc-server` workers reproduced it on every start.

Predicted vs measured on a 3080 Ti:

| model | pp | tg |
|---|---|---|
| gpt-oss-20b | 1745/1915 | 89/103 |
| Qwen3-1.7B | 8918/7088 | 209/192 |
| K2-Horizon-7B | 2305/1734 | 60/56 (was 153/56) |
| Qwen3.5-4B (hybrid) | 3960/2867 | 85/100 (was 232/96) |

pp still reads 20-40% high on three of the four; the error is similar
across devices, so the chosen split is affected less than the numbers.

**rpc-server admission:** a connection takes a `--max-clients` slot only
once its HELLO arrives, within 10 s. Probes and silent sockets never hold a
slot. At most 64 connections wait for their HELLO; the oldest is closed to
make room. A client refused at the cap gets an all-zero HELLO version and
reports "all of its client slots are in use".

So for a model that fits on one node, `auto` keeps it there. RPC helps
throughput only for long-prompt workloads, and only over Thunderbolt:
over 1 GbE, prefill loses 9-21% and tg ~1.4%.

**`GGML_VK_MAX_NODES_PER_SUBMIT=1` on gabesrv06/10 (B4):** stress-tested
without it on this build, with the incident profile of 36-51K-token prompts:

- gabesrv10: production Qwen3.8-27B flags (`--parallel 3`, draft-mtp,
  `-b/-ub 1024`, 786K ctx). 6 prompts, 3 at a time.
- gabesrv06: production gpt-oss-20b flags. 8 prompts.

No DeviceLost and no amdgpu reset/timeout in dmesg. Both hosts already run
`amdgpu.lockup_timeout=10000`, and each submit is capped far below that.
The env can go from the 06/10 values. Keep it on gfx90c (gabesrv01/04),
where the incident happened and the 2 s default timeout applies.

**Not done, measured:**

- **The KQ mask over RPC** costs ~0.13% of tg at 12K context over LAN
  (below the 1% bar).
- **`-sm tensor`** aborts on qwen35 hybrid ("view of permuted tensor not
  implemented", `ggml-backend-meta.cpp:702`). It would also need an
  all-reduce per layer over RPC.

## Known-fragile areas (real bugs found here, not upstream-tracked until filed)

- **Views over transposed tensors** (2026-09-24, `41b2ad40d`): `ggml_view_*`
  always gives dim 0 the element stride, so slicing a transposed tensor
  along dim 0 with more than one element reads the wrong memory. The
  SSM_CONV_SPLIT patch's `build_conv_window` did this, and saved a garbage
  conv state after every multi-token ubatch. draft-mtp verification (a
  3-token ubatch) hit it on every step: Qwen3.8 leaked repeated `</think>`
  and draft acceptance fell to ~0.37. To slice one, view the untransposed
  tensor and transpose the view. A useful check for any speculative
  decoding change: re-score its output with the plain target model and
  count emitted tokens the target gives p < 0.002 (22 broken, 0 fixed).

- **Prompt cache update on a busy slot** (2026-09-25, `0ab13d2bb`, upstream
  bug, still in upstream master): a request pinned with `id_slot` to a slot
  that is still generating is deferred, but slot selection first ran the
  prompt cache update on that slot, loading the cached prompt that best
  matches the pinned request into the running task. The running task then
  finished on another conversation's context. Any model, any
  `--cache-ram` > 0 (the default), triggered by the LiteLLM slot-persistence
  hook's `id_slot` pins. Symptom: a reply that quotes another client's
  conversation, and a slot whose `n_tokens` at release does not equal its
  prompt plus generated tokens. When testing a server fix against an
  unfixed binary, copy the whole `bin/` directory: `llama-server` loads
  `libllama-server-impl.so` through its build-tree RUNPATH, so a copied
  binary alone runs whatever library is currently built.

- **K2-Horizon reasoning tags** (2026-09-25, `8519f34e6`): the template
  opens the reasoning with the tag for the request's `reasoning_effort`, but
  the model does not always close with the same one. At the fleet's "low"
  it often ends with the "high" tag `</ifm|think>`. The parser workaround in
  `chat-diff-analyzer.cpp` accepts all three close tags (`end_alts`).
  Symptom when it breaks: empty `content`, the whole reply plus a raw
  `</ifm|...>` tag in `reasoning_content`. The model also sometimes answers,
  emits a second close tag and starts over with a garbled copy. Content ends
  at any close tag after the reasoning (`analyze_content::stray_ends`), and
  generation stops there too. The server gets the same tags as
  `stop_after_reasoning`: stop strings that count only once one of them has
  closed the reasoning, so the dropped remainder is never decoded. Neither
  applies with `reasoning_format: none`, which returns the raw text.

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
