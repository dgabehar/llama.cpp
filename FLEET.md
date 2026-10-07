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
conflict. At ~100 fork commits that rebase cannot succeed unattended; do a
deliberate sync every 2-3 weeks (see "Upstream sync" below) and treat the weekly
job as a notifier. `fleet-patches/regenerate.sh` takes `BASE=<sha> TIP=HEAD` for a
branch built on a pinned upstream commit.

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

## Upstream sync 2026-10 (rebased onto upstream `4f5406761`, added 2026-10-06)

Branch `upstream-sync-2026-10` = upstream `4f5406761517648c23dbd60ea5ade37f77a316c9`
(`HIP: use -O0 for host code in debug builds (#29795)`, tag **b11444**; the
trial notes called it b11439, the SHA is authoritative) + 100 fork commits
(`fleet-patches/` was rebuilt from scratch; patch numbers changed, refer to
patches by subject). It was built by cherry-picking the 103 code/test fork
commits of `fleet-patches` @ `25d47f832` and skipping the ~56 mirror/FLEET.md
bookkeeping commits. Rollback anchor: tags `pre-merge-2026-10-06`
(`25d47f832`) and `pre-merge-2026-10-06-deployed` (`b16c58a80`, the deployed
`LLAMACPP_REF`).

**Dropped patches (3):**
- `ggml-alloc: finalize view init in a pass after all buft max_size splits`
  -- superseded by upstream #23671 (`ggml_backend_buft_alloc_buffer_n_default`
  initialises views inside each split item, `ggml/src/ggml-backend.cpp`). Needs
  the MTP + `--parallel 3` checkpoint-creation canary (the original
  `GGML_ASSERT(tensor->data != NULL)` repro) before it is considered verified.
- `server: fix 2 test_completion.py tests broken by test-only bugs` -- upstream
  rewrote the same assertion.
- the second copy of `server: don't update the prompt cache of a busy slot picked
  by id_slot` (hotfix cherry-pick with identical content).

**Re-derived against upstream (no behaviour change intended):** `-fit` probe/clamp
(`common/fit.cpp`) uses upstream's `n_streams` again (#29437 revert); `n_ctx_max`
stays non-const. K2-Horizon pre-tokenizer enum renumbered to
`LLAMA_VOCAB_PRE_TYPE_K2_HORIZON = 61` (upstream MMBERT took 60). `conversion/base.py`
keeps our `pytorch_model*.safetensors` fallback plus upstream's
`model.safetensors.index.json` clause. Compile fixes folded into their owning
commits: CPU locality `layer_buft` initialiser gets `.alloc_buffer_n`/`.get_alloc_size_n`
(#23671); `fs_get_cache_directory()` returns `std::filesystem::path` (#29595);
`tests/test-state-capture.cpp` fills `llama_batch` directly (`common_batch_add`
removed by the `llama_batch_ext` migration).

**New slot-save format (breaking):** upstream #28498 bumped `LLAMA_STATE_SEQ_VERSION`
3 -> 4 and `LLAMA_SESSION_VERSION` 10 -> 11 and writes two extra u32 (`n_rot_k`,
`n_rot_v`) after `v_trans`/`n_layer` in every attention KV block. v3 and v4
`.bin` files are not interchangeable in either direction; flush `slotSavePath`
on rollout and on rollback, and update `kvrepl_nstream_convert.py` before the
image bump (see checklist). The fork's SCKP checkpoint appendix (version 1) is unchanged.

### Upstream-sync compatibility checklist (run on every sync, before the image bump)

Our patches must stay 100% compatible with what home-infrastructure consumes.
On every sync, diff `OLD_BASE..NEW_BASE` for the items below; any change means a
home-infrastructure artifact must change in the same series.

1. `include/llama.h`: `LLAMA_STATE_SEQ_VERSION`, `LLAMA_SESSION_VERSION`,
   `LLAMA_STATE_SEQ_FLAGS_*`. `git diff OLD..NEW -- include/llama.h | rg 'VERSION|FLAGS'`.
2. `src/llama-kv-cache.cpp`: `state_write_data` / `state_read_data` field order
   (v_trans, n_layer, then K rows, V rows) and `state_write_meta`.
3. `src/llama-memory-recurrent.cpp`, `llama-memory-hybrid*.cpp`,
   `llama-kv-cache-iswa.cpp`: `state_write*` order and what each writes (r/s
   rows, per-seq headers).
4. `src/llama-context.cpp` `state_seq_*` header (magic `ggsq`, version check,
   `.dft` companion handling) and `tools/server/server-context.cpp`: our SCKP
   appendix (`SLOT_CKPT_VERSION`, `save_slot_checkpoints`) and the `/slots`,
   `/slots/<id>?action=save|restore` and `/props` JSON keys
   (`id_slot`, `is_processing`, `n_saved`, `n_restored`, `filename`, `n_ctx`).
5. Flags the fleet uses (values-*.yaml `extraArgs`, chart templates): extract with
   `rg -oN -h -- '--?[a-z][a-z0-9-]+' deployments/llamacpp-rpc/values*.yaml deployments/llamacpp-rpc/templates`
   in home-infrastructure (non-comment lines), then confirm every
   flag appears in the new `llama-server --help` (current set: `--parallel/-np`,
   `--batch-size`, `--ubatch-size`, `--spec-type {ngram-mod,draft-dflash}`,
   `--spec-draft-n-max`, `--model-draft`, `--chat-template-kwargs`, `--cache-ram`,
   `--no-cache-idle-slots`, `--slot-save-path`, `--cpu-split`, `--split-balance`,
   `--split-workload`, `--split-calibrate`, `--cache-type-k/v`, `-fa`, `-ngl`,
   `--n-cpu-moe`, `-ts`, `--rpc`, `--fit`, `--reasoning-format`, `--embeddings`).
6. Buffer-type / backend interface (`ggml/src/ggml-backend-impl.h`), batch API
   (`llama_batch_ext`) and `common/` helper signatures: these produce compile breaks
   in our out-of-tree initialisers and tests; a clean `git merge`/rebase is not proof.
   Always build (`GGML_VULKAN=ON GGML_RPC=ON`, tests on) and run `ctest` + the
   llama-server pytest suite (`tools/server/tests`, `-m "not slow"`).
7. Conflicts are resolved against upstream's new structure, never by preserving old
   shape (operator rule); drop a patch outright when upstream made it redundant.

home-infrastructure artifacts to update **before** the `LLAMACPP_REF` bump lands:
- `deployments/llamacpp-rpc/scripts/kvrepl_nstream_convert.py` (`BIN_VERSION`, any new
  per-attention-block fields, `_copy_attn_section`, layout doc) and
  `tests/test-kvrepl-nstream-convert.py` fixtures; real round trip with
  `tests/kvrepl-real-restore-proof.py`.
- `slot-cache-sync` / `slot-cache-gc` CronJobs (`templates/master/`): version gate so
  mixed-version nodes do not exchange saves; plan the `slotSavePath` flush.
- `scripts/litellm_custom_callbacks.py` (consumes `id_slot`, `/slots` `is_processing`,
  slot save/restore `filename`) and the `tests/test_litellm_*slot*.py` tests.
- `values-*.yaml` `extraArgs` (flag check above).
- Then bump `LLAMACPP_REF` in `deployments/llamacpp-rpc/image/Dockerfile` (40-char SHA of
  the merged fork commit) and let CI build the image.

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

**Calibration overrated a Vulkan device 6-8x under `GGML_VK_MAX_NODES_PER_SUBMIT=1`
(fixed 2026-09-25, `vk-calib-overrate` branch):** on gabesrv01 (gfx90c, where
production sets the env var as the DeviceLost mitigation above), per-layer
calibration predicted K2-Horizon-7B tg 7.1 and Qwen3.5-4B tg 10.9, while real
`llama-server` measured 0.89 and 1.63 -- `auto` put every layer on it. Root
cause: the untimed-op probe (the `t_op_tg`/`t_op_pp` chain of scale ops that
prices the layer's norms/RoPE/elementwise nodes per node) was only 64 nodes
long. On a backend that submits one command buffer per node, that's too short
to reach the same submit-queue depth a real decode/prefill graph runs at, so
it measured the cheap, unsaturated per-submit cost instead of the real one --
the same class of problem `GGML_VK_MAX_NODES_PER_SUBMIT=1`'s own fleet-wide
note above describes for pipeline parallelism, just inside calibration's own
timing instead of the real graph. Fixed by lengthening the chain to 256 nodes
(8x amdgpu's 32-job `sched_jobs` ring), which required accounting for its
tensors in `calibrate_device`'s `ggml_context` sizing too -- the untouched
budget was tight enough that a MoE model with an output layer configured
(gpt-oss-20b) overflowed it and crashed (`GGML_ASSERT` in `ggml_scale` /
`ggml_new_graph_custom`); `tests/test-split-balance.cpp` gained a regression
test for that (`test_calibrate_context_memory_budget`, a synthetic one-layer
MoE model on the CPU backend, no GPU needed). Verified on an RTX 3080 Ti
Laptop GPU (real per-node submit cost is much smaller there than on gfx90c's
weak amdgpu ring, so the predicted/measured gap the env var causes is only
~10-19%, not 6-8x, but it moves in the right direction and by roughly the
right amount for K2-Horizon-7B, Qwen3.5-4B and gpt-oss-20b); the severity on
gfx90c itself needs the cluster QA loop (Dawn) to confirm against a real
`--split-balance auto` start with the env var set.

**pp "regression" after `dd5756d58` (round 9 QA, 2026-09-26) was a QA
measurement bug, not a cost-model bug.** Round 9 (`layer-calibration-dawn-qa-
round9-2026-09-26.md` section 1, gabesrv01 master + gabesrv07 RPC worker,
`GGML_VK_MAX_NODES_PER_SUBMIT=1`) reported pp off by -74% to -93% (e.g.
K2-Horizon-7B: predicted 197, measured 50.8) on the same 3 models whose tg
the decode-calibration fix had just brought inside +8-18%. That pp figure
came from **one `/completion` call with a short, few-dozen-token prompt**
(`n_predict 64, fresh prompt`) -- `prompt_per_second` from a prompt that
short is dominated by the fixed per-request cost (slot init, batch build,
one graph submit/fence), not steady-state prefill throughput, so it reads
far below the model's real rate. Reproduced live on the identical
gabesrv01+07/r9-image setup: an 17-18 token prompt measured `prompt_per_
second` 23-73 t/s on K2-Horizon-7B and Qwen3.5-4B (matching round 9's 50.8 /
20.5 almost exactly), while a fresh, genuinely long prompt (~2400 tokens,
`cache_prompt: false`, median of 5 reps) on the *same pod, same calibration,
same placement* measured:

| model | predicted pp4096 | short-prompt (~18 tok) pp | long-prompt (~2400 tok) pp, median of 5 |
|---|---|---|---|
| K2-Horizon-7B | 202 | 23-73 (med. 72.7), matches round 9's 50.8 | 119.8-190.0 (med. 138.3, -32%) |
| Qwen3.5-4B | 309 | 23-27 (med. 27.1), matches round 9's 20.5 | 197.4-281.5 (med. 281.0, -9%) |

So with a realistic prompt the error collapses from -74%/-93% to -9%/-32% --
Qwen3.5-4B lands inside the test plan's +-25% dense bar outright, and
K2-Horizon-7B's best individual rep is -6%, with the median dragged down by
real run-to-run throughput variance on this iGPU under
`GGML_VK_MAX_NODES_PER_SUBMIT=1` (successive identical requests measured
~120 t/s and ~190 t/s in a roughly bimodal pattern, unexplained by node CPU
load, which stayed under 1.0 loadavg throughout) rather than by anything the
cost model gets wrong. Code-level suspects named for this investigation were
checked and are NOT the cause: `9ecf429da`'s `byte_scale` is already applied
to prefill (`common/split-balance.cpp`, the `case 1`/`case 2` lines in
`calibrate_device`), and the untimed-op probe already has a ubatch-sized,
256-node-long prefill variant (`t_op_pp`, built from `x_hop` rather than a
single-token view) mirroring the one `vk-calib-overrate` added for decode --
prefill's own real hardware numbers above confirm that probe is adequate,
unlike decode's before `dd5756d58`. No code change follows from this: the
existing cost model is accurate to within the same ballpark FLEET.md already
documents for other hardware (the 3080 Ti's post-`9ecf429da` +-9%).

**Corrected pp measurement method for QA** (supersedes the single-short-
prompt method used through round 9): measure `prompt_per_second` from the
server's own `/completion` response over a **fresh prompt of at least 2000
tokens** (`cache_prompt: false`, new content per rep so llama.cpp's prefix
cache can't shortcut it), and take the **median of at least 5 reps**, not
one -- a single rep on this class of hardware can land anywhere in a ~50%
band. `llama-bench -p 2048` on the same placement is the ideal cross-check
but is **not available in the `llamacpp-rpc-server` runtime image**
(`/app` ships only `llama-server` and `rpc-server`, no `llama-bench`
binary) -- a local build (e.g. this repo's own `build/bin/llama-bench`) is
the only way to run it, and only reaches non-fleet hardware unless a debug
image is built specifically for it.

**Placement verified correct for Qwen3.5-4B** (round 9 flagged the `auto`
placement change -- old 9/23 (worker/master) split vs new 0/32 all-master --
as a literal violation of that round's "placement unchanged" pass criterion
and asked for confirmation this isn't leaving throughput on the table).
Forced the old split back with `-ts 9,23` on the identical gabesrv01+07/r9
pod and measured both ways:

| placement | tg (n_predict 64) | pp (median of 5, ~2400-tok prompt) |
|---|---|---|
| `auto` (0/32, all-master) | 13.23 (round 9) | 281.0 |
| forced `-ts 9,23` (old split) | 10.41 | 164.8 |

`auto`'s new choice is faster on **both** axes -- +27% tg, +70% pp -- not a
regression. This confirms `dd5756d58`'s own reasoning: once decode
calibration correctly prices RPC0's real per-hop cost under
`GGML_VK_MAX_NODES_PER_SUBMIT=1`, keeping this pairing's layers off the RPC
worker is the right call, not a side effect to work around. gpt-oss-20b's
half of this same check was **not** run here -- its GGUF was not staged in
`/var/tmp/dawn` on gabesrv01/07 at the time of this investigation (only
`m6/{K2-Horizon-7B,Qwen3.5-4B,Qwen_Qwen3-1.7B}-*.gguf` were present) -- the
same conclusion very likely holds (same worker/master pair, same underlying
fix), but treat it as unverified until re-run once the model is staged
again.

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

## RPC client receive timeout (added 2026-09-26)

`rpc_dispatcher`, the RPC client that a llama-server master uses for each
`--rpc` worker, had no receive timeout and no TCP keepalive.
`socket_t::connect()` sets neither, and every dispatcher call blocks in
`std::future::wait()` with no deadline. The server side already drops dead
or idle clients, but nothing protected the client from a worker that
accepts a request and then stops answering without closing the socket.
The master then hangs forever. The stack is in `rpc_dispatcher::send()` →
`future.wait()`, and the log freezes with no `predicted` line.

- **Fix:** `rpc_dispatcher::start()` sets `SO_RCVTIMEO` and TCP keepalive on
  the client socket right after `connect()`, before the HELLO handshake.
  - The timeout bounds only a `recv()` made while a request is
    outstanding, so an idle client is never cut off.
  - Default 180 s. `GGML_RPC_CLIENT_TIMEOUT_SEC` overrides it, and `0`
    restores the old unbounded behavior.
  - On timeout the master logs `Lost the connection to the RPC server` /
    `recv failed` and aborts through the existing `RPC_STATUS_ASSERT`. So a
    dead worker now causes a pod restart, not a wedged pod.
- **How it was found:** QA round 8 (R1) saw K2-Horizon-7B hang the RPC
  transfer against two CPU-only `rpc-server` workers. That happened while
  the workstation was thrashing under OOM. On a clean machine the hang
  would not reproduce by just loading or calibrating. The mechanism was
  proved by SIGSTOP-ing one worker mid-transfer: the master hung before
  the fix and failed within the timeout after it.
- **Regression test:** `tests/test-rpc-client-timeout.{cpp,sh}`, label
  `main`. A stub server completes the handshake and then goes silent, and
  the client must fail within the window.
- **Before raising or disabling the timeout:** the 180 s bound is per
  request/response exchange, not per model load. The largest single
  exchange in practice is one weight chunk or one ubatch compute, and both
  are far shorter. If a real workload ever hits it, increase
  `GGML_RPC_CLIENT_TIMEOUT_SEC` for that release rather than setting it
  to 0.
- **Not done:** making RPC errors recoverable instead of aborting, which
  would mean replacing `RPC_STATUS_ASSERT`'s `GGML_ABORT` at every call
  site. Aborting is acceptable under k8s, which restarts the pod.
- **ASan gotcha:** `test-rpc-multi-server` and `test-rpc-server-multiclient`
  report the RPC backend's intentionally never-freed registries as leaks.
  Run ctest under ASan with `ASAN_OPTIONS=detect_leaks=0:abort_on_error=1`.

## Truncated slot-save restore no longer aborts (D1, added 2026-10-06)

Branch `upstream-sync-2026-10-d1` (one commit on `upstream-sync-2026-10`).
Dawn QA defect D1: `/slots/<id>?action=restore` of a save file truncated by 1
byte, 4096 bytes or half hit `GGML_ASSERT(nread <= state_size)` in
`llama_context::state_seq_load_file` and killed llama-server (the file stays
on disk, so a restart does not clear it; this fleet replicates saves between
nodes, so a half-written copy is a real input). Reproduces on the old fleet
build (state v3) and the 2026-10 sync (state v4); the fix is format-independent.

Root cause: `llama_file::read_raw` deliberately tolerates reading past EOF
(zero-fills, so model loading can read direct-I/O alignment padding).
`llama_io_read_file` (the sequence-state file reader) inherited that, so a
short file "read" fine, `nread` overran `state_size` and the assert fired. The
in-memory reader (`llama_io_read_buffer`) already bounds-checks and throws.

Fix (no on-disk format change, v3 and v4 files behave the same):
- `src/llama-context.cpp`: `llama_io_read_file::read` / `read_tensor` throw
  when the request exceeds the bytes left in the file (checked before the
  `read_tensor` staging allocation, so a corrupt length cannot drive a huge
  allocation). The existing catch in `llama_state_seq_load_file` turns it into
  a return of 0, which the server reports as HTTP 400 "Unable to restore
  slot: ...". `state_seq_read_data` now also `seq_rm`s the sequence when the
  restore throws part-way, so a short read leaves no partial sequence (this
  was visible on DeepSeek-V4, whose last-read component left cells behind).
- `tools/server/server-context.cpp`: the SCKP checkpoint appendix reader
  bounds a buffer length by the bytes left in the file instead of a fixed
  16 GiB cap (a corrupt length of a few GiB could OOM a memory-limited pod).
  A bad appendix is still just ignored (restore succeeds without
  checkpoints, as before). The `.dft` companion is read whole by size and
  already validated, so it needed no change.
- Tests: `test-save-load-state` (case "file, truncated by 1 byte / 4096 bytes
  / half" across all 129 arch models) and `test_slot_save.py::test_slot_restore_truncated_file`.
  Both fail before the fix (abort) and pass after.

Upstream check (b11445+11, `51ce9c11a`, 2026-10-06): no fix or open PR for the
truncated-file assert (related but different: #27530 merged, #29723/#29700
closed). Would be offered upstream as a PR; not opened.

## Known-fragile areas (real bugs found here, not upstream-tracked until filed)

- **gpt-oss-20b writes `<|channel|>commentary (analysis)<|message|>`** (2026-09-26):
  at long context (44K-token agent prompts) the model sometimes writes the
  analysis header with a stray "commentary" and "(analysis)". The strict
  gpt-oss PEG parser rejected the whole reply, so llama-server returned an
  error ("unparsed peg-native output") and LiteLLM retried in a loop.
  `common/parsers/gpt-oss.cpp` now accepts that exact header as reasoning.
  Seen on gpt-oss-20b via `gpt-oss-20b-split`: 5 of 6 logged failures; the
  sixth was `<|channel|>commentary**commentary**<|end|>`, not handled.
  `test-chat` has the real output as a case.

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

- **K2-Horizon-7B official IFM weights: three output quirks** (2026-09-26,
  branch `k2-official-parse`). Symptoms in opencode via LiteLLM: "it kept
  stopping" (finish=stop, 0 tokens, the whole tool call inside
  `reasoning_content`; or a 1-token stop with nothing at all), and a literal
  `<ifm|think>` as `reasoning_content` on most turns. Evidence: opencode DB
  copy `/home/dgabehar/k2-evidence/opencode.db`, session
  `ses_f20feb08dffeFXf2LgNbTMHzCd`. Reproduced locally (3080 Ti, opencode-shaped
  requests built from that session, `-fa on -ctk/-ctv q4_0 --spec-type ngram-mod`,
  reasoning_effort low) with the raw text from `/completion`. What the model
  really emits after the template's `<ifm|think_faster>\n`:
  - `</ifm|think_faster>` + answer or `<ifm|tool_calls>` (the normal case,
    ~70%);
  - `<ifm|tool_calls>...` with NO close tag at all (~20% in agentic turns with
    tool results in the history): the parser was still inside reasoning, so the
    call became `reasoning_content`, nothing was emitted, and the client saw a
    plain stop. Fix: `analyze_reasoning::implicit_ends_with_tools` -- with tools
    offered, `<ifm|tool_calls>` ends the reasoning (zero-width, the tool parser
    then reads it);
  - `<ifm|think>\n</ifm|think>...` (the "high" open tag repeated, then closed):
    the parser only knew the start tag of the effort in the prompt, so the
    repeat landed in `reasoning_content`. Fix: `analyze_reasoning::start_alts`,
    repeated open tags (any of the three) after the start are consumed. The
    generation prompt's own `\n` sits between them, hence `space()` before each;
  - an end-of-turn token as the very FIRST token (id 1 `<|ifm|endoftext|>` or
    id 250019 `<|ifm|im_end|>`, both EOG): 1 output token, empty reply. It is
    the model, not the parser: measured 4-12% first-token probability (temp 1,
    no truncation) after an empty assistant turn in the history, i.e. it
    cascades once problem 1 has put empty turns into the conversation. The
    "close tag then EOG" variant is ~0.2-3%. Fix: sampler-side
    `common_params_sampling::no_empty_reply` (`common_sampler::eguard` in
    `common/sampling.cpp`): EOG tokens are masked until a token that is neither
    whitespace nor one of the reasoning tags has been generated. Fed from the
    K2 workaround via `common_chat_params::no_empty_reply_inert` -> the
    `no_empty_reply_inert` request field. It sits beside the grammar and the
    reasoning budget rather than in the chain because the chain also accepts
    prompt tokens (a first attempt in the chain saw "output" before generation
    started and never masked anything).
  Numbers (42 opencode-shaped runs per row, stream and non-stream each, low):
  official file before the fix 13 leaks / 9 tool calls stuck in reasoning / 9
  empty stops; after 0 / 0 / 0. The community quant is NOT clean in this
  harness either (before: 15 / 4 / 4), so the quirks are model behavior, not a
  regression of the official weights; the earlier "0 in 359 messages" on the old
  quant does not reproduce with an opencode-shaped 16k-token context and
  temperature 0.6. The two GGUFs have identical tokenizer, EOG/special-token
  metadata and chat template (only the imatrix keys differ), so there is no
  file-level difference to work around. Tests: `tests/test-chat.cpp` K2 block
  ("official weights" cases, real raw output, streaming + non-streaming).
  Known remaining edge: with NO tools offered, a `<ifm|tool_calls>` right after
  the open tag still ends up in `reasoning_content` (harmless, no leak into
  content). A required tool argument the model leaves out (opencode's bash has
  `description` optional, but a schema requiring it) makes the server return
  500 "does not match the expected peg-native format" -- schema strictness, not
  a K2 issue.

- **K2-Horizon-7B: Kimi-style tool-call opener, then EOG** (2026-09-28, branch
  `k2-kimi-opener`). Live incident (opencode via LiteLLM, ctx ~28K, effort
  low, 3 stalls in ~1 minute): a reasoning part that is exactly
  `\n<|tool_calls_section_begin|><|tool_call_begin|>`, then EOG (output 0,
  reasoning 15, finish=stop). Those markers are NOT vocab tokens of the GGUF
  (they are plain pieces; `<ifm|tool_calls>` is the model's own format), so
  they cannot be banned by id, and the empty-reply guard already treated them
  as "real output" and let EOG through. Fix, part 1: the same guard
  (`common_empty_reply_guard`) keeps a rolling tail of the generated text and
  masks EOG inside an unfinished Kimi opener (`<|tool_calls_section_begin|>`
  until `<|tool_calls_section_end|>`, or a bare `<|tool_call_begin|>` until
  `<|tool_call_end|>`). Part 2 (Doug, 2026-09-29; request field
  `no_empty_reply_hold`, set by the chat handling only when the request offers
  tools): from a native `<ifm|tool_calls>` or a Kimi opener, EOG stays masked
  until the native section is closed AND every inner tag pair is balanced
  (`tool_call`, `arg_key`, `arg_type`, `arg_value`; the json call format only
  uses `tool_call`), because after a Kimi opener the model usually rewrites
  the call in its own format, which the parser understands. A close without
  its opener closes nothing. Hard cap `k_hold_cap` = 2048 generated tokens in
  an open section (real opencode tool-call arguments: p50 157 / p95 1635 /
  p99 3500 chars, ~1200 tokens at p99, x2), then EOG is released again and a
  debug line is logged; the cap only restores the old behaviour, it never
  truncates. Still out of the sampler chain, cloned/reset with the sampler,
  never masks the last candidate. The tag list lives once, in the K2 block of
  `common/chat-diff-analyzer.cpp` (this fork's parser is derived from the
  template, so there is no separate constants file; the names match upstream
  PR ggml-org/llama.cpp#29535's k2-horizon.cpp).
  Reproduction: NOT reproduced naturally in ~220 opencode-shaped runs (replay
  of the stalled turns, padded 0-50K ctx, stream + non-stream, effort low,
  temp 0.6-1.0: 0 Kimi openers; the trigger probably needs opencode's real
  system prompt/tool schemas). Reproduced by forcing the opener into the
  stalled turn's prompt (`/completion`, 40 samples, temp 0.8): 23/40 EOG
  immediately (the incident), 7/40 native call, 4/40 Kimi call only, 6/40
  other. With the guard state seeded as "inside the opener": kimi-only guard
  0/40 EOG but 14/40 Kimi-form call left as reasoning text (still a stall);
  hold-until-native-close 34-35/40 complete native call, 5/40 (500-token
  window) Kimi loops; with the balanced-tags hold and 2600 tokens: 36/40
  complete native call, 4/40 Kimi loops/garbage that ran to the cap, 0/40 EOG.
  Follow-up (2026-09-29, live): with the hold, the fleet turned the silent stall
  into a ~1500-token loop of the opener (`ses_f16214d2...`), so the Kimi hold now
  has its own cap `k_kimi_hold_cap` = 192 tokens (117 complete Kimi calls in the
  harness: median 42, longest 115; native switch after the opener p90 118, max
  255) and a REPEATED unfinished opener releases EOG at once (real loop text is
  a test fixture). Release only permits EOG: forced Kimi-garbage samples that
  never choose EOG still run to the client limit (5/40 ran to 2600 tokens in the
  forced harness, same as before), so this is a safety net, not the fix; the
  root-cause work is why the model leaves the native format at all.
  Residual: those ~10% never switch to the native format, run to the cap and
  then end the turn; parsing the Kimi form (which varies: `<|sep|>`, missing
  `functions.` prefix) would be the next step. Inner-tag balance in the
  harness: no unclosed inner tag in any complete native call; 2/40 native
  calls emitted an EXTRA `</ifm|tool_calls>` (harmless). Tests:
  `tests/test-empty-reply-guard.cpp` (vocab-only, real incident text).

- **K2-Horizon-7B root cause of the Kimi-opener stalls: replayed history** (2026-09-29,
  commits `5864976f9`, `282ab87a9`). The EOG holds (above) are safety nets; the
  cause is in the prompt. (1) The template renders every past assistant turn's
  reasoning with `<ifm|think>` whatever the effort, while the low-effort
  generation prompt opens `<ifm|think_faster>`; (2) opencode replays a stalled
  turn's junk reasoning (`<|tool_calls_section_begin|>...`, `<|close|>`) as
  `reasoning_content`, which the model imitates: junk begets junk (live session
  `ses_f16214d2...`: 25 of 26 replayed reasonings were junk). Evidence, raw
  `/completion` sampling of that session's real history, 40 runs each, same
  cuts/seed: as sent 10/40 Kimi openers; reasoning stripped 0/40; junk
  reasoning stripped 0/40; history tag matched to the generation tag with the
  junk kept 0/40. Fix 1 (`common/chat.cpp` `k2_history_reasoning`, K2 templates,
  effort low/medium): history reasoning goes to the template's `think_faster` /
  `think_fast` field, `<|...|>` / `<ifm|...>` markup is stripped from it, and
  assistant turns left with nothing are dropped. Side effect measured with
  first-token probes: with matching tags the model's first token after the open
  tag skips ahead ("<ifm|tool_call>" 0.10-0.62, "</ifm|tool_calls>" 0.07-0.22,
  vs "<ifm|think>" 0.45-0.78 repeat before), 11/40 empty turns on the garage
  replay. Fix 2 (`common/sampling.cpp`): the tags are single vocab tokens, so
  the armed guard masks every tool tag except the section opener while no
  native section is open (a strict, token-level, non-lazy piece of the tool
  grammar). Ablation on the garage replay (40 stream runs, empty turns): all off
  0, tag match only 10, tag match + strip + drop 11, strip + drop only 0 but
  live-session Kimi 5/60 (new spellings `<|open|>...`). A full non-lazy
  grammar was not built: the trigger is lazy on `<ifm|tool_calls>` by design
  (`chat-auto-parser-generator.cpp`), but with the token mask the failing
  spellings are unreachable at the cheap layer. Result (chat endpoint, tools
  offered, effort low, guard + normalisation): garage replay 0/40 stream, 0/40
  non-stream; live-session replay 0/60 stream, 0/60 non-stream empty / stuck /
  Kimi (before the fix: live stream 16/60 Kimi, 1 empty, 2 stuck).

- **K2-Horizon-7B fix build F2: parser gaps, reasoning-block guard, repetition
  breaker** (2026-09-30, patches 0146-0152, fork commits `11e9b4b63`,
  `0088f823d`, `3449022c9`, `d5812222e`, `f893af144`, `59b910d75`, `ed3ab13ad`).
  - 0146 (`11e9b4b63`): the armed empty-reply guard / tag mask did a
    `std::find` over the EOG and tag id lists per candidate per token (about
    +1.5 ms/token on a ~250k vocab); now a per-vocab flag table, same behavior.
    Test: `test-empty-reply-guard` (200 randomized candidate layouts vs the old
    rule, plus a 250k-candidate timing print).
  - 0147 (`0088f823d`): K2 parser gaps. Reasoning open tag taken from the
    generation prompt (low/medium/high all parse); bare `<ifm|tool_call>` block
    (no section opener/end) is a call; a complete Kimi-form call is surfaced;
    stray `</ifm|think*>`, `<|close|>`, `<|sep|>`, EOG text dropped; a parse
    failure falls back to reasoning + content instead of throwing. Test:
    `tests/test-chat.cpp` (efforts, bare/Kimi calls, stray tags, fallback, 18000-case
    no-throw fuzz).
  - 0148 (`3449022c9`): with tools offered, EOG inside an unclosed reasoning
    block is held; only known K2/Kimi markup is stripped from replayed reasoning
    (was: any `<|...|>`).
  - 0149 (`d5812222e`): a call cut off by EOG/max_tokens (arguments healed to
    `{`) is no longer surfaced as a call; falls back to reasoning/content at the
    end of generation.
  - 0150 (`f893af144`): tagged-args parser takes the arguments in any order
    (was required-first in schema order; K2 writes e.g. `path` before the
    required `pattern`, which surfaced a call with arguments `{`, Grafana F7 t3).
  - 0151 (`59b910d75`): masking EOG alone in an open reasoning block pushed
    replayed turns into 4096-token loops; the close tag now takes the best EOG
    logit while the block is open (capped), so "stop" becomes "close reasoning".
  - 0152 (`ed3ab13ad`): repetition breaker: a reasoning line of 16+ chars written
    three times in the block the prompt opened forces the close tag (R17 live
    seed 0 looped on "Let me check the GitHub releases..." from turn 31).
    Known gap: only identical repeated lines are caught.
  Tests for 0147-0152: `tests/test-chat.cpp`, `tests/test-empty-reply-guard.cpp`.
  QA (Dawn, K2 results doc in home-infrastructure
  `docs/architecture/model-test-plan-k2-horizon-7b-results.md`): GO for a canary.
  R17 live 4 independent seeds 0 empty / 0 runaway / 0 badcall; F7 t3 19/20 OK;
  R01 0/48, R01b 0/48 (baseline fleet build: 4/144 on junk-history replay, 20/20
  leaks at medium/high effort). Pending: garage R17 and a decode-speed check.
  **Upstream rebase point:** ggml-org/llama.cpp PR
  [#29535](https://github.com/ggml-org/llama.cpp/pull/29535) (open, K2 Horizon
  support). When it lands, rebase onto it; our patches still needed on top: the
  EOG guard, bare `<ifm|tool_call>`, Kimi-form opener, `<|close|>` leak,
  optional-before-required argument order, and the repetition breaker.

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

- **`-fit`'s own oversized-context probe aborted the whole server on a
  laptop with an RTX 3080 Ti + an Intel iGPU** (2026-09-25, `fit-abort`
  branch), on plain default startup (`llama-server -m K2-Horizon-7B...
  --jinja -c 8192`, no special flags) -- not upstream-tracked. Not actually
  about having two GPUs: reproduced identically with `--device Vulkan1`
  alone (still crashed), and this fork's own device-selection code already
  drops the iGPU whenever a discrete GPU is present (`gpus.empty()` check in
  `llama_prepare_model_devices`, `src/llama.cpp` -- upstream PR #23897), so
  K2-Horizon-7B was already running on the 3080 Ti alone either way. Root
  cause: llama-server's `n_parallel=auto` picked 4, and K2-Horizon-7B's
  native `n_ctx_train` is 524288, so `-fit`'s own auto-context measurement
  probed at `524288 * 4 = 2097152`. With `kv_unified`, layer 0's K cache
  tensor at that size is `1024 * 2097152 * 2 bytes = 4294967296` -- exactly
  one byte over Vulkan's `maxStorageBufferRange` (`4294967295` on this
  GPU). `ggml_backend_vk_device_supports_op` correctly rejects the op for
  being oversized, `ggml_backend_sched_backend_id_from_cur` finds no backend
  willing to run it, and hits the "pre-allocated tensor ... that cannot run
  the operation" `GGML_ABORT` (`ggml-backend.cpp:941`) -- a hard process
  abort, not a catchable error, the first time that KV cache tensor is
  scheduled (traced live with gdb: `ggml_backend_supports_buft` returns true
  for the GPU backend, `ggml_backend_supports_op` is what returns false).
  Same mechanism would hit any backend with a per-op size ceiling and any
  model whose `n_ctx_train * n_parallel` crosses it; Vulkan/this GPU is just
  where the numbers lined up. Fixed two places:
  - `src/llama-kv-cache.cpp`: check `ggml_backend_dev_supports_op` on each
    layer's K/V tensor right after creating it, before it becomes a
    pre-allocated leaf the scheduler can hard-abort on. Throws a normal,
    descriptive `std::runtime_error` instead (already caught by
    `llama_init_from_model`'s existing try/catch, which logs it and returns
    `nullptr` -- this alone turns the abort into a clean failed-to-load
    instead of a crash, for both `--fit`'s probe and a real model load).
  - `common/fit.cpp`: `common_params_fit_impl`'s auto-context measurement
    (`n_ctx_auto && n_seq_max > 1`) now catches that failure and backs off
    geometrically (halving, floored at `n_ctx_min_total`) until the probe
    succeeds, so `-fit` actually finds a working context size instead of
    just failing to start. Verified: K2-Horizon-7B and Qwen3.5-4B both start
    clean on default flags (no `-c`, no `--device`, no `-fit off`) after the
    fix; K2-Horizon-7B's probe now logs one backoff (2097152 -> 1048576) and
    settles on a real `n_ctx` sized by the existing free-memory-margin logic
    (unchanged). `ctest -L main` (59/59) and `ctest -R split-balance` (1/1)
    both pass.
