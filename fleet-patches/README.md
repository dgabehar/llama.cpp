# Fleet patches (mirror, not source of truth)

This directory is a **disk-backed mirror** of the commits this fork carries on
top of `upstream/master`, generated with `git format-patch`. The `fleet-patches`
branch itself (its git history) is still the source of truth -- these files
exist so a patch can be inspected, diffed, or reapplied without a working
clone of the branch, and so a rebase conflict has a static reference to work
from instead of only `git show <sha>` on a branch that's mid-rebase.

**Regenerate after any change to the fork-exclusive commit set** (a new patch,
a squash, a rebase that changes commit hashes):

```sh
./fleet-patches/regenerate.sh
```

**Reapply onto a fresh checkout** (disaster recovery, or rebuilding
`fleet-patches` from scratch against a newer `upstream/master`):

```sh
git checkout -b fleet-patches upstream/master
git am fleet-patches/*.patch
```

If `git am` conflicts on a patch, that means upstream changed the same code
this patch touches -- resolve it the same way any rebase conflict is
resolved (read the patch's own commit message for *why* the change exists,
re-derive the equivalent change against the new upstream code, `git am
--continue`).

## Current patches (regenerated 2026-10-07, rebased onto upstream `4625240437` = tag b11454)

| # | Commit | Summary |
|---|---|---|
| 0001 | `54b555076` | grammar: translate PCRE shorthand escapes (\d \w \s) to GBNF classes |
| 0002 | `9a97b6f0f` | server: fix slot save/restore losing checkpoint-based cache reuse |
| 0003 | `d77152d47` | common: guard checkpoint update_dft against an empty draft sequence |
| 0004 | `805a22896` | ci: add sync-with-upstream workflow (mirror of the master copy) |
| 0005 | `c9f337434` | fix: repair YAML block-scalar indentation bug in sync workflow |
| 0006 | `8b9d76cba` | server: include id_slot in OAI-compatible chat completion responses |
| 0007 | `7f64d1946` | ggml: add GGML_OP_SSM_CONV_SPLIT (CPU+Vulkan) to skip the per-layer conv-state concat |
| 0008 | `5bf774527` | fleet: mirror fork patches to disk, document fleet workflow separately |
| 0009 | `2cdf6bce0` | server: disable speculative decoding for grammar-constrained requests |
| 0010 | `2d13e476c` | qwen3-coder: bound xml-arg-string with until_one_of, not a bare until |
| 0011 | `9a214a091` | vulkan: occupancy-aware S/M/L tile selector for non-coopmat2 path (AMD/Intel) |
| 0012 | `4fefb9f07` | server: end generation cleanly on a tool-call-diff inconsistency |
| 0013 | `f786fca14` | peg-parser: don't prematurely close an until_one_of span on an ambiguous partial delimiter match |
| 0014 | `6e8a9cd25` | server: bound slot-action queue wait, and explain empty child-slot saves |
| 0015 | `101cf3f9b` | ggml-cpu: extend CPU backend device registry to N locality domains |
| 0016 | `2b618a1ca` | common: new --cpu-split auto\|off flag with startup throughput calibration |
| 0017 | `db1d1af9a` | llama-model: weighted per-layer CPU split across locality domains |
| 0018 | `ad2999e8b` | llama-context: make backend_cpu locality-domain aware |
| 0019 | `ab80c5338` | tests: cover the CPU locality-domain split |
| 0020 | `927b1f540` | docs: document --cpu-split and the CPU locality-domain split feature |
| 0021 | `a30d9adbb` | fix: route ggml_backend_cpu_device_get_locality_mask through the DL proc-address table |
| 0022 | `de98be5d4` | fix: don't reject a GPU's host buffer type in ggml_backend_cpu_device_supports_buft |
| 0023 | `d1d8de372` | =?UTF-8?q?fix:=20correct=20units=20in=20CPU=20split=20cal?= =?UTF-8?q?ibration=20log=20line=20(=C2=B5s,=20not=20seconds)?= |
| 0024 | `15c56e32c` | tests: skip test-cpu-device-split on non-Linux platforms |
| 0025 | `e3089ca40` | tests: fix stale expected output in the shorthand-class regexp test |
| 0026 | `d87cfc01c` | tests: loosen slot-action-timeout test's join timeout margin |
| 0027 | `66de0db9e` | rpc : serve each client connection on its own thread |
| 0028 | `c75c3cd6c` | rpc : serialize backend compute across connection threads |
| 0029 | `15760f8bd` | rpc : per-connection backends, tracked connections, client cap, keepalive |
| 0030 | `b4139e80f` | rpc : keep one bad client from taking down the whole server |
| 0031 | `55d72f3c6` | tests : add test-rpc-server-multiclient |
| 0032 | `fb010047f` | rpc : don't reject a client because a just-closed probe still holds a slot |
| 0033 | `cd4c427b0` | vulkan : size submit batches from the current graph's flops |
| 0034 | `d6eefe221` | llama : restore the worst-case sched plan before large ubatches |
| 0035 | `31f5f3e49` | common : --split-balance for speed-aware layer splits |
| 0036 | `fc93bb21f` | llama, server : capture context checkpoints inside a batch |
| 0037 | `24066e0a8` | common : split-balance: time the dominant weight type, model n_batch drains |
| 0038 | `0e2213c42` | vulkan : keep the flop-sized submits of a graph below the kernel job queue |
| 0039 | `93791945b` | common : split-balance: model imperfect stage overlap in prefill |
| 0040 | `69519c153` | common : split-balance: key the calibration cache by build, expire after 7 days |
| 0041 | `3912cba48` | split-balance: ignore overrides that match nothing, validate and re-check the cache, calibrated overheads |
| 0042 | `e903b4d2c` | server: only an explicit off value disables LLAMA_SERVER_CKPT_CAPTURE |
| 0043 | `cee0edd5b` | rpc: take a client slot at HELLO, bound the handshake, report a full server |
| 0044 | `8c72d93eb` | rpc: say that the connection to the server was lost when a request fails |
| 0045 | `5d7c6d6b1` | vulkan: on integrated GPUs, cap the free memory of host heaps at MemAvailable |
| 0046 | `5e48eaeee` | split-balance: re-check local devices' cached timings too |
| 0047 | `c40b9251b` | vulkan: find the host-RAM heap of an integrated GPU by size |
| 0048 | `7952bc165` | llama: cheaper state captures: no host-side materialization, no context sync |
| 0049 | `280c80db9` | llama: add llama_model_n_devices_used |
| 0050 | `3d4d1bc31` | server: capture checkpoints in-batch only across devices, read them after the next decode |
| 0051 | `47c315fb0` | llama: direct I/O into memory that refuses DMA goes through a bounce buffer |
| 0052 | `f22d7416e` | split-balance: time MoE models with routed expert matmuls, reject inconsistent entries |
| 0053 | `f94f3d1a3` | server tests: keep the slot busy longer than the save timeout on fast hardware |
| 0054 | `0359ec5d4` | models : fix the conv-state window over the transposed new tokens |
| 0055 | `3fa3818b9` | vulkan: cap an integrated GPU's total free memory at MemAvailable |
| 0056 | `fd103a519` | tests : rpc-server multiclient: cast RAND_MAX to float |
| 0057 | `f2228c0f2` | common : split-balance: calibrate on the model's own layers |
| 0058 | `c48dec9c9` | common : split-balance: fix the calibration use-after-free, detect busy nodes against a reference |
| 0059 | `a942dea5d` | common : split-balance: survive malformed cache entries, log why an entry is not used, merge on write |
| 0060 | `c74e0716a` | server, chat : stop generating at a stray K2-Horizon close tag; keep reasoning_format none raw |
| 0061 | `b7962aa96` | server : a captured checkpoint may outlive its task: don't read slot.task for it |
| 0062 | `df5e75daf` | server : don't update the prompt cache of a busy slot picked by id_slot |
| 0063 | `0cd50798c` | common : split-balance: scale prefill by byte_scale too, not just decode |
| 0064 | `9730b2f68` | common : split-balance: fix calibration overrating a device under per-node Vulkan submits |
| 0065 | `b95797a74` | chat : drop whitespace between reasoning end tag and content |
| 0066 | `f4bd6113a` | llama, common: fix -fit's oversized-context probe hard-aborting the server |
| 0067 | `98685de0f` | common: size -fit's auto-context probe from measured free memory, not just a caught exception |
| 0068 | `409cb1006` | common: close the n_seq_max == 1 gap in --fit's auto-context probe, cap iGPU free memory consistently |
| 0069 | `b0f3623c8` | common : split-balance: calibrate a layer's decode ops long enough to pay the real per-submit cost, not just its own norms/RoPE nodes |
| 0070 | `06b374ae2` | ggml-rpc: give the RPC client a bounded receive timeout and keepalive |
| 0071 | `532a6fc25` | chat, sampling: K2-Horizon official weights: tool call in reasoning, repeated open tag, empty first-token stop |
| 0072 | `857323a2a` | chat: gpt-oss parser accepts the "commentary (analysis)" analysis header |
| 0073 | `34a52d11a` | sampling: K2-Horizon: empty-reply guard also holds EOG inside an unfinished Kimi-style tool-call opener |
| 0074 | `049528cf2` | sampling, chat: K2-Horizon: hold EOG from a tool-call opener until the section is closed and its inner tags balance |
| 0075 | `d1ed6605d` | sampling: K2-Horizon: Kimi hold gets its own small cap and releases on a repeated opener |
| 0076 | `414c5e80a` | chat: K2-Horizon: render history reasoning with the generation tag, strip foreign markup, drop emptied turns |
| 0077 | `7f07a1b06` | sampling: K2-Horizon: mask the structural tool-call tags while no native section is open |
| 0078 | `1a9f36e79` | sampling: empty-reply guard/tag mask: O(1) per-candidate masking |
| 0079 | `3aaaa53af` | chat: K2-Horizon parser gaps (all reasoning efforts, bare/Kimi calls, stray tags, never throw) |
| 0080 | `7afa013d3` | K2: hold EOG inside an unclosed reasoning block (tools offered); strip only known K2/Kimi markup from replayed reasoning |
| 0081 | `faf40ac07` | K2 parser: surface a tool call only once its arguments are complete JSON |
| 0082 | `430ec4572` | K2 guard: turn an end of turn inside an unclosed reasoning block into the block's close tag |
| 0083 | `6c00e84f1` | K2 guard: repetition breaker for an open reasoning block |
| 0084 | `cec58d8b2` | llama : fail restoring a truncated sequence state file instead of aborting |
| 0085 | `6cc2ec938` | chat: K2-Horizon: port the fleet's parser tolerance onto upstream's k2-horizon.cpp |
| 0086 | `0619cebbf` | tests: K2-Horizon: port the fleet's parser, history and guard-field tests |
| 0087 | `9473da157` | fleet: rebuild patch mirror and FLEET.md for the K2-Horizon adoption sync (base 4625240437, b11454) |
| 0088 | `2f1949ec8` | chat: K2-Horizon: startup self-check that the sampler guard is fed |

The `split-balance` / `vulkan: size submit batches` patches are the speed-aware
split and pipeline fixes (see FLEET.md, "Speed-aware split and pipeline
fixes"). Fork-only, not submitted upstream.

This mirror was rebuilt on 2026-10-07 for the K2-Horizon adoption sync (88
patches; 12 K2 model/conversion/tokenizer patches were dropped or replaced by
upstream's, see FLEET.md "Upstream sync 2026-10 K2 adoption"). Patch numbers
changed; refer to patches by subject.

See `~/src/llama.cpp/CLAUDE.md`'s "Fleet patch maintenance" section for the
full workflow this fits into (weekly upstream rebase, when to add a new
patch vs. amend an existing one, the `sync-with-upstream` CI workflow).
