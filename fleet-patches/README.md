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

## Current patches (regenerated 2026-10-17, merged with upstream `ea0c4c2c8`; the slot-save checkpoint patch is superseded by upstream and no longer mirrored)

| # | Commit | Summary |
|---|---|---|
| 0001 | `54b555076` | grammar: translate PCRE shorthand escapes (\d \w \s) to GBNF classes |
| 0002 | `d77152d47` | common: guard checkpoint update_dft against an empty draft sequence |
| 0003 | `805a22896` | ci: add sync-with-upstream workflow (mirror of the master copy) |
| 0004 | `c9f337434` | fix: repair YAML block-scalar indentation bug in sync workflow |
| 0005 | `8b9d76cba` | server: include id_slot in OAI-compatible chat completion responses |
| 0006 | `7f64d1946` | ggml: add GGML_OP_SSM_CONV_SPLIT (CPU+Vulkan) to skip the per-layer conv-state concat |
| 0007 | `5bf774527` | fleet: mirror fork patches to disk, document fleet workflow separately |
| 0008 | `2cdf6bce0` | server: disable speculative decoding for grammar-constrained requests |
| 0009 | `2d13e476c` | qwen3-coder: bound xml-arg-string with until_one_of, not a bare until |
| 0010 | `9a214a091` | vulkan: occupancy-aware S/M/L tile selector for non-coopmat2 path (AMD/Intel) |
| 0011 | `4fefb9f07` | server: end generation cleanly on a tool-call-diff inconsistency |
| 0012 | `f786fca14` | peg-parser: don't prematurely close an until_one_of span on an ambiguous partial delimiter match |
| 0013 | `6e8a9cd25` | server: bound slot-action queue wait, and explain empty child-slot saves |
| 0014 | `101cf3f9b` | ggml-cpu: extend CPU backend device registry to N locality domains |
| 0015 | `2b618a1ca` | common: new --cpu-split auto\|off flag with startup throughput calibration |
| 0016 | `db1d1af9a` | llama-model: weighted per-layer CPU split across locality domains |
| 0017 | `ad2999e8b` | llama-context: make backend_cpu locality-domain aware |
| 0018 | `ab80c5338` | tests: cover the CPU locality-domain split |
| 0019 | `927b1f540` | docs: document --cpu-split and the CPU locality-domain split feature |
| 0020 | `a30d9adbb` | fix: route ggml_backend_cpu_device_get_locality_mask through the DL proc-address table |
| 0021 | `de98be5d4` | fix: don't reject a GPU's host buffer type in ggml_backend_cpu_device_supports_buft |
| 0022 | `d1d8de372` | =?UTF-8?q?fix:=20correct=20units=20in=20CPU=20split=20cal?= =?UTF-8?q?ibration=20log=20line=20(=C2=B5s,=20not=20seconds)?= |
| 0023 | `15c56e32c` | tests: skip test-cpu-device-split on non-Linux platforms |
| 0024 | `e3089ca40` | tests: fix stale expected output in the shorthand-class regexp test |
| 0025 | `d87cfc01c` | tests: loosen slot-action-timeout test's join timeout margin |
| 0026 | `66de0db9e` | rpc : serve each client connection on its own thread |
| 0027 | `c75c3cd6c` | rpc : serialize backend compute across connection threads |
| 0028 | `15760f8bd` | rpc : per-connection backends, tracked connections, client cap, keepalive |
| 0029 | `b4139e80f` | rpc : keep one bad client from taking down the whole server |
| 0030 | `55d72f3c6` | tests : add test-rpc-server-multiclient |
| 0031 | `fb010047f` | rpc : don't reject a client because a just-closed probe still holds a slot |
| 0032 | `cd4c427b0` | vulkan : size submit batches from the current graph's flops |
| 0033 | `d6eefe221` | llama : restore the worst-case sched plan before large ubatches |
| 0034 | `31f5f3e49` | common : --split-balance for speed-aware layer splits |
| 0035 | `fc93bb21f` | llama, server : capture context checkpoints inside a batch |
| 0036 | `24066e0a8` | common : split-balance: time the dominant weight type, model n_batch drains |
| 0037 | `0e2213c42` | vulkan : keep the flop-sized submits of a graph below the kernel job queue |
| 0038 | `93791945b` | common : split-balance: model imperfect stage overlap in prefill |
| 0039 | `69519c153` | common : split-balance: key the calibration cache by build, expire after 7 days |
| 0040 | `3912cba48` | split-balance: ignore overrides that match nothing, validate and re-check the cache, calibrated overheads |
| 0041 | `e903b4d2c` | server: only an explicit off value disables LLAMA_SERVER_CKPT_CAPTURE |
| 0042 | `cee0edd5b` | rpc: take a client slot at HELLO, bound the handshake, report a full server |
| 0043 | `8c72d93eb` | rpc: say that the connection to the server was lost when a request fails |
| 0044 | `5d7c6d6b1` | vulkan: on integrated GPUs, cap the free memory of host heaps at MemAvailable |
| 0045 | `5e48eaeee` | split-balance: re-check local devices' cached timings too |
| 0046 | `c40b9251b` | vulkan: find the host-RAM heap of an integrated GPU by size |
| 0047 | `7952bc165` | llama: cheaper state captures: no host-side materialization, no context sync |
| 0048 | `280c80db9` | llama: add llama_model_n_devices_used |
| 0049 | `3d4d1bc31` | server: capture checkpoints in-batch only across devices, read them after the next decode |
| 0050 | `47c315fb0` | llama: direct I/O into memory that refuses DMA goes through a bounce buffer |
| 0051 | `f22d7416e` | split-balance: time MoE models with routed expert matmuls, reject inconsistent entries |
| 0052 | `f94f3d1a3` | server tests: keep the slot busy longer than the save timeout on fast hardware |
| 0053 | `0359ec5d4` | models : fix the conv-state window over the transposed new tokens |
| 0054 | `3fa3818b9` | vulkan: cap an integrated GPU's total free memory at MemAvailable |
| 0055 | `fd103a519` | tests : rpc-server multiclient: cast RAND_MAX to float |
| 0056 | `f2228c0f2` | common : split-balance: calibrate on the model's own layers |
| 0057 | `c48dec9c9` | common : split-balance: fix the calibration use-after-free, detect busy nodes against a reference |
| 0058 | `a942dea5d` | common : split-balance: survive malformed cache entries, log why an entry is not used, merge on write |
| 0059 | `c74e0716a` | server, chat : stop generating at a stray K2-Horizon close tag; keep reasoning_format none raw |
| 0060 | `b7962aa96` | server : a captured checkpoint may outlive its task: don't read slot.task for it |
| 0061 | `df5e75daf` | server : don't update the prompt cache of a busy slot picked by id_slot |
| 0062 | `0cd50798c` | common : split-balance: scale prefill by byte_scale too, not just decode |
| 0063 | `9730b2f68` | common : split-balance: fix calibration overrating a device under per-node Vulkan submits |
| 0064 | `b95797a74` | chat : drop whitespace between reasoning end tag and content |
| 0065 | `f4bd6113a` | llama, common: fix -fit's oversized-context probe hard-aborting the server |
| 0066 | `98685de0f` | common: size -fit's auto-context probe from measured free memory, not just a caught exception |
| 0067 | `409cb1006` | common: close the n_seq_max == 1 gap in --fit's auto-context probe, cap iGPU free memory consistently |
| 0068 | `b0f3623c8` | common : split-balance: calibrate a layer's decode ops long enough to pay the real per-submit cost, not just its own norms/RoPE nodes |
| 0069 | `06b374ae2` | ggml-rpc: give the RPC client a bounded receive timeout and keepalive |
| 0070 | `532a6fc25` | chat, sampling: K2-Horizon official weights: tool call in reasoning, repeated open tag, empty first-token stop |
| 0071 | `857323a2a` | chat: gpt-oss parser accepts the "commentary (analysis)" analysis header |
| 0072 | `34a52d11a` | sampling: K2-Horizon: empty-reply guard also holds EOG inside an unfinished Kimi-style tool-call opener |
| 0073 | `049528cf2` | sampling, chat: K2-Horizon: hold EOG from a tool-call opener until the section is closed and its inner tags balance |
| 0074 | `d1ed6605d` | sampling: K2-Horizon: Kimi hold gets its own small cap and releases on a repeated opener |
| 0075 | `414c5e80a` | chat: K2-Horizon: render history reasoning with the generation tag, strip foreign markup, drop emptied turns |
| 0076 | `7f07a1b06` | sampling: K2-Horizon: mask the structural tool-call tags while no native section is open |
| 0077 | `1a9f36e79` | sampling: empty-reply guard/tag mask: O(1) per-candidate masking |
| 0078 | `3aaaa53af` | chat: K2-Horizon parser gaps (all reasoning efforts, bare/Kimi calls, stray tags, never throw) |
| 0079 | `7afa013d3` | K2: hold EOG inside an unclosed reasoning block (tools offered); strip only known K2/Kimi markup from replayed reasoning |
| 0080 | `faf40ac07` | K2 parser: surface a tool call only once its arguments are complete JSON |
| 0081 | `430ec4572` | K2 guard: turn an end of turn inside an unclosed reasoning block into the block's close tag |
| 0082 | `6c00e84f1` | K2 guard: repetition breaker for an open reasoning block |
| 0083 | `cec58d8b2` | llama : fail restoring a truncated sequence state file instead of aborting |
| 0084 | `6cc2ec938` | chat: K2-Horizon: port the fleet's parser tolerance onto upstream's k2-horizon.cpp |
| 0085 | `0619cebbf` | tests: K2-Horizon: port the fleet's parser, history and guard-field tests |
| 0086 | `2f1949ec8` | chat: K2-Horizon: startup self-check that the sampler guard is fed |
| 0087 | `44d4952e1` | K2-Horizon/server: fix tool_choice none content leak, bound call args, reject bad slot ids |
| 0088 | `5d9445b34` | sampling: value-initialize the empty-reply guard (fixes -Werror=missing-field-initializers on gcc arm64 CI) |
| 0089 | `ef002595f` | ci: remove the disabled sync-fleet-patches workflow |
| 0090 | `c291b7775` | tests: slot-action timeout test runs a 6000-token busy request (12000 outran its 90 s HTTP timeout on the arm64 runner) |
| 0091 | `3d8731b2d` | tests: test-state-capture tolerates by-design re-reserve under GGML_SCHED_NO_REALLOC |
| 0092 | `8b22e1515` | tests: Windows-portable env set in test-state-capture; retry noisy split-balance ratio on busy runners |
| 0093 | `c351de77a` | tests: skip split-balance op-count ratio check when calibration is throttled (hosted macOS runners) |
| 0094 | `dc4d6d84d` | tests: render the K2 parser's chat template once per make_peg_parser, not once per parse |
| 0095 | `4c17b1108` | qwen4exp: use ggml_ssm_conv_split for the GDN conv, dropping its concat |
| 0096 | `ec0884436` | vulkan: ssm_conv_split nc==4 uses the same dot() reduction as ssm_conv, for bit-identical output |
| 0097 | `a6193d8f4` | vulkan: shared-memory tile path for concat of a transposed src1 along dim 0 |
| 0098 | `b9e449fdc` | vulkan/tests: guard ssm_conv_split d_conv <= 16, widen concat_transposed coverage |
| 0099 | `8120c3ced` | chat: tool_choice none no longer returns a raw tool call as content (Qwen3.x) |
| 0100 | `64a4084b3` | chat: tool_choice none drops a call for every tag-delimited template, in one place |
| 0101 | `727dd3ead` | chat: tool_choice none keeps the none parser's plain replies and cuts rejected calls |

The `split-balance` / `vulkan: size submit batches` patches are the speed-aware
split and pipeline fixes (see FLEET.md, "Speed-aware split and pipeline
fixes"). Fork-only, not submitted upstream.

This mirror was rebuilt on 2026-10-07 for the K2-Horizon adoption sync and extended on 2026-10-08
with the K2 fixes (0088, 0089) and the sync-workflow removal (0090), then on 2026-10-22 with the merged test/vulkan patches (0091-0098)
and the tool_choice none fix (0099-0101): 101 patches (87 from the adoption sync; 12 K2 model/conversion/tokenizer patches were dropped or replaced by
upstream's, see FLEET.md "Upstream sync 2026-10 K2 adoption"). Patch numbers
changed; refer to patches by subject.

See `~/src/llama.cpp/CLAUDE.md`'s "Fleet patch maintenance" section for the
full workflow this fits into (when to add a new patch vs. amend an existing one; upstream syncs are manual merges since the
K2 adoption, the `sync-with-upstream` CI workflow is disabled).
