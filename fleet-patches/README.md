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

## Current patches (regenerated 2026-10-06, rebased onto upstream `4f5406761` = tag b11444)

| # | Commit | Summary |
|---|---|---|
| 0001 | `b416db4f3` | grammar: translate PCRE shorthand escapes (\d \w \s) to GBNF classes |
| 0002 | `54b9a9e2d` | server: fix slot save/restore losing checkpoint-based cache reuse |
| 0003 | `6fcd12020` | common: guard checkpoint update_dft against an empty draft sequence |
| 0004 | `9f9044d38` | ci: add sync-with-upstream workflow (mirror of the master copy) |
| 0005 | `c4fffee0a` | fix: repair YAML block-scalar indentation bug in sync workflow |
| 0006 | `9849c0935` | server: include id_slot in OAI-compatible chat completion responses |
| 0007 | `b91aa1bae` | ggml: add GGML_OP_SSM_CONV_SPLIT (CPU+Vulkan) to skip the per-layer conv-state concat |
| 0008 | `d5a1ba215` | fleet: mirror fork patches to disk, document fleet workflow separately |
| 0009 | `3e7b73528` | server: disable speculative decoding for grammar-constrained requests |
| 0010 | `7bc133cd9` | qwen3-coder: bound xml-arg-string with until_one_of, not a bare until |
| 0011 | `be29ab3c5` | vulkan: occupancy-aware S/M/L tile selector for non-coopmat2 path (AMD/Intel) |
| 0012 | `76c6c1970` | model: K2 Horizon gguf conversion code |
| 0013 | `06e845fee` | model: loading hparams and tensors in k2-horizon.cpp |
| 0014 | `1b9e56f70` | model: K2 Horizon compute graph |
| 0015 | `b0d2eb7b9` | model: K2 Horizon compute graph adjustment and registering tokenizers |
| 0016 | `1a9c34cce` | model: K2 Horizon chat template and accomodate safetensors naming |
| 0017 | `366723b6a` | k2-horizon: adapt to hparams.n_ff_exp API drift since fork point |
| 0018 | `bc9abe13a` | server: end generation cleanly on a tool-call-diff inconsistency |
| 0019 | `16be7a5ea` | peg-parser: don't prematurely close an until_one_of span on an ambiguous partial delimiter match |
| 0020 | `fcbf32673` | server: bound slot-action queue wait, and explain empty child-slot saves |
| 0021 | `ea97ff428` | ggml-cpu: extend CPU backend device registry to N locality domains |
| 0022 | `f0ac5853e` | common: new --cpu-split auto\|off flag with startup throughput calibration |
| 0023 | `cd404aaa0` | llama-model: weighted per-layer CPU split across locality domains |
| 0024 | `adda6485c` | llama-context: make backend_cpu locality-domain aware |
| 0025 | `d12bbb408` | tests: cover the CPU locality-domain split |
| 0026 | `d4258c007` | docs: document --cpu-split and the CPU locality-domain split feature |
| 0027 | `5a047c591` | fix: route ggml_backend_cpu_device_get_locality_mask through the DL proc-address table |
| 0028 | `1e574238c` | fix: don't reject a GPU's host buffer type in ggml_backend_cpu_device_supports_buft |
| 0029 | `b2fcf83bf` | =?UTF-8?q?fix:=20correct=20units=20in=20CPU=20split=20cal?= =?UTF-8?q?ibration=20log=20line=20(=C2=B5s,=20not=20seconds)?= |
| 0030 | `1c5a47df5` | unicode : add the K2-Horizon pre-tokenizer splitter |
| 0031 | `d07181073` | tests: expand K2 Horizon unicode splitter coverage |
| 0032 | `1ebc2131b` | unicode: handle K2 Horizon case folding and empty input |
| 0033 | `97c77dadd` | vocab: recognize K2-Horizon's <\|ifm\|im_end\|> as an EOG token |
| 0034 | `9a99f9bef` | chat: add K2-Horizon workaround for reasoning-tag auto-detection |
| 0035 | `5e093a4d3` | tests: skip test-cpu-device-split on non-Linux platforms |
| 0036 | `840ee5bfd` | tests: fix stale expected output in the shorthand-class regexp test |
| 0037 | `d3837b205` | tests: loosen slot-action-timeout test's join timeout margin |
| 0038 | `bd9d4a950` | conversion: fix flake8 lint violations in k2_horizon.py |
| 0039 | `414817e16` | rpc : serve each client connection on its own thread |
| 0040 | `5bacb855d` | rpc : serialize backend compute across connection threads |
| 0041 | `afa73b359` | rpc : per-connection backends, tracked connections, client cap, keepalive |
| 0042 | `150031531` | rpc : keep one bad client from taking down the whole server |
| 0043 | `de64ed90e` | tests : add test-rpc-server-multiclient |
| 0044 | `a37aafb0d` | rpc : don't reject a client because a just-closed probe still holds a slot |
| 0045 | `f5af52c9f` | vulkan : size submit batches from the current graph's flops |
| 0046 | `4d7bde79f` | llama : restore the worst-case sched plan before large ubatches |
| 0047 | `99a069a61` | common : --split-balance for speed-aware layer splits |
| 0048 | `aa300693c` | llama, server : capture context checkpoints inside a batch |
| 0049 | `f9e705a2c` | common : split-balance: time the dominant weight type, model n_batch drains |
| 0050 | `0710f3b18` | vulkan : keep the flop-sized submits of a graph below the kernel job queue |
| 0051 | `7a8a4bff3` | common : split-balance: model imperfect stage overlap in prefill |
| 0052 | `2f4af6d45` | common : split-balance: key the calibration cache by build, expire after 7 days |
| 0053 | `4797e21fb` | split-balance: ignore overrides that match nothing, validate and re-check the cache, calibrated overheads |
| 0054 | `4de5bbf4e` | server: only an explicit off value disables LLAMA_SERVER_CKPT_CAPTURE |
| 0055 | `42d3a854d` | rpc: take a client slot at HELLO, bound the handshake, report a full server |
| 0056 | `be01639e0` | rpc: say that the connection to the server was lost when a request fails |
| 0057 | `333cce375` | vulkan: on integrated GPUs, cap the free memory of host heaps at MemAvailable |
| 0058 | `a45e1bcac` | split-balance: re-check local devices' cached timings too |
| 0059 | `730fae701` | vulkan: find the host-RAM heap of an integrated GPU by size |
| 0060 | `f5d7c1521` | llama: cheaper state captures: no host-side materialization, no context sync |
| 0061 | `5bff21b80` | llama: add llama_model_n_devices_used |
| 0062 | `54920cfed` | server: capture checkpoints in-batch only across devices, read them after the next decode |
| 0063 | `ab1268a2f` | llama: direct I/O into memory that refuses DMA goes through a bounce buffer |
| 0064 | `cd99ff7e5` | split-balance: time MoE models with routed expert matmuls, reject inconsistent entries |
| 0065 | `d06735f4c` | server tests: keep the slot busy longer than the save timeout on fast hardware |
| 0066 | `b9b9ecbd6` | models : fix the conv-state window over the transposed new tokens |
| 0067 | `3f9b80c45` | vulkan: cap an integrated GPU's total free memory at MemAvailable |
| 0068 | `71719479e` | tests : rpc-server multiclient: cast RAND_MAX to float |
| 0069 | `8410a16c3` | common : split-balance: calibrate on the model's own layers |
| 0070 | `5916940d6` | chat : K2-Horizon: accept any effort's close tag as the end of reasoning |
| 0071 | `723dd204c` | common : split-balance: fix the calibration use-after-free, detect busy nodes against a reference |
| 0072 | `527cd259d` | chat : K2-Horizon: end content at a close tag after the answer |
| 0073 | `d93fef38a` | common : split-balance: survive malformed cache entries, log why an entry is not used, merge on write |
| 0074 | `7e2b81517` | server, chat : stop generating at a stray K2-Horizon close tag; keep reasoning_format none raw |
| 0075 | `7b1dd205c` | server : a captured checkpoint may outlive its task: don't read slot.task for it |
| 0076 | `b2e0794e1` | server : don't update the prompt cache of a busy slot picked by id_slot |
| 0077 | `9768fff5b` | chat : K2-Horizon: don't leak tool-call markup into content with no tools |
| 0078 | `cd6ab339c` | chat : K2-Horizon tests and comment: use a neutral example hostname |
| 0079 | `45775961c` | common : split-balance: scale prefill by byte_scale too, not just decode |
| 0080 | `121e417b2` | common : split-balance: fix calibration overrating a device under per-node Vulkan submits |
| 0081 | `599dd43b0` | chat : drop whitespace between reasoning end tag and content |
| 0082 | `d260d6288` | llama, common: fix -fit's oversized-context probe hard-aborting the server |
| 0083 | `442f1b2dd` | common: size -fit's auto-context probe from measured free memory, not just a caught exception |
| 0084 | `80783c14b` | common: close the n_seq_max == 1 gap in --fit's auto-context probe, cap iGPU free memory consistently |
| 0085 | `591d66a36` | common : split-balance: calibrate a layer's decode ops long enough to pay the real per-submit cost, not just its own norms/RoPE nodes |
| 0086 | `1ead2878a` | ggml-rpc: give the RPC client a bounded receive timeout and keepalive |
| 0087 | `3c994840e` | chat, sampling: K2-Horizon official weights: tool call in reasoning, repeated open tag, empty first-token stop |
| 0088 | `6e18dc416` | chat: gpt-oss parser accepts the "commentary (analysis)" analysis header |
| 0089 | `19761cacb` | sampling: K2-Horizon: empty-reply guard also holds EOG inside an unfinished Kimi-style tool-call opener |
| 0090 | `e6c28fdfa` | sampling, chat: K2-Horizon: hold EOG from a tool-call opener until the section is closed and its inner tags balance |
| 0091 | `1fd293626` | sampling: K2-Horizon: Kimi hold gets its own small cap and releases on a repeated opener |
| 0092 | `bdb2125fc` | chat: K2-Horizon: render history reasoning with the generation tag, strip foreign markup, drop emptied turns |
| 0093 | `a2ae8f321` | sampling: K2-Horizon: mask the structural tool-call tags while no native section is open |
| 0094 | `876e04668` | sampling: empty-reply guard/tag mask: O(1) per-candidate masking |
| 0095 | `5bfb85f86` | chat: K2-Horizon parser gaps (all reasoning efforts, bare/Kimi calls, stray tags, never throw) |
| 0096 | `65d46da14` | K2: hold EOG inside an unclosed reasoning block (tools offered); strip only known K2/Kimi markup from replayed reasoning |
| 0097 | `a2868918e` | K2 parser: surface a tool call only once its arguments are complete JSON |
| 0098 | `71e17322d` | K2 parser: accept the tool arguments in any order |
| 0099 | `640423072` | K2 guard: turn an end of turn inside an unclosed reasoning block into the block's close tag |
| 0100 | `5074d0df4` | K2 guard: repetition breaker for an open reasoning block |

0030-0032 are cherry-picked from `MBZUAI-IFM/llama.cpp@model/K2Horizon`
(commits `69d3a4e82`/`a8104b553`/`e78bd9435` there). They fix an MSVC `std::regex`
crash that blocked K2-Horizon GGUFs from loading on Windows, plus
ZWNJ/ZWJ/case-folding/empty-input edge cases in the pre-tokenizer word
splitter. They do **not** touch BPE merge-table granularity.

0012-0017 are cherry-picked from the vendor's own architecture-support branch
(`MBZUAI-IFM/llama.cpp@model/K2Horizon`) -- model-definition/conversion/vocab
layer only, no backend code touched; 0017 is this fork's own fix for API drift
in `llama_hparams::n_ff_exp`. The K2-Horizon pre-tokenizer enum is
`LLAMA_VOCAB_PRE_TYPE_K2_HORIZON = 61` on this base (upstream took 60 for
MMBERT); GGUFs carry the pre-tokenizer as the string `k2-horizon`, so files are
unaffected by the renumbering.

**K2-Horizon upstream status:** upstream PR
[ggml-org/llama.cpp#29535](https://github.com/ggml-org/llama.cpp/pull/29535)
("model : add K2 Horizon dense and MoVA support") is open as of 2026-10-06 and
tracking issue [#28361](https://github.com/ggml-org/llama.cpp/issues/28361).
If #29535 merges, replace the K2 model/conversion/vocab patches with upstream's
and keep only the chat/sampling stack.

0039-0040 are cherry-picked from upstream PR
[ggml-org/llama.cpp#28916](https://github.com/ggml-org/llama.cpp/pull/28916)
(rpc-server serving only one client at a time), original authorship kept; the
other rpc patches (0041-0044, 0055-0056, 0086) build on them (see FLEET.md,
"rpc-server multi-client"). If #28916 merges upstream, drop 0039-0040 on the
next rebase and resolve the rest against whatever shape upstream landed.

The `split-balance` / `vulkan: size submit batches` patches are the speed-aware
split and pipeline fixes (see FLEET.md, "Speed-aware split and pipeline
fixes"). Fork-only, not submitted upstream.

This mirror was rebuilt from scratch on 2026-10-06 for the upstream sync
(100 patches; 3 earlier patches were dropped, see FLEET.md "Upstream sync
2026-10"). Patch numbers changed; refer to patches by subject.

See `~/src/llama.cpp/CLAUDE.md`'s "Fleet patch maintenance" section for the
full workflow this fits into (weekly upstream rebase, when to add a new
patch vs. amend an existing one, the `sync-with-upstream` CI workflow).
