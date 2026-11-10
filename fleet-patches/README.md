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

## Current patches (regenerated 2026-11-10; fleet-patches merged with upstream `202c1a229` on 2026-11-10, see FLEET.md "Upstream sync 2026-11-10")

| # | Commit | Summary |
|---|---|---|
| 0001 | `4449b934f` | grammar: translate PCRE shorthand escapes (\d \w \s) to GBNF classes |
| 0002 | `f4ed0a034` | common: guard checkpoint update_dft against an empty draft sequence |
| 0003 | `20ac9695d` | ggml-alloc: finalize view init in a pass after all buft max_size splits |
| 0004 | `cbf6b1831` | ci: add sync-with-upstream workflow (mirror of the master copy) |
| 0005 | `0323f3340` | fix: repair YAML block-scalar indentation bug in sync workflow |
| 0006 | `fea69e077` | server: include id_slot in OAI-compatible chat completion responses |
| 0007 | `d0cd6ad32` | ggml: add GGML_OP_SSM_CONV_SPLIT (CPU+Vulkan) to skip the per-layer conv-state concat |
| 0008 | `94fb9b77d` | fleet: mirror fork patches to disk, document fleet workflow separately |
| 0009 | `0818a8fd3` | server: disable speculative decoding for grammar-constrained requests |
| 0010 | `b4f77289c` | qwen3-coder: bound xml-arg-string with until_one_of, not a bare until |
| 0011 | `42a54bd31` | vulkan: occupancy-aware S/M/L tile selector for non-coopmat2 path (AMD/Intel) |
| 0012 | `45c96fc15` | model: K2 Horizon gguf conversion code |
| 0013 | `3f172ae8f` | model: loading hparams and tensors in k2-horizon.cpp |
| 0014 | `c6126e292` | model: K2 Horizon compute graph |
| 0015 | `d12a4910b` | model: K2 Horizon compute graph adjustment and registering tokenizers |
| 0016 | `f7cf09604` | model: K2 Horizon chat template and accomodate safetensors naming |
| 0017 | `a8e59d1a4` | k2-horizon: adapt to hparams.n_ff_exp API drift since fork point |
| 0018 | `017177b59` | fleet-patches: reference upstream tracking issue for K2-Horizon support |
| 0019 | `8e23577a7` | server: end generation cleanly on a tool-call-diff inconsistency |
| 0020 | `27e786074` | FLEET.md: flag pre-existing qwen3-coder peg-parser test crash |
| 0021 | `dc1510e38` | peg-parser: don't prematurely close an until_one_of span on an ambiguous partial delimiter match |
| 0022 | `c7b01f7db` | server: bound slot-action queue wait, and explain empty child-slot saves |
| 0023 | `63877150f` | FLEET.md: require Murat/Dawn QA loop for every fork-exclusive change |
| 0024 | `4d4b442dd` | ggml-cpu: extend CPU backend device registry to N locality domains |
| 0025 | `3c3764c39` | common: new --cpu-split auto\|off flag with startup throughput calibration |
| 0026 | `552cd5aba` | llama-model: weighted per-layer CPU split across locality domains |
| 0027 | `b28e5a923` | llama-context: make backend_cpu locality-domain aware |
| 0028 | `96c9bf74b` | tests: cover the CPU locality-domain split |
| 0029 | `97080cf7b` | docs: document --cpu-split and the CPU locality-domain split feature |
| 0030 | `31abfbbf7` | docs: record ship-as-is decision + TODO for FA/resolve_fused_ops caveat |
| 0031 | `ff6e937d3` | fix: route ggml_backend_cpu_device_get_locality_mask through the DL proc-address table |
| 0032 | `3a5e3c498` | fix: don't reject a GPU's host buffer type in ggml_backend_cpu_device_supports_buft |
| 0033 | `613de6d57` | FLEET.md: document the 2026-09-18 CPU-split rollout outage and its fix |
| 0034 | `ff256fe8d` | =?UTF-8?q?fix:=20correct=20units=20in=20CPU=20split=20cal?= =?UTF-8?q?ibration=20log=20line=20(=C2=B5s,=20not=20seconds)?= |
| 0035 | `5446ea5cc` | unicode : add the K2-Horizon pre-tokenizer splitter |
| 0036 | `524c26c7a` | tests: expand K2 Horizon unicode splitter coverage |
| 0037 | `fe31aa6f8` | unicode: handle K2 Horizon case folding and empty input |
| 0038 | `184e4d5ac` | vocab: recognize K2-Horizon's <\|ifm\|im_end\|> as an EOG token |
| 0039 | `4a25953cb` | chat: add K2-Horizon workaround for reasoning-tag auto-detection |
| 0040 | `3a713c1c4` | tests: skip test-cpu-device-split on non-Linux platforms |
| 0041 | `0f261f564` | tests: fix stale expected output in the shorthand-class regexp test |
| 0042 | `07d2e0e67` | tests: loosen slot-action-timeout test's join timeout margin |
| 0043 | `9fc9046f4` | conversion: fix flake8 lint violations in k2_horizon.py |
| 0044 | `32038f723` | rpc : serve each client connection on its own thread |
| 0045 | `a8c9c9973` | rpc : serialize backend compute across connection threads |
| 0046 | `22ce3a307` | rpc : per-connection backends, tracked connections, client cap, keepalive |
| 0047 | `52170fc4c` | rpc : keep one bad client from taking down the whole server |
| 0048 | `727f39e3e` | tests : add test-rpc-server-multiclient |
| 0049 | `b917b2aec` | rpc : don't reject a client because a just-closed probe still holds a slot |
| 0050 | `b1ff71165` | vulkan : size submit batches from the current graph's flops |
| 0051 | `d19689876` | llama : restore the worst-case sched plan before large ubatches |
| 0052 | `2fd18e2a1` | common : --split-balance for speed-aware layer splits |
| 0053 | `ac9411b42` | llama, server : capture context checkpoints inside a batch |
| 0054 | `4727d61e0` | common : split-balance: time the dominant weight type, model n_batch drains |
| 0055 | `a0543ca83` | vulkan : keep the flop-sized submits of a graph below the kernel job queue |
| 0056 | `bed797907` | common : split-balance: model imperfect stage overlap in prefill |
| 0057 | `570db1ff4` | fleet: document speed-aware split and pipeline fixes |
| 0058 | `815779ae6` | fleet: patch table: 0065-0071 are fork-only |
| 0059 | `d7cd53de7` | common : split-balance: key the calibration cache by build, expire after 7 days |
| 0060 | `980ea324a` | split-balance: ignore overrides that match nothing, validate and re-check the cache, calibrated overheads |
| 0061 | `8b92aab27` | server: only an explicit off value disables LLAMA_SERVER_CKPT_CAPTURE |
| 0062 | `46568f521` | rpc: take a client slot at HELLO, bound the handshake, report a full server |
| 0063 | `bfc5ee295` | FLEET: QA before/after table, hostNetwork for Thunderbolt RPC, rpc-server admission |
| 0064 | `b32f04a9b` | rpc: say that the connection to the server was lost when a request fails |
| 0065 | `c3705709a` | vulkan: on integrated GPUs, cap the free memory of host heaps at MemAvailable |
| 0066 | `9826fa24e` | split-balance: re-check local devices' cached timings too |
| 0067 | `1cce05e7c` | vulkan: find the host-RAM heap of an integrated GPU by size |
| 0068 | `3b45d35dc` | llama: cheaper state captures: no host-side materialization, no context sync |
| 0069 | `0f453a6f3` | llama: add llama_model_n_devices_used |
| 0070 | `5fca3701f` | server: capture checkpoints in-batch only across devices, read them after the next decode |
| 0071 | `571d2f523` | llama: direct I/O into memory that refuses DMA goes through a bounce buffer |
| 0072 | `280a0f95f` | split-balance: time MoE models with routed expert matmuls, reject inconsistent entries |
| 0073 | `32a4c4aa5` | server tests: keep the slot busy longer than the save timeout on fast hardware |
| 0074 | `139c960bc` | FLEET: capture policy, direct I/O bounce buffer, MoE calibration |
| 0075 | `41b2ad40d` | models : fix the conv-state window over the transposed new tokens |
| 0076 | `c0f21d0b1` | vulkan: cap an integrated GPU's total free memory at MemAvailable |
| 0077 | `adb26194f` | tests : rpc-server multiclient: cast RAND_MAX to float |
| 0078 | `773e1bef6` | common : split-balance: calibrate on the model's own layers |
| 0079 | `8519f34e6` | chat : K2-Horizon: accept any effort's close tag as the end of reasoning |
| 0080 | `f418ade54` | FLEET.md: per-layer calibration, the conv-window view bug and K2-Horizon close tags |
| 0081 | `a6ded58e6` | common : split-balance: fix the calibration use-after-free, detect busy nodes against a reference |
| 0082 | `3de07b36f` | chat : K2-Horizon: end content at a close tag after the answer |
| 0083 | `8294d54c9` | FLEET.md: calibration grouping, busy reference, calibration cache, K2 stray close tag |
| 0084 | `34da5e374` | common : split-balance: survive malformed cache entries, log why an entry is not used, merge on write |
| 0085 | `c725c5d7e` | server, chat : stop generating at a stray K2-Horizon close tag; keep reasoning_format none raw |
| 0086 | `ede0eed18` | FLEET.md: K2 stop_after_reasoning |
| 0087 | `649e5e50a` | server : a captured checkpoint may outlive its task: don't read slot.task for it |
| 0088 | `0ab13d2bb` | server : don't update the prompt cache of a busy slot picked by id_slot |
| 0089 | `c4234d605` | FLEET.md: record the busy-slot prompt cache fix |
| 0090 | `e9319f635` | server : don't update the prompt cache of a busy slot picked by id_slot |
| 0091 | `1b177d4e5` | chat : K2-Horizon: don't leak tool-call markup into content with no tools |
| 0092 | `3a15bf975` | chat : K2-Horizon tests and comment: use a neutral example hostname |
| 0093 | `9ecf429da` | common : split-balance: scale prefill by byte_scale too, not just decode |
| 0094 | `b4a0f6c85` | common : split-balance: fix calibration overrating a device under per-node Vulkan submits |
| 0095 | `74405080c` | chat : drop whitespace between reasoning end tag and content |
| 0096 | `22a240a7a` | fleet: fix the patch mirror table numbering |
| 0097 | `3d33f7e68` | fleet: fix the patch mirror table rows |
| 0098 | `70f0d0168` | llama, common: fix -fit's oversized-context probe hard-aborting the server |
| 0099 | `1bf5f2473` | server: fix 2 test_completion.py tests broken by test-only bugs, not the server |
| 0100 | `ae3897657` | common: size -fit's auto-context probe from measured free memory, not just a caught exception |
| 0101 | `ccc60e62a` | common: close the n_seq_max == 1 gap in --fit's auto-context probe, cap iGPU free memory consistently |
| 0102 | `dd5756d58` | common : split-balance: calibrate a layer's decode ops long enough to pay the real per-submit cost, not just its own norms/RoPE nodes |
| 0103 | `d7e7f3668` | ggml-rpc: give the RPC client a bounded receive timeout and keepalive |
| 0104 | `8e37aa9a8` | FLEET.md: root-cause the round-9 pp regression as a QA measurement bug |
| 0105 | `11b60fcc7` | fleet: fix the 0131 row title in the patch mirror table |
| 0106 | `c4d2d91de` | FLEET.md: document the RPC client receive timeout |
| 0107 | `1ea0f1029` | chat, sampling: K2-Horizon official weights: tool call in reasoning, repeated open tag, empty first-token stop |
| 0108 | `9a0ac0e5c` | chat: gpt-oss parser accepts the "commentary (analysis)" analysis header |
| 0109 | `01c5ff45c` | fleet: add patch 0139 (gpt-oss commentary (analysis) header) to the mirror |
| 0110 | `84d7611e9` | sampling: K2-Horizon: empty-reply guard also holds EOG inside an unfinished Kimi-style tool-call opener |
| 0111 | `2d04bb954` | fleet: add patch mirror 0140 for the K2 Kimi-opener EOG guard |
| 0112 | `bf6ef50ff` | sampling, chat: K2-Horizon: hold EOG from a tool-call opener until the section is closed and its inner tags balance |
| 0113 | `b324bf274` | fleet: add patch mirror 0141 for the K2 balanced-tag EOG hold |
| 0114 | `fb6b0159d` | sampling: K2-Horizon: Kimi hold gets its own small cap and releases on a repeated opener |
| 0115 | `b212eba79` | fleet: add patch mirror 0142 for the Kimi cap / repeat release |
| 0116 | `5864976f9` | chat: K2-Horizon: render history reasoning with the generation tag, strip foreign markup, drop emptied turns |
| 0117 | `282ab87a9` | sampling: K2-Horizon: mask the structural tool-call tags while no native section is open |
| 0118 | `3b7d51c4d` | FLEET.md: K2 history-replay root cause and the tag mask |
| 0119 | `3c420f69f` | fleet: add patch mirrors 0143-0145 for the K2 history normalisation and tag mask |
| 0120 | `11e9b4b63` | sampling: empty-reply guard/tag mask: O(1) per-candidate masking |
| 0121 | `0088f823d` | chat: K2-Horizon parser gaps (all reasoning efforts, bare/Kimi calls, stray tags, never throw) |
| 0122 | `3449022c9` | K2: hold EOG inside an unclosed reasoning block (tools offered); strip only known K2/Kimi markup from replayed reasoning |
| 0123 | `d5812222e` | K2 parser: surface a tool call only once its arguments are complete JSON |
| 0124 | `f893af144` | K2 parser: accept the tool arguments in any order |
| 0125 | `59b910d75` | K2 guard: turn an end of turn inside an unclosed reasoning block into the block's close tag |
| 0126 | `ed3ab13ad` | K2 guard: repetition breaker for an open reasoning block |
| 0127 | `b16c58a80` | fleet: add patch mirrors 0146-0152 and FLEET.md entry for K2 fix build F2 |
| 0128 | `b416db4f3` | grammar: translate PCRE shorthand escapes (\d \w \s) to GBNF classes |
| 0129 | `6fcd12020` | common: guard checkpoint update_dft against an empty draft sequence |
| 0130 | `9f9044d38` | ci: add sync-with-upstream workflow (mirror of the master copy) |
| 0131 | `c4fffee0a` | fix: repair YAML block-scalar indentation bug in sync workflow |
| 0132 | `9849c0935` | server: include id_slot in OAI-compatible chat completion responses |
| 0133 | `b91aa1bae` | ggml: add GGML_OP_SSM_CONV_SPLIT (CPU+Vulkan) to skip the per-layer conv-state concat |
| 0134 | `d5a1ba215` | fleet: mirror fork patches to disk, document fleet workflow separately |
| 0135 | `3e7b73528` | server: disable speculative decoding for grammar-constrained requests |
| 0136 | `7bc133cd9` | qwen3-coder: bound xml-arg-string with until_one_of, not a bare until |
| 0137 | `be29ab3c5` | vulkan: occupancy-aware S/M/L tile selector for non-coopmat2 path (AMD/Intel) |
| 0138 | `76c6c1970` | model: K2 Horizon gguf conversion code |
| 0139 | `06e845fee` | model: loading hparams and tensors in k2-horizon.cpp |
| 0140 | `1b9e56f70` | model: K2 Horizon compute graph |
| 0141 | `b0d2eb7b9` | model: K2 Horizon compute graph adjustment and registering tokenizers |
| 0142 | `1a9c34cce` | model: K2 Horizon chat template and accomodate safetensors naming |
| 0143 | `366723b6a` | k2-horizon: adapt to hparams.n_ff_exp API drift since fork point |
| 0144 | `bc9abe13a` | server: end generation cleanly on a tool-call-diff inconsistency |
| 0145 | `16be7a5ea` | peg-parser: don't prematurely close an until_one_of span on an ambiguous partial delimiter match |
| 0146 | `fcbf32673` | server: bound slot-action queue wait, and explain empty child-slot saves |
| 0147 | `ea97ff428` | ggml-cpu: extend CPU backend device registry to N locality domains |
| 0148 | `f0ac5853e` | common: new --cpu-split auto\|off flag with startup throughput calibration |
| 0149 | `cd404aaa0` | llama-model: weighted per-layer CPU split across locality domains |
| 0150 | `adda6485c` | llama-context: make backend_cpu locality-domain aware |
| 0151 | `d12bbb408` | tests: cover the CPU locality-domain split |
| 0152 | `d4258c007` | docs: document --cpu-split and the CPU locality-domain split feature |
| 0153 | `5a047c591` | fix: route ggml_backend_cpu_device_get_locality_mask through the DL proc-address table |
| 0154 | `1e574238c` | fix: don't reject a GPU's host buffer type in ggml_backend_cpu_device_supports_buft |
| 0155 | `b2fcf83bf` | =?UTF-8?q?fix:=20correct=20units=20in=20CPU=20split=20cal?= =?UTF-8?q?ibration=20log=20line=20(=C2=B5s,=20not=20seconds)?= |
| 0156 | `1c5a47df5` | unicode : add the K2-Horizon pre-tokenizer splitter |
| 0157 | `d07181073` | tests: expand K2 Horizon unicode splitter coverage |
| 0158 | `1ebc2131b` | unicode: handle K2 Horizon case folding and empty input |
| 0159 | `97c77dadd` | vocab: recognize K2-Horizon's <\|ifm\|im_end\|> as an EOG token |
| 0160 | `9a99f9bef` | chat: add K2-Horizon workaround for reasoning-tag auto-detection |
| 0161 | `5e093a4d3` | tests: skip test-cpu-device-split on non-Linux platforms |
| 0162 | `840ee5bfd` | tests: fix stale expected output in the shorthand-class regexp test |
| 0163 | `d3837b205` | tests: loosen slot-action-timeout test's join timeout margin |
| 0164 | `bd9d4a950` | conversion: fix flake8 lint violations in k2_horizon.py |
| 0165 | `414817e16` | rpc : serve each client connection on its own thread |
| 0166 | `5bacb855d` | rpc : serialize backend compute across connection threads |
| 0167 | `afa73b359` | rpc : per-connection backends, tracked connections, client cap, keepalive |
| 0168 | `150031531` | rpc : keep one bad client from taking down the whole server |
| 0169 | `de64ed90e` | tests : add test-rpc-server-multiclient |
| 0170 | `a37aafb0d` | rpc : don't reject a client because a just-closed probe still holds a slot |
| 0171 | `f5af52c9f` | vulkan : size submit batches from the current graph's flops |
| 0172 | `4d7bde79f` | llama : restore the worst-case sched plan before large ubatches |
| 0173 | `99a069a61` | common : --split-balance for speed-aware layer splits |
| 0174 | `aa300693c` | llama, server : capture context checkpoints inside a batch |
| 0175 | `f9e705a2c` | common : split-balance: time the dominant weight type, model n_batch drains |
| 0176 | `0710f3b18` | vulkan : keep the flop-sized submits of a graph below the kernel job queue |
| 0177 | `7a8a4bff3` | common : split-balance: model imperfect stage overlap in prefill |
| 0178 | `2f4af6d45` | common : split-balance: key the calibration cache by build, expire after 7 days |
| 0179 | `4797e21fb` | split-balance: ignore overrides that match nothing, validate and re-check the cache, calibrated overheads |
| 0180 | `4de5bbf4e` | server: only an explicit off value disables LLAMA_SERVER_CKPT_CAPTURE |
| 0181 | `42d3a854d` | rpc: take a client slot at HELLO, bound the handshake, report a full server |
| 0182 | `be01639e0` | rpc: say that the connection to the server was lost when a request fails |
| 0183 | `333cce375` | vulkan: on integrated GPUs, cap the free memory of host heaps at MemAvailable |
| 0184 | `a45e1bcac` | split-balance: re-check local devices' cached timings too |
| 0185 | `730fae701` | vulkan: find the host-RAM heap of an integrated GPU by size |
| 0186 | `f5d7c1521` | llama: cheaper state captures: no host-side materialization, no context sync |
| 0187 | `5bff21b80` | llama: add llama_model_n_devices_used |
| 0188 | `54920cfed` | server: capture checkpoints in-batch only across devices, read them after the next decode |
| 0189 | `ab1268a2f` | llama: direct I/O into memory that refuses DMA goes through a bounce buffer |
| 0190 | `cd99ff7e5` | split-balance: time MoE models with routed expert matmuls, reject inconsistent entries |
| 0191 | `d06735f4c` | server tests: keep the slot busy longer than the save timeout on fast hardware |
| 0192 | `b9b9ecbd6` | models : fix the conv-state window over the transposed new tokens |
| 0193 | `3f9b80c45` | vulkan: cap an integrated GPU's total free memory at MemAvailable |
| 0194 | `71719479e` | tests : rpc-server multiclient: cast RAND_MAX to float |
| 0195 | `8410a16c3` | common : split-balance: calibrate on the model's own layers |
| 0196 | `5916940d6` | chat : K2-Horizon: accept any effort's close tag as the end of reasoning |
| 0197 | `723dd204c` | common : split-balance: fix the calibration use-after-free, detect busy nodes against a reference |
| 0198 | `527cd259d` | chat : K2-Horizon: end content at a close tag after the answer |
| 0199 | `d93fef38a` | common : split-balance: survive malformed cache entries, log why an entry is not used, merge on write |
| 0200 | `7e2b81517` | server, chat : stop generating at a stray K2-Horizon close tag; keep reasoning_format none raw |
| 0201 | `7b1dd205c` | server : a captured checkpoint may outlive its task: don't read slot.task for it |
| 0202 | `b2e0794e1` | server : don't update the prompt cache of a busy slot picked by id_slot |
| 0203 | `9768fff5b` | chat : K2-Horizon: don't leak tool-call markup into content with no tools |
| 0204 | `cd6ab339c` | chat : K2-Horizon tests and comment: use a neutral example hostname |
| 0205 | `45775961c` | common : split-balance: scale prefill by byte_scale too, not just decode |
| 0206 | `121e417b2` | common : split-balance: fix calibration overrating a device under per-node Vulkan submits |
| 0207 | `599dd43b0` | chat : drop whitespace between reasoning end tag and content |
| 0208 | `d260d6288` | llama, common: fix -fit's oversized-context probe hard-aborting the server |
| 0209 | `442f1b2dd` | common: size -fit's auto-context probe from measured free memory, not just a caught exception |
| 0210 | `80783c14b` | common: close the n_seq_max == 1 gap in --fit's auto-context probe, cap iGPU free memory consistently |
| 0211 | `591d66a36` | common : split-balance: calibrate a layer's decode ops long enough to pay the real per-submit cost, not just its own norms/RoPE nodes |
| 0212 | `1ead2878a` | ggml-rpc: give the RPC client a bounded receive timeout and keepalive |
| 0213 | `3c994840e` | chat, sampling: K2-Horizon official weights: tool call in reasoning, repeated open tag, empty first-token stop |
| 0214 | `6e18dc416` | chat: gpt-oss parser accepts the "commentary (analysis)" analysis header |
| 0215 | `19761cacb` | sampling: K2-Horizon: empty-reply guard also holds EOG inside an unfinished Kimi-style tool-call opener |
| 0216 | `e6c28fdfa` | sampling, chat: K2-Horizon: hold EOG from a tool-call opener until the section is closed and its inner tags balance |
| 0217 | `1fd293626` | sampling: K2-Horizon: Kimi hold gets its own small cap and releases on a repeated opener |
| 0218 | `bdb2125fc` | chat: K2-Horizon: render history reasoning with the generation tag, strip foreign markup, drop emptied turns |
| 0219 | `a2ae8f321` | sampling: K2-Horizon: mask the structural tool-call tags while no native section is open |
| 0220 | `876e04668` | sampling: empty-reply guard/tag mask: O(1) per-candidate masking |
| 0221 | `5bfb85f86` | chat: K2-Horizon parser gaps (all reasoning efforts, bare/Kimi calls, stray tags, never throw) |
| 0222 | `65d46da14` | K2: hold EOG inside an unclosed reasoning block (tools offered); strip only known K2/Kimi markup from replayed reasoning |
| 0223 | `a2868918e` | K2 parser: surface a tool call only once its arguments are complete JSON |
| 0224 | `71e17322d` | K2 parser: accept the tool arguments in any order |
| 0225 | `640423072` | K2 guard: turn an end of turn inside an unclosed reasoning block into the block's close tag |
| 0226 | `5074d0df4` | K2 guard: repetition breaker for an open reasoning block |
| 0227 | `38e46b015` | llama : fail restoring a truncated sequence state file instead of aborting |
| 0228 | `54b555076` | grammar: translate PCRE shorthand escapes (\d \w \s) to GBNF classes |
| 0229 | `d77152d47` | common: guard checkpoint update_dft against an empty draft sequence |
| 0230 | `805a22896` | ci: add sync-with-upstream workflow (mirror of the master copy) |
| 0231 | `c9f337434` | fix: repair YAML block-scalar indentation bug in sync workflow |
| 0232 | `8b9d76cba` | server: include id_slot in OAI-compatible chat completion responses |
| 0233 | `7f64d1946` | ggml: add GGML_OP_SSM_CONV_SPLIT (CPU+Vulkan) to skip the per-layer conv-state concat |
| 0234 | `5bf774527` | fleet: mirror fork patches to disk, document fleet workflow separately |
| 0235 | `2cdf6bce0` | server: disable speculative decoding for grammar-constrained requests |
| 0236 | `2d13e476c` | qwen3-coder: bound xml-arg-string with until_one_of, not a bare until |
| 0237 | `9a214a091` | vulkan: occupancy-aware S/M/L tile selector for non-coopmat2 path (AMD/Intel) |
| 0238 | `4fefb9f07` | server: end generation cleanly on a tool-call-diff inconsistency |
| 0239 | `f786fca14` | peg-parser: don't prematurely close an until_one_of span on an ambiguous partial delimiter match |
| 0240 | `6e8a9cd25` | server: bound slot-action queue wait, and explain empty child-slot saves |
| 0241 | `101cf3f9b` | ggml-cpu: extend CPU backend device registry to N locality domains |
| 0242 | `2b618a1ca` | common: new --cpu-split auto\|off flag with startup throughput calibration |
| 0243 | `db1d1af9a` | llama-model: weighted per-layer CPU split across locality domains |
| 0244 | `ad2999e8b` | llama-context: make backend_cpu locality-domain aware |
| 0245 | `ab80c5338` | tests: cover the CPU locality-domain split |
| 0246 | `927b1f540` | docs: document --cpu-split and the CPU locality-domain split feature |
| 0247 | `a30d9adbb` | fix: route ggml_backend_cpu_device_get_locality_mask through the DL proc-address table |
| 0248 | `de98be5d4` | fix: don't reject a GPU's host buffer type in ggml_backend_cpu_device_supports_buft |
| 0249 | `d1d8de372` | =?UTF-8?q?fix:=20correct=20units=20in=20CPU=20split=20cal?= =?UTF-8?q?ibration=20log=20line=20(=C2=B5s,=20not=20seconds)?= |
| 0250 | `15c56e32c` | tests: skip test-cpu-device-split on non-Linux platforms |
| 0251 | `e3089ca40` | tests: fix stale expected output in the shorthand-class regexp test |
| 0252 | `d87cfc01c` | tests: loosen slot-action-timeout test's join timeout margin |
| 0253 | `66de0db9e` | rpc : serve each client connection on its own thread |
| 0254 | `c75c3cd6c` | rpc : serialize backend compute across connection threads |
| 0255 | `15760f8bd` | rpc : per-connection backends, tracked connections, client cap, keepalive |
| 0256 | `b4139e80f` | rpc : keep one bad client from taking down the whole server |
| 0257 | `55d72f3c6` | tests : add test-rpc-server-multiclient |
| 0258 | `fb010047f` | rpc : don't reject a client because a just-closed probe still holds a slot |
| 0259 | `cd4c427b0` | vulkan : size submit batches from the current graph's flops |
| 0260 | `d6eefe221` | llama : restore the worst-case sched plan before large ubatches |
| 0261 | `31f5f3e49` | common : --split-balance for speed-aware layer splits |
| 0262 | `fc93bb21f` | llama, server : capture context checkpoints inside a batch |
| 0263 | `24066e0a8` | common : split-balance: time the dominant weight type, model n_batch drains |
| 0264 | `0e2213c42` | vulkan : keep the flop-sized submits of a graph below the kernel job queue |
| 0265 | `93791945b` | common : split-balance: model imperfect stage overlap in prefill |
| 0266 | `69519c153` | common : split-balance: key the calibration cache by build, expire after 7 days |
| 0267 | `3912cba48` | split-balance: ignore overrides that match nothing, validate and re-check the cache, calibrated overheads |
| 0268 | `e903b4d2c` | server: only an explicit off value disables LLAMA_SERVER_CKPT_CAPTURE |
| 0269 | `cee0edd5b` | rpc: take a client slot at HELLO, bound the handshake, report a full server |
| 0270 | `8c72d93eb` | rpc: say that the connection to the server was lost when a request fails |
| 0271 | `5d7c6d6b1` | vulkan: on integrated GPUs, cap the free memory of host heaps at MemAvailable |
| 0272 | `5e48eaeee` | split-balance: re-check local devices' cached timings too |
| 0273 | `c40b9251b` | vulkan: find the host-RAM heap of an integrated GPU by size |
| 0274 | `7952bc165` | llama: cheaper state captures: no host-side materialization, no context sync |
| 0275 | `280c80db9` | llama: add llama_model_n_devices_used |
| 0276 | `3d4d1bc31` | server: capture checkpoints in-batch only across devices, read them after the next decode |
| 0277 | `47c315fb0` | llama: direct I/O into memory that refuses DMA goes through a bounce buffer |
| 0278 | `f22d7416e` | split-balance: time MoE models with routed expert matmuls, reject inconsistent entries |
| 0279 | `f94f3d1a3` | server tests: keep the slot busy longer than the save timeout on fast hardware |
| 0280 | `0359ec5d4` | models : fix the conv-state window over the transposed new tokens |
| 0281 | `3fa3818b9` | vulkan: cap an integrated GPU's total free memory at MemAvailable |
| 0282 | `fd103a519` | tests : rpc-server multiclient: cast RAND_MAX to float |
| 0283 | `f2228c0f2` | common : split-balance: calibrate on the model's own layers |
| 0284 | `c48dec9c9` | common : split-balance: fix the calibration use-after-free, detect busy nodes against a reference |
| 0285 | `a942dea5d` | common : split-balance: survive malformed cache entries, log why an entry is not used, merge on write |
| 0286 | `c74e0716a` | server, chat : stop generating at a stray K2-Horizon close tag; keep reasoning_format none raw |
| 0287 | `b7962aa96` | server : a captured checkpoint may outlive its task: don't read slot.task for it |
| 0288 | `df5e75daf` | server : don't update the prompt cache of a busy slot picked by id_slot |
| 0289 | `0cd50798c` | common : split-balance: scale prefill by byte_scale too, not just decode |
| 0290 | `9730b2f68` | common : split-balance: fix calibration overrating a device under per-node Vulkan submits |
| 0291 | `b95797a74` | chat : drop whitespace between reasoning end tag and content |
| 0292 | `f4bd6113a` | llama, common: fix -fit's oversized-context probe hard-aborting the server |
| 0293 | `98685de0f` | common: size -fit's auto-context probe from measured free memory, not just a caught exception |
| 0294 | `409cb1006` | common: close the n_seq_max == 1 gap in --fit's auto-context probe, cap iGPU free memory consistently |
| 0295 | `b0f3623c8` | common : split-balance: calibrate a layer's decode ops long enough to pay the real per-submit cost, not just its own norms/RoPE nodes |
| 0296 | `06b374ae2` | ggml-rpc: give the RPC client a bounded receive timeout and keepalive |
| 0297 | `532a6fc25` | chat, sampling: K2-Horizon official weights: tool call in reasoning, repeated open tag, empty first-token stop |
| 0298 | `857323a2a` | chat: gpt-oss parser accepts the "commentary (analysis)" analysis header |
| 0299 | `34a52d11a` | sampling: K2-Horizon: empty-reply guard also holds EOG inside an unfinished Kimi-style tool-call opener |
| 0300 | `049528cf2` | sampling, chat: K2-Horizon: hold EOG from a tool-call opener until the section is closed and its inner tags balance |
| 0301 | `d1ed6605d` | sampling: K2-Horizon: Kimi hold gets its own small cap and releases on a repeated opener |
| 0302 | `414c5e80a` | chat: K2-Horizon: render history reasoning with the generation tag, strip foreign markup, drop emptied turns |
| 0303 | `7f07a1b06` | sampling: K2-Horizon: mask the structural tool-call tags while no native section is open |
| 0304 | `1a9f36e79` | sampling: empty-reply guard/tag mask: O(1) per-candidate masking |
| 0305 | `3aaaa53af` | chat: K2-Horizon parser gaps (all reasoning efforts, bare/Kimi calls, stray tags, never throw) |
| 0306 | `7afa013d3` | K2: hold EOG inside an unclosed reasoning block (tools offered); strip only known K2/Kimi markup from replayed reasoning |
| 0307 | `faf40ac07` | K2 parser: surface a tool call only once its arguments are complete JSON |
| 0308 | `430ec4572` | K2 guard: turn an end of turn inside an unclosed reasoning block into the block's close tag |
| 0309 | `6c00e84f1` | K2 guard: repetition breaker for an open reasoning block |
| 0310 | `cec58d8b2` | llama : fail restoring a truncated sequence state file instead of aborting |
| 0311 | `6cc2ec938` | chat: K2-Horizon: port the fleet's parser tolerance onto upstream's k2-horizon.cpp |
| 0312 | `0619cebbf` | tests: K2-Horizon: port the fleet's parser, history and guard-field tests |
| 0313 | `2f1949ec8` | chat: K2-Horizon: startup self-check that the sampler guard is fed |
| 0314 | `44d4952e1` | K2-Horizon/server: fix tool_choice none content leak, bound call args, reject bad slot ids |
| 0315 | `5d9445b34` | sampling: value-initialize the empty-reply guard (fixes -Werror=missing-field-initializers on gcc arm64 CI) |
| 0316 | `ef002595f` | ci: remove the disabled sync-fleet-patches workflow |
| 0317 | `c291b7775` | tests: slot-action timeout test runs a 6000-token busy request (12000 outran its 90 s HTTP timeout on the arm64 runner) |
| 0318 | `3d8731b2d` | tests: test-state-capture tolerates by-design re-reserve under GGML_SCHED_NO_REALLOC |
| 0319 | `8b22e1515` | tests: Windows-portable env set in test-state-capture; retry noisy split-balance ratio on busy runners |
| 0320 | `c351de77a` | tests: skip split-balance op-count ratio check when calibration is throttled (hosted macOS runners) |
| 0321 | `dc4d6d84d` | tests: render the K2 parser's chat template once per make_peg_parser, not once per parse |
| 0322 | `4c17b1108` | qwen4exp: use ggml_ssm_conv_split for the GDN conv, dropping its concat |
| 0323 | `ec0884436` | vulkan: ssm_conv_split nc==4 uses the same dot() reduction as ssm_conv, for bit-identical output |
| 0324 | `a6193d8f4` | vulkan: shared-memory tile path for concat of a transposed src1 along dim 0 |
| 0325 | `b9e449fdc` | vulkan/tests: guard ssm_conv_split d_conv <= 16, widen concat_transposed coverage |
| 0326 | `8120c3ced` | chat: tool_choice none no longer returns a raw tool call as content (Qwen3.x) |
| 0327 | `64a4084b3` | chat: tool_choice none drops a call for every tag-delimited template, in one place |
| 0328 | `727dd3ead` | chat: tool_choice none keeps the none parser's plain replies and cuts rejected calls |
| 0329 | `de3105f80` | chat: tool_choice none parses with the none parser and cuts at the call opener |
| 0330 | `da19f59e8` | chat: tool_choice none cuts DeepSeek V4 reasoning calls and Kimi-K2 section tags |
| 0331 | `516485132` | chat: tool_choice none renders the prompt without the tools |
| 0332 | `4eecf8d28` | tests: keep each tool_choice none test under its own doc comment |
| 0333 | `d4fa7afd8` | ggml: GGML_OP_COUNT 103 after merging upstream TOP_K_MASK with the fork's SSM_CONV_SPLIT |
| 0334 | `609f33718` | tests/cpu: cover tiled concat with transposed i64 src1 |
| 0335 | `597e8cc31` | chat: K2-Horizon tool_choice none renders the prompt without the tools too |
| 0336 | `c5a8d3485` | chat: K2-Horizon takes the generic tool_choice none path |
| 0337 | `aac1d90d9` | server: write slot saves to a temp file and rename it into place |
| 0338 | `051cd3a76` | qwen3-coder: let a string argument hold the format's own tags as text |
| 0339 | `f6a35691b` | qwen3-coder: end the reasoning only at a real call opener, not a bare <tool_call> |
| 0340 | `90f0e772f` | k2-horizon: end the reasoning only at a real call start, not a bare call tag |
| 0341 | `a6a56f6a9` | server: add --spec-grammar to draft for grammar-constrained requests |
| 0342 | `ae33b89b0` | server: draft for grammar-constrained requests by default (--spec-grammar on) |
| 0343 | `3d5f8253c` | server: keep evicted prompt cache entries in the slot dir and restore them on a miss |
| 0344 | `5c97dad6e` | server: flush the prompt cache to the slot dir on SIGTERM/SIGINT |
| 0345 | `2f7d95d15` | server: fix review findings on the SIGTERM prompt-cache flush |
| 0346 | `ea3f2dc9a` | Fix compile breaks the 2026-11-10 upstream merge left outside conflict markers |
| 0347 | `7571530fa` | Restore the fork's concat-free conv-state store, fix three more merge regressions |
| 0348 | `842d35e4f` | FLEET: Upstream sync 2026-11-10 entry, regenerated patch mirror, refreshed flag checklist |
| 0349 | `d54e7fdd0` | FLEET: correct the 2026-11-10 sync's wrong grammar-regression attribution |

The `split-balance` / `vulkan: size submit batches` patches are the speed-aware
split and pipeline fixes (see FLEET.md, "Speed-aware split and pipeline
fixes"). Fork-only, not submitted upstream.

This mirror was regenerated on 2026-11-10 (349 patches: the 347 of the previous mirror, plus the FLEET.md doc
commit that mirror regen itself did not yet cover, plus the same-day correction of that doc's wrong grammar-regression
attribution; `BASE=202c1a229f5a83d1c9d758ed0f2d43760fa4bab0 TIP=HEAD`, no `EXCLUDE`, as for the 347-patch mirror).
The merge commit itself carries no patch; its conflict resolutions are listed in FLEET.md. The CPU strided-concat
hunk of 0334 is now covered by upstream's concat row kernel (only the test part of that patch is still live).
Patch numbers changed from the previous table; refer to patches by subject.

See `~/src/llama.cpp/CLAUDE.md`'s "Fleet patch maintenance" section for the
full workflow this fits into (when to add a new patch vs. amend an existing one; upstream syncs are manual merges since the
K2 adoption, the `sync-with-upstream` CI workflow is disabled).
