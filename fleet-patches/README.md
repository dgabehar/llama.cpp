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

## Current patches (regenerated 2026-09-23)

| # | Commit | Summary |
|---|---|---|
| 0001 | `4449b934f` | grammar: translate PCRE shorthand escapes (`\d \w \s`) to GBNF classes |
| 0002 | `4790afadf` | server: fix slot save/restore losing checkpoint-based cache reuse |
| 0003 | `f4ed0a034` | common: guard checkpoint `update_dft` against an empty draft sequence |
| 0004 | `20ac9695d` | ggml-alloc: finalize view init in a pass after all buft max_size splits |
| 0005 | `cbf6b1831` | ci: add sync-with-upstream workflow (mirror of the master copy) |
| 0006 | `0323f3340` | fix: repair YAML block-scalar indentation bug in sync workflow |
| 0007 | `fea69e077` | server: include `id_slot` in OAI-compatible chat completion responses |
| 0008 | `d0cd6ad32` | ggml: add `GGML_OP_SSM_CONV_SPLIT` (CPU+Vulkan) to skip the per-layer conv-state concat |
| 0009 | `94fb9b77d` | fleet: mirror fork patches to disk, document fleet workflow separately |
| 0010 | `0818a8fd3` | server: disable speculative decoding for grammar-constrained requests |
| 0011 | `b4f77289c` | qwen3-coder: bound xml-arg-string with until_one_of, not a bare until |
| 0013 | `42a54bd31` | vulkan: occupancy-aware S/M/L tile selector for non-coopmat2 path (AMD/Intel) |
| 0015 | `45c96fc15` | model: K2 Horizon gguf conversion code |
| 0016 | `3f172ae8f` | model: loading hparams and tensors in k2-horizon.cpp |
| 0017 | `c6126e292` | model: K2 Horizon compute graph |
| 0018 | `d12a4910b` | model: K2 Horizon compute graph adjustment and registering tokenizers |
| 0019 | `f7cf09604` | model: K2 Horizon chat template and accomodate safetensors naming |
| 0020 | `a8e59d1a4` | k2-horizon: adapt to hparams.n_ff_exp API drift since fork point |
| 0022 | `017177b59` | fleet-patches: reference upstream tracking issue for K2-Horizon support |
| 0023 | `8e23577a7` | server: end generation cleanly on a tool-call-diff inconsistency |
| 0024 | `27e786074` | FLEET.md: flag pre-existing qwen3-coder peg-parser test crash |
| 0026 | `dc1510e38` | peg-parser: don't prematurely close an until_one_of span on an ambiguous partial delimiter match |
| 0027 | `c7b01f7db` | server: bound slot-action queue wait, and explain empty child-slot saves |
| 0029 | `63877150f` | FLEET.md: require Murat/Dawn QA loop for every fork-exclusive change |
| 0030 | `4d4b442dd` | ggml-cpu: extend CPU backend device registry to N locality domains |
| 0031 | `3c3764c39` | common: new `--cpu-split auto|off` flag with startup throughput calibration |
| 0032 | `552cd5aba` | llama-model: weighted per-layer CPU split across locality domains |
| 0033 | `b28e5a923` | llama-context: make backend_cpu locality-domain aware |
| 0034 | `96c9bf74b` | tests: cover the CPU locality-domain split |
| 0035 | `97080cf7b` | docs: document `--cpu-split` and the CPU locality-domain split feature |
| 0036 | `31abfbbf7` | docs: record ship-as-is decision + TODO for FA/resolve_fused_ops caveat |
| 0038 | `ff6e937d3` | fix: route `ggml_backend_cpu_device_get_locality_mask` through the DL proc-address table |
| 0040 | `3a5e3c498` | fix: don't reject a GPU's host buffer type in `ggml_backend_cpu_device_supports_buft` |
| 0042 | `613de6d57` | FLEET.md: document the 2026-09-18 CPU-split rollout outage and its fix |
| 0044 | `ff256fe8d` | fix: correct units in CPU split calibration log line (µs, not seconds) |
| 0046 | `5446ea5cc` | unicode: add the K2-Horizon pre-tokenizer splitter |
| 0047 | `524c26c7a` | tests: expand K2 Horizon unicode splitter coverage |
| 0048 | `fe31aa6f8` | unicode: handle K2 Horizon case folding and empty input |
| 0050 | `184e4d5ac` | vocab: recognize K2-Horizon's <\|ifm\|im_end\|> as an EOG token |
| 0051 | `4a25953cb` | chat: add K2-Horizon workaround for reasoning-tag auto-detection |
| 0052 | `3a713c1c4` | tests: skip test-cpu-device-split on non-Linux platforms |
| 0053 | `0f261f564` | tests: fix stale expected output in the shorthand-class regexp test |
| 0054 | `07d2e0e67` | tests: loosen slot-action-timeout test's join timeout margin |
| 0055 | `9fc9046f4` | conversion: fix flake8 lint violations in k2_horizon.py |
| 0057 | `32038f723` | rpc : serve each client connection on its own thread |
| 0058 | `a8c9c9973` | rpc : serialize backend compute across connection threads |
| 0059 | `22ce3a307` | rpc : per-connection backends, tracked connections, client cap, keepalive |
| 0060 | `52170fc4c` | rpc : keep one bad client from taking down the whole server |
| 0061 | `727f39e3e` | tests : add test-rpc-server-multiclient |
| 0063 | `b917b2aec` | rpc : don't reject a client because a just-closed probe still holds a slot |
| 0065 | `b1ff71165` | vulkan : size submit batches from the current graph's flops |
| 0066 | `d19689876` | llama : restore the worst-case sched plan before large ubatches |
| 0067 | `2fd18e2a1` | common : --split-balance for speed-aware layer splits |
| 0068 | `ac9411b42` | llama, server : capture context checkpoints inside a batch |
| 0069 | `4727d61e0` | common : split-balance: time the dominant weight type, model n_batch drains |
| 0070 | `a0543ca83` | vulkan : keep the flop-sized submits of a graph below the kernel job queue |
| 0071 | `bed797907` | common : split-balance: model imperfect stage overlap in prefill |
| 0074 | `d7cd53de7` | common : split-balance: key the calibration cache by build, expire after 7 days |
| 0075 | `980ea324a` | split-balance: ignore overrides that match nothing, validate and re-check the cache, calibrated overheads |
| 0076 | `8b92aab27` | server: only an explicit off value disables LLAMA_SERVER_CKPT_CAPTURE |
| 0077 | `46568f521` | rpc: take a client slot at HELLO, bound the handshake, report a full server |
| 0079 | `b32f04a9b` | rpc: say that the connection to the server was lost when a request fails |
| 0080 | `c3705709a` | vulkan: on integrated GPUs, cap the free memory of host heaps at MemAvailable |
| 0081 | `9826fa24e` | split-balance: re-check local devices' cached timings too |
| 0082 | (this commit) | vulkan: find the host-RAM heap of an integrated GPU by size |

0046-0048 are cherry-picked from `MBZUAI-IFM/llama.cpp@model/K2Horizon`
(commits `69d3a4e82`/`a8104b553`/`e78bd9435` there), landed 3 commits ahead
of this fork's original 0015-0019 fork point. Fixes an MSVC `std::regex`
crash that blocked K2-Horizon GGUFs from loading on Windows, plus
ZWNJ/ZWJ/case-folding/empty-input edge cases in the pre-tokenizer word
splitter. Does **not** touch BPE merge-table granularity -- unrelated to the
"hostname splits into odd subword pieces" tokenization behavior observed
live on gabesrv06 2026-09-21 (that's normal BPE-vocab behavior, shared with
gpt-oss-20b's tokenizer on the same string, not a bug this fixes).

Patches 0015-0019 are cherry-picked from the vendor's own architecture-support
branch (`MBZUAI-IFM/llama.cpp@model/K2Horizon`, forked from upstream
2026-08-28) -- model-definition/conversion/vocab layer only, no backend code
touched. 0020 is this fork's own fix for API drift in `llama_hparams::n_ff_exp`
(plain field -> per-layer accessor) that landed upstream after the vendor's
fork point. See `home-infrastructure`'s TODO.md ("K2-Horizon Ascent" entry,
2026-09-16) for the full port/build/canary writeup.

**No upstream PR exists for this yet.** The model card claims "PR to
llama.cpp is in progress," but as of this writing nothing has actually been
filed against `ggml-org/llama.cpp` -- only a tracking bug report exists:
[ggml-org/llama.cpp#28361](https://github.com/ggml-org/llama.cpp/issues/28361)
("Eval bug: K2-Horizon models fail to load"), still open, whose own comments
point back to the same vendor branch these patches are cherry-picked from
(https://github.com/MBZUAI-IFM/llama.cpp/tree/model/K2Horizon). Re-check that
issue before this fork's next upstream rebase -- if a real PR lands and
merges, these patches become redundant and should be dropped rather than
carried forward as a rebase conflict.

0057-0058 are cherry-picked from upstream PR
[ggml-org/llama.cpp#28916](https://github.com/ggml-org/llama.cpp/pull/28916)
(fix for issue #28908, rpc-server serving only one client at a time), original
authorship kept; 0059-0061 and 0063 build on them (see FLEET.md, "rpc-server
multi-client"). If #28916 merges upstream, drop 0057-0058 on the next rebase
and resolve 0059 against whatever shape upstream landed.

0065-0071 and 0074 are the speed-aware split and pipeline fixes (see FLEET.md, "Speed-aware
split and pipeline fixes"). Fork-only, not submitted upstream.

See `~/src/llama.cpp/CLAUDE.md`'s "Fleet patch maintenance" section for the
full workflow this fits into (weekly upstream rebase, when to add a new
patch vs. amend an existing one, the `sync-with-upstream` CI workflow).
