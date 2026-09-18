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

## Current patches (regenerated 2026-09-18)

| # | Commit | Summary |
|---|---|---|
| 0001 | `467589f6b` | grammar: translate PCRE shorthand escapes (`\d \w \s`) to GBNF classes |
| 0002 | `a1e979568` | server: fix slot save/restore losing checkpoint-based cache reuse |
| 0003 | `b74b2e31c` | common: guard checkpoint `update_dft` against an empty draft sequence |
| 0004 | `5078faa5e` | ggml-alloc: finalize view init in a pass after all buft max_size splits |
| 0005 | `9950b9dd4` | ci: add sync-with-upstream workflow (mirror of the master copy) |
| 0006 | `290b1023d` | fix: repair YAML block-scalar indentation bug in sync workflow |
| 0007 | `c5f91197b` | server: include `id_slot` in OAI-compatible chat completion responses |
| 0008 | `87b0c16d4` | ggml: add `GGML_OP_SSM_CONV_SPLIT` (CPU+Vulkan) to skip the per-layer conv-state concat |
| 0009 | `b7fd9fc37` | fleet: revert qwen3-coder complex-type parsing regression (upstream #28736/#28742) |
| 0010 | `f60302abe` | fleet: mirror fork patches to disk, document fleet workflow separately |
| 0011 | `17ecd9d52` | server: disable speculative decoding for grammar-constrained requests |
| 0012 | `3c4e5bf4b` | qwen3-coder: bound xml-arg-string with until_one_of, not a bare until |
| 0014 | `a1b3b4199` | vulkan: occupancy-aware S/M/L tile selector for non-coopmat2 path (AMD/Intel) |
| 0016 | `2bef5fb62` | model: K2 Horizon gguf conversion code |
| 0017 | `04dec3511` | model: loading hparams and tensors in k2-horizon.cpp |
| 0018 | `24b5a208f` | model: K2 Horizon compute graph |
| 0019 | `d1e6830aa` | model: K2 Horizon compute graph adjustment and registering tokenizers |
| 0020 | `40fd73c38` | model: K2 Horizon chat template and accomodate safetensors naming |
| 0021 | `7f9e25ec4` | k2-horizon: adapt to hparams.n_ff_exp API drift since fork point |
| 0024 | `35e651927` | server: end generation cleanly on a tool-call-diff inconsistency |
| 0027 | `9e917cddf` | peg-parser: don't prematurely close an until_one_of span on an ambiguous partial delimiter match |
| 0028 | `72a88ea77` | server: bound slot-action queue wait, and explain empty child-slot saves |
| 0030 | `faee26a4c` | FLEET.md: require Murat/Dawn QA loop for every fork-exclusive change |
| 0031 | `84b660c0d` | ggml-cpu: extend CPU backend device registry to N locality domains |
| 0032 | `6fe8f7c86` | common: new `--cpu-split auto|off` flag with startup throughput calibration |
| 0033 | `1d8ec4bc5` | llama-model: weighted per-layer CPU split across locality domains |
| 0034 | `849dacc67` | llama-context: make backend_cpu locality-domain aware |
| 0035 | `7e43e4a6d` | tests: cover the CPU locality-domain split |
| 0036 | `1335f74e1` | docs: document `--cpu-split` and the CPU locality-domain split feature |
| 0039 | `33581fbd8` | fix: route `ggml_backend_cpu_device_get_locality_mask` through the DL proc-address table |

Patches 0016-0020 are cherry-picked from the vendor's own architecture-support
branch (`MBZUAI-IFM/llama.cpp@model/K2Horizon`, forked from upstream
2026-08-28) -- model-definition/conversion/vocab layer only, no backend code
touched. 0021 is this fork's own fix for API drift in `llama_hparams::n_ff_exp`
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

See `~/src/llama.cpp/CLAUDE.md`'s "Fleet patch maintenance" section for the
full workflow this fits into (weekly upstream rebase, when to add a new
patch vs. amend an existing one, the `sync-with-upstream` CI workflow).
