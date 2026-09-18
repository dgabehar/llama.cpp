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
bug -- the existing fallback is doing exactly what it's designed to do. Not
fixed here: whether the right fix is keeping a layer's fused ops
deterministically co-located with its other CPU tensors (a `dev_layer`-level
change) or something else is a real design decision, not obvious from the
code alone -- needs Doug's input before changing `resolve_fused_ops` or the
per-layer split logic further. Flag to Murat/Dawn explicitly when testing
this on gabesrv10.

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
