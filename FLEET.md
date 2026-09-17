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
