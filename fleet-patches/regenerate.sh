#!/usr/bin/env bash
# Regenerate the fleet-patches/*.patch mirror from the fork-exclusive commit set.
# Run from the repo root (or anywhere inside the repo). Overwrites every
# *.patch file in this directory -- commit the result.
#
# Default: BASE=upstream/master (freshly fetched; needs an `upstream` remote = ggml-org/llama.cpp), TIP=origin/fleet-patches.
# Since the 2026-10 K2 adoption the branch is merge-based: pass BASE explicitly (the upstream commit last merged, e.g.
# 4625240437c6821ee5c2da99e12d4817c6a8f07f = b11454) and EXCLUDE (below), or upstream-only and old-lineage commits get counted.
# For a sync branch built on a pinned upstream commit, pass both explicitly:
#   BASE=<upstream sha the branch is rebased on> TIP=HEAD ./fleet-patches/regenerate.sh
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [ -z "${BASE:-}" ]; then
    git fetch upstream master
    BASE=upstream/master
fi
if [ -z "${TIP:-}" ]; then
    git fetch origin fleet-patches
    TIP=origin/fleet-patches
fi

rm -f fleet-patches/*.patch
# Mirror/landing bookkeeping commits ("fleet: rebuild/regenerate patch mirror ...", "Land ...") are not fork
# patches: they only carry this directory and FLEET.md, so they are left out of the mirror.
n=0
# Merge commits (PR merges, upstream-sync merges) carry no patch of their own: --no-merges.
# EXCLUDE: a lineage to leave out. The 2026-10 K2 adoption landing (88f47132b) kept the old rebase-based fleet-patches
# as its second parent, so BASE..TIP also reaches ~230 old commits; pass EXCLUDE=38773736370e66abffbdb1af4805612055e87953.
for sha in $(git rev-list --reverse --no-merges "$BASE..$TIP" ${EXCLUDE:+"^$EXCLUDE"}); do
    case "$(git log -1 --format=%s "$sha")" in
        "fleet: rebuild patch mirror"*|"fleet: regenerate patch mirror"*|"fleet: regenerate.sh"*|"Land "*) continue ;;
    esac
    n=$((n + 1))
    git format-patch --no-numbered --start-number "$n" -1 -o fleet-patches "$sha" >/dev/null
done

echo "Regenerated $(ls fleet-patches/*.patch | wc -l) patch files."
echo "Update the table in fleet-patches/README.md if the patch set changed, then commit."
