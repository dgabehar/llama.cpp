#!/usr/bin/env bash
# Regenerate the fleet-patches/*.patch mirror from the fork-exclusive commit set.
# Run from the repo root (or anywhere inside the repo). Overwrites every
# *.patch file in this directory -- commit the result.
#
# Default: BASE=upstream/master (freshly fetched), TIP=origin/fleet-patches.
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
git format-patch --no-numbered -o fleet-patches "$BASE..$TIP"

echo "Regenerated $(ls fleet-patches/*.patch | wc -l) patch files."
echo "Update the table in fleet-patches/README.md if the patch set changed, then commit."
