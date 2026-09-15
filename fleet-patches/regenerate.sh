#!/usr/bin/env bash
# Regenerate the fleet-patches/*.patch mirror from the current fleet-patches
# branch. Run from the repo root (or anywhere inside the repo). Overwrites
# every *.patch file in this directory -- commit the result.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

git fetch upstream master
git fetch origin fleet-patches

rm -f fleet-patches/*.patch
git format-patch --no-numbered -o fleet-patches upstream/master..origin/fleet-patches

echo "Regenerated $(ls fleet-patches/*.patch | wc -l) patch files."
echo "Update the table in fleet-patches/README.md if the patch set changed, then commit."
