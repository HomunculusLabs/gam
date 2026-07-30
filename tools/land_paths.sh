#!/usr/bin/env bash
# Land specific files onto origin/main without touching anyone else's work.
#
# The repo has ONE shared working tree driven by several concurrent lanes, so
# `git add -A` sweeps other lanes' in-flight edits into your commit and a
# whole-tree push silently reverts whatever landed while you were editing.
# This builds a tree from the LIVE origin/main plus only the paths you name,
# so a concurrent land is preserved unless it touched the same file.
#
#   tools/land_paths.sh "commit message" path/a.rs path/b.rs ...
#   tools/land_paths.sh --delete "msg" path/gone.rs      # stage a deletion
#
# Reads file CONTENT from the working tree. Never resets, stashes, or branches.
set -euo pipefail

DELETE=0
if [ "${1:-}" = "--delete" ]; then DELETE=1; shift; fi
MSG="${1:?commit message required}"; shift
[ $# -gt 0 ] || { echo "no paths given" >&2; exit 2; }

cd "$(git rev-parse --show-toplevel)"
git fetch -q origin main
BASE=$(git rev-parse origin/main)

IDX=$(mktemp -t landidx.XXXXXX)
trap 'rm -f "$IDX"' EXIT
GIT_INDEX_FILE=$IDX git read-tree "$BASE"

for p in "$@"; do
  if [ "$DELETE" = 1 ]; then
    GIT_INDEX_FILE=$IDX git update-index --force-remove "$p"
    echo "  - $p (deleted)"
  else
    [ -f "$p" ] || { echo "missing: $p" >&2; exit 3; }
    MODE=100644; [ -x "$p" ] && MODE=100755
    BLOB=$(git hash-object -w -- "$p")
    GIT_INDEX_FILE=$IDX git update-index --add --cacheinfo "$MODE,$BLOB,$p"
    echo "  + $p ($(wc -c <"$p") bytes)"
  fi
done

TREE=$(GIT_INDEX_FILE=$IDX git write-tree)
if [ "$TREE" = "$(git rev-parse "$BASE^{tree}")" ]; then
  echo "no change against $BASE — nothing to land"; exit 0
fi
COMMIT=$(git commit-tree "$TREE" -p "$BASE" -m "$MSG")
git push -q origin "$COMMIT:main"
echo "landed $COMMIT on main (base $BASE)"
