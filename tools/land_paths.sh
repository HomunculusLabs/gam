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
#
# main moves every few minutes, so a single push loses to any lane that landed
# between the fetch and the push. That loss is safe -- git refuses a
# non-fast-forward -- but it is not rare, and a caller reading only the exit
# code cannot tell "another lane won the race" from "nothing to land". So the
# WHOLE sequence retries: refetch, re-resolve the base, rebuild the tree on the
# NEW base, re-commit. A base carried across attempts is exactly the stale-tree
# bug this script exists to prevent.
#
# Success is confirmed by reading back what origin/main points at, not by
# trusting the push's exit status.
set -uo pipefail

ATTEMPTS=${LAND_ATTEMPTS:-8}

DELETE=0
if [ "${1:-}" = "--delete" ]; then DELETE=1; shift; fi
MSG="${1:?commit message required}"; shift
[ $# -gt 0 ] || { echo "no paths given" >&2; exit 2; }

cd "$(git rev-parse --show-toplevel)"

# Fail on a missing path before doing any ref work, so a typo cannot produce a
# half-built attempt.
if [ "$DELETE" = 0 ]; then
  for p in "$@"; do
    [ -f "$p" ] || { echo "missing: $p" >&2; exit 3; }
  done
fi

attempt=1
while [ "$attempt" -le "$ATTEMPTS" ]; do
  git fetch -q origin main || { echo "fetch failed" >&2; exit 4; }
  BASE=$(git rev-parse origin/main)

  IDX=$(mktemp -t landidx.XXXXXX)
  GIT_INDEX_FILE=$IDX git read-tree "$BASE"
  for p in "$@"; do
    if [ "$DELETE" = 1 ]; then
      GIT_INDEX_FILE=$IDX git update-index --force-remove "$p"
    else
      MODE=100644; [ -x "$p" ] && MODE=100755
      BLOB=$(git hash-object -w -- "$p")
      GIT_INDEX_FILE=$IDX git update-index --add --cacheinfo "$MODE,$BLOB,$p"
    fi
  done
  TREE=$(GIT_INDEX_FILE=$IDX git write-tree)
  rm -f "$IDX"

  if [ "$TREE" = "$(git rev-parse "$BASE^{tree}")" ]; then
    echo "no change against $BASE — nothing to land"; exit 0
  fi
  COMMIT=$(git commit-tree "$TREE" -p "$BASE" -m "$MSG")

  # Assert the commit touches ONLY the named paths. Building the tree from
  # origin/main should make this impossible; check anyway, because the failure
  # it guards -- reverting a concurrent lane's landed work -- is silent, and a
  # numstat that a human never reads is not a guard.
  UNEXPECTED=$(comm -23 \
    <(git diff --name-only "$BASE" "$COMMIT" | sort -u) \
    <(printf '%s\n' "$@" | sort -u))
  if [ -n "$UNEXPECTED" ]; then
    echo "REFUSING: commit would touch paths that were not named:" >&2
    printf '  %s\n' $UNEXPECTED >&2
    exit 5
  fi

  if git push -q origin "$COMMIT:main" 2>/dev/null; then
    git fetch -q origin main
    if [ "$(git rev-parse origin/main)" = "$COMMIT" ]; then
      echo "landed $COMMIT on main (base $BASE, attempt $attempt)"
      git diff --numstat "$BASE" "$COMMIT"
      exit 0
    fi
  fi

  echo "attempt $attempt: main moved past $BASE; rebuilding on the new tip" >&2
  attempt=$((attempt + 1))
done

echo "gave up after $ATTEMPTS attempts; main is moving faster than this lane" >&2
exit 6
