#!/usr/bin/env bash
# Merge a lane's tree-side work back together with what landed upstream.
#
# Context: `land_paths.sh` commits by plumbing and never updates the working
# tree, so a lane's local copy goes STALE the moment anything lands. A tree copy
# is therefore usually `some old base + that lane's new edits`. Landing it
# wholesale reverts upstream; restoring upstream wholesale discards the lane's
# edits. Both are wrong; the answer is a three-way merge.
#
# The only hard part is finding the BASE, because nobody recorded which commit
# the lane checked out. We recover it: walk this file's history and pick the
# blob with the smallest diff to the tree version. That commit is, by
# construction, the one the lane started from.
#
#   tools/merge_lane_work.sh <path> <commit-holding-the-tree-version>
#
# Writes the merged result into the working tree. Conflicts are left in the file
# with markers and reported, never auto-resolved.
set -uo pipefail
F="${1:?path required}"
TREE_COMMIT="${2:?commit holding the tree version required}"

git fetch -q origin main
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

git show "${TREE_COMMIT}:${F}"   > "$TMP/theirs" 2>/dev/null || { echo "no tree version of $F at $TREE_COMMIT" >&2; exit 2; }
git show "origin/main:${F}"      > "$TMP/ours"   2>/dev/null || { echo "$F not on main" >&2; exit 3; }

# Recover the base: the historical blob closest to the tree version.
#
# EXCLUDE the bulk-land and restore commits. They are in main's history and one
# of them contains the tree version byte-for-byte, so an unfiltered search picks
# it as the base at distance 0 -- which makes base==theirs and the merge a
# silent no-op that still prints "MERGED CLEAN". That happened on the first run
# of this script and produced ten vacuous successes.
BEST=""; BESTN=999999
for c in $(git log --format='%H %s' -60 origin/main -- "$F" \
             | grep -vE '^[0-9a-f]+ (land: incorporate lane work|fix: restore what my bulk land reverted)' \
             | cut -d' ' -f1); do
  git show "${c}:${F}" > "$TMP/cand" 2>/dev/null || continue
  n=$(diff -u "$TMP/cand" "$TMP/theirs" 2>/dev/null | grep -c '^[-+][^-+]')
  if [ "$n" -lt "$BESTN" ]; then BESTN=$n; BEST=$c; fi
done
[ -n "$BEST" ] || { echo "could not recover a base for $F" >&2; exit 4; }
git show "${BEST}:${F}" > "$TMP/base"
echo "base=$(git log -1 --format=%h $BEST)  distance_to_tree=${BESTN}"

cp "$TMP/ours" "$TMP/merged"
if git merge-file -L main -L base -L lane "$TMP/merged" "$TMP/base" "$TMP/theirs"; then
  cp "$TMP/merged" "$F"
  echo "MERGED CLEAN  $F"
  git diff --numstat origin/main -- "$F"
else
  echo "CONFLICTS in $F — left unmerged, resolve by hand:" >&2
  grep -c '^<<<<<<<' "$TMP/merged" >&2
  cp "$TMP/merged" "$F.merge-conflict"
  exit 5
fi
