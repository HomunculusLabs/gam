#!/usr/bin/env bash
# Merge the shared working tree's version of a file with what is on main.
#
# `land_paths.sh` commits by plumbing and never touches the working tree, so a
# lane's local copy goes STALE the instant anything lands. A dirty file is
# therefore almost always `some older base + that lane's edits`, and BOTH naive
# moves are wrong: landing it wholesale reverts upstream (that cost 31 files
# earlier today), and restoring upstream wholesale discards the lane's edits.
#
# The base nobody recorded is recoverable: walk this file's history and pick the
# blob closest to the tree version. Exclude the recovery commits, or the search
# picks one of them -- they contain the tree version byte-for-byte, so the
# "merge" silently becomes a no-op that still reports success.
#
#   tools/merge_tree_onto_main.sh <path>
#
# Leaves the merged result in the working tree. Conflicts are reported and left
# for a human; nothing is auto-resolved.
set -uo pipefail
F="${1:?path required}"
[ -f "$F" ] || { echo "SKIP  $F (not a file)"; exit 0; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
git show "origin/main:${F}" > "$TMP/ours" 2>/dev/null || { echo "NEW   $F (not on main -- land as-is)"; exit 0; }
cp "$F" "$TMP/theirs"

cmp -s "$TMP/ours" "$TMP/theirs" && { echo "SAME  $F"; exit 0; }

BEST=""; BESTN=999999
for c in $(git log --format='%H %s' -60 origin/main -- "$F" \
             | grep -vE '^[0-9a-f]+ (land: incorporate lane work|fix: restore what my bulk land reverted|revert\(#979\))' \
             | cut -d' ' -f1); do
  git show "${c}:${F}" > "$TMP/cand" 2>/dev/null || continue
  n=$(diff -u "$TMP/cand" "$TMP/theirs" 2>/dev/null | grep -c '^[-+][^-+]')
  if [ "$n" -lt "$BESTN" ]; then BESTN=$n; BEST=$c; fi
done
[ -n "$BEST" ] || { echo "NOBASE $F"; exit 0; }
git show "${BEST}:${F}" > "$TMP/base"

cp "$TMP/ours" "$TMP/merged"
if git merge-file -q -L main -L base -L tree "$TMP/merged" "$TMP/base" "$TMP/theirs" 2>/dev/null; then
  cp "$TMP/merged" "$F"
  echo "MERGED $F (base $(git log -1 --format=%h $BEST), dist $BESTN)"
else
  echo "CONFLICT $F -- left unmerged" >&2
  exit 5
fi
