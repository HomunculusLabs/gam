#!/usr/bin/env bash
#
# Does every crate that can be documented still document? (#2711)
#
# WHY THIS EXISTS
# ---------------
# `cargo doc` exits 101 on this workspace and no CI job had ever noticed,
# because no workflow ran `cargo doc` or `rustdoc` at all. The surface was dark
# by omission, not by a gate that skipped it: `docs.yml` is mkdocs, so "docs is
# green" was true every night and said nothing about rustdoc. The errors exist
# because `[lints.rust] warnings = "deny"` is forwarded to rustdoc as well as
# rustc, which promotes rustdoc's default-`warn` lints (mostly
# `rustdoc::private_intra_doc_links`) to hard errors -- a thing `cargo build`
# and `cargo test` never see, which is why every other lane reads green.
#
# WHY IT IS A RATCHET AND NOT A PASS/FAIL ON ZERO
# -----------------------------------------------
# There are hundreds of these errors across most of the workspace. A job that
# demanded zero would be red the day it landed, and a gate that is red on
# arrival trains everyone to ignore it -- which makes it the same non-instrument
# as the job that never ran. So this compares against a committed ledger of the
# crates KNOWN to be red, and fails in exactly two directions:
#
#   * a crate NOT in the ledger is red  -> regression. Someone broke rustdoc for
#     a crate that documented cleanly. This is the gate.
#   * a crate IN the ledger is clean    -> the ledger is stale. Whoever fixed the
#     crate must delete its line, in the fix's own commit, so the covered set
#     only ever grows.
#
# The second direction is what keeps this from rotting into a rubber stamp, and
# it is also the POSITIVE CONTROL: the ledger crates are known-failing inputs
# run through the same command on every invocation. If the command silently did
# nothing -- wrong flags, no toolchain, an empty crate list -- those crates would
# come back "clean" and the job fails loudly. A green run here therefore cannot
# be confused with a run that measured nothing, which is the failure mode that
# hid the whole surface in the first place.
#
# It prints a per-crate table and a `scanned N crates` line for the same reason:
# silent success and silent non-execution must not look alike.
#
# USAGE
#   scripts/rustdoc_ratchet.sh [ledger-path]
# Exit 0 = ledger matches reality. Exit 1 = a direction above fired. Exit 2 =
# the script could not measure (missing tool, no crates, unreadable ledger).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEDGER="${1:-${REPO_ROOT}/scripts/rustdoc_red_crates.txt}"

# `gam-pyffi` is excluded for the same reason `cross-check.yml` excludes it from
# both compile jobs: documenting it builds `pyo3-ffi`'s build script, which needs
# a configured Python interpreter. Its absence here is a known hole, not a claim
# of cleanliness -- say so out loud rather than letting the exclusion pass as
# coverage.
EXCLUDED_CRATES=("gam-pyffi")

die() {
  echo "rustdoc-ratchet: FATAL: $*" >&2
  exit 2
}

command -v cargo >/dev/null 2>&1 || die "cargo not on PATH"
command -v jq >/dev/null 2>&1 || die "jq not on PATH (needed to read cargo metadata)"
[ -r "${LEDGER}" ] || die "ledger not readable at ${LEDGER}. This file is required: without it there is nothing to compare against, and a run with nothing to compare against is not a measurement."

cd "${REPO_ROOT}" || die "cannot cd to ${REPO_ROOT}"

mapfile -t ALL_CRATES < <(cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name' | sort)
[ "${#ALL_CRATES[@]}" -gt 0 ] || die "cargo metadata returned no workspace packages"

mapfile -t LEDGER_CRATES < <(grep -vE '^\s*(#|$)' "${LEDGER}" | tr -d '\r' | awk '{print $1}' | sort -u)

is_excluded() {
  local needle="$1" c
  for c in "${EXCLUDED_CRATES[@]}"; do
    [ "$c" = "$needle" ] && return 0
  done
  return 1
}

in_ledger() {
  local needle="$1" c
  for c in "${LEDGER_CRATES[@]:-}"; do
    [ "$c" = "$needle" ] && return 0
  done
  return 1
}

# A ledger line naming a crate that no longer exists is a silent hole: it would
# sit there forever satisfying nothing.
UNKNOWN_LEDGER=()
for c in "${LEDGER_CRATES[@]:-}"; do
  found=1
  for k in "${ALL_CRATES[@]}"; do
    [ "$k" = "$c" ] && found=0 && break
  done
  [ "$found" -eq 0 ] || UNKNOWN_LEDGER+=("$c")
done

SCANNED=0
RED=()
CLEAN=()
declare -A ERROR_COUNT=()

echo "rustdoc-ratchet: ledger=${LEDGER} (${#LEDGER_CRATES[@]} crate(s) recorded red)"
echo "rustdoc-ratchet: excluded from this scan: ${EXCLUDED_CRATES[*]}"
echo

for crate in "${ALL_CRATES[@]}"; do
  if is_excluded "$crate"; then
    printf '  %-24s EXCLUDED (not measured, not a cleanliness claim)\n' "$crate"
    continue
  fi
  log="$(mktemp)"
  cargo doc -p "$crate" --no-deps >"$log" 2>&1
  rc=$?
  errors="$(grep -cE '^error' "$log" || true)"
  ERROR_COUNT["$crate"]="$errors"
  SCANNED=$((SCANNED + 1))
  if [ "$rc" -eq 0 ]; then
    CLEAN+=("$crate")
    printf '  %-24s ok        rc=0\n' "$crate"
  else
    RED+=("$crate")
    printf '  %-24s RED       rc=%s errors=%s\n' "$crate" "$rc" "$errors"
    # The first few lines are what a fixer needs; the whole log is noise here.
    grep -E '^error' "$log" | head -n 3 | sed 's/^/      /'
  fi
  rm -f "$log"
done

echo
echo "rustdoc-ratchet: scanned ${SCANNED} crates: ${#CLEAN[@]} clean, ${#RED[@]} red"

status=0

# Direction 1: a crate outside the ledger went red. This is the gate.
REGRESSED=()
for c in "${RED[@]:-}"; do
  in_ledger "$c" || REGRESSED+=("$c")
done
if [ "${#REGRESSED[@]}" -gt 0 ]; then
  status=1
  echo
  echo "::error title=rustdoc regression::These crates documented cleanly and no longer do: ${REGRESSED[*]}"
  for c in "${REGRESSED[@]}"; do
    echo "  ${c}: ${ERROR_COUNT[$c]} rustdoc error(s). Reproduce with: cargo doc -p ${c} --no-deps"
  done
fi

# Direction 2: a ledger crate is clean. Someone fixed it and left the ledger
# behind; the covered set must only grow.
PROMOTABLE=()
for c in "${CLEAN[@]:-}"; do
  in_ledger "$c" && PROMOTABLE+=("$c")
done
if [ "${#PROMOTABLE[@]}" -gt 0 ]; then
  status=1
  echo
  echo "::error title=rustdoc ledger is stale::These crates now document cleanly and must be removed from ${LEDGER}: ${PROMOTABLE[*]}"
  echo "  Delete those lines in the same commit as the fix. Until then this crate is not gated, so a later regression in it would pass unnoticed."
fi

if [ "${#UNKNOWN_LEDGER[@]}" -gt 0 ]; then
  status=1
  echo
  echo "::error title=rustdoc ledger names unknown crates::${UNKNOWN_LEDGER[*]} are in the ledger but are not workspace members. A ledger line that matches nothing is satisfied by nothing."
fi

if [ "$status" -eq 0 ]; then
  echo "rustdoc-ratchet: OK -- no crate outside the ledger is red, and every ledger crate is still red (so this run really did run rustdoc)."
fi
exit "$status"
