#!/usr/bin/env bash
#
# Does every crate whose doctests pass still pass them? (#2732)
#
# WHY THIS IS SEPARATE FROM scripts/rustdoc_ratchet.sh
# ----------------------------------------------------
# That one runs `cargo doc -p <crate> --no-deps` -- doc GENERATION. This runs
# `cargo test --doc -p <crate>` -- doc TESTS. Different command, different
# surface, and neither result predicts the other: measured at c3635e04c,
# `cargo test --doc -p gam-solve` was `0 passed; 12 failed` while
# `git grep -rn "test --doc" -- .github scripts` returned nothing at all.
#
# They are NOT merged into one ledger on purpose. A ledger that means "fails one
# of two different commands" cannot say which, and its promotion direction ("this
# crate is clean now") stops being interpretable. One ledger, one command, one
# meaning. The two will also shrink at different rates even though one root
# cause (Unicode in indented `///` blocks that rustdoc compiles as Rust) feeds
# both, because fencing a block as ```text fixes the doctest without touching
# whatever the doc build objects to.
#
# WHY IT IS A RATCHET
# -------------------
# Same reasoning as #2711, and the rule that shaped it: never enable a job that
# is red on arrival, because a gate red the day it lands is ignored from the day
# it lands -- the same non-instrument as one that never ran. So this compares
# against a committed ledger of crates KNOWN to fail, and fails in two
# directions:
#
#   * a crate NOT in the ledger fails    -> regression. This is the gate.
#   * a crate IN the ledger passes       -> stale ledger; delete its line in the
#                                           commit that fixed it.
#
# The second direction is also the POSITIVE CONTROL: the ledger crates are
# known-failing inputs pushed through the identical command on every run, so a
# scan that silently measured nothing reports them passing and fails loudly. A
# green here cannot be a non-run. A `scanned N crates` line exists for the same
# reason -- silent success and silent non-execution must never look alike.
#
# MODES
#   scripts/doctest_ratchet.sh                 gate against the ledger
#   scripts/doctest_ratchet.sh --census        measure and PRINT, gate nothing
#
# `--census` exists only to populate the ledger from measurement rather than
# from a guess, and to triage later. It is refused under `push` so it can never
# become a job that always passes.
#
# Exit 0 = ledger matches reality (or census completed). 1 = a direction fired.
# 2 = could not measure.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEDGER="${REPO_ROOT}/scripts/doctest_red_crates.txt"
CENSUS=0
if [ "${1:-}" = "--census" ]; then
  CENSUS=1
fi

# Same exclusion as cross-check.yml's compile jobs and the rustdoc ratchet:
# building gam-pyffi needs a configured Python interpreter for pyo3-ffi. Excluded
# is NOT a claim of cleanliness, and the scan says so per crate.
EXCLUDED_CRATES=("gam-pyffi")

die() {
  echo "doctest-ratchet: FATAL: $*" >&2
  exit 2
}

command -v cargo >/dev/null 2>&1 || die "cargo not on PATH"
command -v jq >/dev/null 2>&1 || die "jq not on PATH (needed to read cargo metadata)"

if [ "${CENSUS}" -eq 1 ] && [ "${GITHUB_EVENT_NAME:-}" = "push" ]; then
  die "--census is a measurement, not a verdict, and must not run as a push gate"
fi

if [ "${CENSUS}" -eq 0 ] && [ ! -r "${LEDGER}" ]; then
  die "ledger not readable at ${LEDGER}. Run with --census first and commit the measured ledger; a run with nothing to compare against is not a measurement."
fi

cd "${REPO_ROOT}" || die "cannot cd to ${REPO_ROOT}"

mapfile -t ALL_CRATES < <(cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name' | sort)
[ "${#ALL_CRATES[@]}" -gt 0 ] || die "cargo metadata returned no workspace packages"

LEDGER_CRATES=()
if [ -r "${LEDGER}" ]; then
  mapfile -t LEDGER_CRATES < <(grep -vE '^\s*(#|$)' "${LEDGER}" | tr -d '\r' | awk '{print $1}' | sort -u)
fi

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

if [ "${CENSUS}" -eq 1 ]; then
  echo "=============================================================="
  echo " doctest-ratchet CENSUS -- THIS IS A MEASUREMENT, NOT A VERDICT"
  echo " Nothing below gates anything. Its purpose is to produce the"
  echo " ledger from measured fact instead of from a guess."
  echo "=============================================================="
else
  echo "doctest-ratchet: ledger=${LEDGER} (${#LEDGER_CRATES[@]} crate(s) recorded failing)"
fi
echo "doctest-ratchet: excluded from this scan: ${EXCLUDED_CRATES[*]}"
echo

SCANNED=0
FAILING=()
PASSING=()
declare -A SUMMARY=()

for crate in "${ALL_CRATES[@]}"; do
  if is_excluded "$crate"; then
    printf '  %-24s EXCLUDED (not measured, not a cleanliness claim)\n' "$crate"
    continue
  fi
  log="$(mktemp)"
  cargo test --doc -p "$crate" >"$log" 2>&1
  rc=$?
  # Strip ANSI before matching: cargo colours its output when it decides the
  # sink is a terminal, and a `^test result:` match then fails against a line
  # that begins with an escape sequence. The rustdoc ratchet reported `errors=0`
  # beside `rc=101` for exactly that reason before it was fixed.
  sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' "$log" >"${log}.plain"
  result_line="$(grep -E '^test result:' "${log}.plain" | tail -n 1)"
  SUMMARY["$crate"]="${result_line:-(no test result line)}"
  SCANNED=$((SCANNED + 1))
  if [ "$rc" -eq 0 ]; then
    PASSING+=("$crate")
    printf '  %-24s pass   rc=0   %s\n' "$crate" "${result_line:-(no doctests)}"
  else
    FAILING+=("$crate")
    printf '  %-24s FAIL   rc=%s   %s\n' "$crate" "$rc" "${result_line:-(no test result line)}"
    if [ -z "${result_line}" ]; then
      # A non-zero exit with no `test result:` line means the doctest harness
      # never reported -- a build failure, not a failing doctest. Those are
      # different findings with different owners, so do not let one read as the
      # other.
      echo "      (no test result line: the harness did not run. Build failure tail:)"
      grep -E '^error' "${log}.plain" | tail -n 3 | sed 's/^/      /'
    fi
  fi
  rm -f "$log" "${log}.plain"
done

echo
echo "doctest-ratchet: scanned ${SCANNED} crates: ${#PASSING[@]} passing, ${#FAILING[@]} failing"

if [ "${CENSUS}" -eq 1 ]; then
  echo
  echo "----- paste into scripts/doctest_red_crates.txt (below the header) -----"
  for c in "${FAILING[@]:-}"; do
    [ -n "$c" ] && echo "$c"
  done
  echo "-----------------------------------------------------------------------"
  echo "CENSUS COMPLETE -- gated nothing."
  exit 0
fi

status=0

REGRESSED=()
for c in "${FAILING[@]:-}"; do
  [ -n "$c" ] || continue
  in_ledger "$c" || REGRESSED+=("$c")
done
if [ "${#REGRESSED[@]}" -gt 0 ]; then
  status=1
  echo
  echo "::error title=Doctest regression::These crates' doctests passed and no longer do: ${REGRESSED[*]}"
  for c in "${REGRESSED[@]}"; do
    echo "  ${c}: ${SUMMARY[$c]}"
    echo "    Reproduce: cargo test --doc -p ${c}"
    echo "    Note an INDENTED block in a /// comment is compiled as Rust. If it is prose or math, fence it as a text block."
  done
fi

PROMOTABLE=()
for c in "${PASSING[@]:-}"; do
  [ -n "$c" ] || continue
  in_ledger "$c" && PROMOTABLE+=("$c")
done
if [ "${#PROMOTABLE[@]}" -gt 0 ]; then
  status=1
  echo
  echo "::error title=Doctest ledger is stale::These crates' doctests now pass and must be removed from ${LEDGER}: ${PROMOTABLE[*]}"
  echo "  Delete those lines in the same commit as the fix. Until then the crate is not gated, so a later regression in it passes unnoticed."
fi

UNKNOWN_LEDGER=()
for c in "${LEDGER_CRATES[@]:-}"; do
  [ -n "$c" ] || continue
  found=1
  for k in "${ALL_CRATES[@]}"; do
    [ "$k" = "$c" ] && found=0 && break
  done
  [ "$found" -eq 0 ] || UNKNOWN_LEDGER+=("$c")
done
if [ "${#UNKNOWN_LEDGER[@]}" -gt 0 ]; then
  status=1
  echo
  echo "::error title=Doctest ledger names unknown crates::${UNKNOWN_LEDGER[*]} are in the ledger but are not workspace members. A ledger line that matches nothing is satisfied by nothing."
fi

if [ "$status" -eq 0 ]; then
  echo "doctest-ratchet: OK -- no crate outside the ledger fails, and every ledger crate still fails (so this run really did run the doctests)."
fi
exit "$status"
