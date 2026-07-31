#!/usr/bin/env bash
# Negative control for the ban scanner: prove it still FAILS on a real violation.
#
# WHY THIS EXISTS. `forbid_build_rs_self_tampering` in the root `build.rs` checks
# the scanner's SHAPE — that exactly one report site exists and that an
# unconditional `process::exit` follows it. Shape is not behaviour. Gut the body
# of a rule so it matches nothing and every shape assertion still passes, which
# is precisely the failure the self-integrity gate exists to prevent and the one
# thing it structurally cannot see. A gate that has never been observed failing
# is not a gate.
#
# The control cannot live inside `build.rs`. The scanner's file list is memoised
# (`SCANNABLE_FILES: OnceLock`, populated on first call from whatever root is
# passed), so an in-process self-test either reads the cached real tree and
# passes vacuously, or — if it ran first — poisons the cache with a synthetic
# tree and leaves the REAL scan looking at nothing. Both outcomes are worse than
# no self-test. Planting on disk and running the scanner as a subprocess has
# neither hazard: it exercises the real `build.rs`, compiled from the tree under
# test, exactly as CI and cargo run it.
#
# The probe file is deliberately NOT added to any `mod` tree: the scanner walks
# files on disk, so it is seen by the scanner and ignored by rustc. That keeps
# the control from perturbing any build.
#
# Usage:  scripts/ban_scanner_selftest.sh [repo-root]
# Exit:   0 both arms behaved, 1 the scanner failed to catch a planted violation
#         (or rejected a clean tree), 2 harness/setup problem.
#
# `ban_scanner.sh` distinguishes its exit codes and so must this control:
# 0 = clean, 1 = violations found, 2 = it could not run at all (no build.rs, or
# rustc failed to compile it). Collapsing 2 into "violation" would make arm B
# pass whenever the scanner is merely BROKEN — a false green in the one check
# whose whole purpose is to catch a scanner that reports nothing.

set -uo pipefail

root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
scanner="${root}/scripts/ban_scanner.sh"
if [ ! -x "$scanner" ] && [ ! -f "$scanner" ]; then
  echo "ban_scanner_selftest: no scanner at ${scanner}" >&2
  exit 2
fi

probe="${root}/crates/gam-math/src/zz_ban_scanner_selftest_probe.rs"
if [ -e "$probe" ]; then
  echo "ban_scanner_selftest: ${probe} already exists; refusing to clobber it" >&2
  exit 2
fi

# Remove the probe on ANY exit path. Leaving it behind would plant a real
# violation in the tree and fail every subsequent build.
cleanup() { rm -f "$probe"; }
trap cleanup EXIT INT TERM

# --- Arm A: the tree as committed must be clean -----------------------------
bash "$scanner" "$root" >/dev/null 2>&1
clean_rc=$?
if [ "$clean_rc" -eq 2 ]; then
  echo "ban_scanner_selftest: SETUP PROBLEM — the scanner could not run (rc=2)." >&2
  echo "  Either ${root}/build.rs is missing or rustc failed to compile it." >&2
  echo "  This is not a verdict about the tree; re-run once the toolchain is available." >&2
  exit 2
fi
if [ "$clean_rc" -ne 0 ]; then
  echo "ban_scanner_selftest: ARM A FAILED — the scanner rejects the committed tree (rc=${clean_rc})." >&2
  echo "  The real scanner step reports the offending file:line; fix that first." >&2
  exit 1
fi

# --- Arm B: a planted violation must be caught ------------------------------
# `#[allow(...)]` is banned anywhere by the umbrella rule, so this is the
# cheapest construct that exercises a real rule end to end.
cat > "$probe" <<'PROBE'
// Planted by scripts/ban_scanner_selftest.sh. If you are reading this in a
// committed tree, the self-test died before its cleanup trap ran — delete it.
pub fn ban_scanner_selftest_probe() {
    #[allow(dead_code)]
    let planted = 1;
    let _keep = planted;
}
PROBE

bash "$scanner" "$root" >/dev/null 2>&1
planted_rc=$?
cleanup

# Require exactly 1 (violations found). Anything else is a failure of this
# control, INCLUDING 2 — a scanner that cannot run has not caught anything, and
# treating that as success is the false green this whole file exists to prevent.
if [ "$planted_rc" -eq 2 ]; then
  echo "ban_scanner_selftest: SETUP PROBLEM — the scanner could not run on the planted arm (rc=2)." >&2
  exit 2
fi
if [ "$planted_rc" -ne 1 ]; then
  echo "ban_scanner_selftest: ARM B FAILED — the scanner did not report a violation for a planted" >&2
  echo "  \`#[allow(dead_code)]\` (rc=${planted_rc}, expected 1)." >&2
  echo "  The scanner is not detecting violations. Its shape gate in build.rs can still pass while" >&2
  echo "  this is true; that is exactly why this control exists. Do not silence it — repair the rule." >&2
  exit 1
fi

echo "ban_scanner_selftest: OK (clean tree rc=0, planted violation rc=${planted_rc})"
exit 0
