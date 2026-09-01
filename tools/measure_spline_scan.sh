#!/usr/bin/env bash
# What did restoring the affine-arithmetic enclosure actually buy?
#
# `655a418aa` put the Zonotope mean/dP enclosure back on main after a diagnostic
# push had reverted it. Its author measured w(Sum v^2/F) moving 1.1e43 -> 4.9e-10
# and the order-2 search reaching the interior optimum, but nobody has run the
# spline_scan tests against main since it landed. This does exactly that, and
# nothing else, so the answer is attributable to that commit.
set -uo pipefail
export CARGO_TARGET_DIR=/scratch.global/sauer354/zverify/target
export CARGO_BUILD_JOBS=8
source /projects/standard/hsiehph/sauer354/lane_env.sh
export CARGO_TARGET_DIR=/scratch.global/sauer354/zverify/target
export CARGO_BUILD_JOBS=8

LANE=/scratch.global/sauer354/zverify/gam
OUT=/scratch.global/sauer354/zverify
cd "$LANE"
exec > "$OUT/postland.log" 2>&1
git fetch -q origin main
git checkout -f -q origin/main
echo "POSTLAND_SHA=$(git rev-parse --short HEAD)"
echo "zonotope_present=$(grep -c 'struct Zonotope' crates/gam-solve/src/spline_scan.rs)"

date +%s
cargo nextest run -p gam-solve --all-features --lib --no-fail-fast --test-threads=4 \
  -E 'test(spline_scan)' > "$OUT/postland_test.log" 2>&1
RC=$?
date +%s
sed 's/\x1b\[[0-9;]*m//g' "$OUT/postland_test.log" > "$OUT/postland_test.clean.log"
grep -E '^error' "$OUT/postland_test.clean.log" | head -5
grep -E 'Summary' "$OUT/postland_test.clean.log" | tail -1
grep -E '^ +(FAIL|TIMEOUT)' "$OUT/postland_test.clean.log" | sed -E 's/^ +//' | cut -c1-110
echo "POSTLAND_DONE_${RC}"
