#!/bin/bash
# #2283 / #1017 — cheap scaling witness for the SAE curved-tier serial wall.
#
# Drives `crates/gam-sae/examples/curved_tier_scaling_2283.rs` over two
# one-dimensional sweeps that separate the per-row packed row-jet buffer from
# everything else in the curved fit:
#
#   * `p` at fixed chart count — the buffer is linear in `p`; the chart-side
#     work (basis, penalties, atlas) is not.
#   * chart count at fixed `p` and fixed `top_k` — the buffer is linear in the
#     chart count through `n_beta`, while the *live* work is linear in `top_k`
#     and therefore constant across this sweep.
#
# A cost that grows with the second sweep at fixed `top_k` is, by construction,
# work done on borders whose atom is inactive on the row: structural zeros.
#
# Usage:
#   BIN=<path to release example> bash bench/sae_curved_border_scaling_2283.sh <label>
#
# Emits one `[2283-scaling]` line per point on stdout, each already carrying its
# own shape, plus a `%CPU` reading per point so a serial section cannot be
# reported as a parallel one.
set -u

BIN=${BIN:?set BIN to the built curved_tier_scaling_2283 example}
LABEL=${1:-unlabelled}

N=${N:-256}
TOPK=${TOPK:-2}
MAXITER=${MAXITER:-8}
# The outer rho search makes the trajectory LENGTH data-dependent, which would
# turn a per-iterate cost measurement into a measurement of how many criterion
# evaluations the optimizer happened to take. Fixed inner budget, no outer
# search: both arms then walk the identical trajectory and `r2` must agree to
# the last digit, which is simultaneously the correctness gate for the change.
RHO=${RHO:-0}

echo "== #2283 curved-tier border scaling: ${LABEL}"
echo "== binary ${BIN}"
echo "== fixed n=${N} top_k=${TOPK} max_iter=${MAXITER} rho_search=${RHO}"

run_point() {
    local p=$1 charts=$2
    local out
    out=$(GAM_2283_N="${N}" GAM_2283_P="${p}" GAM_2283_CHARTS="${charts}" \
        GAM_2283_TOPK="${TOPK}" GAM_2283_MAXITER="${MAXITER}" \
        GAM_2283_RHO="${RHO}" GAM_2283_LOG=warn \
        /usr/bin/time -f "cpu_percent=%P max_rss_kb=%M" "${BIN}" 2>&1)
    echo "arm=${LABEL} ${out}" | tr '\n' ' '
    echo
}

echo "-- sweep A: p at charts=8"
for p in 64 128 256 512 1024; do
    run_point "${p}" 8
done

echo "-- sweep B: charts at p=128"
for charts in 4 8 16 32; do
    run_point 128 "${charts}"
done
