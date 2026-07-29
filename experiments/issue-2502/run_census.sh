#!/usr/bin/env bash
# One self-contained run of the Gemma Scope 2 shattered-circle census.
#
# Every number in the write-up comes from this script: it provisions nothing it
# cannot re-provision, so a machine disappearing mid-run costs the wall-clock and
# nothing else. Results land in ~/census_out, small enough to copy off as soon as
# each stage finishes.
#
#   bash run_census.sh            # full sweep
#   LAYERS="17" bash run_census.sh   # just the flagship layer
set -euo pipefail

LAYERS=${LAYERS:-"17 9 22 29"}
FLAGSHIP=${FLAGSHIP:-17}
ROWS=${ROWS:-100000}
ATOMS=${ATOMS:-15040}
MINCO=${MINCO:-300}
PARSE=${PARSE:-2000}
# e-BH over ~2e4 screened pairs needs e >= m/(alpha*rank); with alpha = 0.05 this
# resolution lets the ledger reject down to rank ~2.
DRAWS=${DRAWS:-200000}
ALPHA=${ALPHA:-0.05}
OUT=$HOME/census_out
mkdir -p "$OUT"

if [ ! -x "$HOME/.cargo/bin/cargo" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
    sh -s -- -y --default-toolchain 1.97.1 --profile minimal
fi
. "$HOME/.cargo/env"

if [ ! -d "$HOME/venv" ]; then
  python3 -m venv "$HOME/venv"
  "$HOME/venv/bin/pip" install -q --upgrade pip
  "$HOME/venv/bin/pip" install -q torch transformers datasets safetensors matplotlib numpy
fi
PY=$HOME/venv/bin/python

if [ ! -d "$HOME/gam" ]; then
  git clone --depth 50 https://github.com/SauersML/gam.git "$HOME/gam"
fi
cd "$HOME/gam"
git fetch --depth 50 origin main && git checkout -f origin/main
CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-30} cargo build --release -p gam-sae \
  --example curl_census_foreign
BIN=$HOME/gam/target/release/examples/curl_census_foreign
DUMP=$HOME/gam/experiments/issue-2502/gemma_scope_dump.py

for L in $LAYERS; do
  D=$HOME/gs2L$L
  [ -f "$D/meta.json" ] || $PY "$DUMP" "$D" --layer "$L" --rows "$ROWS"
  # arm 1: the dictionary as released
  $BIN "$D" "$ATOMS" "$MINCO" "$PARSE" 0 -0.85 "$DRAWS" "$ALPHA" "$OUT/cen_L${L}.json"
  if [ "$L" = "$FLAGSHIP" ]; then
    # arm 2: coalescing off — the screen a transcription of curl.rs ships
    $BIN "$D" "$ATOMS" "$MINCO" "$PARSE" 0 -1.5 "$DRAWS" "$ALPHA" "$OUT/ctrl_nocoal.json"
    # arm 3: whole-run permutation nulls, on top of the per-pair ledger
    for S in 20502 20503 20504; do
      $BIN "$D" "$ATOMS" "$MINCO" "$PARSE" "$S" -0.85 "$DRAWS" "$ALPHA" "$OUT/cen_null${S}.json"
    done
  fi
  echo "LAYER $L DONE"
done

cp "$HOME/gs2L$FLAGSHIP/vocab.json" "$OUT/vocab.json"
cp "$HOME/gs2L$FLAGSHIP/tokens.i32" "$OUT/tokens.i32"
$PY "$HOME/gam/experiments/issue-2502/census_figs.py" "$OUT" "$OUT" "$OUT" || true
echo CENSUS_ALL_DONE
