#!/bin/bash
# chain7: round-2 parallel wheel -> doc-split flagship.
set -u
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH
export CARGO_INCREMENTAL=0
export RUSTFLAGS="-C linker=clang -C link-arg=-fuse-ld=mold"
log() { echo "[chain7 $(date -u +%H:%M:%S)] $*" >> ~/i2502/chain.log; }
cd ~/i2502
ln -sf test_doc.npy prep_L16_p128_doc/test.npy
ln -sf rows_test_doc.npy prep_L16_p128_doc/rows_test.npy

until grep -qE "Finished|error\[" ~/i2502/example_build4.log; do sleep 10; done
if ! grep -q Finished ~/i2502/example_build4.log; then
  log "ROUND2 BUILD FAILED"; exit 1
fi
log "round2 example built"

# stop old-round fits; quick A/B at flagship shape (90s of cycles)
OLD=$(pgrep -f "flag_chart.bin" | head -1); [ -n "$OLD" ] && kill $OLD
PYOLD=$(pgrep -f "flagship_pipeline" | head -1); [ -n "$PYOLD" ] && kill $PYOLD
cd ~/lane-2502
RUST_LOG=info nohup ./target/release/examples/support_real_chart \
  ~/i2502/flag_chart.bin 25000 128 32000 8 400 > ~/i2502/flag_rust2.log 2>&1 &
ABPID=$!
sleep 240
C=$(grep -c "support fixed-point cycle" ~/i2502/flag_rust2.log || true)
log "round2 A/B: $C cycles in ~110s of fit time (seed ~130s)"
kill $ABPID 2>/dev/null

# wheels9 + doc-split flagship relaunch
if ! ~/venv2283/bin/maturin build --release --out ~/i2502/wheels9 -i ~/venv2283/bin/python > ~/i2502/build9.log 2>&1; then
  log "WHEELS9 BUILD FAILED"; exit 1
fi
W=$(ls -t ~/i2502/wheels9/*.whl | head -1)
uv pip install --python ~/venv2502/bin/python --quiet --force-reinstall --no-deps "$W"
cd ~
~/venv2502/bin/python -c "import gamfit" || { log "WHEELS9 IMPORT FAILED"; exit 1; }
log "wheels9 installed; launching doc-split flagship"
cd ~/i2502
RUST_LOG=warn nohup ~/venv2502/bin/python flagship_pipeline.py \
  --prep ~/i2502/prep_L16_p128_doc --n-iter 400 > flagship.log 2>&1 &
sudo renice -n -5 -p $(pgrep -f flagship_pipeline) >/dev/null 2>&1
log "doc-split flagship launched (n=44818, test=12000 held-out docs)"
