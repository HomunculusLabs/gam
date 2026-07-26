#!/bin/bash
# #2502 chain2: parallel-sweep wheel -> relaunch flagship + validation micro.
set -u
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH
export CARGO_INCREMENTAL=0
export RUSTFLAGS="-C linker=clang -C link-arg=-fuse-ld=mold"
log() { echo "[chain2 $(date -u +%H:%M:%S)] $*" >> ~/i2502/chain.log; }
cd ~/lane-2502
if false; then
  log "CARGO CHECK FAILED (see check2.log)"
  exit 1
fi
log "check green; building wheels5"
if ! ~/venv2283/bin/maturin build --release --out ~/i2502/wheels5 -i ~/venv2283/bin/python > ~/i2502/build4.log 2>&1; then
  log "WHEEL BUILD FAILED (see build4.log)"
  exit 1
fi
W=$(ls -t ~/i2502/wheels5/*.whl | head -1)
uv pip install --python ~/venv2502/bin/python --quiet --force-reinstall --no-deps "$W"
cd ~
if ! ~/venv2502/bin/python -c "import gamfit" 2>> ~/i2502/chain.log; then
  log "WHEEL IMPORT FAILED"
  exit 1
fi
log "wheels5 installed; relaunching fits"
cd ~/i2502
RUST_LOG=info nohup nice -n 10 ~/venv2502/bin/python fit_flagship.py --arms pilot \
  --prep ~/i2502/prep_L16_p128 --pilot-k 256 --pilot-rows 3000 --n-iter 250 \
  --pilot-lanes topk --tag _par_micro > probe_par_micro.log 2>&1 &
RUST_LOG=info nohup ~/venv2502/bin/python flagship_pipeline.py \
  --prep ~/i2502/prep_L16_p128_25k --n-iter 400 > flagship.log 2>&1 &
sudo renice -n -5 -p $(pgrep -f flagship_pipeline) >/dev/null 2>&1
log "flagship + micro relaunched on parallel wheel"
