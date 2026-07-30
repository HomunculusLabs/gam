#!/bin/bash
# shrink-to-flat fix chain: harvest, then flat anchor vs gamma-zero field variants.
set -u
cd ~
echo "[CHAIN] harvest starting $(date -u +%H:%M)"
python3 harvest.py > ~/harvest.log 2>&1
grep -q "HARVEST DONE" ~/harvest.log || { echo "CHAIN_ABORT harvest"; exit 1; }
echo "[CHAIN] harvest done $(date -u +%H:%M)"

# pair 1: flat anchor vs shrink-to-flat field (warmup, no decay)
python3 curved_steelman.py 10525 0 60 8 flat > ~/f_flat8.log 2>&1 &
GAMMA_INIT=0 CURV_WARM_EPOCHS=12 python3 forms_arm.py field 10460 0 60 8 > ~/f_ff_w12.log 2>&1 &
wait
# pair 2: shrinkage variants (with and without warmup)
GAMMA_INIT=0 CURV_WARM_EPOCHS=12 CURV_DECAY=1e-3 python3 forms_arm.py field 10460 0 60 8 > ~/f_ff_w12_wd3.log 2>&1 &
GAMMA_INIT=0 CURV_WARM_EPOCHS=0 CURV_DECAY=1e-3 python3 forms_arm.py field 10460 0 60 8 > ~/f_ff_w0_wd3.log 2>&1 &
wait
echo "CHAIN_DONE"
