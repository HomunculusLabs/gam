#!/bin/bash
# #2502 autonomous executor: fires each stage the moment its precondition lands.
set -u
cd ~/i2502
log() { echo "[chain $(date -u +%H:%M:%S)] $*" >> ~/i2502/chain.log; }
log "chain armed"

# 1. wait for the patched wheel + venv2502
until grep -q "PATCHED WHEEL OK" rebuild.log 2>/dev/null; do sleep 5; done
log "patched wheel ready"

# 2a. MICRO canary — certificate verdict in ~1 min (K=256 > P=128, 3k rows)
RUST_LOG=info nohup nice -n 5 ~/venv2502/bin/python fit_flagship.py --arms pilot \
  --prep ~/i2502/prep_L16_p128 --pilot-k 256 --pilot-rows 3000 --n-iter 250 \
  --pilot-lanes topk --tag _patched_micro > probe_micro.log 2>&1 &
log "micro canary launched"

# 2b. mid probe (K=1024, 25k rows — the scaling datum)
RUST_LOG=info nohup nice -n 10 ~/venv2502/bin/python fit_flagship.py --arms pilot \
  --prep ~/i2502/prep_L16_p128 --pilot-k 1024 --pilot-rows 25000 --n-iter 250 \
  --pilot-lanes topk --tag _patched_i250 > probe_patched.log 2>&1 &
log "probe launched"

# 3. flagship (K=32000, 25k rows, big budget)
RUST_LOG=info nohup ~/venv2502/bin/python flagship_pipeline.py \
  --prep ~/i2502/prep_L16_p128_25k --n-iter 400 > flagship.log 2>&1 &
log "flagship launched"

# 4. torch baseline retrain on the matched 25k train set (GPU, ~1 min)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup ~/venv2283/bin/python \
  torch_topk_sae_p128.py --prep ~/i2502/prep_L16_p128_25k > topk_p128_25k.log 2>&1 &
log "torch retrain launched"

# 5. second rebuild with the row-skip speed patch (for restart headroom + PR)
~/venv2283/bin/python ~/i2502/patch_rust2.py >> chain.log 2>&1
nohup bash rebuild2502b.sh > rebuild2.log 2>&1 &
log "speed-patch rebuild launched"

# 6. flagship done -> stage B -> figures
until grep -qE "PIPELINE DONE|Traceback|GamError" flagship.log 2>/dev/null; do sleep 20; done
if grep -q "PIPELINE DONE" flagship.log; then
  log "pipeline done; stage B starting"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ~/venv2283/bin/python \
    gpu_stage_b.py > stage_b.log 2>&1
  log "stage B done; figures"
  ~/venv2283/bin/python steer_figs.py > figs.log 2>&1
  ~/venv2283/bin/python benchmark_fig.py --suffix _p128 \
    --out ~/i2502/flagship/fig_benchmark.png >> figs.log 2>&1
  log "ALL DONE"
else
  log "FLAGSHIP FAILED (see flagship.log)"
fi
