#!/bin/bash
# 60-epoch screening matrix over atom-geometry forms, two arms per GPU slot.
# All at ~1.347M decoder params; the contest rung is 8 scalars/token.
set -x
cd ~

run2() {  # run two arms concurrently, wait for both
  bash -c "$1" &
  A=$!
  bash -c "$2" &
  B=$!
  wait $A $B
}

run2 "python3 curved_steelman.py 10525 0 60 8 flat > ~/s_flat8.log 2>&1; echo RC=\$? >> ~/s_flat8.log" \
     "python3 ray_arm.py 5262 0 60 8 > ~/s_ray8.log 2>&1; echo RC=\$? >> ~/s_ray8.log"

run2 "python3 curved_steelman.py 5262 0 60 4 > ~/s_seg4.log 2>&1; echo RC=\$? >> ~/s_seg4.log" \
     "python3 forms_arm.py cubic 3492 0 60 8 > ~/s_cubic8.log 2>&1; echo RC=\$? >> ~/s_cubic8.log"

run2 "python3 forms_arm.py offset 5262 0 60 8 > ~/s_offset8.log 2>&1; echo RC=\$? >> ~/s_offset8.log" \
     "python3 curved_steelman.py 5262 0 60 8 > ~/s_seg8.log 2>&1; echo RC=\$? >> ~/s_seg8.log"

python3 rate_router_eval.py ~/i2502v2/curved_k5262_s0_l8.npz 8,16 > ~/s_router.log 2>&1
echo "RC=$?" >> ~/s_router.log

echo SCREEN_DONE
