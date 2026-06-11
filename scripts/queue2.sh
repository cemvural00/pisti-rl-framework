#!/bin/bash
# Follow-up queue: waits for ppo_main (pid $1), then seed runs + ablation (6M each).
set -u
cd "$(dirname "$0")/.."
PY=venv/bin/python
export OMP_NUM_THREADS=4

while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "=== ppo_main finished, starting follow-ups $(date) ==="
$PY -m training.train --config configs/default.yaml --set run_name=ppo_s1 seed=1
echo "=== ppo_s1 done $(date) ==="
$PY -m training.train --config configs/default.yaml --set run_name=ppo_s2 seed=2
echo "=== ppo_s2 done $(date) ==="
$PY -m training.train --config configs/default.yaml --set run_name=ppo_nomem seed=3 observer.memory=false
echo "=== queue2 finished $(date) ==="
