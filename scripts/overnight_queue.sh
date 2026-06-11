#!/bin/bash
# Overnight training queue: main run, 2 extra seeds, memory ablation.
set -u
cd "$(dirname "$0")/.."
PY=venv/bin/python
export OMP_NUM_THREADS=4

echo "=== queue started $(date) ==="
$PY -m training.train --config configs/default.yaml --set run_name=ppo_main seed=0
echo "=== ppo_main done $(date) ==="
$PY -m training.train --config configs/default.yaml --set run_name=ppo_s1 seed=1
echo "=== ppo_s1 done $(date) ==="
$PY -m training.train --config configs/default.yaml --set run_name=ppo_s2 seed=2
echo "=== ppo_s2 done $(date) ==="
$PY -m training.train --config configs/default.yaml --set run_name=ppo_nomem seed=0 observer.memory=false
echo "=== queue finished $(date) ==="
