#!/bin/bash
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
echo "=== beast eval started $(date) ==="
venv/bin/python -m training.evaluate \
  --agents "beast:runs/ppo_main/final_model:32" "ppo:runs/ppo_main/final_model" "expectimax:16,6" "dqn:runs/dqn_main/final_model" \
  --n-deals 120 --seed 321 --out results/beast_eval.json
echo "=== beast eval finished $(date) ==="
