#!/bin/bash
set -u
cd "$(dirname "$0")/.."
PY=venv/bin/python
export OMP_NUM_THREADS=4

echo "=== dqn_main training $(date) ==="
$PY -m training.train --config configs/dqn.yaml
echo "=== dqn_main done $(date) ==="

echo "=== tournament with dqn $(date) ==="
$PY -m training.evaluate --agents random greedy hunter "expectimax:16,6" \
    "ppo:runs/ppo_main/final_model" "ppo:runs/ppo_s1/final_model" \
    "ppo:runs/ppo_s2/final_model" "ppo-nomem:runs/ppo_nomem/final_model" \
    "dqn:runs/dqn_main/final_model" \
    --n-deals 250 --seed 123 --out results/tournament_v2.json
$PY -m analysis.significance results/tournament_v2.json

echo "=== dqn exploitability $(date) ==="
$PY -m training.exploitability --target dqn:runs/dqn_main/final_model.zip \
    --steps 1500000 --seed 42 --eval-deals 400 \
    --init-from runs/ppo_main/final_model.zip \
    --out results/exploit/dqn_main_final.json
echo "=== dqn pipeline finished $(date) ==="
