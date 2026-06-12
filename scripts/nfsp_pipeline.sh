#!/bin/bash
set -u
cd "$(dirname "$0")/.."
PY=venv/bin/python
export OMP_NUM_THREADS=4

echo "=== nfsp training $(date) ==="
$PY -m training.train_nfsp --run-name nfsp_main --steps 10_000_000
echo "=== nfsp training done $(date) ==="

echo "=== tournament v3 $(date) ==="
$PY -m training.evaluate --agents random greedy hunter "expectimax:16,6" \
    "ppo:runs/ppo_main/final_model" "ppo:runs/ppo_s1/final_model" \
    "ppo:runs/ppo_s2/final_model" "ppo-nomem:runs/ppo_nomem/final_model" \
    "dqn:runs/dqn_main/final_model" "nfsp:runs/nfsp_main/final.pt" \
    --n-deals 250 --seed 123 --out results/tournament_v3.json
$PY -m analysis.significance results/tournament_v3.json

echo "=== nfsp exploitability $(date) ==="
$PY -m training.exploitability --target nfsp:runs/nfsp_main/final.pt \
    --steps 1500000 --seed 42 --eval-deals 400 \
    --init-from runs/ppo_main/final_model.zip \
    --out results/exploit/nfsp_main_final.json

echo "=== nfsp internal BR gap $(date) ==="
$PY - <<'PYEOF'
import json
from training.evaluate import build_agent
from training.match import play_match
_, br, _ = build_agent("nfsp-br:runs/nfsp_main/final.pt", seed=1)
_, avg, _ = build_agent("nfsp:runs/nfsp_main/final.pt", seed=2)
res = play_match(br, avg, n_deals=400, seed=99, name_a="nfsp_br", name_b="nfsp_avg")
s = res.summary()
print("internal BR gap:", s)
json.dump(s, open("results/nfsp_internal_gap.json", "w"), indent=2)
PYEOF
echo "=== nfsp pipeline finished $(date) ==="
