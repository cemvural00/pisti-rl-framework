#!/usr/bin/env bash
# Evaluate the confirmatory memory study after all 20 target models exist.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
mkdir -p logs results plots

for seed in $(seq 0 9); do
    for condition in on off; do
        model="runs/study_mem_${condition}_s${seed}/final_model.zip"
        if [[ ! -f "$model" ]]; then
            echo "Missing model: $model" >&2
            exit 2
        fi
    done
done

for result in results/memory_study_matches.json results/memory_study_crossplay.json; do
    if [[ -e "$result" ]]; then
        echo "Refusing to overwrite result: $result" >&2
        exit 3
    fi
done

venv/bin/python scripts/validate_study_runs.py \
    >logs/study_validation.log
venv/bin/python -m analysis.study_training \
    >logs/study_training_analysis.log

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    venv/bin/python scripts/evaluate_memory_study.py \
    >logs/memory_study_evaluation.log 2>&1 &
evaluation_pid=$!

specs=()
for seed in $(seq 0 9); do
    specs+=("ppo:runs/study_mem_on_s${seed}/final_model.zip")
    specs+=("ppo-nomem:runs/study_mem_off_s${seed}/final_model.zip")
done
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    venv/bin/python -m training.evaluate \
    --agents "${specs[@]}" --n-deals 500 --seed 20260808 \
    --out results/memory_study_crossplay.json \
    >logs/memory_study_crossplay.log 2>&1 &
crossplay_pid=$!

failures=0
wait "$evaluation_pid" || failures=$((failures + 1))
wait "$crossplay_pid" || failures=$((failures + 1))
if (( failures > 0 )); then
    echo "Evaluation failed; inspect logs/memory_study_*.log" >&2
    exit 1
fi

venv/bin/python -m analysis.memory_study results/memory_study_matches.json \
    >logs/memory_study_analysis.log
venv/bin/python -m analysis.significance results/memory_study_crossplay.json --no-exploits \
    >logs/memory_study_significance.log

echo "memory study evaluation complete"
