#!/usr/bin/env bash
# Paired fixed-budget attacks declared in research/experiment_preregistration.md.
set -uo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir" || exit 1
mkdir -p logs results/robustness runs/robustness_attackers

for seed in $(seq 0 4); do
    for run in "study_mem_on_s${seed}" "study_fixed_s${seed}"; do
        if [[ ! -f "runs/${run}/final_model.zip" ]]; then
            echo "Missing target model: runs/${run}/final_model.zip" >&2
            exit 2
        fi
    done
done

pids=()
labels=()
for seed in $(seq 0 4); do
    attacker_seed=$((100 + seed))
    for condition in league fixed; do
        target_run="study_mem_on_s${seed}"
        if [[ "$condition" == "fixed" ]]; then
            target_run="study_fixed_s${seed}"
        fi
        label="attack_${condition}_s${seed}_a${attacker_seed}"
        out="results/robustness/${label}.json"
        attacker_model="runs/robustness_attackers/${label}.zip"
        if [[ -e "$out" || -e "$attacker_model" ]]; then
            echo "Refusing to overwrite existing attack: $label" >&2
            exit 3
        fi
        env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
            venv/bin/python -m training.exploitability \
            --target "ppo-stoch:runs/${target_run}/final_model.zip" \
            --steps 1000000 --seed "$attacker_seed" --eval-deals 500 \
            --init-from runs/ppo_main/final_model.zip \
            --save-br "$attacker_model" --out "$out" \
            >"logs/${label}.log" 2>&1 &
        pids+=("$!")
        labels+=("$label")
    done
done

# Two additional attacks plus the seed-100 attack above give the principal
# league target three independent attackers.
for attacker_seed in 201 202; do
    label="attack_league_s0_a${attacker_seed}"
    out="results/robustness/${label}.json"
    attacker_model="runs/robustness_attackers/${label}.zip"
    if [[ -e "$out" || -e "$attacker_model" ]]; then
        echo "Refusing to overwrite existing attack: $label" >&2
        exit 3
    fi
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        venv/bin/python -m training.exploitability \
        --target ppo-stoch:runs/study_mem_on_s0/final_model.zip \
        --steps 1000000 --seed "$attacker_seed" --eval-deals 500 \
        --init-from runs/ppo_main/final_model.zip \
        --save-br "$attacker_model" --out "$out" \
        >"logs/${label}.log" 2>&1 &
    pids+=("$!")
    labels+=("$label")
done

failures=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "completed ${labels[$i]}"
    else
        echo "failed ${labels[$i]} (see logs/${labels[$i]}.log)" >&2
        failures=$((failures + 1))
    fi
done
exit "$failures"
