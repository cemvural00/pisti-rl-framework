#!/usr/bin/env bash
# Secondary fixed-opponent controls declared in the preregistration.
set -uo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir" || exit 1
mkdir -p logs

pids=()
labels=()
for seed in $(seq 0 4); do
    run_name="study_fixed_s${seed}"
    run_dir="runs/${run_name}"
    if [[ -e "$run_dir" ]]; then
        echo "Refusing to overwrite existing run: $run_dir" >&2
        exit 2
    fi
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        venv/bin/python -m training.train \
        --config configs/fixed_opponents.yaml \
        --set "run_name=${run_name}" "seed=${seed}" \
        >"logs/${run_name}.log" 2>&1 &
    pids+=("$!")
    labels+=("$run_name")
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
