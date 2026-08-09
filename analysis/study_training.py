"""Aggregate confirmatory learning curves across independent training seeds."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {
    "Memory": [f"study_mem_on_s{seed}" for seed in range(10)],
    "No memory": [f"study_mem_off_s{seed}" for seed in range(10)],
    "Fixed opponents": [f"study_fixed_s{seed}" for seed in range(5)],
}
METRICS = ("diff_greedy", "diff_hunter", "diff_expectimax")


def read_curve(run_name: str) -> List[Dict[str, float]]:
    path = ROOT / "runs" / run_name / "eval.csv"
    with path.open() as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def interval(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    sem = matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])
    half_width = stats.t.ppf(0.975, matrix.shape[0] - 1) * sem
    return mean, mean - half_width, mean + half_width


def analyze() -> Dict:
    output: Dict[str, object] = {"conditions": {}}
    for condition, names in CONDITIONS.items():
        curves = [read_curve(name) for name in names]
        steps = [int(row["timesteps"]) for row in curves[0]]
        if not all([int(row["timesteps"]) for row in curve] == steps for curve in curves):
            raise ValueError(f"unaligned evaluations for {condition}")
        condition_result = {"n_seeds": len(names), "timesteps": steps, "metrics": {}}
        for metric in METRICS:
            matrix = np.asarray([[row[metric] for row in curve] for curve in curves])
            mean, low, high = interval(matrix)
            condition_result["metrics"][metric] = {
                "mean": mean.tolist(),
                "ci95_low": low.tolist(),
                "ci95_high": high.tolist(),
            }
        output["conditions"][condition] = condition_result
    return output


def plot(summary: Dict, path: Path) -> None:
    colors = {"Memory": "#1769aa", "No memory": "#d1495b", "Fixed opponents": "#4c956c"}
    titles = {
        "diff_greedy": "vs greedy",
        "diff_hunter": "vs pişti-hunter",
        "diff_expectimax": "vs determinization search",
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), sharey=True)
    for axis, metric in zip(axes, METRICS):
        for condition, result in summary["conditions"].items():
            steps = np.asarray(result["timesteps"]) / 1_000_000
            values = result["metrics"][metric]
            mean = np.asarray(values["mean"])
            low = np.asarray(values["ci95_low"])
            high = np.asarray(values["ci95_high"])
            axis.plot(steps, mean, label=condition, color=colors[condition])
            axis.fill_between(steps, low, high, color=colors[condition], alpha=0.14)
        axis.axhline(0, color="0.5", linewidth=0.8)
        axis.set_title(titles[metric])
        axis.set_xlabel("Environment steps (millions)")
    axes[0].set_ylabel("Score differential per game")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Confirmatory learning curves (mean and 95% seed interval)")
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    summary = analyze()
    output = ROOT / "results/study_training_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    plot(summary, ROOT / "plots/study_training_curves.png")
    print(f"saved {output.relative_to(ROOT)} and plots/study_training_curves.png")


if __name__ == "__main__":
    main()
