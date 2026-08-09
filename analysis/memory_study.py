"""Seed-aware analysis for the preregistered memory study."""

import argparse
import itertools
import json
import os
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def paired_deal_diffs(match: Dict) -> np.ndarray:
    by_deal: Dict[int, List[float]] = {}
    for record in match["records"]:
        diff = record["score_a"] - record["score_b"]
        by_deal.setdefault(record["deal"], []).append(diff)
    if not all(len(values) == 2 for values in by_deal.values()):
        raise ValueError("every evaluation deal must have both seatings")
    return np.asarray([np.mean(values) for _, values in sorted(by_deal.items())])


def exact_sign_flip_pvalue(effects: np.ndarray) -> float:
    observed = abs(float(effects.mean()))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(effects)):
        values.append(abs(float(np.mean(effects * np.asarray(signs)))))
    values = np.asarray(values)
    return float(np.mean(values >= observed - 1e-12))


def crossed_interval(
    deal_effects: List[np.ndarray], seed: int = 20260808, n_boot: int = 20_000
) -> Tuple[float, float]:
    """Bootstrap crossed training seeds and shared evaluation deals."""
    rng = np.random.default_rng(seed)
    matrix = np.stack(deal_effects)
    n_seeds, n_deals = matrix.shape
    samples = np.empty(n_boot)
    for index in range(n_boot):
        selected_seeds = rng.integers(0, n_seeds, size=n_seeds)
        selected_deals = rng.integers(0, n_deals, size=n_deals)
        samples[index] = matrix[np.ix_(selected_seeds, selected_deals)].mean()
    return tuple(float(value) for value in np.percentile(samples, [2.5, 97.5]))


def summarize(matches: Iterable[Dict]) -> Dict:
    deal_effects = [paired_deal_diffs(match) for match in matches]
    effects = np.asarray([values.mean() for values in deal_effects])
    n = len(effects)
    sem = effects.std(ddof=1) / np.sqrt(n)
    critical = stats.t.ppf(0.975, df=n - 1)
    mean = float(effects.mean())
    return {
        "seed_effects": [round(float(value), 4) for value in effects],
        "mean": round(mean, 4),
        "median": round(float(np.median(effects)), 4),
        "sd_across_seeds": round(float(effects.std(ddof=1)), 4),
        "seed_t_ci95": [
            round(float(mean - critical * sem), 4),
            round(float(mean + critical * sem), 4),
        ],
        "crossed_bootstrap_ci95": [round(value, 4) for value in crossed_interval(deal_effects)],
        "exact_sign_flip_p_two_sided": round(exact_sign_flip_pvalue(effects), 6),
        "positive_seeds": int(np.sum(effects > 0)),
        "n_seeds": n,
        "n_deals_per_seed": int(len(deal_effects[0])),
    }


def anchor_summary(entries: List[Dict]) -> Dict:
    grouped: Dict[Tuple[int, str], List[float]] = {}
    for entry in entries:
        key = (entry["training_seed"], entry["condition"])
        grouped.setdefault(key, []).append(float(paired_deal_diffs(entry).mean()))
    seeds = sorted({seed for seed, _ in grouped})
    effects = np.asarray(
        [np.mean(grouped[(seed, "on")]) - np.mean(grouped[(seed, "off")]) for seed in seeds]
    )
    return {
        "seed_effects": [round(float(value), 4) for value in effects],
        "mean": round(float(effects.mean()), 4),
        "exact_sign_flip_p_two_sided": round(exact_sign_flip_pvalue(effects), 6),
        "positive_seeds": int(np.sum(effects > 0)),
    }


def plot_effects(summary: Dict, path: str) -> None:
    labels = ["Retrained\n(on − off)", "Acute\n(normal − zeroed)", "Stochastic\n(on − off)"]
    keys = ["direct", "acute", "stochastic_sensitivity"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for x, key in enumerate(keys):
        effects = np.asarray(summary[key]["seed_effects"])
        jitter = np.linspace(-0.12, 0.12, len(effects))
        ax.scatter(np.full(len(effects), x) + jitter, effects, alpha=0.75, s=28)
        low, high = summary[key]["seed_t_ci95"]
        mean = summary[key]["mean"]
        ax.errorbar(x, mean, yerr=[[mean - low], [high - mean]], fmt="o", color="black", capsize=5)
    ax.axhline(0, color="0.45", linewidth=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Score differential per game")
    ax.set_title("Played-card memory effects across independent training seeds")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(input_path: str) -> Dict:
    with open(input_path) as handle:
        data = json.load(handle)
    summary = {
        "protocol": data["protocol"],
        "direct": summarize(data["direct"]),
        "acute": summarize(data["acute"]),
        "stochastic_sensitivity": summarize(data["stochastic_sensitivity"]),
        "anchors": anchor_summary(data["anchors"]),
    }
    direct = np.asarray(summary["direct"]["seed_effects"])
    acute = np.asarray(summary["acute"]["seed_effects"])
    summary["adaptation"] = {
        "acute_minus_retrained_by_seed": [round(float(value), 4) for value in acute - direct],
        "mean": round(float(np.mean(acute - direct)), 4),
        "positive_seeds": int(np.sum(acute - direct > 0)),
        "exact_sign_flip_p_two_sided": round(exact_sign_flip_pvalue(acute - direct), 6),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="results/memory_study_matches.json")
    parser.add_argument("--out", default="results/memory_study_summary.json")
    parser.add_argument("--plot", default="plots/memory_study_effects.png")
    args = parser.parse_args()
    summary = analyze(args.input)
    with open(args.out, "w") as handle:
        json.dump(summary, handle, indent=2)
    plot_effects(summary, args.plot)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
