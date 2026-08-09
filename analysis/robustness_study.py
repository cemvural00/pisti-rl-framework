"""Analyze paired fixed-budget attacks on league and fixed-opponent policies."""

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analysis.memory_study import exact_sign_flip_pvalue


def load_result(path: str) -> Dict:
    with open(path) as handle:
        return json.load(handle)


def attack_edge(result: Dict) -> float:
    """Unrounded mean edge over mirrored deal pairs."""
    by_deal: Dict[int, List[float]] = {}
    for record in result["records"]:
        diff = record["score_a"] - record["score_b"]
        by_deal.setdefault(record["deal"], []).append(diff)
    if not all(len(values) == 2 for values in by_deal.values()):
        raise ValueError("attack result is missing a mirrored seating")
    return float(np.mean([np.mean(values) for values in by_deal.values()]))


def analyze(results_dir: str) -> Dict:
    league: List[float] = []
    fixed: List[float] = []
    pairs = []
    for seed in range(5):
        attacker_seed = 100 + seed
        league_result = load_result(
            os.path.join(results_dir, f"attack_league_s{seed}_a{attacker_seed}.json")
        )
        fixed_result = load_result(
            os.path.join(results_dir, f"attack_fixed_s{seed}_a{attacker_seed}.json")
        )
        league.append(attack_edge(league_result))
        fixed.append(attack_edge(fixed_result))
        pairs.append(
            {
                "target_seed": seed,
                "attacker_seed": attacker_seed,
                "league_attack_edge": league[-1],
                "fixed_attack_edge": fixed[-1],
                "fixed_minus_league": fixed[-1] - league[-1],
            }
        )

    effects = np.asarray(fixed) - np.asarray(league)
    sem = effects.std(ddof=1) / np.sqrt(len(effects))
    critical = stats.t.ppf(0.975, len(effects) - 1)
    mean = float(effects.mean())
    principal = []
    for attacker_seed in (100, 201, 202):
        result = load_result(os.path.join(results_dir, f"attack_league_s0_a{attacker_seed}.json"))
        principal.append(
            {
                "attacker_seed": attacker_seed,
                "edge": attack_edge(result),
                "deal_ci95_half_width": result["ci95"],
            }
        )
    return {
        "protocol": {
            "target_seeds": 5,
            "attack_steps": 1_000_000,
            "evaluation_deals": 500,
            "paired_target_comparison": True,
        },
        "pairs": pairs,
        "fixed_minus_league": {
            "seed_effects": [round(float(value), 4) for value in effects],
            "mean": round(mean, 4),
            "seed_t_ci95": [
                round(float(mean - critical * sem), 4),
                round(float(mean + critical * sem), 4),
            ],
            "positive_seeds": int(np.sum(effects > 0)),
            "exact_sign_flip_p_two_sided": round(exact_sign_flip_pvalue(effects), 6),
        },
        "principal_target_replicated_attacks": principal,
        "principal_max_discovered_edge": max(item["edge"] for item in principal),
    }


def plot(summary: Dict, path: str) -> None:
    pairs = summary["pairs"]
    league = np.asarray([pair["league_attack_edge"] for pair in pairs])
    fixed = np.asarray([pair["fixed_attack_edge"] for pair in pairs])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for seed, (league_edge, fixed_edge) in enumerate(zip(league, fixed)):
        ax.plot([0, 1], [league_edge, fixed_edge], marker="o", alpha=0.7, label=f"seed {seed}")
    ax.axhline(0, color="0.45", linewidth=1)
    ax.set_xticks([0, 1], ["League", "Fixed opponents"])
    ax.set_ylabel("Discovered attack edge (points/game)")
    ax.set_title("Fixed-budget approximate best responses")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/robustness")
    parser.add_argument("--out", default="results/robustness_study_summary.json")
    parser.add_argument("--plot", default="plots/robustness_study.png")
    args = parser.parse_args()
    summary = analyze(args.results_dir)
    with open(args.out, "w") as handle:
        json.dump(summary, handle, indent=2)
    plot(summary, args.plot)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
