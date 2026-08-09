"""Statistical significance analysis for tournament results.

For every match we test the mirror-paired deal differences against zero:
  - paired t-test (parametric)
  - Wilcoxon signed-rank (non-parametric, robust to outliers)
  - Holm-Bonferroni correction across all matches in the tournament

Bradley-Terry rating uncertainty comes from a deal-level bootstrap:
resample deals (keeping both seatings together) within every match,
recompute the win matrix and refit ratings.

Usage:
    venv/bin/python -m analysis.significance results/tournament.json
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, List

import numpy as np
from scipy import stats

from training.evaluate import bradley_terry


def paired_deal_diffs(records: List[dict], a: str, b: str) -> np.ndarray:
    by_deal = defaultdict(dict)
    for r in records:
        if r["a"] == a and r["b"] == b:
            by_deal[r["deal"]][r["a_leads"]] = r["score_a"] - r["score_b"]
    return np.array([(v[True] + v[False]) / 2 for v in by_deal.values() if len(v) == 2])


def match_tests(tournament: Dict) -> List[Dict]:
    """Per-match significance tests on mirror-paired deal differences."""
    out = []
    for m in tournament["matches"]:
        a, b = m["agent_a"], m["agent_b"]
        d = paired_deal_diffs(tournament["records"], a, b)
        t_stat, p_t = stats.ttest_1samp(d, 0.0)
        try:
            _, p_w = stats.wilcoxon(d)
        except ValueError:  # all-zero diffs
            p_w = 1.0
        se = d.std(ddof=1) / np.sqrt(len(d))
        out.append(
            {
                "agent_a": a,
                "agent_b": b,
                "n_deals": int(len(d)),
                "mean_diff": float(d.mean()),
                "se": float(se),
                "ci95": float(1.96 * se),
                "t": float(t_stat),
                "p_ttest": float(p_t),
                "p_wilcoxon": float(p_w),
            }
        )
    # Holm-Bonferroni across all matches (on the t-test p-values)
    order = np.argsort([r["p_ttest"] for r in out])
    k = len(out)
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (k - rank) * out[idx]["p_ttest"])
        adj = max(adj, prev)  # enforce monotonicity
        out[idx]["p_holm"] = float(adj)
        prev = adj
    for r in out:
        r["significant_05"] = r["p_holm"] < 0.05
    return out


def bootstrap_ratings(tournament: Dict, n_boot: int = 1000, seed: int = 0) -> Dict[str, Dict]:
    """Deal-level bootstrap CIs for Bradley-Terry ratings."""
    rng = np.random.default_rng(seed)
    names = tournament["agents"]

    # Pre-index: per match, per deal -> (win_a_contrib, win_b_contrib)
    matches = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for r in tournament["records"]:
        key = (r["a"], r["b"])
        d = r["score_a"] - r["score_b"]
        wa = 1.0 if d > 0 else (0.5 if d == 0 else 0.0)
        cell = matches[key][r["deal"]]
        cell[0] += wa
        cell[1] += 1.0 - wa
    match_arrays = {key: np.array(list(deals.values())) for key, deals in matches.items()}

    samples = {n: [] for n in names}
    for _ in range(n_boot):
        results = {}
        for key, arr in match_arrays.items():
            idx = rng.integers(0, len(arr), len(arr))
            wa, wb = arr[idx].sum(axis=0)
            results[key] = (wa, wb)
        ratings = bradley_terry(names, results)
        for n in names:
            samples[n].append(ratings[n])

    out = {}
    for n in names:
        s = np.array(samples[n])
        out[n] = {
            "rating": tournament["ratings"][n],
            "ci_low": float(np.percentile(s, 2.5)),
            "ci_high": float(np.percentile(s, 97.5)),
        }
    return out


def exploitability_tests(paths: List[str]) -> List[Dict]:
    """Normal-approximation p-values for BR edge != 0 (from saved CI)."""
    out = []
    for p in paths:
        d = json.load(open(p))
        se = d["ci95"] / 1.96
        z = d["exploitability_pts"] / se
        out.append(
            {
                "target": d["target"],
                "target_steps": d.get("target_steps"),
                "exploitability_pts": d["exploitability_pts"],
                "ci95": d["ci95"],
                "z": float(z),
                "p_value": float(2 * (1 - stats.norm.cdf(abs(z)))),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="results/tournament.json")
    parser.add_argument(
        "--no-exploits",
        action="store_true",
        help="do not attach historical results/exploit/*.json tests",
    )
    args = parser.parse_args()
    path = args.input
    tournament = json.load(open(path))

    tests = match_tests(tournament)
    ratings = bootstrap_ratings(tournament)

    import glob

    exploit_paths = [] if args.no_exploits else sorted(glob.glob("results/exploit/*.json"))
    exploits = exploitability_tests(exploit_paths)

    result = {"match_tests": tests, "rating_ci": ratings, "exploitability": exploits}
    out = path.replace(".json", "_significance.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"{'matchup':>34}  {'diff':>7}  {'p(t)':>8} {'p(W)':>8} {'p(Holm)':>8}  sig")
    for r in sorted(tests, key=lambda x: x["p_holm"]):
        print(
            f"{r['agent_a']:>16} vs {r['agent_b']:<16} {r['mean_diff']:+7.2f}  "
            f"{r['p_ttest']:8.2g} {r['p_wilcoxon']:8.2g} {r['p_holm']:8.2g}  "
            f"{'*' if r['significant_05'] else ''}"
        )
    print("\nBradley-Terry ratings with bootstrap 95% CI:")
    for n, d in sorted(ratings.items(), key=lambda x: -x[1]["rating"]):
        print(f"  {n:>16}: {d['rating']:7.1f}  [{d['ci_low']:7.1f}, {d['ci_high']:7.1f}]")
    print("\nExploitability (H0: edge = 0):")
    for e in exploits:
        steps = f"@{e['target_steps']:,}" if e["target_steps"] else ""
        print(
            f"  {e['target']:>40}{steps:>12}: {e['exploitability_pts']:+6.2f} "
            f"±{e['ci95']:.2f}  p={e['p_value']:.2g}"
        )
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
