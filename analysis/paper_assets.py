"""Generate manuscript macros and tables from the final study artifacts."""

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Dict:
    with path.open() as handle:
        return json.load(handle)


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def p_fmt(value: float) -> str:
    return f"{value:.4f}"


def aggregate_direct(raw: Dict) -> Dict[str, float]:
    records = [record for match in raw["direct"] for record in match["records"]]
    diffs = np.asarray([record["score_a"] - record["score_b"] for record in records])
    win_rate = float(np.mean((diffs > 0) + 0.5 * (diffs == 0)))
    return {
        "win_rate": win_rate,
        "pisti_diff": float(
            np.mean([record["pistis_a"] - record["pistis_b"] for record in records])
        ),
        "captured_diff": float(
            np.mean([record["captured_a"] - record["captured_b"] for record in records])
        ),
    }


def paired_rating_difference(crossplay: Dict) -> float:
    ratings = crossplay["ratings"]
    differences: List[float] = []
    for seed in range(10):
        on = ratings[f"study_mem_on_s{seed}"]
        off = ratings[f"nomem-study_mem_off_s{seed}"]
        differences.append(on - off)
    return float(np.mean(differences))


def command(name: str, value: object) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def memory_table(summary: Dict) -> str:
    direct = summary["direct"]["seed_effects"]
    acute = summary["acute"]["seed_effects"]
    stochastic = summary["stochastic_sensitivity"]["seed_effects"]
    rows = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Score effects by independent training seed (points per game). Positive values favor memory-on or ordinary memory access.}",
        r"\label{tab:memory}",
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"Seed & Retrained & Acute removal & Stochastic \\",
        r"\midrule",
    ]
    for seed, values in enumerate(zip(direct, acute, stochastic)):
        rows.append(f"{seed} & {values[0]:+.2f} & {values[1]:+.2f} & " f"{values[2]:+.2f} " + r"\\")
    rows.extend(
        [
            r"\midrule",
            (
                f"Mean & {summary['direct']['mean']:+.2f} & "
                f"{summary['acute']['mean']:+.2f} & "
                f"{summary['stochastic_sensitivity']['mean']:+.2f} " + r"\\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(rows) + "\n"


def robustness_table(summary: Dict) -> str:
    rows = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Discovered attack edges for paired targets (points per game). Larger values indicate a more successful fixed-budget attack.}",
        r"\label{tab:robustness}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Target seed & Attacker seed & League & Fixed & Fixed $-$ league \\",
        r"\midrule",
    ]
    for pair in summary["pairs"]:
        rows.append(
            f"{pair['target_seed']} & {pair['attacker_seed']} & "
            f"{pair['league_attack_edge']:+.2f} & {pair['fixed_attack_edge']:+.2f} & "
            f"{pair['fixed_minus_league']:+.2f} " + r"\\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(rows) + "\n"


def macro_lines(memory: Dict, direct: Dict, crossplay: Dict, robustness: Dict) -> Iterable[str]:
    d = memory["direct"]
    a = memory["acute"]
    s = memory["stochastic_sensitivity"]
    adaptation = memory["adaptation"]
    anchor = memory["anchors"]
    robust = robustness["fixed_minus_league"]
    values = {
        "DirectMean": fmt(d["mean"]),
        "DirectMedian": fmt(d["median"]),
        "DirectCILow": fmt(d["seed_t_ci95"][0]),
        "DirectCIHigh": fmt(d["seed_t_ci95"][1]),
        "DirectBootLow": fmt(d["crossed_bootstrap_ci95"][0]),
        "DirectBootHigh": fmt(d["crossed_bootstrap_ci95"][1]),
        "DirectP": p_fmt(d["exact_sign_flip_p_two_sided"]),
        "DirectPositive": d["positive_seeds"],
        "DirectWinRate": fmt(direct["win_rate"], 3),
        "DirectPistiDiff": fmt(direct["pisti_diff"], 3),
        "DirectCapturedDiff": fmt(direct["captured_diff"], 2),
        "AcuteMean": fmt(a["mean"]),
        "AcuteCILow": fmt(a["seed_t_ci95"][0]),
        "AcuteCIHigh": fmt(a["seed_t_ci95"][1]),
        "AcuteP": p_fmt(a["exact_sign_flip_p_two_sided"]),
        "AcutePositive": a["positive_seeds"],
        "AdaptMean": fmt(adaptation["mean"]),
        "AdaptP": p_fmt(adaptation["exact_sign_flip_p_two_sided"]),
        "StochasticMean": fmt(s["mean"]),
        "StochasticCILow": fmt(s["seed_t_ci95"][0]),
        "StochasticCIHigh": fmt(s["seed_t_ci95"][1]),
        "StochasticP": p_fmt(s["exact_sign_flip_p_two_sided"]),
        "AnchorMean": fmt(anchor["mean"]),
        "AnchorP": p_fmt(anchor["exact_sign_flip_p_two_sided"]),
        "RatingDiff": fmt(paired_rating_difference(crossplay), 1),
        "RobustnessMean": fmt(robust["mean"]),
        "RobustnessCILow": fmt(robust["seed_t_ci95"][0]),
        "RobustnessCIHigh": fmt(robust["seed_t_ci95"][1]),
        "RobustnessP": p_fmt(robust["exact_sign_flip_p_two_sided"]),
        "RobustnessPositive": robust["positive_seeds"],
        "PrincipalMax": fmt(robustness["principal_max_discovered_edge"]),
    }
    for name, value in values.items():
        yield command(name, value)


def main() -> None:
    memory = read_json(ROOT / "results/memory_study_summary.json")
    raw = read_json(ROOT / "results/memory_study_matches.json")
    crossplay = read_json(ROOT / "results/memory_study_crossplay.json")
    robustness = read_json(ROOT / "results/robustness_study_summary.json")

    paper = ROOT / "paper"
    paper.mkdir(exist_ok=True)
    (paper / "generated_results.tex").write_text(
        "\n".join(macro_lines(memory, aggregate_direct(raw), crossplay, robustness)) + "\n"
    )
    (paper / "generated_memory_table.tex").write_text(memory_table(memory))
    (paper / "generated_robustness_table.tex").write_text(robustness_table(robustness))
    print("generated manuscript macros and tables")


if __name__ == "__main__":
    main()
