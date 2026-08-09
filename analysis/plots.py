"""Plots and statistical analysis for the Pişti study.

All functions write PNG files into an output directory and return the
figure path. Tournament JSONs come from training.evaluate; eval.csv files
come from training.train.
"""

import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
def training_curves(run_dirs: List[str], out_dir: str, label_map=None) -> str:
    """Win rate and score diff vs steps for each run's eval.csv."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = sns.color_palette("deep")
    for ri, run in enumerate(run_dirs):
        with open(os.path.join(run, "eval.csv")) as f:
            rows = list(csv.DictReader(f))
        label = (label_map or {}).get(run, os.path.basename(run))
        steps = [int(r["timesteps"]) for r in rows]
        for ax, metric, title in (
            (axes[0], "win", "win rate"),
            (axes[1], "diff", "score differential (pts/game)"),
        ):
            for li, opp in enumerate(("greedy", "expectimax")):
                vals = [float(r[f"{metric}_{opp}"]) for r in rows]
                ax.plot(
                    steps,
                    vals,
                    color=colors[ri],
                    alpha=1.0 if opp == "greedy" else 0.45,
                    linestyle="-" if opp == "greedy" else "--",
                    label=f"{label} vs {opp}",
                )
            ax.set_xlabel("training steps")
            ax.set_title(title)
    axes[0].axhline(0.5, color="gray", lw=1, ls=":")
    axes[1].axhline(0.0, color="gray", lw=1, ls=":")
    axes[0].legend(fontsize=10)
    return _save(fig, out_dir, "training_curves.png")


# ----------------------------------------------------------------------
def _pair_matrix(tournament: Dict, key: str):
    names = tournament["agents"]
    idx = {n: i for i, n in enumerate(names)}
    mat = np.full((len(names), len(names)), np.nan)
    for m in tournament["matches"]:
        a, b = m["agent_a"], m["agent_b"]
        if key == "win":
            mat[idx[a], idx[b]] = m["win_rate_a"]
            mat[idx[b], idx[a]] = 1 - m["win_rate_a"]
        else:
            mat[idx[a], idx[b]] = m["mean_diff"]
            mat[idx[b], idx[a]] = -m["mean_diff"]
    return names, mat


def tournament_heatmap(tournament: Dict, out_dir: str) -> str:
    order = list(tournament["ratings"].keys())  # strongest first
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, key, fmt, cmap, title in (
        (axes[0], "win", ".2f", "RdYlGn", "win rate (row vs col)"),
        (axes[1], "diff", "+.1f", "RdBu_r", "score diff (row − col, pts/game)"),
    ):
        names, mat = _pair_matrix(tournament, key)
        re = [names.index(n) for n in order]
        mat = mat[np.ix_(re, re)]
        center = 0.5 if key == "win" else 0.0
        sns.heatmap(
            mat,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            center=center,
            xticklabels=order,
            yticklabels=order,
            ax=ax,
            cbar=False,
        )
        ax.set_title(title)
    return _save(fig, out_dir, "tournament_heatmap.png")


def ratings_bar(tournament: Dict, out_dir: str) -> str:
    ratings = tournament["ratings"]
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(ratings.keys())[::-1]
    vals = [ratings[n] for n in names]
    ax.barh(names, vals, color=sns.color_palette("viridis", len(names)))
    ax.set_xlabel("Bradley–Terry rating (Elo-like, 1500 = mean)")
    ax.set_xlim(min(vals) - 60, max(vals) + 60)
    for i, v in enumerate(vals):
        ax.text(v + 5, i, f"{v:.0f}", va="center", fontsize=12)
    ax.set_title("Pişti agent ladder")
    return _save(fig, out_dir, "ratings.png")


# ----------------------------------------------------------------------
def luck_vs_skill(tournament: Dict, out_dir: str) -> Dict:
    """Mirror-pair variance decomposition.

    For each deal played in both seatings: deal effect = mean of the two
    diffs (luck of the cards), seat+noise = half the difference. The share
    of outcome variance explained by the deal quantifies how luck-driven
    each matchup is. Also reports the first-mover (seat) advantage.
    """
    by_pair = defaultdict(lambda: defaultdict(dict))
    for r in tournament["records"]:
        d = r["score_a"] - r["score_b"]
        by_pair[(r["a"], r["b"])][r["deal"]][r["a_leads"]] = d

    stats = {}
    for (a, b), deals in by_pair.items():
        deal_fx, seat_fx, alldiffs = [], [], []
        for _, seatings in deals.items():
            if True in seatings and False in seatings:
                d1, d2 = seatings[True], seatings[False]
                deal_fx.append((d1 + d2) / 2)
                # positive = leading helps agent A
                seat_fx.append((d1 - d2) / 2)
                alldiffs += [d1, d2]
        deal_fx, seat_fx = np.array(deal_fx), np.array(seat_fx)
        total_var = np.var(alldiffs)
        stats[f"{a} vs {b}"] = {
            "deal_var_share": float(np.var(deal_fx) / total_var) if total_var else 0.0,
            "seat_advantage_pts": float(seat_fx.mean()),
            "seat_adv_ci95": float(1.96 * seat_fx.std(ddof=1) / np.sqrt(len(seat_fx))),
            "skill_gap_pts": float(deal_fx.mean()),
        }

    # Plot: deal-variance share per matchup
    fig, ax = plt.subplots(figsize=(11, max(4, 0.5 * len(stats))))
    names = list(stats.keys())
    shares = [stats[n]["deal_var_share"] for n in names]
    ax.barh(names, shares, color="steelblue")
    ax.set_xlabel("share of outcome variance explained by the deal (luck)")
    ax.set_xlim(0, 1)
    ax.set_title("How much of a Pişti game is luck?")
    _save(fig, out_dir, "luck_share.png")
    return stats


# ----------------------------------------------------------------------
def exploitability_curve(points: List[Dict], out_dir: str, baselines: Optional[Dict] = None) -> str:
    """points: [{steps, exploitability_pts, ci95}], baselines: name->pts."""
    fig, ax = plt.subplots(figsize=(10, 6))
    pts = sorted(points, key=lambda p: p["steps"])
    x = [p["steps"] for p in pts]
    y = [p["exploitability_pts"] for p in pts]
    ci = [p["ci95"] for p in pts]
    ax.errorbar(x, y, yerr=ci, marker="o", capsize=4, color="crimson", lw=2)
    if baselines:
        for i, (name, v) in enumerate(baselines.items()):
            ax.axhline(v, ls="--", lw=1.5, color=f"C{i+2}", label=f"{name}: {v:+.1f}")
        ax.legend(fontsize=11)
    ax.set_xlabel("self-play training steps of target policy")
    ax.set_ylabel("best-response edge (pts/game)")
    ax.set_title("Approximate exploitability during self-play training")
    return _save(fig, out_dir, "exploitability.png")
