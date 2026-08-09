"""Fun-but-rigorous statistics for the story report.

Everything is computed from the saved tournament records (no new games):
  - match-to-151 simulation: how often does the better player win a real
    Pişti match (first to 151 points), vs a single hand?
  - "the deck decides": share of decks won by the same side in both
    seatings (cards beat skill+seat) vs split decks
  - score-swing histogram: the skill signal vs deal-noise storm
"""

import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def load(path="results/tournament.json"):
    return json.load(open(path))


def pair_records(t, a, b):
    return [r for r in t["records"] if r["a"] == a and r["b"] == b]


def match_to_151(records, n_matches=20000, seed=0):
    """Simulate first-to-151 matches by resampling hands with replacement."""
    rng = np.random.default_rng(seed)
    sa = np.array([r["score_a"] for r in records])
    sb = np.array([r["score_b"] for r in records])
    wins_a = 0
    hands_played = []
    for _ in range(n_matches):
        ta = tb = 0
        n = 0
        while ta < 151 and tb < 151:
            i = rng.integers(0, len(sa))
            ta += sa[i]
            tb += sb[i]
            n += 1
        if ta == tb:  # both crossed equally; replay decider
            continue
        wins_a += ta > tb
        hands_played.append(n)
    return wins_a / n_matches, float(np.mean(hands_played))


def single_game_winrate(records):
    d = np.array([r["score_a"] - r["score_b"] for r in records])
    return float((np.sum(d > 0) + 0.5 * np.sum(d == 0)) / len(d))


def deck_decides(records):
    """Classify each deck: same side wins both seatings, or split."""
    by_deal = defaultdict(dict)
    for r in records:
        by_deal[r["deal"]][r["a_leads"]] = np.sign(r["score_a"] - r["score_b"])
    same = split = tied = 0
    for v in by_deal.values():
        if len(v) != 2:
            continue
        s1, s2 = v[True], v[False]
        if s1 == 0 or s2 == 0:
            tied += 1
        elif s1 == s2:
            same += 1
        else:
            split += 1
    total = same + split + tied
    return same / total, split / total, tied / total


def main(out_dir="plots", tournament_path="results/tournament.json"):
    t = load(tournament_path)
    os.makedirs(out_dir, exist_ok=True)
    fun = {}

    matchups = [
        ("ppo_main", "greedy", "RL vs greedy"),
        ("ppo_main", "nomem-ppo_nomem", "memory vs amnesia"),
        ("expectimax16", "ppo_main", "search vs RL"),
        ("greedy", "hunter", "greedy vs hunter"),
    ]

    # ---- 1. single game vs match to 151 ------------------------------
    labels, single, match = [], [], []
    for a, b, label in matchups:
        recs = pair_records(t, a, b) or pair_records(t, b, a)
        if not recs:
            continue
        if not pair_records(t, a, b):  # swapped orientation
            a, b = b, a
        wr1 = single_game_winrate(recs)
        wr151, mean_hands = match_to_151(recs)
        # orient toward the stronger side
        if wr1 < 0.5:
            wr1, wr151 = 1 - wr1, 1 - wr151
        labels.append(label)
        single.append(wr1)
        match.append(wr151)
        fun[f"match151:{label}"] = {
            "single_game": round(wr1, 3),
            "match_151": round(wr151, 3),
            "mean_hands_per_match": round(mean_hands, 1),
        }

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.bar(x - 0.2, single, 0.4, label="one hand", color="#9ecae1")
    ax.bar(x + 0.2, match, 0.4, label="match to 151", color="#08519c")
    for i, (s, m) in enumerate(zip(single, match)):
        ax.text(i - 0.2, s + 0.012, f"{s:.0%}", ha="center", fontsize=12)
        ax.text(i + 0.2, m + 0.012, f"{m:.0%}", ha="center", fontsize=12, fontweight="bold")
    ax.axhline(0.5, color="gray", ls=":", lw=1.5)
    ax.set_xticks(x, labels, fontsize=12)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("better side wins")
    ax.set_title("Luck dies in long matches: one hand vs first-to-151")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fun_match151.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 2. the deck decides -----------------------------------------
    labels2, sames, splits, ties = [], [], [], []
    for a, b, label in matchups:
        recs = pair_records(t, a, b) or pair_records(t, b, a)
        if not recs:
            continue
        s, sp, ti = deck_decides(recs)
        labels2.append(label)
        sames.append(s)
        splits.append(sp)
        ties.append(ti)
        fun[f"deck_decides:{label}"] = {
            "same_winner_both_seatings": round(s, 3),
            "split_by_seat_or_skill": round(sp, 3),
            "involves_tie": round(ti, 3),
        }
    y = np.arange(len(labels2))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh(y, sames, color="#d95f02", label="same side wins both seatings\n(the deck decided)")
    ax.barh(
        y,
        splits,
        left=sames,
        color="#1b9e77",
        label="winner flips with the seats\n(skill/seat decided)",
    )
    ax.barh(
        y, ties, left=np.array(sames) + np.array(splits), color="#cccccc", label="a tie involved"
    )
    for i, s in enumerate(sames):
        ax.text(
            s / 2,
            i,
            f"{s:.0%}",
            ha="center",
            va="center",
            color="white",
            fontsize=13,
            fontweight="bold",
        )
    ax.set_yticks(y, labels2, fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of decks")
    ax.set_title("Who really wins a hand of Pişti? Often, the deck.")
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fun_deck_decides.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 3. the skill whisper in the luck storm ----------------------
    recs = pair_records(t, "greedy", "ppo_main")
    d = np.array([r["score_b"] - r["score_a"] for r in recs])  # RL perspective
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.histplot(d, bins=np.arange(-25.5, 26.5, 2), ax=ax, color="#6baed6", edgecolor="white")
    ax.axvline(0, color="gray", lw=1.5, ls=":")
    ax.axvline(d.mean(), color="crimson", lw=3)
    ax.annotate(
        f"the skill edge: {d.mean():+.1f} pts",
        xy=(d.mean(), ax.get_ylim()[1] * 0.92),
        xytext=(d.mean() + 6, ax.get_ylim()[1] * 0.92),
        color="crimson",
        fontsize=14,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="crimson"),
    )
    ax.set_xlabel("RL agent score − greedy score (one hand)")
    ax.set_title(
        f"A +{d.mean():.1f} whisper in a ±{d.std():.0f} storm " f"(std of single-hand outcomes)"
    )
    fun["storm"] = {
        "mean": round(float(d.mean()), 2),
        "std": round(float(d.std()), 2),
        "min": int(d.min()),
        "max": int(d.max()),
    }
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fun_storm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 4. pişti economics ------------------------------------------
    all_p = []
    for r in t["records"]:
        all_p += [r["pistis_a"], r["pistis_b"]]
    all_p = np.array(all_p)
    fun["pisti"] = {
        "mean_per_player_per_game": round(float(all_p.mean()), 3),
        "share_of_player_games_with_pisti": round(float((all_p > 0).mean()), 3),
        "max_in_one_game": int(all_p.max()),
    }

    with open("results/fun_stats.json", "w") as f:
        json.dump(fun, f, indent=2)
    print(json.dumps(fun, indent=2))


if __name__ == "__main__":
    main()
