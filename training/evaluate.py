"""Round-robin tournament with mirrored deals + Bradley-Terry ratings.

Usage:
    python -m training.evaluate --agents random greedy hunter \
        expectimax "ppo:runs/ppo_main/final_model" \
        --n-deals 300 --out results/tournament.json

Agent specs:
    random | greedy | hunter
    expectimax[:n_samples,rollout_plies]   e.g. expectimax:32,8
    ppo:<model_path>                       MaskablePPO checkpoint
    ppo-nomem:<model_path>                 same, with the memory-ablated observer
    ppo-stoch:<model_path>                 stochastic (sampled) PPO policy
    dqn:<model_path>                       MaskedDQN checkpoint
    nfsp:<path> | nfsp-br:<path>           NFSP average policy / best response
    beast:<model_path>[:n_samples]         policy + batched PIMC rollouts
"""

import argparse
import itertools
import json
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np

from agents.baselines import GreedyAgent, PistiHunterAgent, RandomAgent
from agents.expectimax import ExpectimaxAgent
from encoding.obs import Observer
from training.match import play_match


def build_agent(spec: str, seed: int = 0):
    """Returns (name, agent, observer)."""
    if spec == "random":
        return spec, RandomAgent(seed=seed), Observer()
    if spec == "greedy":
        return spec, GreedyAgent(seed=seed), Observer()
    if spec == "hunter":
        return spec, PistiHunterAgent(seed=seed), Observer()
    if spec.startswith("expectimax"):
        n, r = 16, 6
        if ":" in spec:
            n, r = (int(x) for x in spec.split(":", 1)[1].split(","))
        return f"expectimax{n}", ExpectimaxAgent(n_samples=n, rollout_plies=r, seed=seed), Observer()
    if spec.startswith("ppo-nomem:"):
        from agents.frozen import FrozenPolicyAgent

        path = spec.split(":", 1)[1]
        name = "nomem-" + os.path.basename(os.path.dirname(path))
        return name, FrozenPolicyAgent.load(path), Observer(memory=False)
    if spec.startswith("ppo:"):
        from agents.frozen import FrozenPolicyAgent

        path = spec.split(":", 1)[1]
        name = os.path.basename(os.path.dirname(path))
        return name, FrozenPolicyAgent.load(path), Observer()
    if spec.startswith("ppo-stoch:"):
        from agents.frozen import FrozenPolicyAgent

        path = spec.split(":", 1)[1]
        name = "stoch-" + os.path.basename(path).replace(".zip", "")
        return name, FrozenPolicyAgent.load(path, deterministic=False), Observer()
    if spec.startswith("beast:"):
        from agents.beast import BeastAgent

        parts = spec.split(":")
        path = parts[1]
        n = int(parts[2]) if len(parts) > 2 else 32
        return f"beast{n}", BeastAgent(path, n_samples=n, seed=seed), Observer()
    if spec.startswith("nfsp:"):
        from agents.nfsp import NFSPAgent

        path = spec.split(":", 1)[1]
        name = os.path.basename(os.path.dirname(path)) or "nfsp"
        return name, NFSPAgent.load(path, mode="avg", seed=seed), Observer()
    if spec.startswith("nfsp-br:"):
        from agents.nfsp import NFSPAgent

        path = spec.split(":", 1)[1]
        name = "br-" + (os.path.basename(os.path.dirname(path)) or "nfsp")
        return name, NFSPAgent.load(path, mode="br", seed=seed), Observer()
    if spec.startswith("dqn:"):
        from agents.frozen import FrozenPolicyAgent
        from agents.masked_dqn import MaskedDQN

        path = spec.split(":", 1)[1]
        name = os.path.basename(os.path.dirname(path))
        model = MaskedDQN.load(path, device="cpu")
        model.policy.set_training_mode(False)
        return name, FrozenPolicyAgent(model.policy, deterministic=True), Observer()
    raise ValueError(f"unknown agent spec: {spec}")


def bradley_terry(
    names: List[str], results: Dict[Tuple[str, str], Tuple[float, float]]
) -> Dict[str, float]:
    """Fit Bradley-Terry strengths from (wins_a, wins_b) per pair (ties
    counted as half). Returns Elo-like ratings centered at 1500."""
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)
    wins = np.zeros((k, k))
    for (a, b), (wa, wb) in results.items():
        wins[idx[a], idx[b]] += wa
        wins[idx[b], idx[a]] += wb
    strengths = np.ones(k)
    for _ in range(500):
        new = np.zeros(k)
        for i in range(k):
            num = wins[i].sum()
            den = sum(
                (wins[i, j] + wins[j, i]) / (strengths[i] + strengths[j])
                for j in range(k)
                if j != i
            )
            new[i] = num / den if den > 0 else strengths[i]
        new /= np.exp(np.mean(np.log(new)))  # geometric-mean normalize
        if np.allclose(new, strengths, atol=1e-10):
            strengths = new
            break
        strengths = new
    return {n: 1500 + 400 * math.log10(strengths[idx[n]]) for n in names}


def run_tournament(specs: List[str], n_deals: int, seed: int, out: str) -> Dict:
    agents = [build_agent(s, seed=seed) for s in specs]
    names = [a[0] for a in agents]
    print(f"tournament: {names}, {n_deals} mirrored deals per pair")

    summaries = []
    pair_wins: Dict[Tuple[str, str], Tuple[float, float]] = {}
    records = []
    for (na, a, oa), (nb, b, ob) in itertools.combinations(agents, 2):
        t0 = time.perf_counter()
        res = play_match(
            a, b, n_deals=n_deals, seed=seed, name_a=na, name_b=nb,
            observer_a=oa, observer_b=ob,
        )
        s = res.summary()
        summaries.append(s)
        pair_wins[(na, nb)] = (
            res.wins_a + 0.5 * res.ties,
            res.wins_b + 0.5 * res.ties,
        )
        records.extend(
            {
                "a": na, "b": nb, "deal": r.deal, "a_leads": r.a_leads,
                "score_a": r.score_a, "score_b": r.score_b,
                "pistis_a": r.pistis_a, "pistis_b": r.pistis_b,
                "captured_a": r.captured_a, "captured_b": r.captured_b,
            }
            for r in res.records
        )
        print(
            f"  {na} vs {nb}: win={s['win_rate_a']:.3f} "
            f"diff={s['mean_diff']:+.2f}±{s['diff_ci95']:.2f} "
            f"({time.perf_counter() - t0:.0f}s)"
        )

    ratings = bradley_terry(names, pair_wins)
    result = {
        "agents": names,
        "n_deals": n_deals,
        "seed": seed,
        "ratings": dict(sorted(ratings.items(), key=lambda x: -x[1])),
        "matches": summaries,
        "records": records,
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f)
    print("\nBradley-Terry ratings (Elo-like):")
    for n, r in result["ratings"].items():
        print(f"  {n:>20}: {r:7.1f}")
    print(f"saved -> {out}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="+", required=True)
    p.add_argument("--n-deals", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/tournament.json")
    args = p.parse_args()
    run_tournament(args.agents, args.n_deals, args.seed, args.out)


if __name__ == "__main__":
    main()
