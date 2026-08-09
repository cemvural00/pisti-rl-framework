"""Run the preregistered evaluation for the paired memory study."""

import argparse
import json
import os
import random
from dataclasses import asdict
from typing import Dict

import numpy as np
import torch

from agents.baselines import GreedyAgent, PistiHunterAgent
from agents.expectimax import ExpectimaxAgent
from agents.frozen import FrozenPolicyAgent
from encoding.obs import Observer
from training.match import MatchResult, play_match


def serialize_match(result: MatchResult) -> Dict:
    return {
        "summary": result.summary(),
        "records": [asdict(record) for record in result.records],
    }


def load_policy(condition: str, seed: int, deterministic: bool = True):
    path = f"runs/study_mem_{condition}_s{seed}/final_model.zip"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return FrozenPolicyAgent.load(path, deterministic=deterministic)


def evaluate(n_seeds: int, n_deals: int, eval_seed: int) -> Dict:
    output: Dict[str, object] = {
        "protocol": {
            "n_training_seeds": n_seeds,
            "n_deals_per_match": n_deals,
            "games_per_match": 2 * n_deals,
            "evaluation_seed": eval_seed,
            "stochastic_policy_seed_rule": "evaluation_seed + training_seed",
            "mirrored": True,
            "primary_policy_mode": "deterministic",
        },
        "direct": [],
        "acute": [],
        "stochastic_sensitivity": [],
        "anchors": [],
    }
    full_observer = Observer(memory=True)
    zero_observer = Observer(memory=False)

    for seed in range(n_seeds):
        on = load_policy("on", seed)
        off = load_policy("off", seed)
        direct = play_match(
            on,
            off,
            n_deals=n_deals,
            seed=eval_seed,
            name_a=f"mem_on_s{seed}",
            name_b=f"mem_off_s{seed}",
            observer_a=full_observer,
            observer_b=zero_observer,
        )
        output["direct"].append({"training_seed": seed, **serialize_match(direct)})

        on_normal = load_policy("on", seed)
        on_zeroed = load_policy("on", seed)
        acute = play_match(
            on_normal,
            on_zeroed,
            n_deals=n_deals,
            seed=eval_seed,
            name_a=f"mem_on_s{seed}",
            name_b=f"mem_on_s{seed}_seen_zeroed",
            observer_a=full_observer,
            observer_b=zero_observer,
        )
        output["acute"].append({"training_seed": seed, **serialize_match(acute)})

        stochastic_seed = eval_seed + seed
        on_stochastic = load_policy("on", seed, deterministic=False)
        off_stochastic = load_policy("off", seed, deterministic=False)
        random.seed(stochastic_seed)
        np.random.seed(stochastic_seed)
        torch.manual_seed(stochastic_seed)
        sensitivity = play_match(
            on_stochastic,
            off_stochastic,
            n_deals=n_deals,
            seed=eval_seed,
            name_a=f"stoch_mem_on_s{seed}",
            name_b=f"stoch_mem_off_s{seed}",
            observer_a=full_observer,
            observer_b=zero_observer,
        )
        output["stochastic_sensitivity"].append(
            {"training_seed": seed, **serialize_match(sensitivity)}
        )

        anchor_factories = {
            "greedy": lambda: GreedyAgent(seed=eval_seed),
            "hunter": lambda: PistiHunterAgent(seed=eval_seed),
            "expectimax16": lambda: ExpectimaxAgent(n_samples=16, rollout_plies=6, seed=eval_seed),
        }
        for condition, observer in (("on", full_observer), ("off", zero_observer)):
            for anchor_name, factory in anchor_factories.items():
                policy = load_policy(condition, seed)
                match = play_match(
                    policy,
                    factory(),
                    n_deals=n_deals,
                    seed=eval_seed,
                    name_a=f"mem_{condition}_s{seed}",
                    name_b=anchor_name,
                    observer_a=observer,
                )
                output["anchors"].append(
                    {
                        "training_seed": seed,
                        "condition": condition,
                        "anchor": anchor_name,
                        **serialize_match(match),
                    }
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-deals", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--out", default="results/memory_study_matches.json")
    args = parser.parse_args()
    result = evaluate(args.n_seeds, args.n_deals, args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
