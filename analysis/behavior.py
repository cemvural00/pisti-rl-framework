"""Behavioral statistics: what tactics does an agent actually use?

Plays instrumented games and measures, per agent:
  - captures/pistis/double pistis per game
  - jack discipline: average pile size (and points) captured with a Jack,
    and how often a Jack is wasted on an empty-ish pile
  - bait rate: discarding a rank of which the agent holds a duplicate
    (classic pişti setup: if the opponent matches it, you re-match)
  - risky discards: putting a scoring card (A/J/2♣/10♦) on the table
    without capturing
  - leftover sweeps: points gained from the end-of-game leftover pile
"""

import json
from collections import defaultdict
from typing import Dict, List

import numpy as np

from encoding.obs import Observer
from engine.game import CARD_POINTS, JACK, PistiGame, new_deck, rank_of
import random


def behavior_stats(
    agent,
    opponent,
    n_games: int = 400,
    seed: int = 0,
    observer_agent: Observer = None,
    observer_opp: Observer = None,
) -> Dict:
    """Agent plays as both seats vs opponent; stats collected for agent."""
    rng = random.Random(seed)
    obs_a = observer_agent or Observer()
    obs_o = observer_opp or Observer()
    stats = defaultdict(float)
    jack_pile_sizes: List[int] = []
    jack_pile_points: List[int] = []

    for ep in range(n_games):
        deck = new_deck(rng)
        agent_seat = ep % 2
        game = PistiGame(deck=deck, first_player=ep % 2)
        for a in (agent, opponent):
            if hasattr(a, "reset"):
                a.reset()

        while not game.done:
            p = game.current
            is_agent = p == agent_seat
            actor = agent if is_agent else opponent
            observer = obs_a if is_agent else obs_o
            obs = observer.encode(game, p)
            if getattr(actor, "wants_game", False):
                action = actor.predict(obs, obs["action_mask"], game=game, player=p)
            else:
                action = actor.predict(obs, obs["action_mask"])
            action = int(action)

            if is_agent:
                hand = list(game.hands[p])
                table = list(game.table)

            info = game.step(action)

            if is_agent:
                stats["moves"] += 1
                if info["captured"]:
                    stats["captures"] += 1
                    if rank_of(action) == JACK and (not table or rank_of(table[-1]) != JACK):
                        jack_pile_sizes.append(len(table))
                        jack_pile_points.append(sum(CARD_POINTS[c] for c in table))
                    if info["pisti"] == 1:
                        stats["pistis"] += 1
                    elif info["pisti"] == 2:
                        stats["double_pistis"] += 1
                else:
                    # discard onto the table
                    same_rank_in_hand = sum(1 for c in hand if rank_of(c) == rank_of(action))
                    if not table and same_rank_in_hand >= 2:
                        stats["baits"] += 1
                    if not table:
                        stats["empty_table_discards"] += 1
                    if CARD_POINTS[action] > 0:
                        stats["risky_discards"] += 1
                    if rank_of(action) == JACK:
                        stats["jack_discards"] += 1

        # leftover sweep points for the agent
        s = game.scores()
        stats["games"] += 1
        stats["wins"] += (
            1.0 if game.winner() == agent_seat else (0.5 if game.winner() is None else 0.0)
        )
        stats["total_pts"] += game.points[agent_seat]
        stats["total_diff"] += s[agent_seat] - s[1 - agent_seat]

    g = stats["games"]
    return {
        "games": int(g),
        "win_rate": round(stats["wins"] / g, 3),
        "mean_diff": round(stats["total_diff"] / g, 2),
        "captures_per_game": round(stats["captures"] / g, 2),
        "pistis_per_game": round(stats["pistis"] / g, 3),
        "double_pistis_per_game": round(stats["double_pistis"] / g, 4),
        "bait_rate": round(stats["baits"] / max(stats["empty_table_discards"], 1), 3),
        "risky_discards_per_game": round(stats["risky_discards"] / g, 2),
        "jack_discards_per_game": round(stats["jack_discards"] / g, 3),
        "jack_capture_mean_pile": (
            round(float(np.mean(jack_pile_sizes)), 2) if jack_pile_sizes else None
        ),
        "jack_capture_mean_points": (
            round(float(np.mean(jack_pile_points)), 2) if jack_pile_points else None
        ),
    }


def compare_agents(
    agent_specs: List[str], n_games: int = 400, seed: int = 0, opponent_spec: str = "greedy"
) -> Dict[str, Dict]:
    """Behavior of each agent measured against a common reference opponent."""
    from training.evaluate import build_agent

    _, ref_opp, ref_obs = build_agent(opponent_spec, seed=seed + 1)
    out = {}
    for spec in agent_specs:
        name, agent, obs = build_agent(spec, seed=seed)
        out[name] = behavior_stats(
            agent,
            ref_opp,
            n_games=n_games,
            seed=seed,
            observer_agent=obs,
            observer_opp=ref_obs,
        )
        print(name, "->", json.dumps(out[name]))
    return out
