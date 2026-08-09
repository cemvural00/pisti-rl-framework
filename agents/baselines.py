"""Baseline policies for Pişti.

Protocol: predict(obs: Dict[str, np.ndarray], action_mask: np.ndarray) -> int
All baselines are stateless between moves; reset() is a no-op hook.
"""

from typing import Dict

import numpy as np

from engine.game import CARD_POINTS, JACK, rank_of


def _legal(action_mask: np.ndarray) -> np.ndarray:
    legal = np.flatnonzero(action_mask)
    if len(legal) == 0:
        raise ValueError("no legal actions in mask")
    return legal


class RandomAgent:
    """Plays a uniformly random card from hand."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def predict(self, obs: Dict, action_mask: np.ndarray) -> int:
        return int(self.rng.choice(_legal(action_mask)))


class GreedyAgent:
    """Captures whenever possible; otherwise discards its least valuable card.

    Capture preference: rank match over Jack (saves Jacks for later).
    Discard preference: lowest card-point value, avoiding Jacks, then low rank.
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def predict(self, obs: Dict, action_mask: np.ndarray) -> int:
        legal = _legal(action_mask)
        top = np.flatnonzero(obs["table_top"] > 0.5)

        if len(top) > 0:
            top_rank = rank_of(int(top[0]))
            rank_matches = [c for c in legal if rank_of(int(c)) == top_rank]
            if rank_matches:
                return int(rank_matches[0])
            jacks = [c for c in legal if rank_of(int(c)) == JACK]
            if jacks:
                return int(jacks[0])

        def discard_cost(c):
            c = int(c)
            return (CARD_POINTS[c], rank_of(c) == JACK, rank_of(c))

        return int(min(legal, key=discard_cost))


class PistiHunterAgent:
    """Heuristic focused on pişti opportunities.

    1. Take a pişti / double pişti when available.
    2. Capture by rank match; use a Jack only on valuable or large piles.
    3. When discarding, prefer a rank it holds in duplicate (so it can
       re-capture if the opponent matches) and never bait with Jacks or
       scoring cards.
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def predict(self, obs: Dict, action_mask: np.ndarray) -> int:
        legal = _legal(action_mask)
        top = np.flatnonzero(obs["table_top"] > 0.5)
        table_count = float(obs["stats"][0]) * 26.0  # un-normalize

        if len(top) > 0:
            top_card = int(top[0])
            top_rank = rank_of(top_card)
            rank_matches = [c for c in legal if rank_of(int(c)) == top_rank]
            if rank_matches:
                return int(rank_matches[0])  # includes pişti when pile == 1
            jacks = [c for c in legal if rank_of(int(c)) == JACK]
            pile_attractive = table_count >= 3 or CARD_POINTS[top_card] > 0
            if jacks and pile_attractive:
                return int(jacks[0])

        ranks_held = {}
        for c in legal:
            ranks_held.setdefault(rank_of(int(c)), []).append(int(c))

        def discard_cost(c):
            c = int(c)
            r = rank_of(c)
            duplicate = len(ranks_held[r]) > 1
            return (
                CARD_POINTS[c],  # don't give away points
                r == JACK,  # never bait with a Jack
                not duplicate,  # prefer ranks we can re-match
                r,
            )

        return int(min(legal, key=discard_cost))
