"""Determinized expectimax agent (Perfect Information Monte Carlo).

For each legal action it samples N "determinizations" — completions of the
hidden information (opponent hand, stock order, hidden center cards) drawn
uniformly from the cards this player has NOT seen — then plays the action
in each sample, rolls out a few plies with a greedy policy for both sides,
and averages the resulting score differential. Picks the best action
(optionally softmax-sampled with a temperature, used to weaken it).

This agent needs the real game object to determinize from, but it only
ever reads information available to its own information set:
`game.seen_cards(player)`, hand sizes, pile contents that were played
face-up. Set `wants_game = True` so drivers pass the game in.
"""

from typing import Dict, List, Optional

import numpy as np

from engine.game import (
    CARD_POINTS,
    JACK,
    MAJORITY_POINTS,
    PistiGame,
    rank_of,
)


def _greedy_action(game: PistiGame) -> int:
    """Fast in-engine greedy policy used for rollouts."""
    hand = game.hands[game.current]
    if game.table:
        top_rank = rank_of(game.table[-1])
        for c in hand:
            if rank_of(c) == top_rank:
                return c
        for c in hand:
            if rank_of(c) == JACK:
                return c
    return min(
        hand, key=lambda c: (CARD_POINTS[c], rank_of(c) == JACK, rank_of(c))
    )


def _evaluate(game: PistiGame, player: int) -> float:
    """Score differential plus a majority-progress shaping term."""
    diff = float(game.score_diff(player))
    if not game.done:
        opp = 1 - player
        cap_lead = game.captured_count[player] - game.captured_count[opp]
        diff += MAJORITY_POINTS * np.tanh(cap_lead / 13.0)
        # Cards on the table are up for grabs; mildly value board points
        if game.table and game.last_capturer == player:
            diff += 0.1 * sum(CARD_POINTS[c] for c in game.table)
    return diff


class ExpectimaxAgent:
    """PIMC expectimax over determinized games.

    Args:
        n_samples: determinizations per legal action.
        rollout_plies: greedy rollout length after the candidate action.
        temperature: 0 = argmax; >0 = softmax over action values
            (higher = weaker/easier opponent).
    """

    wants_game = True

    def __init__(
        self,
        n_samples: int = 16,
        rollout_plies: int = 6,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.n_samples = n_samples
        self.rollout_plies = rollout_plies
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    # ------------------------------------------------------------------
    def _rollout_value(self, game: PistiGame, player: int) -> float:
        g = game
        for _ in range(self.rollout_plies):
            if g.done:
                break
            g.step(_greedy_action(g))
        return _evaluate(g, player)

    def predict(
        self,
        obs: Dict,
        action_mask: np.ndarray,
        game: Optional[PistiGame] = None,
        player: Optional[int] = None,
    ) -> int:
        if game is None:
            raise ValueError("ExpectimaxAgent requires the game object")
        if player is None:
            player = game.current
        legal: List[int] = list(game.hands[player])
        if len(legal) == 1:
            return legal[0]

        values = np.zeros(len(legal))
        for s in range(self.n_samples):
            det = game.determinize(player, self.rng)
            for i, action in enumerate(legal):
                g = det.clone()
                g.step(action)
                values[i] += self._rollout_value(g, player)
        values /= self.n_samples

        if self.temperature > 0:
            z = values / self.temperature
            z -= z.max()
            p = np.exp(z)
            p /= p.sum()
            return int(self.rng.choice(legal, p=p))
        return int(legal[int(np.argmax(values))])
