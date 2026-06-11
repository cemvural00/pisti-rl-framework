"""Observation builder for the rebuilt Pişti engine.

Produces a Gymnasium Dict observation from the perspective of one player:
  - hand:        (52,) multi-hot of cards currently held
  - table_top:   (52,) one-hot of the table's top card (zeros if empty)
  - seen:        (52,) multi-hot of all cards this player has observed
                 (own hand + every card played face-up). Zeroed out when
                 memory=False (the card-counting ablation).
  - stats:       (12,) scalar features, roughly normalized to [0, 1]
  - action_mask: (52,) bool — playable cards (== hand)

Every card in hand is always playable in Pişti, so the action mask equals
the hand vector; it is kept separate because MaskablePPO consumes it.
"""

from typing import Dict

import numpy as np
from gymnasium import spaces

from engine.game import PistiGame

STATS_DIM = 12


class Observer:
    """Encodes a PistiGame into a Dict observation for one player."""

    def __init__(self, memory: bool = True):
        self.memory = memory

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "hand": spaces.Box(0, 1, shape=(52,), dtype=np.float32),
                "table_top": spaces.Box(0, 1, shape=(52,), dtype=np.float32),
                "seen": spaces.Box(0, 1, shape=(52,), dtype=np.float32),
                "stats": spaces.Box(0, 1, shape=(STATS_DIM,), dtype=np.float32),
                "action_mask": spaces.MultiBinary(52),
            }
        )

    def encode(self, game: PistiGame, player: int) -> Dict[str, np.ndarray]:
        opp = 1 - player

        hand = np.zeros(52, dtype=np.float32)
        hand[game.hands[player]] = 1.0

        table_top = np.zeros(52, dtype=np.float32)
        if game.table:
            table_top[game.table[-1]] = 1.0

        seen = np.zeros(52, dtype=np.float32)
        if self.memory:
            seen[game.seen_cards(player)] = 1.0

        last_cap = game.last_capturer
        stats = np.array(
            [
                len(game.table) / 26.0,
                len(game.stock) / 40.0,
                len(game.hands[player]) / 4.0,
                len(game.hands[opp]) / 4.0,
                game.captured_count[player] / 52.0,
                game.captured_count[opp] / 52.0,
                game.points[player] / 30.0,
                game.points[opp] / 30.0,
                (game.pistis[player] + 2 * game.double_pistis[player]) / 4.0,
                (game.pistis[opp] + 2 * game.double_pistis[opp]) / 4.0,
                1.0 if last_cap == player else 0.0,
                1.0 if last_cap == opp else 0.0,
            ],
            dtype=np.float32,
        )

        return {
            "hand": hand,
            "table_top": table_top,
            "seen": seen,
            "stats": stats,
            "action_mask": hand.astype(np.int8),
        }
