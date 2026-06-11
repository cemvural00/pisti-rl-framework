"""Gymnasium environment for single-agent Pişti training.

The learning agent is always player 0; a pluggable opponent (any object
with `predict(obs_dict, action_mask) -> int`) plays player 1 internally.

Reward modes:
  - "delta" (default): change in the agent's score differential between
    consecutive agent decision points. Includes everything the opponent's
    reply did to the score, and the terminal majority bonus. Telescopes to
    the exact final score differential over an episode.
  - "sparse": 0 everywhere except the terminal step, which pays the final
    score differential.

Both modes therefore have identical episode returns; "delta" just
distributes the signal. (The old codebase lost the terminal reward
entirely whenever the opponent made the last move of the game.)

Mirrored-deal evaluation: pass `options={"deck": [...], "agent_leads": bool}`
to reset() to control the deal exactly.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import Env, spaces

from engine.game import PistiGame, new_deck
from encoding.obs import Observer
import random as _random


class PistiEnv(Env):
    metadata = {"render_modes": ["human"], "name": "Pisti-v1"}

    def __init__(
        self,
        opponent=None,
        observer: Optional[Observer] = None,
        opponent_observer: Optional[Observer] = None,
        reward_mode: str = "delta",
        reward_scale: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        from agents.baselines import RandomAgent

        self.opponent = opponent or RandomAgent(seed=seed)
        self.observer = observer or Observer()
        # Opponent gets full memory by default even if the agent is ablated
        self.opponent_observer = opponent_observer or Observer()
        if reward_mode not in ("delta", "sparse"):
            raise ValueError(f"unknown reward_mode: {reward_mode}")
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale

        self.observation_space = self.observer.observation_space()
        self.action_space = spaces.Discrete(52)

        self._rng = _random.Random(seed)
        self._episode = 0
        self.game: Optional[PistiGame] = None
        self._prev_diff = 0.0

    # ------------------------------------------------------------------
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = _random.Random(seed)
        options = options or {}

        deck = options.get("deck")
        if deck is None:
            deck = new_deck(self._rng)
        if "agent_leads" in options:
            agent_leads = options["agent_leads"]
        else:
            agent_leads = self._episode % 2 == 0  # alternate seats
        self._episode += 1

        self.game = PistiGame(deck=list(deck), first_player=0 if agent_leads else 1)

        if hasattr(self.opponent, "reset"):
            self.opponent.reset()

        # Anchor at 0 so episode return telescopes to the exact final score
        # differential — including whatever the opponent's lead move does.
        self._prev_diff = 0.0

        # If the opponent leads, let it move first
        if self.game.current == 1:
            self._opponent_move()

        return self.observer.encode(self.game, 0), {}

    def step(
        self, action
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        game = self.game
        if game is None or game.done:
            raise RuntimeError("call reset() before step()")
        action = int(action)

        game.step(action)
        # Opponent replies until it's the agent's turn again or the game ends
        while not game.done and game.current == 1:
            self._opponent_move()

        diff = float(game.score_diff(0))
        if self.reward_mode == "delta":
            reward = (diff - self._prev_diff) * self.reward_scale
        else:
            reward = diff * self.reward_scale if game.done else 0.0
        self._prev_diff = diff

        terminated = game.done
        info: Dict[str, Any] = {}
        if terminated:
            s0, s1 = game.scores()
            info["scores"] = (s0, s1)
            info["score_diff"] = s0 - s1
            info["winner"] = game.winner()
            info["pistis"] = (
                game.pistis[0] + game.double_pistis[0],
                game.pistis[1] + game.double_pistis[1],
            )
            info["captured"] = tuple(game.captured_count)

        obs = self.observer.encode(game, 0)
        return obs, reward, terminated, False, info

    def _opponent_move(self) -> None:
        obs = self.opponent_observer.encode(self.game, 1)
        if getattr(self.opponent, "wants_game", False):
            action = self.opponent.predict(
                obs, obs["action_mask"], game=self.game, player=1
            )
        else:
            action = self.opponent.predict(obs, obs["action_mask"])
        self.game.step(int(action))

    # ------------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        """Mask for MaskablePPO: True where the action is a card in hand."""
        mask = np.zeros(52, dtype=bool)
        if self.game is not None and not self.game.done:
            mask[self.game.hands[0]] = True
        return mask

    def set_opponent(self, opponent) -> None:
        self.opponent = opponent

    def render(self):
        if self.game is not None:
            print(self.game.render())
