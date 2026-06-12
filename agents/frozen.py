"""Opponents built from trained policies, plus the self-play league.

FrozenPolicyAgent wraps a MaskableActorCriticPolicy (a deep copy of the
training policy, or one loaded from disk) behind the standard
predict(obs, mask) protocol.

League holds the self-play state shared by all training envs:
  - a pool of frozen snapshot policies
  - the live training policy (for true mirror self-play)
  - the current curriculum phase's opponent mixture weights

MixtureOpponent is the per-env opponent: on each episode reset it samples
an opponent type from the league's current mixture.
"""

import copy
from typing import Dict, List, Optional

import numpy as np

from agents.baselines import GreedyAgent, PistiHunterAgent, RandomAgent
from agents.expectimax import ExpectimaxAgent


class FrozenPolicyAgent:
    """A frozen policy (MaskablePPO or MaskedDQN) behind predict(obs, mask).

    MaskablePPO policies take an `action_masks` kwarg; DQN policies mask
    inside the Q-network (the mask is part of the observation), so we
    detect once which calling convention the policy supports.
    """

    def __init__(self, policy, deterministic: bool = False, name: str = "frozen"):
        import inspect

        self.policy = policy
        self.deterministic = deterministic
        self.name = name
        self._takes_masks = (
            "action_masks" in inspect.signature(policy.predict).parameters
        )

    @classmethod
    def snapshot(cls, model, name: str = "snapshot") -> "FrozenPolicyAgent":
        """Deep-copy the model's current policy onto CPU, eval mode."""
        policy = copy.deepcopy(model.policy).to("cpu")
        policy.set_training_mode(False)
        return cls(policy, deterministic=False, name=name)

    @classmethod
    def load(cls, path: str, deterministic: bool = True) -> "FrozenPolicyAgent":
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(path, device="cpu")
        model.policy.set_training_mode(False)
        return cls(model.policy, deterministic=deterministic, name=str(path))

    def reset(self):
        pass

    def predict(self, obs: Dict, action_mask: np.ndarray, **_) -> int:
        if self._takes_masks:
            action, _state = self.policy.predict(
                obs,
                deterministic=self.deterministic,
                action_masks=np.asarray(action_mask, dtype=bool),
            )
        else:
            action, _state = self.policy.predict(
                obs, deterministic=self.deterministic
            )
        return int(action)


class League:
    """Shared self-play state: snapshot pool + mixture weights."""

    def __init__(self, pool_size: int = 12, seed: int = 0):
        self.pool_size = pool_size
        self.snapshots: List[FrozenPolicyAgent] = []
        self.live_model = None  # set by the trainer for "latest" self-play
        self.weights: Dict[str, float] = {"random": 0.5, "greedy": 0.5}
        self.rng = np.random.default_rng(seed)

    def add_snapshot(self, model, step: int) -> None:
        self.snapshots.append(
            FrozenPolicyAgent.snapshot(model, name=f"snap_{step}")
        )
        if len(self.snapshots) > self.pool_size:
            # Drop a random old snapshot, keep the newest always
            drop = self.rng.integers(0, len(self.snapshots) - 1)
            self.snapshots.pop(int(drop))

    def set_weights(self, weights: Dict[str, float]) -> None:
        self.weights = dict(weights)


class MixtureOpponent:
    """Per-env opponent: re-samples its identity from the league mixture
    at every episode reset. Falls back to greedy when the pool is empty."""

    wants_game = True  # sub-agents may need the game (expectimax)

    def __init__(self, league: League, seed: int = 0,
                 expectimax_kwargs: Optional[dict] = None):
        self.league = league
        self.rng = np.random.default_rng(seed)
        self._fixed = {
            "random": RandomAgent(seed=seed),
            "greedy": GreedyAgent(seed=seed),
            "hunter": PistiHunterAgent(seed=seed),
            "expectimax": ExpectimaxAgent(
                **(expectimax_kwargs or {"n_samples": 6, "rollout_plies": 4}),
                seed=seed,
            ),
        }
        self.active = self._fixed["random"]
        self.active_name = "random"

    def reset(self):
        names = list(self.league.weights.keys())
        probs = np.array([self.league.weights[n] for n in names], dtype=float)
        probs /= probs.sum()
        choice = str(self.rng.choice(names, p=probs))

        if choice == "pool" and not self.league.snapshots:
            choice = "greedy"
        if choice == "latest" and self.league.live_model is None:
            choice = "greedy"

        if choice == "pool":
            idx = int(self.rng.integers(0, len(self.league.snapshots)))
            self.active = self.league.snapshots[idx]
        elif choice == "latest":
            self.active = FrozenPolicyAgent(
                self.league.live_model.policy, deterministic=False, name="latest"
            )
        else:
            self.active = self._fixed[choice]
        self.active_name = choice
        if hasattr(self.active, "reset"):
            self.active.reset()

    def predict(self, obs, action_mask, game=None, player=None) -> int:
        if getattr(self.active, "wants_game", False):
            return self.active.predict(obs, action_mask, game=game, player=player)
        return self.active.predict(obs, action_mask)
