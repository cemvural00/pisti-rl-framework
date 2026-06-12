"""The Beast: trained policy + decision-time determinized search.

AlphaZero's lesson applied to Pişti: a learned policy is strongest when
used inside search, not instead of it. For every legal action we sample N
determinizations of the hidden information (honest information-set
sampling via PistiGame.determinize), play the action, then roll every
resulting game to the end with the trained MaskablePPO policy playing
BOTH seats (stochastic rollouts — averaging over plausible continuations,
not assuming one). The action with the best mean final score
differential wins.

All rollouts advance in lockstep so policy inference is batched: one
network call per ply for up to 4*N games, ~0.3s per decision at N=32.
"""

from typing import Dict, Optional

import numpy as np

from encoding.obs import Observer
from engine.game import PistiGame


class BeastAgent:
    wants_game = True

    def __init__(self, model_path: str, n_samples: int = 32,
                 seed: Optional[int] = None, name: str = "beast"):
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(model_path, device="cpu")
        self.policy = model.policy
        self.policy.set_training_mode(False)
        self.observer = Observer()
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self.name = name

    def reset(self):
        pass

    def _rollout_all(self, sims):
        """Advance all games to terminal, batching policy calls per ply."""
        while True:
            alive = [g for g in sims if not g.done]
            if not alive:
                return
            encs = [self.observer.encode(g, g.current) for g in alive]
            obs_b = {k: np.stack([e[k] for e in encs]) for k in encs[0]}
            masks = obs_b["action_mask"].astype(bool)
            actions, _ = self.policy.predict(
                obs_b, deterministic=False, action_masks=masks
            )
            for g, a in zip(alive, actions):
                g.step(int(a))

    def predict(
        self,
        obs: Dict,
        action_mask: np.ndarray,
        game: Optional[PistiGame] = None,
        player: Optional[int] = None,
    ) -> int:
        if game is None:
            raise ValueError("BeastAgent requires the game object")
        if player is None:
            player = game.current
        legal = list(game.hands[player])
        if len(legal) == 1:
            return legal[0]

        sims, owner = [], []
        for _ in range(self.n_samples):
            det = game.determinize(player, self.rng)
            for ai, a in enumerate(legal):
                g = det.clone()
                g.step(a)
                sims.append(g)
                owner.append(ai)

        self._rollout_all(sims)

        values = np.zeros(len(legal))
        for g, ai in zip(sims, owner):
            values[ai] += g.score_diff(player)
        return int(legal[int(np.argmax(values))])
