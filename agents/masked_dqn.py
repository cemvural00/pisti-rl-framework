"""DQN with action masking for Pişti.

The action mask is part of the observation (`obs["action_mask"]`, equal to
the hand vector), so masking is implemented *inside the Q-network forward
pass*: illegal actions get Q ≈ -1e8. Standard DQN machinery then respects
legality everywhere it matters for free:

  - greedy action selection (argmax over masked Q)
  - the TD target's max over next-state actions (target network also masks,
    because the mask is in the stored next observation)

The only places SB3 samples unmasked random actions — ε-greedy exploration
and the learning-starts warm-up — are overridden to sample uniformly from
legal actions instead.

Scientific note: a trained DQN plays a *deterministic* greedy policy. In
imperfect-information games this is a structural handicap (predictability
is exploitable) — which is exactly the hypothesis this agent exists to test.
"""

from typing import Optional

import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MultiInputPolicy, QNetwork


class MaskedQNetwork(QNetwork):
    def forward(self, obs) -> th.Tensor:
        q = super().forward(obs)
        mask = obs["action_mask"].float()
        return q + (mask - 1.0) * 1e8


class MaskedDQNPolicy(MultiInputPolicy):
    def make_q_net(self) -> MaskedQNetwork:
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None
        )
        return MaskedQNetwork(**net_args).to(self.device)


def _masked_random(masks: np.ndarray) -> np.ndarray:
    if masks.ndim == 1:
        return np.array(np.random.choice(np.flatnonzero(masks)))
    return np.array([int(np.random.choice(np.flatnonzero(m))) for m in masks])


class MaskedDQN(DQN):
    policy_aliases = {**DQN.policy_aliases, "MaskedMultiInputPolicy": MaskedDQNPolicy}

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
    ):
        if not deterministic and np.random.rand() < self.exploration_rate:
            return _masked_random(np.asarray(observation["action_mask"])), state
        return self.policy.predict(observation, state, episode_start, deterministic)

    def _sample_action(
        self,
        learning_starts: int,
        action_noise=None,
        n_envs: int = 1,
    ):
        if self.num_timesteps < learning_starts:
            action = _masked_random(np.asarray(self._last_obs["action_mask"]))
            action = action.reshape(n_envs)
            return action, action
        return super()._sample_action(learning_starts, action_noise, n_envs)
