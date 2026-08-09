"""NFSP networks and the evaluation-time agent wrapper.

NFSP (Heinrich & Silver, 2016) trains two networks from self-play:
  - Q (best response): DQN trained on all transitions,
  - Pi (average policy): supervised on the best response's own actions,
    sampled into a reservoir buffer — its time-average converges toward a
    Nash equilibrium strategy in the fictitious-play sense.

The average policy Pi is the deliverable: an inherently *stochastic*
policy (masked softmax) that we evaluate and attack like every other
agent in the study.
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

FEAT_DIM = 52 * 3 + 12  # hand, table_top, seen, stats


def featurize(obs: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([obs["hand"], obs["table_top"], obs["seen"], obs["stats"]]).astype(
        np.float32
    )


class MLP(nn.Module):
    def __init__(self, hidden=(256, 256)):
        super().__init__()
        layers, d = [], FEAT_DIM
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 52))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NFSPNets:
    """Container: Q, Q-target, Pi + persistence."""

    def __init__(self, hidden=(256, 256), device="cpu"):
        self.device = device
        self.q = MLP(hidden).to(device)
        self.q_target = MLP(hidden).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.pi = MLP(hidden).to(device)
        self.hidden = tuple(hidden)

    def sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())

    def save(self, path: str):
        torch.save(
            {
                "q": self.q.state_dict(),
                "pi": self.pi.state_dict(),
                "hidden": self.hidden,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device="cpu") -> "NFSPNets":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        nets = cls(hidden=ckpt["hidden"], device=device)
        nets.q.load_state_dict(ckpt["q"])
        nets.q_target.load_state_dict(ckpt["q"])
        nets.pi.load_state_dict(ckpt["pi"])
        return nets


class NFSPAgent:
    """Plays the average policy Pi (or the greedy best response).

    mode="avg":  masked softmax sample from Pi — the Nash candidate.
    mode="br":   masked argmax of Q — the exploitative best response.
    """

    def __init__(
        self, nets: NFSPNets, mode: str = "avg", seed: Optional[int] = None, name: str = "nfsp"
    ):
        self.nets = nets
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.name = name

    @classmethod
    def load(cls, path: str, mode: str = "avg", seed=None) -> "NFSPAgent":
        return cls(NFSPNets.load(path), mode=mode, seed=seed, name=path)

    def reset(self):
        pass

    @torch.no_grad()
    def predict(self, obs: Dict, action_mask: np.ndarray, **_) -> int:
        x = torch.from_numpy(featurize(obs)).unsqueeze(0)
        mask = np.asarray(action_mask, dtype=bool)
        if self.mode == "br":
            q = self.nets.q(x).squeeze(0).numpy()
            q[~mask] = -1e9
            return int(np.argmax(q))
        logits = self.nets.pi(x).squeeze(0).numpy()
        logits[~mask] = -1e9
        z = logits - logits.max()
        p = np.exp(z)
        p /= p.sum()
        return int(self.rng.choice(52, p=p))
