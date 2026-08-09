"""NFSP self-play training for Pişti.

Both seats are controlled by the same NFSP agent (parameter sharing). At
every episode each seat independently picks its policy for that episode:
the best response (prob. eta) or the average policy (prob. 1 - eta).
All decisions produce RL transitions (per-seat telescoping score-diff
rewards, identical objective to the PPO/DQN runs); best-response
decisions additionally feed the (state, action) reservoir that trains the
average policy.

Usage:
    python -m training.train_nfsp --steps 6_000_000 --run-name nfsp_main
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from agents.baselines import GreedyAgent, PistiHunterAgent
from agents.expectimax import ExpectimaxAgent
from agents.nfsp import FEAT_DIM, NFSPAgent, NFSPNets, featurize
from encoding.obs import Observer
from engine.game import PistiGame, new_deck
from training.match import play_match


class ReplayBuffer:
    """Circular (s, a, r, s', done) buffer; next-state mask is s'[:52]."""

    def __init__(self, size: int):
        self.size = size
        self.s = np.zeros((size, FEAT_DIM), dtype=np.float16)
        self.a = np.zeros(size, dtype=np.int64)
        self.r = np.zeros(size, dtype=np.float32)
        self.s2 = np.zeros((size, FEAT_DIM), dtype=np.float16)
        self.done = np.zeros(size, dtype=np.float32)
        self.n = 0
        self.idx = 0

    def add(self, s, a, r, s2, done):
        i = self.idx
        self.s[i], self.a[i], self.r[i], self.done[i] = s, a, r, float(done)
        self.s2[i] = s2
        self.idx = (i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, batch, rng):
        j = rng.integers(0, self.n, batch)
        return (
            torch.from_numpy(self.s[j].astype(np.float32)),
            torch.from_numpy(self.a[j]),
            torch.from_numpy(self.r[j]),
            torch.from_numpy(self.s2[j].astype(np.float32)),
            torch.from_numpy(self.done[j]),
        )


class Reservoir:
    """Reservoir-sampled (s, a) buffer for the average policy."""

    def __init__(self, size: int):
        self.size = size
        self.s = np.zeros((size, FEAT_DIM), dtype=np.float16)
        self.a = np.zeros(size, dtype=np.int64)
        self.n_seen = 0
        self.n = 0

    def add(self, s, a, rng):
        self.n_seen += 1
        if self.n < self.size:
            self.s[self.n], self.a[self.n] = s, a
            self.n += 1
        else:
            j = rng.integers(0, self.n_seen)
            if j < self.size:
                self.s[j], self.a[j] = s, a

    def sample(self, batch, rng):
        j = rng.integers(0, self.n, batch)
        return (
            torch.from_numpy(self.s[j].astype(np.float32)),
            torch.from_numpy(self.a[j]),
        )


def masked_q(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return q + (mask - 1.0) * 1e9


def diff_for(game: PistiGame, seat: int) -> float:
    return float(game.score_diff(seat))


def train(args):
    torch.set_num_threads(args.threads)
    run_dir = os.path.join("runs", args.run_name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)
    observer = Observer()

    nets = NFSPNets(hidden=(256, 256))
    opt_q = torch.optim.Adam(nets.q.parameters(), lr=args.lr_q)
    opt_pi = torch.optim.Adam(nets.pi.parameters(), lr=args.lr_pi)
    rl_buf = ReplayBuffer(args.rl_buffer)
    sl_buf = Reservoir(args.sl_buffer)

    eval_opps = {
        "greedy": GreedyAgent(0),
        "hunter": PistiHunterAgent(0),
        "expectimax": ExpectimaxAgent(n_samples=16, rollout_plies=6, seed=0),
    }
    eval_rows = []

    def new_episode(ep):
        game = PistiGame(deck=new_deck(py_rng), first_player=ep % 2)
        modes = ["br" if rng.random() < args.eta else "avg" for _ in range(2)]
        pending = [None, None]  # (feat, action, diff_at_decision) per seat
        return game, modes, pending

    @torch.no_grad()
    def act(feat: np.ndarray, mask: np.ndarray, mode: str, eps: float) -> int:
        legal = np.flatnonzero(mask)
        x = torch.from_numpy(feat).unsqueeze(0)
        if mode == "br":
            if rng.random() < eps:
                return int(rng.choice(legal))
            q = nets.q(x).squeeze(0).numpy()
            q[mask < 0.5] = -1e9
            return int(np.argmax(q))
        logits = nets.pi(x).squeeze(0).numpy()
        logits[mask < 0.5] = -1e9
        z = logits - logits.max()
        p = np.exp(z)
        p /= p.sum()
        return int(rng.choice(52, p=p))

    def train_step():
        if rl_buf.n >= args.batch:
            s, a, r, s2, done = rl_buf.sample(args.batch, rng)
            with torch.no_grad():
                q2 = masked_q(nets.q_target(s2), s2[:, :52])
                target = r + args.gamma * (1 - done) * q2.max(dim=1).values
            q = nets.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss_q = F.smooth_l1_loss(q, target)
            opt_q.zero_grad()
            loss_q.backward()
            opt_q.step()
        if sl_buf.n >= args.batch:
            s, a = sl_buf.sample(args.batch, rng)
            loss_pi = F.cross_entropy(nets.pi(s), a)
            opt_pi.zero_grad()
            loss_pi.backward()
            opt_pi.step()

    def evaluate(step):
        agent = NFSPAgent(nets, mode="avg", seed=0)
        row = {"timesteps": step}
        for name, opp in eval_opps.items():
            n = args.eval_deals if name != "expectimax" else max(args.eval_deals // 3, 20)
            res = play_match(
                agent, opp, n_deals=n, seed=args.seed + 7777, name_a="nfsp", name_b=name
            )
            s = res.summary()
            row[f"win_{name}"] = s["win_rate_a"]
            row[f"diff_{name}"] = s["mean_diff"]
            row[f"ci_{name}"] = s["diff_ci95"]
        eval_rows.append(row)
        with open(os.path.join(run_dir, "eval.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerows(eval_rows)
        print(
            f"[{step:>9,}] "
            + "  ".join(
                f"{n}: win={row[f'win_{n}']:.2f} diff={row[f'diff_{n}']:+.2f}±{row[f'ci_{n}']:.2f}"
                for n in eval_opps
            ),
            flush=True,
        )

    t0 = time.time()
    step = 0
    episode = 0
    next_eval, next_ckpt = 0, args.ckpt_every
    game, modes, pending = new_episode(0)

    while step < args.steps:
        seat = game.current
        obs = observer.encode(game, seat)
        feat = featurize(obs)

        # close out this seat's previous decision
        if pending[seat] is not None:
            pf, pa, pd = pending[seat]
            r = (diff_for(game, seat) - pd) * args.reward_scale
            rl_buf.add(pf, pa, r, feat, False)

        eps = max(args.eps_final, args.eps_start * (1 - step / args.eps_decay_steps))
        a = act(feat, obs["action_mask"], modes[seat], eps)
        if modes[seat] == "br":
            sl_buf.add(feat, a, rng)
        pending[seat] = (feat, a, diff_for(game, seat))
        game.step(a)
        step += 1

        if game.done:
            for s_ in (0, 1):
                if pending[s_] is not None:
                    pf, pa, pd = pending[s_]
                    r = (diff_for(game, s_) - pd) * args.reward_scale
                    rl_buf.add(pf, pa, r, np.zeros(FEAT_DIM, np.float16), True)
            episode += 1
            game, modes, pending = new_episode(episode)

        if step % args.train_every == 0 and step >= args.learning_starts:
            train_step()
        if step % args.target_every == 0:
            nets.sync_target()
        if step >= next_eval:
            next_eval += args.eval_every
            evaluate(step)
        if step >= next_ckpt:
            next_ckpt += args.ckpt_every
            nets.save(os.path.join(run_dir, "checkpoints", f"ckpt_{step}.pt"))

    nets.save(os.path.join(run_dir, "final.pt"))
    meta = {
        "algorithm": "nfsp",
        "args": vars(args),
        "wall_seconds": round(time.time() - t0, 1),
        "episodes": episode,
        "sl_buffer_seen": sl_buf.n_seen,
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done in {meta['wall_seconds']}s -> {run_dir}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="nfsp_main")
    p.add_argument("--steps", type=int, default=6_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--lr-q", type=float, default=1e-4)
    p.add_argument("--lr-pi", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--reward-scale", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--rl-buffer", type=int, default=200_000)
    p.add_argument("--sl-buffer", type=int, default=500_000)
    p.add_argument("--train-every", type=int, default=16)
    p.add_argument("--target-every", type=int, default=10_000)
    p.add_argument("--learning-starts", type=int, default=20_000)
    p.add_argument("--eps-start", type=float, default=0.12)
    p.add_argument("--eps-final", type=float, default=0.01)
    p.add_argument("--eps-decay-steps", type=int, default=2_000_000)
    p.add_argument("--eval-every", type=int, default=500_000)
    p.add_argument("--eval-deals", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=1_000_000)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
