"""Train a MaskablePPO Pişti agent with curriculum + self-play league.

Usage:
    python -m training.train --config configs/default.yaml
    python -m training.train --config configs/default.yaml --set seed=1 run_name=ppo_s1

Outputs land in runs/<run_name>/:
    config.yaml      resolved config (reproducibility)
    metadata.json    git hash, package versions, timing
    eval.csv         periodic mirrored-eval results
    checkpoints/     periodic model checkpoints
    final_model.zip  the trained model
    best_model.zip   best checkpoint by eval mean diff vs greedy
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import time
from typing import Dict

import numpy as np
import yaml
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.baselines import GreedyAgent, PistiHunterAgent, RandomAgent
from agents.expectimax import ExpectimaxAgent
from agents.frozen import FrozenPolicyAgent, League, MixtureOpponent
from encoding.obs import Observer
from envs.pisti_env import PistiEnv
from training.match import play_match


def load_config(path: str, overrides=None) -> Dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for kv in overrides or []:
        key, value = kv.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def make_vec_env(cfg: Dict, league: League) -> DummyVecEnv:
    n_envs = cfg["n_envs"]
    seed = cfg["seed"]
    obs_cfg = cfg.get("observer", {})

    def make(i):
        def _init():
            return PistiEnv(
                opponent=MixtureOpponent(league, seed=seed * 1000 + i),
                observer=Observer(memory=obs_cfg.get("memory", True)),
                reward_mode=cfg["reward"]["mode"],
                reward_scale=cfg["reward"]["scale"],
                seed=seed * 1000 + i,
            )

        return _init

    return DummyVecEnv([make(i) for i in range(n_envs)])


class LeagueCallback(BaseCallback):
    """Curriculum phase switching + snapshotting + periodic mirrored eval."""

    def __init__(self, cfg: Dict, league: League, run_dir: str):
        super().__init__()
        self.cfg = cfg
        self.league = league
        self.run_dir = run_dir
        self.phases = sorted(cfg["curriculum"], key=lambda p: p["at"])
        self.phase_idx = -1
        self.snapshot_freq = cfg["selfplay"]["snapshot_freq"]
        self.next_snapshot = self.snapshot_freq
        self.eval_freq = cfg["eval"]["freq"]
        self.next_eval = 0
        self.eval_deals = cfg["eval"]["n_deals"]
        self.eval_seed = cfg["seed"] + 7777
        self.best_diff = -1e9
        self.eval_rows = []
        self._eval_opponents = {
            "greedy": GreedyAgent(0),
            "hunter": PistiHunterAgent(0),
            "expectimax": ExpectimaxAgent(n_samples=16, rollout_plies=6, seed=0),
        }
        self.observer = Observer(memory=cfg.get("observer", {}).get("memory", True))

    def _advance_phase(self):
        t = self.num_timesteps
        idx = self.phase_idx
        while idx + 1 < len(self.phases) and t >= self.phases[idx + 1]["at"]:
            idx += 1
        if idx != self.phase_idx:
            self.phase_idx = idx
            weights = self.phases[idx]["weights"]
            self.league.set_weights(weights)
            print(f"[{t:>9,}] curriculum phase {idx}: {weights}")

    def _snapshot(self):
        self.league.add_snapshot(self.model, self.num_timesteps)
        path = os.path.join(
            self.run_dir, "checkpoints", f"ckpt_{self.num_timesteps}"
        )
        self.model.save(path)

    def _evaluate(self):
        t = self.num_timesteps
        agent = FrozenPolicyAgent(self.model.policy, deterministic=True, name="current")
        row = {"timesteps": t}
        for name, opp in self._eval_opponents.items():
            n = self.eval_deals if name != "expectimax" else max(self.eval_deals // 3, 20)
            res = play_match(
                agent, opp, n_deals=n, seed=self.eval_seed,
                name_a="agent", name_b=name, observer_a=self.observer,
            )
            s = res.summary()
            row[f"win_{name}"] = s["win_rate_a"]
            row[f"diff_{name}"] = s["mean_diff"]
            row[f"ci_{name}"] = s["diff_ci95"]
            self.logger.record(f"eval/win_{name}", s["win_rate_a"])
            self.logger.record(f"eval/diff_{name}", s["mean_diff"])
        self.eval_rows.append(row)
        self._write_eval_csv()
        print(
            f"[{t:>9,}] eval: "
            + "  ".join(
                f"{n}: win={row[f'win_{n}']:.2f} diff={row[f'diff_{n}']:+.2f}±{row[f'ci_{n}']:.2f}"
                for n in self._eval_opponents
            )
        )
        if row["diff_greedy"] > self.best_diff:
            self.best_diff = row["diff_greedy"]
            self.model.save(os.path.join(self.run_dir, "best_model"))

    def _write_eval_csv(self):
        if not self.eval_rows:
            return
        path = os.path.join(self.run_dir, "eval.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.eval_rows[-1].keys()))
            writer.writeheader()
            writer.writerows(self.eval_rows)

    def _on_step(self) -> bool:
        self._advance_phase()
        if self.num_timesteps >= self.next_snapshot:
            self.next_snapshot += self.snapshot_freq
            self._snapshot()
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.eval_freq
            self._evaluate()
        return True


def train(cfg: Dict) -> str:
    run_dir = os.path.join(cfg["out_dir"], cfg["run_name"])
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    league = League(pool_size=cfg["selfplay"]["pool_size"], seed=cfg["seed"])
    league.set_weights(cfg["curriculum"][0]["weights"])
    env = make_vec_env(cfg, league)

    algo = cfg.get("algorithm", "maskable_ppo")
    if algo == "maskable_ppo":
        ppo = cfg["ppo"]
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=ppo["learning_rate"],
            n_steps=ppo["n_steps"],
            batch_size=ppo["batch_size"],
            n_epochs=ppo["n_epochs"],
            gamma=ppo["gamma"],
            gae_lambda=ppo["gae_lambda"],
            clip_range=ppo["clip_range"],
            ent_coef=ppo["ent_coef"],
            vf_coef=ppo["vf_coef"],
            max_grad_norm=ppo["max_grad_norm"],
            policy_kwargs={"net_arch": ppo["net_arch"]},
            seed=cfg["seed"],
            verbose=0,
            tensorboard_log=os.path.join(run_dir, "tb"),
            device="cpu",
        )
    elif algo == "dqn":
        from agents.masked_dqn import MaskedDQN, MaskedDQNPolicy

        dqn = cfg["dqn"]
        model = MaskedDQN(
            MaskedDQNPolicy,
            env,
            learning_rate=dqn["learning_rate"],
            buffer_size=dqn["buffer_size"],
            learning_starts=dqn["learning_starts"],
            batch_size=dqn["batch_size"],
            gamma=dqn["gamma"],
            train_freq=dqn["train_freq"],
            gradient_steps=dqn["gradient_steps"],
            target_update_interval=dqn["target_update_interval"],
            exploration_fraction=dqn["exploration_fraction"],
            exploration_final_eps=dqn["exploration_final_eps"],
            policy_kwargs={"net_arch": dqn["net_arch"]},
            seed=cfg["seed"],
            verbose=0,
            tensorboard_log=os.path.join(run_dir, "tb"),
            device="cpu",
        )
    else:
        raise ValueError(f"unknown algorithm: {algo}")
    league.live_model = model

    meta = {
        "git_hash": git_hash(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "config": cfg,
    }

    cb = LeagueCallback(cfg, league, run_dir)
    t0 = time.time()
    model.learn(total_timesteps=cfg["total_timesteps"], callback=cb)
    meta["wall_seconds"] = round(time.time() - t0, 1)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

    model.save(os.path.join(run_dir, "final_model"))
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"done in {meta['wall_seconds']}s -> {run_dir}")
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--set", nargs="*", default=[], help="config overrides: key=value"
    )
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    train(cfg)


if __name__ == "__main__":
    main()
