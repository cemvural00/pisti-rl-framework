"""Tests for the Gym env reward accounting and the match driver."""

import numpy as np

from agents.baselines import GreedyAgent, RandomAgent
from encoding.obs import Observer
from envs.pisti_env import PistiEnv
from training.match import play_match


def _run_episode(env, rng):
    obs, _ = env.reset()
    total, done, info = 0.0, False, {}
    while not done:
        legal = np.flatnonzero(env.action_masks())
        obs, r, done, _, info = env.step(rng.choice(legal))
        total += r
    return total, info


def test_delta_reward_telescopes_to_final_diff():
    env = PistiEnv(opponent=GreedyAgent(0), seed=1, reward_mode="delta")
    rng = np.random.default_rng(0)
    for _ in range(100):
        total, info = _run_episode(env, rng)
        assert abs(total - info["score_diff"]) < 1e-9


def test_sparse_reward_equals_final_diff():
    env = PistiEnv(opponent=GreedyAgent(0), seed=2, reward_mode="sparse")
    rng = np.random.default_rng(0)
    for _ in range(50):
        total, info = _run_episode(env, rng)
        assert abs(total - info["score_diff"]) < 1e-9


def test_env_alternates_seats():
    env = PistiEnv(opponent=RandomAgent(0), seed=3)
    rng = np.random.default_rng(0)
    leads = []
    for _ in range(6):
        env.reset()
        leads.append(env.game.first_player == 0)
        _run = _run_episode  # keep episodes short-circuit free
    assert True in leads and False in leads


def test_memory_ablation_zeroes_seen():
    env = PistiEnv(opponent=RandomAgent(0), observer=Observer(memory=False), seed=4)
    obs, _ = env.reset()
    rng = np.random.default_rng(0)
    for _ in range(10):
        legal = np.flatnonzero(env.action_masks())
        obs, r, done, _, _ = env.step(rng.choice(legal))
        assert obs["seen"].sum() == 0
        if done:
            break


def test_mirrored_match_pairs_deals():
    res = play_match(
        GreedyAgent(0), RandomAgent(0), n_deals=20, seed=5,
        name_a="g", name_b="r",
    )
    assert res.n_games == 40
    # each deal appears exactly twice with both seatings
    by_deal = {}
    for r in res.records:
        by_deal.setdefault(r.deal, []).append(r.a_leads)
    assert all(sorted(v) == [False, True] for v in by_deal.values())
    # greedy should clearly beat random
    assert res.win_rate_a > 0.6


def test_mirrored_variance_reduction():
    """Paired (mirrored) CI should not exceed the naive unpaired CI."""
    res = play_match(GreedyAgent(0), RandomAgent(0), n_deals=200, seed=6)
    diffs = np.array([r.diff for r in res.records])
    naive_se = diffs.std(ddof=1) / np.sqrt(len(diffs))
    _, ci = res.diff_ci95()
    assert ci <= 1.96 * naive_se * 1.10  # allow slack for randomness
