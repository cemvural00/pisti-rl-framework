# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL + game-theory study of the Turkish card game Pişti (2-player, zero-sum, imperfect information). The codebase was rebuilt in June 2026; anything in git history before commit `01c8903` is the old, non-functional iteration — don't resurrect patterns from it. A virtualenv lives at `venv/` (`venv/bin/python`).

## Commands

```bash
venv/bin/python -m pytest                # full test suite (fast, <1s)
venv/bin/python -m pytest tests/test_game.py -q   # engine rules only

# Training (config-driven MaskablePPO + curriculum + self-play league)
venv/bin/python -m training.train --config configs/default.yaml \
    --set run_name=myrun seed=1 total_timesteps=1_000_000

# Mirrored-deal round-robin tournament + Bradley-Terry ratings
venv/bin/python -m training.evaluate --agents greedy hunter \
    "ppo:runs/ppo_main/final_model" --n-deals 300 --out results/t.json

# Approximate exploitability (train a best response vs a frozen target)
venv/bin/python -m training.exploitability --target ppo-stoch:runs/ppo_main/final_model.zip

# Play against an agent in the terminal
venv/bin/python scripts/play.py --agent ppo:runs/ppo_main/final_model

black .   # line-length 100
```

Packages are flat top-level modules (`engine`, `encoding`, `envs`, `agents`, `training`, `analysis`); run everything from the repo root.

## Architecture

Dependency direction: `engine` → `encoding` → `envs`/`agents` → `training`/`analysis`.

- **`engine/game.py`** — the single source of game truth. Cards are ints 0-51 (`suit*13 + rank`; J=rank 9, A=12, 2♣=39, 10♦=34). `PistiGame` is mutable and fast (~1M moves/s); search agents use `clone()`. `determinize(player, rng)` resamples all info hidden from a player (opponent hand, stock, hidden center, captured-hidden cards, with point adjustment) — this is what keeps search agents honest; never let an agent read hidden fields directly.
- **`encoding/obs.py`** — `Observer` produces the Dict observation (hand/table_top/seen multi-hots + stats vector + action_mask). `Observer(memory=False)` zeroes the `seen` vector — the card-counting ablation.
- **`envs/pisti_env.py`** — Gymnasium env; agent is player 0, opponent plays inside `step()`. **Reward invariant: episode return always equals `reward_scale ×` final score differential** (both "delta" and "sparse" modes; tested in `tests/test_env_and_match.py`). If you touch reward logic, keep the telescoping test green — the old codebase died by silently losing terminal rewards.
- **`agents/`** — everything speaks `predict(obs, action_mask) -> int`. Agents with `wants_game = True` (expectimax, MixtureOpponent) additionally receive `game=`/`player=` kwargs and may only use information-set-legal data via `determinize`. `frozen.py` holds the self-play machinery: `League` (snapshot pool + mixture weights, shared across DummyVecEnv envs in-process) and `MixtureOpponent` (re-samples opponent type each episode).
- **`training/match.py`** — mirrored ("duplicate") evaluation: each deck played twice with seats swapped (`PistiGame(first_player=0|1)` swaps hands too). All skill claims use `MatchResult.diff_ci95()`, which computes CIs over mirror-paired deals, not raw games.
- **`training/train.py`** — curriculum phases are timestep-keyed opponent mixtures in `configs/default.yaml`. Outputs to `runs/<run_name>/` (config.yaml, eval.csv, checkpoints/, final_model.zip, metadata.json with git hash).

## Conventions

- Statistical claims need mirrored deals + CIs; ad-hoc win counts over unpaired games are not acceptable for results.
- Exploitability targets must be the *stochastic* policy (`ppo-stoch:` spec), not deterministic argmax.
- Results JSONs go to `results/`, figures to `plots/`, training runs to `runs/` (gitignored except config/eval.csv).
