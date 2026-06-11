# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL framework for the Turkish card game Pişti (2-player, imperfect information). Python 3.8+, built on Gymnasium/PettingZoo/Stable-Baselines3. A virtualenv exists at `venv/` (`venv/bin/python`).

## Commands

```bash
# Install (editable, with dev tools)
pip install -e ".[dev]"

# Tests
pytest                                  # all tests
pytest tests/test_rules.py              # single file
pytest tests/test_probabilistic_quick.py -v   # fast probabilistic-agent checks
pytest tests/test_full_project.py -v    # integration test across all components
pytest --cov=engine --cov=envs --cov=encoding --cov=agents

# Sanity scripts (standalone, no pytest)
python scripts/minimal_check.py
python scripts/test_full_project.py

# Training (config-driven; algorithm chosen in YAML)
python -m training.train_sb3 --config configs/default.yaml      # PPO/MaskablePPO/RecurrentPPO/DQN/RainbowDQN
python -m training.train_nfsp --config configs/default.yaml
python -m training.train_deep_cfr --config configs/default.yaml
python -m training.train_r2d2 --config configs/default.yaml

# Evaluation
python -m training.eval --checkpoint <path> --opponents random,greedy
python -m training.evaluate_comprehensive --checkpoint <path> --opponents random,greedy,pisti_hunter,probabilistic --n-episodes 1000 --n-seeds 10 --output-dir results/<name>
python -m training.generate_report --results-dir results/<name> --checkpoint <path> --format markdown,latex,csv

# Formatting
black .   # line-length 100 (configured in pyproject.toml)
```

Important: packages are flat top-level modules (`engine`, `envs`, `encoding`, `agents`, `training`) — imports are `from engine.cards import ...`, not `from pisti_rl...`. Run everything from the repo root.

## Architecture

Layered design with strict dependency direction: `engine` → `encoding` → `envs` → `agents`/`training`.

- **`engine/`** — pure game logic, no RL dependencies. `state.py` defines `GameState` with conceptually immutable transitions (`apply_action()` returns a new state). `rules.py` has capture/pişti/scoring logic, `rewards.py` has `sparse_reward` (terminal score differential) and `shaped_reward` (per-step bonuses, weights from YAML).
- **`envs/base.py`** — `PistiGameEngine`, the single shared engine that both environment wrappers delegate to. It owns the `GameState`, the encoder, and reward computation.
- **`envs/pisti_gym.py`** — Gymnasium single-agent env: the learning agent is player 0; the opponent is a pluggable policy object stepped internally. **`envs/pisti_pettingzoo.py`** — PettingZoo AEC env for multi-agent/self-play.
- **`encoding/`** — `ObservationEncoder` ABC with `MultiHotEncoder` (default), `CNNEncoder`, `FeatureEncoder`, `SequenceEncoder`. Observations are Dict spaces of 52-length multi-hot vectors (`hand`, `table_top`, `seen_cards`, `action_mask`, counts) — never raw integer card IDs.
- **`agents/`** — everything implements the duck-typed opponent protocol `predict(obs: Dict, action_mask: np.ndarray) -> int`. Includes baselines (`RandomValidAgent`, `GreedyCaptureAgent`, `PistiHunterAgent`), `probabilistic_agent.py` (belief tracking + expectimax sampling, tunable via `max_samples`/`depth`/`temperature`), and `opponents.py` (self-play: `OpponentPool`, `FrozenCheckpointOpponent`).
- **`training/train_sb3.py`** — main entry point. Reads the YAML config and dispatches by `training.algorithm`. Implements curriculum learning (`training.curriculum.phases` switches opponent type at timestep thresholds, ending in self-play) and the opponent pool of frozen checkpoints. `sb3_contrib` imports (MaskablePPO, RecurrentPPO, Rainbow) are guarded try/except — algorithms degrade gracefully if not installed.

### Key conventions

- **Action space**: `Discrete(52)`; `card_id = suit_id * 13 + rank_id` (recover with `divmod`).
- **Action masking**: every observation includes `action_mask`; agents must only pick masked-legal actions.
- **Configuration**: all game rules, reward shaping weights, network architectures, curriculum phases, and hyperparameters live in `configs/default.yaml` — prefer adding config options over hardcoding.
- **Model storage**: `models/{algorithm}/{checkpoints,final,snapshots}/`, managed by `training/model_storage.py`. Each checkpoint gets a `{name}_metadata.json` (config, hyperparameters, git hash) via `training/metadata.py`.

## Reference docs

`ALGORITHMS.md` (per-algorithm details), `EVALUATION_GUIDE.md`, `MODEL_STORAGE.md`, `TESTING.md`, `MANUAL.md` (full game rules).
