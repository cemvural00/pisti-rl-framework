# Pişti RL

*Work in Progress*

A modular reinforcement learning framework for learning (near-)optimal play in the Turkish card game **Pişti** (Pishti).

## Overview

This project provides a complete RL environment for Pişti with clean separation between:
- **Game engine**: Pure, testable game rules and state management
- **Observation encoders**: Modular encoding system with multi-hot vectors (never raw integer IDs)
- **Environment wrappers**: Both PettingZoo (multi-agent) and Gymnasium (single-agent) interfaces
- **Baseline agents**: Random, greedy, and heuristic opponents
- **Training pipelines**: Support for 8 RL algorithms (PPO, MaskablePPO, RecurrentPPO, DQN, RainbowDQN, NFSP, Deep CFR, R2D2) with self-play and league training
- **Evaluation tools**: Comprehensive metrics and opponent evaluation

## Features

- **Modular architecture**: Easy to extend for 4-player partnerships, new encoders, new algorithms
- **Action masking**: Proper handling of invalid actions via action masks
- **Partial observability**: POMDP formulation with hidden opponent hands and stock order
- **Self-play**: Opponent pool mechanism for league training
- **Multiple encoders**: Multi-hot, CNN-friendly reshaped views, feature, and sequence encoders
- **Configurable**: YAML-based configuration for all game rules, rewards, and training parameters

## Installation

```bash
# Clone the repository
git clone <https://github.com/cemvural00/pisti-rl-framework>
cd pisti-rl

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Model Storage

Models are organized in a structured `models/` directory:

```
models/
├── {algorithm}/
│   ├── checkpoints/    # Intermediate checkpoints
│   ├── final/          # Final trained models
│   └── snapshots/       # Snapshots for self-play
```

See [MODEL_STORAGE.md](MODEL_STORAGE.md) for details.

## Quick Start

### Training

Train an agent using the default configuration:

```bash
# SB3-based algorithms (PPO, MaskablePPO, RecurrentPPO, DQN, RainbowDQN)
python -m training.train_sb3 --config configs/default.yaml

# NFSP
python -m training.train_nfsp --config configs/default.yaml

# Deep CFR
python -m training.train_deep_cfr --config configs/default.yaml

# R2D2
python -m training.train_r2d2 --config configs/default.yaml
```

Or use the command-line script:

```bash
pisti-train --config configs/default.yaml
```

#### Training Strategies

The framework supports different training strategies via configuration:

**Training Against Probabilistic Agent (Recommended for Initial Training):**
```yaml
training:
  opponent:
    type: "probabilistic"  # Strong baseline opponent
    switch_to_self_play_at: 500000  # Switch to self-play after N timesteps
    probabilistic_config:
      max_samples: 50
      depth: 1
```

**Other Opponent Options:**
- `"random"` - Random valid moves
- `"greedy"` - Greedy capture strategy
- `"pisti_hunter"` - Heuristic pişti-focused strategy
- `"self_play"` - Train against past checkpoints (requires self-play enabled)

**Training Flow:**
1. Start training against probabilistic agent (strong baseline)
2. After `switch_to_self_play_at` timesteps, automatically switch to self-play
3. Self-play uses opponent pool of past checkpoints for diverse training

### Evaluation

**Simple Evaluation:**
```bash
python -m training.eval --checkpoint ./checkpoints/pisti_model_final --opponents random,greedy
```

**Comprehensive Evaluation (with statistical analysis):**
```bash
python -m training.evaluate_comprehensive \
    --checkpoint models/ppo/final/pisti_model_final \
    --opponents random,greedy,pisti_hunter,probabilistic \
    --n-episodes 1000 \
    --n-seeds 10 \
    --output-dir results/experiment_1 \
    --cleanup-old 5  # Keep only 5 most recent results
```

**Generate Academic Report:**
```bash
python -m training.generate_report \
    --results-dir results/experiment_1 \
    --checkpoint models/ppo/final/pisti_model_final \
    --format markdown,latex,csv
```

### Results Cleanup

**Manual Cleanup:**
```bash
# Delete old results, keeping only 5 most recent
python -m training.cleanup_results --keep-recent 5

# Delete results matching a pattern
python -m training.cleanup_results --pattern "eval_2024*"

# Dry run (see what would be deleted)
python -m training.cleanup_results --keep-recent 5 --dry-run
```

**Auto-Cleanup During Evaluation:**
Use the `--cleanup-old N` flag in `evaluate_comprehensive` to automatically clean up old results before running new evaluation.

## Project Structure

```
pisti_rl/
├── engine/           # Core game logic (pure, testable)
│   ├── cards.py      # Card representation and deck
│   ├── rules.py      # Capture logic, pişti detection, scoring
│   ├── state.py      # GameState with immutable transitions
│   └── rewards.py    # Reward functions (sparse and shaped)
├── envs/             # Environment wrappers
│   ├── base.py       # Shared game engine
│   ├── pisti_pettingzoo.py  # PettingZoo AEC environment
│   └── pisti_gym.py  # Gymnasium wrapper (single-agent)
├── encoding/         # Observation encoding
│   ├── obs_builder.py    # Observation builder
│   └── encoders.py       # Modular encoder interface
├── agents/           # Baseline policies and opponents
│   ├── baselines.py      # Random, greedy, heuristic agents
│   └── opponents.py      # Opponent pool, frozen checkpoints
├── training/         # Training and evaluation
│   ├── train_sb3.py              # Main training script
│   ├── eval.py                    # Simple evaluation script
│   ├── evaluate_comprehensive.py  # Comprehensive evaluation with statistics
│   ├── generate_report.py         # Academic report generator
│   ├── metadata.py                # Model metadata management
│   ├── results.py                 # Results export, visualization, analysis
│   └── callbacks.py               # SB3 callbacks
├── configs/          # YAML configuration files
│   └── default.yaml      # Default configuration
└── tests/            # Unit and integration tests
```

## Configuration

All settings are configured via YAML files. See `configs/default.yaml` for options:

- **Game rules**: Pişti exceptions, expose bottom card, etc.
- **Rewards**: Sparse vs. shaped rewards, bonus weights
- **Encoding**: Encoder type, history length, CNN views
- **Training**: Algorithm (PPO/DQN), hyperparameters, self-play settings
- **Evaluation**: Evaluation frequency, opponents, metrics

## Architecture

### Card Representation

- **Action space**: Discrete(52) using 0-51 card IDs
- **Mapping**: `card_id = suit_id * 13 + rank_id` (recoverable via `divmod`)
- **Observations**: Multi-hot vectors (52-length), **never raw integer IDs**

### Observation Encoding

The framework provides multiple encoder types:

- **MultiHotEncoder** (default): 52-length multi-hot vectors for hands/piles/seen cards
- **CNNEncoder**: Adds (4,13) reshaped tensor views for CNN experiments
- **FeatureEncoder**: Flattens to single vector for MLP policies
- **SequenceEncoder**: Adds move history sequence for RNN/LSTM policies

All encoders implement the `ObservationEncoder` interface for easy swapping.

### Environments

Two environment interfaces are provided:

1. **PettingZoo AEC**: Multi-agent environment for self-play and multi-agent RL
2. **Gymnasium**: Single-agent wrapper with pluggable opponent for SB3 training

Both use the same underlying game engine and support action masking.

### Baseline Agents

- **RandomValidAgent**: Plays random legal cards
- **GreedyCaptureAgent**: Captures if possible, else plays low-value cards
- **PistiHunterAgent**: Heuristic for setting up pişti opportunities

### Self-Play and League Training

The framework supports self-play via:
- **OpponentPool**: Maintains a pool of past checkpoints
- **FrozenCheckpointOpponent**: Loads saved models as opponents
- **SelfPlayOpponent**: Uses current training policy

## Game Rules (Summary)

Pişti is a Turkish card game with the following key rules:

- **Deal**: 4 cards to table center (3 face-down, 1 face-up), 4 cards to each player
- **Capture**: Match rank of top card OR play a Jack (captures any card)
- **Pişti**: Bonus points for capturing a single-card pile by rank match (10 points) or double pişti with Jacks (20 points)
- **Scoring**: Aces (+1), Jacks (+1), 2♣ (+2), 10♦ (+3), majority bonus (+3), pişti bonuses
- **Partial observability**: Opponent hand and stock order are hidden

See the code comments for complete rule specifications.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=engine --cov=envs --cov=encoding --cov=agents

# Run specific test file
pytest tests/test_rules.py

# Quick lightweight tests for probabilistic agent
pytest tests/test_probabilistic_quick.py -v

# Comprehensive project test (tests all components)
pytest tests/test_full_project.py -v
# Or use the standalone script:
python scripts/test_full_project.py

# Minimal check (just verify agent works)
python scripts/minimal_check.py
```

See [TESTING.md](TESTING.md) for detailed testing guide.

## Example Usage

### Basic Training

```python
from envs.pisti_gym import PistiGymEnv
from agents.baselines import RandomValidAgent
from stable_baselines3 import PPO

# Create environment
env = PistiGymEnv(opponent=RandomValidAgent(), seed=42)

# Train agent
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save model
model.save("pisti_model")
```

### Custom Encoder

```python
from encoding.encoders import CNNEncoder
from envs.pisti_gym import PistiGymEnv

# Use CNN encoder with reshaped views
encoder = CNNEncoder()
env = PistiGymEnv(encoder=encoder, seed=42)
```

### Self-Play Training

```python
from agents.opponents import OpponentPool
from training.train_sb3 import train

# Train with self-play (configured in YAML)
train("configs/default.yaml")
```

## Model Metadata & Reproducibility

Each checkpoint automatically saves metadata including:
- Full training configuration (YAML)
- Hyperparameters
- Training statistics (timesteps, best scores)
- System information (Python version, package versions)
- Git commit hash (for reproducibility)
- Model architecture details

Metadata is saved as JSON alongside each checkpoint: `{checkpoint_name}_metadata.json`

## Academic Evaluation & Reporting

The framework includes comprehensive evaluation tools for academic research:

### Comprehensive Evaluation
- **Statistical Analysis**: Mean, standard deviation, 95% confidence intervals
- **Multiple Seeds**: Robust evaluation across multiple random seeds
- **Multiple Metrics**: Win rate, score differential, pişti frequency, capture efficiency
- **Statistical Tests**: t-test and Mann-Whitney U test for significance

### Report Generation
- **Multiple Formats**: Markdown, LaTeX, HTML, CSV
- **Visualizations**: Win rates, score distributions, performance comparisons
- **Reproducibility Section**: Full training config, hyperparameters, system info
- **Statistical Tables**: Publication-ready tables with confidence intervals

### Usage Example
```bash
# Run comprehensive evaluation
python -m training.evaluate_comprehensive \
    --checkpoint checkpoints/pisti_model_final \
    --n-episodes 1000 \
    --n-seeds 10 \
    --output-dir results/experiment_1

# Generate academic report
python -m training.generate_report \
    --results-dir results/experiment_1 \
    --checkpoint checkpoints/pisti_model_final \
    --format markdown,latex
```

## Model Metadata & Reproducibility

Each checkpoint automatically saves metadata including:
- Full training configuration (YAML)
- Hyperparameters
- Training statistics (timesteps, best scores)
- System information (Python version, package versions)
- Git commit hash (for reproducibility)
- Model architecture details

Metadata is saved as JSON alongside each checkpoint: `{checkpoint_name}_metadata.json`

## Academic Evaluation & Reporting

The framework includes comprehensive evaluation tools for academic research:

### Comprehensive Evaluation
- **Statistical Analysis**: Mean, standard deviation, 95% confidence intervals
- **Multiple Seeds**: Robust evaluation across multiple random seeds
- **Multiple Metrics**: Win rate, score differential, pişti frequency, capture efficiency
- **Statistical Tests**: t-test and Mann-Whitney U test for significance

### Report Generation
- **Multiple Formats**: Markdown, LaTeX, HTML, CSV
- **Visualizations**: Win rates, score distributions, performance comparisons
- **Reproducibility Section**: Full training config, hyperparameters, system info
- **Statistical Tables**: Publication-ready tables with confidence intervals

### Usage Example
```bash
# Run comprehensive evaluation
python -m training.evaluate_comprehensive \
    --checkpoint checkpoints/pisti_model_final \
    --n-episodes 1000 \
    --n-seeds 10 \
    --output-dir results/experiment_1

# Generate academic report
python -m training.generate_report \
    --results-dir results/experiment_1 \
    --checkpoint checkpoints/pisti_model_final \
    --format markdown,latex
```

## Supported RL Algorithms

The framework supports 8 RL algorithms:

- **PPO** (Proximal Policy Optimization) - Baseline on-policy
- **MaskablePPO** - PPO with proper action masking
- **RecurrentPPO** - PPO with LSTM for partial observability
- **DQN** (Deep Q-Network) - Baseline off-policy
- **RainbowDQN** - Enhanced DQN with multiple improvements
- **NFSP** (Neural Fictitious Self-Play) - For imperfect information games
- **Deep CFR** (Deep Counterfactual Regret Minimization) - Theoretical optimality
- **R2D2** (Recurrent Replay Distributed DQN) - Recurrent value-based

See [ALGORITHMS.md](ALGORITHMS.md) for detailed documentation on each algorithm, network architectures, use cases, and research background.

## TODO: Future Extensions

The codebase is designed to be easily extended. Marked areas for future work:

- [ ] **4-player partnership mode**: Extend `GameState` to support 4 players with partners opposite
- [ ] **Bluffing variant**: Modify rules to add bluff action and detection
- [ ] **NFSP/DeepCFR integration**: Add training scripts for approximate Nash equilibrium methods
- [ ] **Richer belief modeling**: Implement opponent hand inference and belief state tracking
- [ ] **Transformer encoders**: Add attention-based sequence encoders
- [ ] **True MaskablePPO**: Integrate `sb3-contrib` for proper action masking in PPO
- [ ] **Multi-GPU training**: Support for distributed training
- [ ] **Tournament evaluation**: Automated tournament system for agent evaluation

## Contributing

Contributions are welcome! Please ensure:
- Code follows the existing style and structure
- Tests are added for new features
- Documentation is updated
- Type hints are used

## License

All rights reserved

## Acknowledgments

This project implements the Turkish card game Pişti (Pishti) for reinforcement learning research. The game rules are based on the standard Turkish variant.
