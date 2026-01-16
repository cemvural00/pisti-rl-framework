# RL Algorithms Documentation

This document describes all reinforcement learning algorithms implemented in the Pişti RL framework, their architectures, use cases, and research background.

## Overview

The framework supports 8 RL algorithms, organized into three tiers:

1. **Tier 1: SB3-Based Methods** (Easy integration, well-tested)
   - PPO (baseline)
   - MaskablePPO
   - RecurrentPPO
   - DQN (baseline)
   - RainbowDQN

2. **Tier 2: Imperfect Information Specialists** (Designed for partial observability)
   - NFSP (Neural Fictitious Self-Play)
   - Deep CFR (Deep Counterfactual Regret Minimization)

3. **Tier 3: Advanced Value-Based** (Recurrent value learning)
   - R2D2 (Recurrent Replay Distributed DQN)

## Network Architectures

### Deep Models (Primary)

All deep models use architectures with sufficient capacity for complex game play:

- **Standard Deep**: `[256, 256, 128]` - 3 hidden layers
- **Recurrent**: LSTM(256, 2 layers) + MLP([128, 128])
- **NFSP**: Average strategy `[256, 256, 128]`, Best response `[256, 256, 128]`
- **Deep CFR**: Counterfactual value `[256, 256, 256, 128]`, Strategy `[256, 256, 128]`

### Benchmark Models (For Comparison)

- **Shallow**: `[64, 64]` - 2 layers, smaller width
- **Medium**: `[128, 128]` - 2 layers, medium width

## Algorithm Details

### 1. PPO (Proximal Policy Optimization)

**Type**: On-policy, Policy gradient  
**Library**: Stable-Baselines3  
**Architecture**: Deep `[256, 256, 128]` or shallow `[64, 64]` (benchmark)

**Use Case**: Baseline comparison, general RL

**Key Features**:
- Clipped objective for stable learning
- GAE (Generalized Advantage Estimation)
- Works well with continuous and discrete actions

**Configuration**:
```yaml
training:
  algorithm: "PPO"
  ppo:
    learning_rate: 3.0e-4
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99
```

### 2. MaskablePPO

**Type**: On-policy, Policy gradient with action masking  
**Library**: sb3-contrib  
**Architecture**: Deep `[256, 256, 128]`

**Use Case**: Proper action masking for discrete action spaces

**Key Features**:
- Built-in action masking support
- Prevents invalid actions from being selected
- Better than regular PPO for games with large action spaces

**Configuration**:
```yaml
training:
  algorithm: "MaskablePPO"
  maskable_ppo:
    learning_rate: 3.0e-4
    # ... same as PPO
```

**Research**: [Action Masking in PPO](https://github.com/DLR-RM/stable-baselines3-contrib)

### 3. RecurrentPPO

**Type**: On-policy, Recurrent policy gradient  
**Library**: sb3-contrib  
**Architecture**: LSTM(256, 2 layers) + MLP([128, 128])

**Use Case**: Partial observability via history

**Key Features**:
- LSTM policy network
- Maintains hidden state across steps
- Better for POMDPs

**Configuration**:
```yaml
training:
  algorithm: "RecurrentPPO"
  encoding:
    encoder_type: "SequenceEncoder"  # Recommended
  recurrent_ppo:
    lstm_hidden_size: 256
    lstm_layers: 2
    mlp_layers: [128, 128]
```

### 4. DQN (Deep Q-Network)

**Type**: Off-policy, Value-based  
**Library**: Stable-Baselines3  
**Architecture**: Deep `[256, 256, 128]` or shallow `[64, 64]` (benchmark)

**Use Case**: Baseline comparison, value-based learning

**Key Features**:
- Experience replay
- Target network
- Epsilon-greedy exploration

**Configuration**:
```yaml
training:
  algorithm: "DQN"
  dqn:
    learning_rate: 1.0e-4
    buffer_size: 100000
    exploration_fraction: 0.1
```

**Research**: Mnih et al. (2015) - "Human-level control through deep reinforcement learning"

### 5. RainbowDQN

**Type**: Off-policy, Enhanced value-based  
**Library**: sb3-contrib  
**Architecture**: Deep `[256, 256, 128]`

**Use Case**: Improved value-based learning

**Key Features**:
- Double DQN
- Dueling architecture
- Prioritized experience replay
- Distributional RL
- Multi-step learning

**Configuration**:
```yaml
training:
  algorithm: "RainbowDQN"
  rainbow_dqn:
    learning_rate: 1.0e-4
    # ... similar to DQN
```

**Research**: Hessel et al. (2018) - "Rainbow: Combining Improvements in Deep Reinforcement Learning"

### 6. NFSP (Neural Fictitious Self-Play)

**Type**: Self-play, Imperfect information  
**Library**: Custom implementation  
**Architecture**: 
- Average strategy: `[256, 256, 128]`
- Best response: `[256, 256, 128]`

**Use Case**: Finding approximate Nash equilibria in imperfect information games

**Key Features**:
- Average strategy network (for Nash equilibrium)
- Best response network
- Reservoir sampling for opponent strategies
- Anticipatory parameter (η)

**How It Works**:
1. Train best response network against opponent strategies
2. Train average strategy network to match best response
3. Use reservoir buffer to maintain diverse opponents
4. Alternating training phases

**Configuration**:
```yaml
training:
  algorithm: "NFSP"
  nfsp:
    anticipatory_param: 0.1  # η
    average_strategy_update_freq: 1000
    reservoir_buffer_size: 10000
    learning_rate: 1.0e-4
```

**Training**:
```bash
python -m training.train_nfsp --config configs/default.yaml
```

**Research**: Heinrich & Silver (2016) - "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games"

### 7. Deep CFR (Deep Counterfactual Regret Minimization)

**Type**: Regret minimization, Imperfect information  
**Library**: Custom implementation  
**Architecture**:
- Counterfactual value: `[256, 256, 256, 128]`
- Strategy: `[256, 256, 128]`

**Use Case**: Theoretical optimality, academic research

**Key Features**:
- Information set representation
- Counterfactual value networks
- Regret matching
- Strategy computation from regrets

**How It Works**:
1. Traverse game tree for each player
2. Compute counterfactual values for information sets
3. Accumulate regrets
4. Compute strategies via regret matching
5. Average strategies converge to Nash equilibrium

**Configuration**:
```yaml
training:
  algorithm: "DeepCFR"
  deep_cfr:
    regret_matching_epsilon: 0.001
    traversal_batch_size: 32
    learning_rate: 1.0e-4
```

**Training**:
```bash
python -m training.train_deep_cfr --config configs/default.yaml
```

**Research**: Brown et al. (2019) - "Deep Counterfactual Regret Minimization"

### 8. R2D2 (Recurrent Replay Distributed DQN)

**Type**: Off-policy, Recurrent value-based  
**Library**: Custom implementation  
**Architecture**: LSTM(256, 2 layers) + MLP([128, 128])

**Use Case**: Value-based learning with partial observability

**Key Features**:
- Recurrent Q-network with LSTM
- Prioritized experience replay
- N-step returns
- Recurrent replay buffer

**Configuration**:
```yaml
training:
  algorithm: "R2D2"
  encoding:
    encoder_type: "SequenceEncoder"  # Recommended
  r2d2:
    n_step: 5
    replay_alpha: 0.6
    replay_beta: 0.4
    learning_rate: 1.0e-4
```

**Training**:
```bash
python -m training.train_r2d2 --config configs/default.yaml
```

**Research**: Kapturowski et al. (2019) - "Recurrent Replay Distributed DQN"

## Benchmark Strategy

### Shallow Models (Benchmarks)
- **Purpose**: Baseline comparison, faster training
- **Architecture**: `[64, 64]` - 2 layers
- **Use**: Compare against deep models to measure architecture impact

### Deep Models (Primary)
- **Purpose**: Sufficient capacity for complex strategies
- **Architecture**: `[256, 256, 128]` or deeper
- **Use**: All new methods and production training

## Comparison Matrix

| Algorithm | Type | Imperfect Info | Recurrent | Action Masking | Complexity |
|-----------|------|----------------|-----------|----------------|------------|
| PPO | On-policy | No | No | Manual | Low |
| MaskablePPO | On-policy | No | No | Yes | Low |
| RecurrentPPO | On-policy | Partial | Yes | Manual | Medium |
| DQN | Off-policy | No | No | Manual | Low |
| RainbowDQN | Off-policy | No | No | Manual | Medium |
| NFSP | Self-play | Yes | No | Yes | High |
| Deep CFR | Regret min | Yes | No | Yes | Very High |
| R2D2 | Off-policy | Partial | Yes | Yes | High |

## Recommendations

### For Best Performance
- **MaskablePPO** or **RecurrentPPO** with deep architecture
- Use **SequenceEncoder** for recurrent methods

### For Imperfect Information
- **NFSP** or **Deep CFR** (designed for this)
- **RecurrentPPO** or **R2D2** (handle via history)

### For Academic Research
- **NFSP** and **Deep CFR** (theoretical foundations)
- Compare all methods for comprehensive study

### For Quick Prototyping
- **PPO** or **DQN** (simplest, fastest)
- Use shallow architecture for quick iterations

## Evaluation Metrics

All methods support:
- Win rate (with confidence intervals)
- Score differential
- Pişti frequency
- Capture efficiency
- Game length statistics

Imperfect information methods (NFSP, Deep CFR) additionally support:
- Exploitability (measure of Nash equilibrium convergence)
- Strategy convergence metrics

## References

1. **PPO**: Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
2. **MaskablePPO**: Huang & Ontañón (2022) - "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms"
3. **DQN**: Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
4. **Rainbow**: Hessel et al. (2018) - "Rainbow: Combining Improvements in Deep Reinforcement Learning"
5. **NFSP**: Heinrich & Silver (2016) - "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games"
6. **Deep CFR**: Brown et al. (2019) - "Deep Counterfactual Regret Minimization"
7. **R2D2**: Kapturowski et al. (2019) - "Recurrent Replay Distributed DQN"
