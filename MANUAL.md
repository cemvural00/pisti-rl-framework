# Pişti RL Framework - Technical Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Game Engine](#game-engine)
3. [Environments](#environments)
4. [Observation Encoding](#observation-encoding)
5. [Agents](#agents)
6. [Training System](#training-system)
7. [Evaluation System](#evaluation-system)
8. [Configuration System](#configuration-system)
9. [Data Flow Diagrams](#data-flow-diagrams)
10. [Code Examples](#code-examples)

---

## Introduction

### Framework Overview

The Pişti RL framework is a modular reinforcement learning system designed to learn optimal play in the Turkish card game Pişti. The framework is built with clean separation of concerns, making it easy to extend, test, and experiment with different RL algorithms, observation encodings, and training strategies.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  (Training Scripts, Evaluation, Report Generation)          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Agent Layer                               │
│  (Baseline Agents, RL Agents, Opponents)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  Environment Layer                           │
│  (Gymnasium, PettingZoo Wrappers)                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Observation Encoding Layer                      │
│  (MultiHotEncoder, CNNEncoder, SequenceEncoder, etc.)       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Game Engine Layer                        │
│  (GameState, Rules, Cards, Rewards)                         │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Modularity**: Each component is independent and can be swapped
2. **Immutability**: Game states are immutable; actions return new states
3. **Type Safety**: Extensive use of type hints and dataclasses
4. **Testability**: Pure functions and clear interfaces enable easy testing
5. **Extensibility**: Abstract base classes allow easy addition of new components
6. **Reproducibility**: Deterministic seeding and metadata tracking

### Core Concepts

- **Card ID System**: Cards are represented as integers 0-51, never as raw IDs in neural network inputs
- **Action Masking**: Invalid actions are masked rather than penalized
- **Partial Observability**: Opponent hands and stock order are hidden
- **Multi-Hot Encoding**: Observations use 52-length binary vectors, never raw integers

---

## Game Engine

The game engine (`engine/`) contains the core game logic, completely independent of RL frameworks. This ensures the game rules are pure, testable, and reusable.

### Cards (`engine/cards.py`)

#### Card Representation

```python
@dataclass(frozen=True)
class Card:
    rank: str  # 'A', '2', '3', ..., 'K'
    suit: str  # 'S', 'H', 'D', 'C'
```

Cards are immutable dataclasses, ensuring they cannot be accidentally modified.

#### Card ID Mapping

The framework uses a canonical 0-51 ID system for actions:

```python
def card_to_id(card: Card) -> int:
    """
    Maps Card to 0-51 ID.
    Formula: card_id = suit_id * 13 + rank_id
    
    - suit_id: 0=S, 1=H, 2=D, 3=C
    - rank_id: 0=2, 1=3, ..., 9=10, 10=J, 11=Q, 12=K, 13=A
    """
    rank_id = RANKS.index(card.rank)  # 0-12
    suit_id = SUITS.index(card.suit)  # 0-3
    return suit_id * 13 + rank_id

def id_to_card(card_id: int) -> Card:
    """Inverse mapping: recoverable via divmod(card_id, 13)"""
    suit_id, rank_id = divmod(card_id, 13)
    return Card(rank=RANKS[rank_id], suit=SUITS[suit_id])
```

**Important**: While actions use 0-51 IDs, observations **never** use raw integer IDs. They use multi-hot vectors (52-length binary arrays).

#### Deck Management

```python
class Deck:
    """Standard 52-card deck with shuffling."""
    
    def __init__(self, seed: int = None):
        # Creates all 52 cards and shuffles
        self.cards: List[Card] = [...]
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.cards)
```

The deck supports deterministic shuffling via seeds for reproducibility.

### Rules (`engine/rules.py`)

#### Capture Logic

```python
def check_capture(played_card: Card, top_card: Card) -> bool:
    """
    Check if played card captures the table pile.
    
    Capture occurs if:
    1. Played card matches top card's rank, OR
    2. Played card is a Jack (captures anything)
    """
    if played_card.rank == "J":
        return True
    return played_card.rank == top_card.rank
```

#### Pişti Detection

```python
def calculate_pisti(pile_size, played_card, top_card, is_jack_capture) -> int:
    """
    Calculate pişti bonus.
    
    Returns:
    - 10: Regular pişti (single card captured by rank match)
    - 20: Double pişti (Jack captures single Jack)
    - 0: No pişti
    """
    if pile_size != 1:
        return 0
    
    # Double pişti: Jack captures Jack
    if played_card.rank == "J" and top_card.rank == "J":
        return 20
    
    # Regular pişti: rank match (not Jack capture)
    if not is_jack_capture and played_card.rank == top_card.rank:
        return 10
    
    return 0
```

#### Scoring

Scoring cards:
- **Aces**: +1 each
- **Jacks**: +1 each
- **2♣**: +2
- **10♦**: +3
- **Majority bonus**: +3 to player with more captured cards
- **Pişti**: +10 each
- **Double pişti**: +20 each

### Game State (`engine/state.py`)

#### State Structure

```python
@dataclass(frozen=False)
class GameState:
    hands: Dict[int, List[Card]]           # Player hands
    table_pile: List[Card]                 # Current table pile
    captured: Dict[int, List[Card]]         # Captured cards per player
    center_cards: List[Card]                # 3 face-down initial cards
    stock: List[Card]                       # Remaining deck
    current_player: int                    # 0 or 1
    first_capture_made: bool                # Track first capture
    score_breakdown: Dict[int, Dict]        # Scoring breakdown
    move_history: List[Tuple]              # Move history
```

#### Immutable State Transitions

The state is conceptually immutable. `apply_action()` returns a **new** state:

```python
def apply_action(self, card: Card, config: Optional[Dict] = None) -> "GameState":
    """
    Apply action and return NEW state (immutable pattern).
    
    Process:
    1. Create deep copy of current state
    2. Remove card from player's hand
    3. Check for capture
    4. If capture: move pile to captured, check pişti
    5. If no capture: add card to table pile
    6. Switch current player
    7. Deal new cards if hands empty
    8. Return new state
    """
    new_state = copy.deepcopy(self)  # Immutable pattern
    # ... modify new_state ...
    return new_state
```

#### Legal Actions

```python
def get_legal_actions(self, player_id: int) -> List[int]:
    """Return list of card IDs (0-51) in player's hand."""
    return [card_to_id(card) for card in self.hands[player_id]]
```

#### Terminal Condition

```python
def is_terminal(self) -> bool:
    """Game ends when both hands empty AND stock exhausted."""
    hands_empty = len(self.hands[0]) == 0 and len(self.hands[1]) == 0
    stock_empty = len(self.stock) == 0
    return hands_empty and stock_empty
```

### Rewards (`engine/rewards.py`)

#### Sparse Rewards (Default)

```python
def sparse_reward(state: GameState, player_id: int, prev_state: GameState) -> float:
    """
    Sparse reward: only at terminal state.
    
    Returns:
    - 0.0 during game
    - Final score difference (player0_score - player1_score) at end
    """
    if not state.is_terminal():
        return 0.0
    
    scores = state.get_final_scores()
    return scores[0] - scores[1]  # From player 0's perspective
```

#### Shaped Rewards (Optional)

```python
def shaped_reward(state, player_id, prev_state, config) -> float:
    """
    Shaped reward: immediate rewards for good actions.
    
    Rewards:
    - Capturing scoring cards (A, J, 2♣, 10♦)
    - Pişti bonuses
    - Capturing pile (proxy for card advantage)
    """
    reward = 0.0
    
    # Check for scoring card captures
    if captured_scoring_card:
        reward += config.get("scoring_card_bonus", 1.0)
    
    # Check for pişti
    if pisti_occurred:
        reward += config.get("pisti_bonus", 10.0)
    
    # Terminal reward
    if state.is_terminal():
        scores = state.get_final_scores()
        reward += scores[0] - scores[1]
    
    return reward
```

---

## Environments

The environment layer (`envs/`) provides RL framework interfaces (Gymnasium and PettingZoo) while using the shared game engine.

### Base Game Engine (`envs/base.py`)

`PistiGameEngine` is the shared engine used by both environment wrappers:

```python
class PistiGameEngine:
    """
    Shared game engine for PettingZoo and Gymnasium wrappers.
    
    Responsibilities:
    - Game state management
    - Action application
    - Reward calculation
    - Observation generation
    """
```

#### Initialization Flow

```python
def reset(self, seed: Optional[int] = None) -> GameState:
    """
    Reset game to initial state.
    
    Process:
    1. Create and shuffle deck (with seed)
    2. Deal 4 cards to table center
    3. Flip one card face-up (non-Jack, or handle Jack case)
    4. Deal 4 cards to each player
    5. Remaining cards become stock
    6. Create initial GameState
    7. Return state
    """
```

#### Step Function

```python
def step(self, action: int) -> tuple[GameState, float, bool, Dict]:
    """
    Apply action and return (state, reward, done, info).
    
    Process:
    1. Convert action (0-51) to Card
    2. Validate action is legal
    3. Store previous state
    4. Apply action (get new state)
    5. Calculate reward (sparse or shaped)
    6. Check terminal condition
    7. Return (new_state, reward, done, info)
    """
```

#### Observation Generation

```python
def get_observation(self, player_id: int) -> Dict[str, np.ndarray]:
    """
    Get observation for a player using encoder.
    
    Returns dict with:
    - hand: (52,) multi-hot
    - table_top: (52,) one-hot
    - seen_cards: (52,) multi-hot
    - action_mask: (52,) boolean
    - score_breakdown: (6,) features
    - scalar features: table_count, captured counts, etc.
    """
    return self.encoder.encode(self.state, player_id)
```

### Gymnasium Environment (`envs/pisti_gym.py`)

The Gymnasium wrapper enables single-agent training with a pluggable opponent:

```python
class PistiGymEnv(Env):
    """
    Gymnasium environment for single-agent training.
    
    - Learning agent controls player_0
    - Opponent (pluggable) controls player_1
    - Supports action masking
    """
```

#### Reset Process

```python
def reset(self, seed=None, options=None) -> Tuple[ObsType, Dict]:
    """
    Reset environment.
    
    Process:
    1. Reset game engine
    2. Get observation for player_0 (learning agent)
    3. If opponent starts (player_1), make opponent move first
    4. Return (observation, info)
    """
```

#### Step Process

```python
def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict]:
    """
    Step with learning agent's action.
    
    Process:
    1. Learning agent (player_0) makes move
    2. If not done and opponent's turn, opponent makes move
    3. Return (observation, reward, terminated, truncated, info)
    
    Note: Reward is from player_0's perspective
    """
```

#### Opponent Interaction

The environment handles opponent moves automatically:

```python
def _opponent_step(self):
    """Make opponent move and return (obs, reward, done, info)."""
    # Get opponent's observation
    opp_obs = self.engine.get_observation(1)
    action_mask = opp_obs.get("action_mask", np.ones(52, dtype=bool))
    
    # Get opponent's action
    opp_action = self.opponent.predict(opp_obs, action_mask)
    
    # Step environment
    new_state, opp_reward, done, info = self.engine.step(opp_action)
    
    # Return observation for learning agent
    obs = self.engine.get_observation(0)
    return obs, opp_reward, done, info
```

### PettingZoo Environment (`envs/pisti_pettingzoo.py`)

The PettingZoo AEC (Agent Environment Cycle) wrapper enables multi-agent training:

```python
class PistiPettingZooEnv(AECEnv):
    """
    PettingZoo AEC environment for multi-agent training.
    
    Features:
    - Sequential turns
    - Each agent gets its own observation
    - Supports self-play
    """
```

#### Agent Selection

```python
def reset(self, seed=None, options=None):
    """Reset and set first agent."""
    self.engine.reset(seed=seed)
    self.agents = ["player_0", "player_1"]
    self._agent_selector = agent_selector(self.agents)
    self.agent_selection = self._agent_selector.reset()
```

#### Step Process

```python
def step(self, action):
    """
    Step with current agent's action.
    
    Process:
    1. Current agent makes move
    2. Switch to next agent
    3. Update agent_selection
    4. Set rewards, terminations, truncations
    """
```

### Action Masking

Action masks are included in observations to prevent invalid actions:

```python
def build_action_mask(state: GameState, player_id: int) -> np.ndarray:
    """
    Build action mask: True for legal actions, False for invalid.
    
    Returns:
    (52,) boolean array where True = card is in player's hand
    """
    mask = np.zeros(52, dtype=bool)
    legal_actions = state.get_legal_actions(player_id)
    mask[legal_actions] = True
    return mask
```

**Usage in RL**:
- **MaskablePPO**: Uses mask directly in policy network
- **Other methods**: Mask applied to logits before sampling

---

## Observation Encoding

The encoding system (`encoding/`) converts game states into neural network-friendly representations.

### Observation Builder (`encoding/obs_builder.py`)

`ObsBuilder` constructs the raw observation components:

#### Hand Representation

```python
def build_hand(self, state: GameState, player_id: int) -> np.ndarray:
    """
    Build multi-hot vector for player's hand.
    
    Returns:
    (52,) array where 1.0 = card is in hand, 0.0 = not in hand
    
    Example:
    If player has [2♠, 5♥, J♦]:
    hand[0] = 1.0   # 2♠
    hand[18] = 1.0  # 5♥
    hand[36] = 1.0  # J♦
    All others = 0.0
    """
```

#### Table Top Representation

```python
def build_table_top(self, state: GameState) -> np.ndarray:
    """
    Build one-hot vector for top card of table pile.
    
    Returns:
    (52,) array where 1.0 at top card position, all zeros if empty
    """
```

#### Seen Cards

```python
def build_seen_cards(self, state: GameState, player_id: int) -> np.ndarray:
    """
    Build multi-hot vector for all cards seen by player.
    
    Includes:
    - Cards in player's hand
    - Cards in table pile (all visible)
    - Cards captured by either player (visible after capture)
    
    Used for probabilistic inference (opponent hand tracking)
    """
```

#### Scalar Features

```python
def build_scalar_features(self, state, player_id) -> Dict:
    """
    Build scalar features:
    - table_count: Number of cards in table pile
    - my_captured_count: Cards I've captured
    - opp_captured_count: Cards opponent captured
    - stock_remaining: Cards left in stock
    - hand_size: Cards in my hand
    - last_capture_by: Who made last capture (-1, 0, or 1)
    - running_score_estimate: Estimated score difference
    """
```

### Encoders (`encoding/encoders.py`)

All encoders inherit from `ObservationEncoder`:

```python
class ObservationEncoder(ABC):
    """Abstract base class for all encoders."""
    
    @abstractmethod
    def encode(self, state: GameState, player_id: int) -> Dict[str, np.ndarray]:
        """Encode state into observation dict."""
        pass
    
    def get_observation_space_dict(self) -> Dict:
        """Return Gymnasium spaces.Dict-compatible observation space."""
        pass
```

#### MultiHotEncoder (Default)

```python
class MultiHotEncoder(ObservationEncoder):
    """
    Default encoder: 52-length multi-hot vectors.
    
    Observation dict contains:
    - hand: (52,) multi-hot
    - table_top: (52,) one-hot
    - seen_cards: (52,) multi-hot
    - action_mask: (52,) boolean
    - score_breakdown: (6,) features
    - All scalar features
    """
```

**Key Principle**: Never uses raw integer IDs. All card information is in multi-hot format.

#### CNNEncoder

```python
class CNNEncoder(MultiHotEncoder):
    """
    Extends MultiHotEncoder with (4,13) reshaped views.
    
    Adds:
    - hand_cnn: (4, 13) reshaped view (suit × rank)
    - table_top_cnn: (4, 13) reshaped view
    - seen_cards_cnn: (4, 13) reshaped view
    
    Useful for CNN experiments that benefit from spatial structure.
    """
```

#### FeatureEncoder

```python
class FeatureEncoder(ObservationEncoder):
    """
    Flattens observation to single vector.
    
    Returns:
    - features: (N,) flattened vector of all features
    
    Useful for simple MLP policies.
    """
```

#### SequenceEncoder

```python
class SequenceEncoder(ObservationEncoder):
    """
    Adds move history sequence for recurrent policies.
    
    Returns:
    - All MultiHotEncoder features
    - move_history: (history_length, feature_dim) sequence
    
    Useful for LSTM/GRU policies that need history.
    """
```

---

## Agents

The agent layer (`agents/`) contains all decision-making policies, from simple baselines to complex RL agents.

### Agent Interface

All agents implement a common interface:

```python
class Agent:
    def predict(self, obs: Dict, action_mask: np.ndarray, deterministic: bool = False) -> int:
        """Predict action given observation."""
        pass
    
    def update_state(self, state: GameState):
        """Update internal state (optional, for agents that need full state)."""
        pass
```

### Baseline Agents (`agents/baselines.py`)

#### RandomValidAgent

```python
class RandomValidAgent:
    """
    Plays a random legal card.
    
    Strategy:
    1. Get all legal actions from action_mask
    2. Sample uniformly from legal actions
    3. Return action
    """
    def predict(self, obs: Dict, action_mask: np.ndarray) -> int:
        legal_actions = np.where(action_mask)[0]
        return int(np.random.choice(legal_actions))
```

**Use Case**: Baseline opponent, evaluation metric.

#### GreedyCaptureAgent

```python
class GreedyCaptureAgent:
    """
    Greedy capture strategy.
    
    Strategy:
    1. If can capture (match rank or have Jack), do so
    2. Prefer Jack if available (can capture anything)
    3. Else, play lowest rank card
    """
```

**Use Case**: Strong baseline, evaluation opponent.

#### PistiHunterAgent

```python
class PistiHunterAgent:
    """
    Heuristic for pişti opportunities.
    
    Strategy:
    1. If table has 1 card, try to match for pişti
    2. If table empty/multiple, play cards we have duplicates of
    3. Fallback to greedy capture
    """
```

**Use Case**: Heuristic baseline, evaluation opponent.

### Probabilistic Agent (`agents/probabilistic_agent.py`)

The probabilistic agent plays optimally based on inferred card probabilities.

#### BeliefTracker

```python
class BeliefTracker:
    """
    Tracks seen cards and infers opponent hand probabilities.
    
    Process:
    1. Maintain set of seen cards
    2. Update from observations
    3. Infer unseen cards (52 - seen_cards)
    4. Sample possible opponent hands using hypergeometric distribution
    5. Compute probabilities for each hand
    """
    
    def update_from_observation(self, obs, my_hand_size, opp_hand_size):
        """Update seen cards from observation."""
        # Extract seen cards from observation
        seen_cards = obs.get("seen_cards", np.zeros(52))
        # Update internal tracking
        ...
    
    def get_opponent_hand_probs(self, opp_hand_size, unseen_cards, max_samples):
        """
        Generate sampled opponent hands with probabilities.
        
        Returns:
        List of (hand_tuple, probability) pairs
        """
```

#### ActionEvaluator

```python
class ActionEvaluator:
    """
    Evaluates actions by simulating outcomes.
    
    Process:
    1. For each legal action:
       a. Sample possible opponent hands
       b. Simulate outcomes for each hand
       c. Compute expected value
    2. Return action with highest expected value
    """
    
    def evaluate_action(self, state, action, opponent_hand_probs):
        """
        Compute expected value of action.
        
        For each possible opponent hand:
        1. Simulate game outcome
        2. Estimate state value
        3. Weight by probability
        4. Sum to get expected value
        """
```

#### ProbabilisticOptimalAgent

```python
class ProbabilisticOptimalAgent:
    """
    Combines BeliefTracker and ActionEvaluator.
    
    Process:
    1. Update beliefs from observation
    2. Estimate opponent hand size
    3. Sample opponent hands with probabilities
    4. Evaluate all legal actions
    5. Return best action
    """
```

**Use Case**: Strong baseline, optimal play under uncertainty.

### RL Agents

#### NFSP Agent (`agents/nfsp_agent.py`)

**Neural Fictitious Self-Play** learns approximate Nash equilibria:

```python
class NFSPAgent:
    """
    NFSP agent with two networks:
    1. Average Strategy Network: Learns Nash equilibrium
    2. Best Response Network: Learns best response to opponents
    """
    
    def __init__(self, observation_dim, action_dim, config):
        # Average strategy network: [256, 256, 128]
        self.average_strategy_net = AverageStrategyNetwork(...)
        
        # Best response network: [256, 256, 128]
        self.best_response_net = BestResponseNetwork(...)
        
        # Reservoir buffer for opponent strategies
        self.reservoir_buffer = ReservoirBuffer(...)
```

**Training Process**:
1. **Best Response Phase**: Train best response network against opponent strategies
2. **Average Strategy Phase**: Train average strategy to match best response
3. **Reservoir Sampling**: Maintain diverse opponent strategies

**Action Selection**:
- With probability (1-η): Use average strategy (Nash equilibrium)
- With probability η: Use best response (exploration)

**Architecture**:
- Average Strategy: `[256, 256, 128]`
- Best Response: `[256, 256, 128]` (policy + value networks)

#### Deep CFR Agent (`agents/deep_cfr_agent.py`)

**Deep Counterfactual Regret Minimization** uses regret matching:

```python
class DeepCFRAgent:
    """
    Deep CFR agent with:
    1. Counterfactual Value Networks: Compute counterfactual values
    2. Regret Accumulators: Track regrets per information set
    3. Strategy Computation: Regret matching
    """
    
    def __init__(self, observation_dim, action_dim, config):
        # Counterfactual value networks (one per player)
        self.counterfactual_value_nets = {
            0: CounterfactualValueNetwork(..., [256, 256, 256, 128]),
            1: CounterfactualValueNetwork(..., [256, 256, 256, 128]),
        }
        
        # Regret accumulators: info_set -> action -> regret
        self.regrets = {0: defaultdict(lambda: defaultdict(float)),
                        1: defaultdict(lambda: defaultdict(float))}
```

**Training Process**:
1. **Traverse**: Traverse game tree for each player
2. **Compute Counterfactual Values**: For each information set
3. **Update Regrets**: Accumulate regrets
4. **Compute Strategy**: Regret matching

**Information Sets**:
- Represent states indistinguishable to player
- Hash based on public information (table, seen cards, etc.)

**Architecture**:
- Counterfactual Value: `[256, 256, 256, 128]` (4 layers)
- Strategy: `[256, 256, 128]`

#### R2D2 Agent (`agents/r2d2_agent.py`)

**Recurrent Replay Distributed DQN** for partial observability:

```python
class R2D2Agent:
    """
    R2D2 agent with:
    1. Recurrent Q-network (LSTM)
    2. Prioritized experience replay
    3. N-step returns
    """
    
    def __init__(self, observation_dim, action_dim, config):
        # Recurrent Q-network
        self.q_network = RecurrentQNetwork(
            input_dim, output_dim,
            lstm_hidden_size=256,
            lstm_layers=2,
            mlp_layers=[128, 128]
        )
        
        # Prioritized replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(...)
```

**Architecture**:
- LSTM: 256 hidden units, 2 layers
- MLP after LSTM: `[128, 128]`

**Features**:
- Maintains hidden state across steps
- Prioritized replay (samples important experiences)
- N-step returns (looks ahead N steps)

### Opponents (`agents/opponents.py`)

#### FrozenCheckpointOpponent

```python
class FrozenCheckpointOpponent:
    """
    Loads a saved SB3 model as opponent.
    
    Process:
    1. Load model from checkpoint
    2. Set to evaluation mode (no training)
    3. Use model.predict() for actions
    """
```

**Use Case**: Self-play with past checkpoints.

#### OpponentPool

```python
class OpponentPool:
    """
    Maintains pool of past checkpoints.
    
    Process:
    1. Store checkpoints in pool
    2. Sample uniformly when needed
    3. Load checkpoint as opponent
    """
    
    def sample_opponent(self):
        """Sample random checkpoint from pool."""
        checkpoint = random.choice(self.opponents)
        return FrozenCheckpointOpponent(checkpoint.path, checkpoint.algorithm)
```

**Use Case**: League training, diverse opponents.

#### SelfPlayOpponent

```python
class SelfPlayOpponent:
    """
    Uses current training policy as opponent.
    
    Process:
    1. Wraps current model
    2. Uses same policy for opponent
    3. Updates as model trains
    """
```

**Use Case**: Direct self-play training.

---

## Training System

The training system (`training/`) orchestrates RL algorithm training with callbacks, metadata, and utilities.

### SB3 Training (`training/train_sb3.py`)

Supports all Stable-Baselines3 algorithms:

#### Supported Algorithms

1. **PPO** (Proximal Policy Optimization)
2. **MaskablePPO** (PPO with action masking)
3. **RecurrentPPO** (PPO with LSTM)
4. **DQN** (Deep Q-Network)
5. **RainbowDQN** (Enhanced DQN)

#### Training Flow

```python
def train(config_path: str):
    """
    Main training function.
    
    Process:
    1. Load config from YAML
    2. Set up directories (logs, checkpoints)
    3. Create encoder
    4. Create opponent (if self-play)
    5. Create environment
    6. Create model with network architecture
    7. Set up callbacks (checkpoint, evaluation, league)
    8. Create metadata
    9. Train model
    10. Save final model and metadata
    """
```

#### Network Architecture Selection

```python
# Get architecture from config
use_deep = training_config.get("use_deep_architecture", True)
arch_type = "deep" if use_deep else "shallow"
net_arch_dict = get_network_arch(config, arch_type)

# Apply to model
policy_kwargs = {
    "net_arch": {
        "pi": net_arch_dict["pi"],  # [256, 256, 128] for deep
        "vf": net_arch_dict["vf"],  # [256, 256, 128] for deep
    }
}
model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, ...)
```

#### Model Creation Examples

**MaskablePPO**:
```python
if algorithm == "MaskablePPO":
    from sb3_contrib import MaskablePPO
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,  # Deep architecture
        learning_rate=3e-4,
        ...
    )
```

**RecurrentPPO**:
```python
if algorithm == "RecurrentPPO":
    from sb3_contrib import RecurrentPPO
    policy_kwargs = {
        "lstm_hidden_size": 256,
        "n_lstm_layers": 2,
        "net_arch": [128, 128],  # After LSTM
    }
    model = RecurrentPPO(
        "MlpLstmPolicy",  # LSTM policy
        env,
        policy_kwargs=policy_kwargs,
        ...
    )
```

### NFSP Training (`training/train_nfsp.py`)

Custom training loop for NFSP:

```python
def train_nfsp(config_path: str):
    """
    NFSP training loop.
    
    Process:
    1. Create NFSP agent
    2. Training loop:
       a. Reset environment
       b. Agent predicts action (using anticipatory parameter)
       c. Step environment
       d. Store experience
       e. Train agent (alternating best response / average strategy)
    3. Save checkpoints periodically
    """
```

**Key Features**:
- Alternating training phases
- Reservoir buffer management
- Average strategy updates

### Deep CFR Training (`training/train_deep_cfr.py`)

Custom training loop for Deep CFR:

```python
def train_deep_cfr(config_path: str):
    """
    Deep CFR training loop.
    
    Process:
    1. Create Deep CFR agent
    2. For each traversal:
       a. Traverse game tree for player
       b. Compute counterfactual values
       c. Update regrets
       d. Train counterfactual value networks
    3. Compute strategies from regrets
    """
```

**Key Features**:
- Information set traversals
- Counterfactual value computation
- Regret matching

### R2D2 Training (`training/train_r2d2.py`)

Custom training loop for R2D2:

```python
def train_r2d2(config_path: str):
    """
    R2D2 training loop.
    
    Process:
    1. Create R2D2 agent
    2. Training loop:
       a. Reset environment (reset hidden state)
       b. Predict action (with hidden state)
       c. Step environment
       d. Store experience (with hidden states)
       e. Train from replay buffer (prioritized)
    3. Update target network periodically
    """
```

**Key Features**:
- Recurrent replay buffer
- Hidden state management
- Prioritized experience replay
- N-step returns

### Callbacks (`training/callbacks.py`)

#### CheckpointCallback

```python
class CheckpointCallback(BaseCallback):
    """
    Saves model checkpoints periodically.
    
    Process:
    1. Every save_freq steps:
       a. Save model checkpoint
       b. Save metadata JSON
    2. Track training progress
    """
```

#### EvalCallback

```python
class EvalCallback(BaseCallback):
    """
    Evaluates model periodically.
    
    Process:
    1. Every eval_freq steps:
       a. Run n_eval_episodes
       b. Compute win rate, score diff
       c. Log metrics
       d. Track best score
       e. Update metadata
    """
```

#### LeagueCallback

```python
class LeagueCallback(BaseCallback):
    """
    Manages opponent pool for self-play.
    
    Process:
    1. Every snapshot_freq steps:
       a. Save snapshot
       b. Add to opponent pool
       c. Update pool (maintain max size)
    """
```

### Metadata (`training/metadata.py`)

Tracks all information needed for reproducibility:

```python
@dataclass
class ModelMetadata:
    config_path: str
    training_config: Dict
    algorithm: str
    hyperparameters: Dict
    encoder_type: str
    total_timesteps: int
    training_start_time: str
    training_end_time: str
    best_eval_score: float
    git_commit_hash: str
    python_version: str
    package_versions: Dict
    # ... more fields
```

**Saved with each checkpoint** as `{checkpoint_name}_metadata.json`.

### Utilities (`training/utils/`)

#### Network Architectures

```python
def get_network_arch(config, arch_type):
    """
    Get network architecture from config.
    
    Types:
    - "deep": [256, 256, 128]
    - "medium": [128, 128]
    - "shallow": [64, 64]
    - "recurrent": LSTM(256, 2) + MLP([128, 128])
    - "nfsp": Average [256, 256, 128], Best response [256, 256, 128]
    - "deep_cfr": Value [256, 256, 256, 128], Strategy [256, 256, 128]
    """
```

#### Action Masking

```python
def apply_action_mask(logits, action_mask, mask_value=-1e9):
    """
    Apply action mask to logits.
    
    Sets invalid actions to mask_value (typically -1e9).
    """
    masked_logits = logits.copy()
    masked_logits[~action_mask] = mask_value
    return masked_logits
```

---

## Evaluation System

The evaluation system provides comprehensive metrics and statistical analysis.

### Simple Evaluation (`training/eval.py`)

Basic evaluation against baseline opponents:

```python
def evaluate(checkpoint_path, config_path, opponents, n_episodes):
    """
    Simple evaluation.
    
    Process:
    1. Load model
    2. For each opponent:
       a. Run n_episodes
       b. Compute win rate, avg score diff
       c. Print results
    """
```

### Comprehensive Evaluation (`training/evaluate_comprehensive.py`)

Statistical evaluation with confidence intervals:

```python
def evaluate_comprehensive(checkpoint_path, config_path, opponents, n_episodes, n_seeds):
    """
    Comprehensive evaluation with statistics.
    
    Process:
    1. Load model
    2. For each opponent:
       a. Run n_episodes across n_seeds
       b. Compute metrics (win rate, score diff, pistis, etc.)
       c. Calculate confidence intervals (95% CI)
       d. Perform statistical tests
    3. Save results to JSON
    """
```

**Metrics Tracked**:
- Win rate (with CI)
- Score differential (with CI)
- Pişti frequency
- Capture efficiency
- Game length
- Raw data for further analysis

**Statistical Analysis**:
- Mean, standard deviation
- 95% confidence intervals (t-distribution)
- Multiple seeds for robustness

### Results Export (`training/results.py`)

#### ResultsExporter

```python
class ResultsExporter:
    @staticmethod
    def to_csv(results, output_path):
        """Export to CSV format."""
    
    @staticmethod
    def to_latex_table(results, output_path):
        """Export to LaTeX table format."""
    
    @staticmethod
    def to_markdown(results, output_path):
        """Export to Markdown format."""
```

#### ResultsVisualizer

```python
class ResultsVisualizer:
    @staticmethod
    def plot_win_rates(results, output_path):
        """Plot win rates with confidence intervals."""
    
    @staticmethod
    def plot_score_distributions(results, output_path):
        """Plot score difference distributions."""
    
    @staticmethod
    def plot_performance_comparison(results, output_path):
        """Comprehensive performance comparison plot."""
```

#### ResultsAnalyzer

```python
class ResultsAnalyzer:
    @staticmethod
    def t_test(data1, data2):
        """Perform t-test between two groups."""
    
    @staticmethod
    def mann_whitney_u(data1, data2):
        """Perform Mann-Whitney U test."""
    
    @staticmethod
    def compare_opponents(results, baseline):
        """Compare all opponents against baseline."""
```

### Report Generation (`training/generate_report.py`)

```python
def generate_report(results_dir, output_path, formats, checkpoint_path):
    """
    Generate academic report.
    
    Process:
    1. Load evaluation results
    2. Generate visualizations
    3. Perform statistical analysis
    4. Export to formats (Markdown, LaTeX, HTML, CSV)
    5. Include reproducibility section
    """
```

**Report Contents**:
- Executive summary
- Performance metrics table
- Visualizations (win rates, distributions, comparisons)
- Statistical analysis (significance tests)
- Reproducibility section (config, hyperparameters, system info)

---

## Configuration System

All settings are configured via YAML files (`configs/default.yaml`).

### Structure

```yaml
game:
  expose_bottom_card: false
  pisti_exceptions: []

reward:
  sparse: true
  shaping:
    enabled: false
    capture_bonus: 0.1
    scoring_card_bonus: 1.0
    pisti_bonus: 10.0

encoding:
  encoder_type: "MultiHotEncoder"
  mode: "feature"
  include_history: false
  history_length: 10

training:
  algorithm: "MaskablePPO"
  total_timesteps: 1000000
  seed: 42
  use_deep_architecture: true
  
  network_architectures:
    deep:
      pi: [256, 256, 128]
      vf: [256, 256, 128]
      qf: [256, 256, 128]
    shallow:
      pi: [64, 64]
      vf: [64, 64]
      qf: [64, 64]
    # ... more architectures
  
  maskable_ppo:
    learning_rate: 3.0e-4
    n_steps: 2048
    # ... more hyperparameters
  
  nfsp:
    anticipatory_param: 0.1
    average_strategy_update_freq: 1000
    # ... more hyperparameters
```

### Network Architecture Configuration

Network architectures are defined in `training.network_architectures`:

- **Deep**: `[256, 256, 128]` - Sufficient capacity
- **Medium**: `[128, 128]` - Moderate capacity
- **Shallow**: `[64, 64]` - Benchmark
- **Recurrent**: LSTM(256, 2) + MLP([128, 128])
- **NFSP**: Average `[256, 256, 128]`, Best response `[256, 256, 128]`
- **Deep CFR**: Value `[256, 256, 256, 128]`, Strategy `[256, 256, 128]`

---

## Data Flow Diagrams

### Training Loop Flow

```
┌─────────────┐
│   Config    │
│  (YAML)     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Load Config    │
│  Create Encoder │
│  Create Env     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Create Model   │
│  (with arch)    │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Set Callbacks  │
│  (checkpoint,   │
│   eval, league) │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐      ┌──────────────┐
│  Training Loop  │◄─────│   Callbacks  │
│                 │      │  (periodic)  │
│  env.reset()   │      └──────────────┘
│  obs = env.get_│
│  action = model.│
│  obs, r, d = env│
│  model.learn()  │
└─────────────────┘
       │
       ▼
┌─────────────────┐
│  Save Final     │
│  Model +        │
│  Metadata       │
└─────────────────┘
```

### Agent Prediction Flow

```
┌──────────────┐
│  Game State  │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  Encoder.encode()│
│  (state, player) │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Observation Dict│
│  - hand: (52,)   │
│  - table_top: (52│
│  - action_mask   │
│  - ...           │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Agent.predict() │
│  (obs, mask)     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Action (0-51)   │
└──────────────────┘
```

### State Transition Flow

```
┌──────────────┐
│ Current State│
└──────┬───────┘
       │
       │ action (card ID)
       ▼
┌──────────────────┐
│  Validate Action │
│  (in hand?)      │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  apply_action()  │
│  (returns NEW    │
│   state)         │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Check Capture   │
│  - Match rank?   │
│  - Is Jack?      │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Update State    │
│  - Remove card   │
│  - Capture pile  │
│  - Check pişti   │
│  - Switch player │
│  - Deal cards    │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  New State       │
│  (immutable)     │
└──────────────────┘
```

### Evaluation Flow

```
┌──────────────┐
│  Checkpoint  │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  Load Model      │
│  (detect type)   │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  For each        │
│  opponent:       │
│                  │
│  For each seed:  │
│    For episode:  │
│      obs = reset │
│      while !done:│
│        action =  │
│        model(obs)│
│        obs,r,d = │
│        env.step()│
│      metrics +=  │
│      episode     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Aggregate       │
│  Metrics         │
│  - Mean, Std     │
│  - CI (95%)      │
│  - Tests         │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Export Results  │
│  - JSON          │
│  - CSV           │
│  - LaTeX         │
│  - Visualizations│
└──────────────────┘
```

---

## Code Examples

### Training a Model

```python
# Using SB3 algorithms
python -m training.train_sb3 --config configs/default.yaml

# Using NFSP
python -m training.train_nfsp --config configs/default.yaml

# Using Deep CFR
python -m training.train_deep_cfr --config configs/default.yaml

# Using R2D2
python -m training.train_r2d2 --config configs/default.yaml
```

### Evaluating a Model

```python
# Simple evaluation
python -m training.eval \
    --checkpoint checkpoints/pisti_model_final \
    --opponents random,greedy,pisti_hunter

# Comprehensive evaluation
python -m training.evaluate_comprehensive \
    --checkpoint checkpoints/pisti_model_final \
    --opponents random,greedy,pisti_hunter,probabilistic \
    --n-episodes 1000 \
    --n-seeds 10 \
    --output-dir results/experiment_1
```

### Creating a Custom Agent

```python
from typing import Dict
import numpy as np
from engine.state import GameState

class MyCustomAgent:
    """Custom agent example."""
    
    def predict(self, obs: Dict, action_mask: np.ndarray, deterministic: bool = False) -> int:
        """
        Predict action.
        
        Args:
            obs: Observation dict
            action_mask: Boolean array of legal actions
            deterministic: Whether to use deterministic policy
        
        Returns:
            Action (0-51)
        """
        legal_actions = np.where(action_mask)[0]
        
        # Your custom logic here
        # Example: prefer certain cards
        preferred_cards = [0, 13, 26, 39]  # Example: all 2s
        for card in preferred_cards:
            if card in legal_actions:
                return card
        
        # Fallback: random
        return int(np.random.choice(legal_actions))
    
    def update_state(self, state: GameState):
        """Optional: update internal state if needed."""
        pass
```

### Using Different Encoders

```python
from encoding.encoders import MultiHotEncoder, CNNEncoder, SequenceEncoder
from envs.pisti_gym import PistiGymEnv

# Multi-hot encoder (default)
encoder = MultiHotEncoder()
env = PistiGymEnv(encoder=encoder)

# CNN encoder (with reshaped views)
encoder = CNNEncoder()
env = PistiGymEnv(encoder=encoder)

# Sequence encoder (for recurrent policies)
encoder = SequenceEncoder(config={"history_length": 10})
env = PistiGymEnv(encoder=encoder)
```

### Custom Network Architecture

```python
# In configs/custom.yaml
training:
  use_deep_architecture: true
  network_architectures:
    deep:
      pi: [512, 512, 256, 128]  # Custom deep architecture
      vf: [512, 512, 256, 128]
      qf: [512, 512, 256, 128]
```

### Self-Play Training

```python
# In configs/default.yaml
training:
  self_play:
    enabled: true
    opponent_pool_size: 5
    snapshot_frequency: 50000
```

The framework automatically:
1. Creates opponent pool
2. Saves snapshots periodically
3. Samples from pool for opponents
4. Maintains diverse opponent strategies

### Loading and Using a Trained Model

```python
from stable_baselines3 import PPO
from envs.pisti_gym import PistiGymEnv
from encoding.encoders import MultiHotEncoder

# Load model
model = PPO.load("checkpoints/pisti_model_final")

# Create environment
encoder = MultiHotEncoder()
env = PistiGymEnv(encoder=encoder)

# Play episode
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

# Get final scores
if env.engine.state.is_terminal():
    scores = env.engine.state.get_final_scores()
    print(f"Final scores: {scores}")
```

---

## Summary

This framework provides a complete, modular system for RL research on Pişti:

1. **Game Engine**: Pure, testable game logic
2. **Environments**: Gymnasium and PettingZoo interfaces
3. **Encoding**: Flexible observation encoding system
4. **Agents**: Baseline, probabilistic, and RL agents
5. **Training**: Multiple RL algorithms with deep architectures
6. **Evaluation**: Comprehensive statistical evaluation
7. **Configuration**: YAML-based configuration system

All components are designed to be modular, extensible, and well-documented for academic research and experimentation.
