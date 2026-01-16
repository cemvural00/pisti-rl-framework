"""Deep Counterfactual Regret Minimization (Deep CFR) agent."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import hashlib

from engine.state import GameState
from training.utils.network_architectures import (
    create_deep_policy_network,
    create_value_network,
    get_network_arch,
)


class InformationSet:
    """Represents an information set (set of states indistinguishable to player)."""
    
    def __init__(self, player_id: int, public_info: Dict):
        self.player_id = player_id
        self.public_info = public_info
        self.key = self._compute_key()
    
    def _compute_key(self) -> str:
        """Compute hash key for information set."""
        # Create hash from public information
        info_str = str(sorted(self.public_info.items()))
        return hashlib.md5(info_str.encode()).hexdigest()
    
    def __hash__(self):
        return hash(self.key)
    
    def __eq__(self, other):
        return isinstance(other, InformationSet) and self.key == other.key


class CounterfactualValueNetwork(nn.Module):
    """Network for computing counterfactual values."""
    
    def __init__(self, input_dim: int, output_dim: int, net_arch: List[int] = [256, 256, 256, 128]):
        super().__init__()
        self.network = create_value_network(input_dim, net_arch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class StrategyNetwork(nn.Module):
    """Network for computing strategy from regrets."""
    
    def __init__(self, input_dim: int, output_dim: int, net_arch: List[int] = [256, 256, 128]):
        super().__init__()
        self.network = create_deep_policy_network(input_dim, output_dim, net_arch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class DeepCFRAgent:
    """
    Deep Counterfactual Regret Minimization agent.
    
    Deep CFR learns an approximate Nash equilibrium by:
    1. Computing counterfactual values for information sets
    2. Accumulating regrets
    3. Computing strategies via regret matching
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        """
        Initialize Deep CFR agent.
        
        Args:
            observation_dim: Dimension of observation space
            action_dim: Size of action space (52 for Pişti)
            config: Configuration dict
            device: Device to run on
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        # Hyperparameters
        self.regret_matching_epsilon = config.get("regret_matching_epsilon", 0.001)
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.traversal_batch_size = config.get("traversal_batch_size", 32)
        
        # Get network architectures
        deep_cfr_arch = get_network_arch(config, "deep_cfr")
        cf_value_arch = deep_cfr_arch.get("counterfactual_value", [256, 256, 256, 128])
        strategy_arch = deep_cfr_arch.get("strategy", [256, 256, 128])
        
        # Networks (one per player)
        self.counterfactual_value_nets = {
            0: CounterfactualValueNetwork(observation_dim, 1, cf_value_arch).to(self.device),
            1: CounterfactualValueNetwork(observation_dim, 1, cf_value_arch).to(self.device),
        }
        self.strategy_nets = {
            0: StrategyNetwork(observation_dim, action_dim, strategy_arch).to(self.device),
            1: StrategyNetwork(observation_dim, action_dim, strategy_arch).to(self.device),
        }
        
        # Optimizers
        self.value_optimizers = {
            player_id: optim.Adam(net.parameters(), lr=self.learning_rate)
            for player_id, net in self.counterfactual_value_nets.items()
        }
        self.strategy_optimizers = {
            player_id: optim.Adam(net.parameters(), lr=self.learning_rate)
            for player_id, net in self.strategy_nets.items()
        }
        
        # Regret accumulators (information set -> action -> regret)
        self.regrets: Dict[int, Dict[str, Dict[int, float]]] = {
            0: defaultdict(lambda: defaultdict(float)),
            1: defaultdict(lambda: defaultdict(float)),
        }
        
        # Strategy accumulators (for average strategy)
        self.strategy_sums: Dict[int, Dict[str, Dict[int, float]]] = {
            0: defaultdict(lambda: defaultdict(float)),
            1: defaultdict(lambda: defaultdict(float)),
        }
        
        self.current_state: Optional[GameState] = None
        self.traversal_count = 0
    
    def get_information_set(self, player_id: int, obs: Dict[str, np.ndarray]) -> InformationSet:
        """
        Get information set for current observation.
        
        Args:
            player_id: Player ID
            obs: Observation dict
        
        Returns:
            InformationSet object
        """
        # Extract public information (everything except hidden opponent hand)
        public_info = {
            "table_top": obs.get("table_top", np.zeros(52)).tolist(),
            "table_count": int(obs.get("table_count", 0)),
            "seen_cards": obs.get("seen_cards", np.zeros(52)).tolist(),
            "my_captured_count": int(obs.get("my_captured_count", 0)),
            "opp_captured_count": int(obs.get("opp_captured_count", 0)),
            "stock_remaining": int(obs.get("stock_remaining", 0)),
        }
        return InformationSet(player_id, public_info)
    
    def predict(
        self,
        obs: Dict[str, np.ndarray],
        action_mask: np.ndarray,
        player_id: int = 0,
        deterministic: bool = False,
    ) -> int:
        """
        Predict action using current strategy.
        
        Args:
            obs: Observation dict
            action_mask: Action mask
            player_id: Player ID
            deterministic: Whether to use deterministic policy
        
        Returns:
            Action index
        """
        info_set = self.get_information_set(player_id, obs)
        info_set_key = info_set.key
        
        # Get strategy from regrets
        strategy = self._get_strategy_from_regrets(info_set_key, player_id, action_mask)
        
        # Sample or take argmax
        if deterministic:
            action = int(np.argmax(strategy))
        else:
            action = int(np.random.choice(self.action_dim, p=strategy))
        
        return action
    
    def _get_strategy_from_regrets(
        self, info_set_key: str, player_id: int, action_mask: np.ndarray
    ) -> np.ndarray:
        """
        Compute strategy from regrets using regret matching.
        
        Args:
            info_set_key: Information set key
            player_id: Player ID
            action_mask: Action mask
        
        Returns:
            Strategy distribution (probability vector)
        """
        regrets = self.regrets[player_id][info_set_key]
        
        # Get positive regrets for valid actions
        positive_regrets = {}
        for action in range(self.action_dim):
            if action_mask[action]:
                regret = regrets.get(action, 0.0)
                positive_regrets[action] = max(0.0, regret)
        
        # Regret matching
        sum_positive = sum(positive_regrets.values())
        
        if sum_positive > self.regret_matching_epsilon:
            # Normalize positive regrets
            strategy = np.zeros(self.action_dim)
            for action, regret in positive_regrets.items():
                strategy[action] = regret / sum_positive
            return strategy
        else:
            # Uniform over valid actions
            strategy = np.zeros(self.action_dim)
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                strategy[valid_actions] = 1.0 / len(valid_actions)
            return strategy
    
    def update_state(self, state: GameState):
        """Update internal game state."""
        self.current_state = state
    
    def update_regrets(
        self,
        info_set_key: str,
        player_id: int,
        action: int,
        counterfactual_value: float,
        action_values: Dict[int, float],
    ):
        """
        Update regrets for information set.
        
        Args:
            info_set_key: Information set key
            player_id: Player ID
            action: Action taken
            counterfactual_value: Counterfactual value
            action_values: Values for each action
        """
        for a, value in action_values.items():
            regret = value - counterfactual_value
            self.regrets[player_id][info_set_key][a] += regret
    
    def update_strategy_sum(
        self,
        info_set_key: str,
        player_id: int,
        strategy: np.ndarray,
        reach_prob: float,
    ):
        """
        Update strategy sum for average strategy computation.
        
        Args:
            info_set_key: Information set key
            player_id: Player ID
            strategy: Strategy distribution
            reach_prob: Reach probability
        """
        for action, prob in enumerate(strategy):
            self.strategy_sums[player_id][info_set_key][action] += prob * reach_prob
    
    def train_counterfactual_value(
        self,
        info_sets: List[str],
        observations: List[torch.Tensor],
        counterfactual_values: List[float],
        player_id: int,
    ):
        """
        Train counterfactual value network.
        
        Args:
            info_sets: List of information set keys
            observations: List of observation tensors
            counterfactual_values: List of target counterfactual values
            player_id: Player ID
        """
        if len(observations) == 0:
            return
        
        obs_batch = torch.stack(observations)
        target_values = torch.tensor(
            counterfactual_values, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        
        # Forward pass
        predicted_values = self.counterfactual_value_nets[player_id](obs_batch)
        
        # Loss
        loss = nn.functional.mse_loss(predicted_values, target_values)
        
        # Update
        self.value_optimizers[player_id].zero_grad()
        loss.backward()
        self.value_optimizers[player_id].step()
    
    def _obs_to_tensor(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        """Convert observation dict to tensor."""
        obs_list = []
        for key in sorted(obs.keys()):
            if key != "action_mask":
                obs_list.append(obs[key].flatten())
        obs_array = np.concatenate(obs_list)
        return torch.tensor(obs_array, dtype=torch.float32, device=self.device)
    
    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            "counterfactual_value_nets": {
                pid: net.state_dict() for pid, net in self.counterfactual_value_nets.items()
            },
            "strategy_nets": {
                pid: net.state_dict() for pid, net in self.strategy_nets.items()
            },
            "regrets": dict(self.regrets),
            "strategy_sums": dict(self.strategy_sums),
            "traversal_count": self.traversal_count,
        }, path)
    
    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        for pid in [0, 1]:
            self.counterfactual_value_nets[pid].load_state_dict(
                checkpoint["counterfactual_value_nets"][pid]
            )
            self.strategy_nets[pid].load_state_dict(
                checkpoint["strategy_nets"][pid]
            )
        self.regrets = checkpoint.get("regrets", {0: {}, 1: {}})
        self.strategy_sums = checkpoint.get("strategy_sums", {0: {}, 1: {}})
        self.traversal_count = checkpoint.get("traversal_count", 0)
