"""Neural Fictitious Self-Play (NFSP) agent for imperfect information games."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import random

from engine.state import GameState
from training.utils.network_architectures import (
    create_deep_policy_network,
    create_value_network,
    get_network_arch,
)


class AverageStrategyNetwork(nn.Module):
    """Network for learning average strategy (Nash equilibrium)."""
    
    def __init__(self, input_dim: int, output_dim: int, net_arch: List[int] = [256, 256, 128]):
        super().__init__()
        self.network = create_deep_policy_network(input_dim, output_dim, net_arch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class BestResponseNetwork(nn.Module):
    """Network for learning best response to opponent strategies."""
    
    def __init__(self, input_dim: int, output_dim: int, net_arch: List[int] = [256, 256, 128]):
        super().__init__()
        self.policy_net = create_deep_policy_network(input_dim, output_dim, net_arch)
        self.value_net = create_value_network(input_dim, net_arch)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning policy and value."""
        policy = self.policy_net(x)
        value = self.value_net(x)
        return policy, value


class ReservoirBuffer:
    """Reservoir sampling buffer for opponent strategies."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.size = 0
    
    def add(self, item: Any):
        """Add item using reservoir sampling."""
        if self.size < self.max_size:
            self.buffer.append(item)
            self.size += 1
        else:
            # Reservoir sampling: replace with probability max_size / (size + 1)
            idx = random.randint(0, self.size)
            if idx < self.max_size:
                self.buffer[idx] = item
            self.size += 1
    
    def sample(self, n: int = 1) -> List[Any]:
        """Sample n items uniformly."""
        if len(self.buffer) == 0:
            return []
        return random.sample(list(self.buffer), min(n, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


class NFSPAgent:
    """
    Neural Fictitious Self-Play agent.
    
    NFSP learns an approximate Nash equilibrium by:
    1. Training a best response network against opponent strategies
    2. Training an average strategy network to match the best response
    3. Using reservoir sampling to maintain diverse opponent strategies
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        """
        Initialize NFSP agent.
        
        Args:
            observation_dim: Dimension of observation space
            action_dim: Size of action space (52 for Pişti)
            config: Configuration dict with hyperparameters
            device: Device to run on
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        # Hyperparameters
        self.anticipatory_param = config.get("anticipatory_param", 0.1)  # η
        self.average_strategy_update_freq = config.get("average_strategy_update_freq", 1000)
        self.reservoir_buffer_size = config.get("reservoir_buffer_size", 10000)
        self.learning_rate = config.get("learning_rate", 1e-4)
        
        # Get network architectures
        nfsp_arch = get_network_arch(config, "nfsp")
        avg_strategy_arch = nfsp_arch.get("average_strategy", [256, 256, 128])
        best_response_arch = nfsp_arch.get("best_response", [256, 256, 128])
        
        # Networks
        self.average_strategy_net = AverageStrategyNetwork(
            observation_dim, action_dim, avg_strategy_arch
        ).to(self.device)
        self.best_response_net = BestResponseNetwork(
            observation_dim, action_dim, best_response_arch
        ).to(self.device)
        
        # Optimizers
        self.avg_strategy_optimizer = optim.Adam(
            self.average_strategy_net.parameters(), lr=self.learning_rate
        )
        self.best_response_optimizer = optim.Adam(
            self.best_response_net.parameters(), lr=self.learning_rate
        )
        
        # Reservoir buffer for opponent strategies
        self.reservoir_buffer = ReservoirBuffer(self.reservoir_buffer_size)
        
        # Training state
        self.step_count = 0
        self.training_mode = "best_response"  # or "average_strategy"
        self.current_state: Optional[GameState] = None
        
        # Experience buffers
        self.best_response_buffer: List[Tuple] = []
        self.average_strategy_buffer: List[Tuple] = []
    
    def predict(
        self,
        obs: Dict[str, np.ndarray],
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """
        Predict action given observation.
        
        Args:
            obs: Observation dict
            action_mask: Action mask (boolean array)
            deterministic: Whether to use deterministic policy
        
        Returns:
            Action index
        """
        # Choose network based on anticipatory parameter
        use_average_strategy = np.random.random() < (1 - self.anticipatory_param)
        
        if use_average_strategy:
            network = self.average_strategy_net
        else:
            network = self.best_response_net.policy_net
        
        # Convert observation to tensor
        obs_tensor = self._obs_to_tensor(obs)
        
        # Get logits
        with torch.no_grad():
            logits = network(obs_tensor).cpu().numpy().flatten()
        
        # Apply action mask
        masked_logits = logits.copy()
        masked_logits[~action_mask] = -1e9
        
        # Sample or take argmax
        if deterministic:
            action = int(np.argmax(masked_logits))
        else:
            # Softmax and sample
            probs = np.exp(masked_logits - np.max(masked_logits))
            probs = probs / probs.sum()
            action = int(np.random.choice(self.action_dim, p=probs))
        
        return action
    
    def update_state(self, state: GameState):
        """Update internal game state."""
        self.current_state = state
    
    def add_experience(
        self,
        obs: Dict[str, np.ndarray],
        action: int,
        reward: float,
        next_obs: Dict[str, np.ndarray],
        done: bool,
        action_mask: np.ndarray,
    ):
        """Add experience to appropriate buffer."""
        experience = (obs, action, reward, next_obs, done, action_mask)
        
        if self.training_mode == "best_response":
            self.best_response_buffer.append(experience)
        else:
            self.average_strategy_buffer.append(experience)
    
    def train_step(self, batch_size: int = 32):
        """Perform one training step."""
        self.step_count += 1
        
        # Alternate between best response and average strategy training
        if self.step_count % self.average_strategy_update_freq == 0:
            self.training_mode = "average_strategy"
        else:
            self.training_mode = "best_response"
        
        if self.training_mode == "best_response":
            self._train_best_response(batch_size)
        else:
            self._train_average_strategy(batch_size)
    
    def _train_best_response(self, batch_size: int):
        """Train best response network."""
        if len(self.best_response_buffer) < batch_size:
            return
        
        batch = random.sample(self.best_response_buffer, batch_size)
        
        obs_batch = torch.stack([self._obs_to_tensor(exp[0]) for exp in batch])
        action_batch = torch.tensor([exp[1] for exp in batch], device=self.device)
        reward_batch = torch.tensor([exp[2] for exp in batch], device=self.device, dtype=torch.float32)
        next_obs_batch = torch.stack([self._obs_to_tensor(exp[3]) for exp in batch])
        done_batch = torch.tensor([exp[4] for exp in batch], device=self.device)
        
        # Compute Q-values
        policy_logits, value = self.best_response_net(obs_batch)
        _, next_value = self.best_response_net(next_obs_batch)
        
        # TD target
        target = reward_batch + (1 - done_batch.float()) * 0.99 * next_value.squeeze()
        
        # Policy loss (cross-entropy)
        policy_loss = nn.functional.cross_entropy(policy_logits, action_batch)
        
        # Value loss (MSE)
        value_loss = nn.functional.mse_loss(value.squeeze(), target.detach())
        
        # Total loss
        loss = policy_loss + value_loss
        
        # Update
        self.best_response_optimizer.zero_grad()
        loss.backward()
        self.best_response_optimizer.step()
    
    def _train_average_strategy(self, batch_size: int):
        """Train average strategy network."""
        if len(self.average_strategy_buffer) < batch_size:
            return
        
        batch = random.sample(self.average_strategy_buffer, batch_size)
        
        obs_batch = torch.stack([self._obs_to_tensor(exp[0]) for exp in batch])
        action_batch = torch.tensor([exp[1] for exp in batch], device=self.device)
        
        # Get best response policy
        with torch.no_grad():
            best_response_policy, _ = self.best_response_net(obs_batch)
            best_response_probs = nn.functional.softmax(best_response_policy, dim=1)
        
        # Average strategy should match best response
        avg_strategy_logits = self.average_strategy_net(obs_batch)
        avg_strategy_probs = nn.functional.softmax(avg_strategy_logits, dim=1)
        
        # KL divergence loss (average strategy should match best response)
        loss = nn.functional.kl_div(
            nn.functional.log_softmax(avg_strategy_logits, dim=1),
            best_response_probs,
            reduction="batchmean",
        )
        
        # Update
        self.avg_strategy_optimizer.zero_grad()
        loss.backward()
        self.avg_strategy_optimizer.step()
    
    def _obs_to_tensor(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        """Convert observation dict to tensor."""
        # Flatten observation dict
        obs_list = []
        for key in sorted(obs.keys()):
            if key != "action_mask":  # Exclude action mask from network input
                obs_list.append(obs[key].flatten())
        obs_array = np.concatenate(obs_list)
        return torch.tensor(obs_array, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            "average_strategy_net": self.average_strategy_net.state_dict(),
            "best_response_net": self.best_response_net.state_dict(),
            "avg_strategy_optimizer": self.avg_strategy_optimizer.state_dict(),
            "best_response_optimizer": self.best_response_optimizer.state_dict(),
            "step_count": self.step_count,
        }, path)
    
    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.average_strategy_net.load_state_dict(checkpoint["average_strategy_net"])
        self.best_response_net.load_state_dict(checkpoint["best_response_net"])
        self.avg_strategy_optimizer.load_state_dict(checkpoint["avg_strategy_optimizer"])
        self.best_response_optimizer.load_state_dict(checkpoint["best_response_optimizer"])
        self.step_count = checkpoint.get("step_count", 0)
