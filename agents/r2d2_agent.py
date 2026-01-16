"""R2D2 (Recurrent Replay Distributed DQN) agent components."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple, Deque
from collections import deque
import random

from engine.state import GameState
from training.utils.network_architectures import (
    create_recurrent_policy_network,
    get_network_arch,
)


class RecurrentQNetwork(nn.Module):
    """Recurrent Q-network with LSTM."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lstm_hidden_size: int = 256,
        lstm_layers: int = 2,
        mlp_layers: List[int] = [128, 128],
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            lstm_hidden_size,
            lstm_layers,
            batch_first=True,
        )
        lstm_output_dim = lstm_hidden_size
        
        # MLP after LSTM
        layers = []
        prev_dim = lstm_output_dim
        for dim in mlp_layers:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim) or (batch, input_dim)
            hidden: LSTM hidden state
        
        Returns:
            (Q-values, new_hidden_state)
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        lstm_out, hidden = self.lstm(x, hidden)
        # Take last output
        lstm_out = lstm_out[:, -1, :]
        q_values = self.mlp(lstm_out)
        return q_values, hidden


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer for R2D2."""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)
        self.priorities: Deque[float] = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def add(self, experience: Tuple, priority: Optional[float] = None):
        """Add experience with priority."""
        if priority is None:
            priority = self.max_priority
        self.buffer.append(experience)
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Tuple], np.ndarray, np.ndarray]:
        """
        Sample batch with priorities.
        
        Returns:
            (batch, indices, importance_weights)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Compute sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]
        
        # Compute importance weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return len(self.buffer)


class R2D2Agent:
    """
    R2D2 (Recurrent Replay Distributed DQN) agent.
    
    Features:
    - Recurrent Q-network with LSTM
    - Prioritized experience replay
    - N-step returns
    - Action masking support
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        """
        Initialize R2D2 agent.
        
        Args:
            observation_dim: Dimension of observation space
            action_dim: Size of action space
            config: Configuration dict
            device: Device to run on
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        # Hyperparameters
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.gamma = config.get("gamma", 0.99)
        self.n_step = config.get("n_step", 5)
        self.tau = config.get("tau", 1.0)  # Target network update
        self.batch_size = config.get("batch_size", 32)
        self.buffer_size = config.get("buffer_size", 100000)
        
        # Get recurrent architecture
        recurrent_arch = get_network_arch(config, "recurrent")
        lstm_hidden_size = recurrent_arch.get("lstm_hidden_size", 256)
        lstm_layers = recurrent_arch.get("lstm_layers", 2)
        mlp_layers = recurrent_arch.get("mlp_layers", [128, 128])
        
        # Networks
        self.q_network = RecurrentQNetwork(
            observation_dim, action_dim, lstm_hidden_size, lstm_layers, mlp_layers
        ).to(self.device)
        self.target_network = RecurrentQNetwork(
            observation_dim, action_dim, lstm_hidden_size, lstm_layers, mlp_layers
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.buffer_size,
            alpha=config.get("replay_alpha", 0.6),
            beta=config.get("replay_beta", 0.4),
        )
        
        # Training state
        self.step_count = 0
        self.current_state: Optional[GameState] = None
        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    
    def predict(
        self,
        obs: Dict[str, np.ndarray],
        action_mask: np.ndarray,
        deterministic: bool = False,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict action given observation.
        
        Args:
            obs: Observation dict
            action_mask: Action mask
            deterministic: Whether to use deterministic policy
            hidden: LSTM hidden state
        
        Returns:
            (action, new_hidden_state)
        """
        obs_tensor = self._obs_to_tensor(obs).unsqueeze(0).unsqueeze(0)  # (1, 1, obs_dim)
        
        with torch.no_grad():
            q_values, new_hidden = self.q_network(obs_tensor, hidden)
            q_values = q_values.cpu().numpy().flatten()
        
        # Apply action mask
        masked_q_values = q_values.copy()
        masked_q_values[~action_mask] = -1e9
        
        # Select action
        if deterministic:
            action = int(np.argmax(masked_q_values))
        else:
            # Epsilon-greedy (should be handled by training)
            action = int(np.argmax(masked_q_values))
        
        return action, new_hidden
    
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
        hidden: Optional[Tuple] = None,
        next_hidden: Optional[Tuple] = None,
    ):
        """Add experience to replay buffer."""
        experience = (obs, action, reward, next_obs, done, action_mask, hidden, next_hidden)
        self.replay_buffer.add(experience, priority=self.replay_buffer.max_priority)
    
    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        # Prepare batch
        obs_batch = torch.stack([self._obs_to_tensor(exp[0]) for exp in batch])
        action_batch = torch.tensor([exp[1] for exp in batch], device=self.device)
        reward_batch = torch.tensor([exp[2] for exp in batch], device=self.device, dtype=torch.float32)
        next_obs_batch = torch.stack([self._obs_to_tensor(exp[3]) for exp in batch])
        done_batch = torch.tensor([exp[4] for exp in batch], device=self.device)
        weights_batch = torch.tensor(weights, device=self.device, dtype=torch.float32)
        
        # Compute Q-values
        obs_batch = obs_batch.unsqueeze(1)  # (batch, 1, obs_dim)
        q_values, _ = self.q_network(obs_batch)
        q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values (with target network)
        next_obs_batch = next_obs_batch.unsqueeze(1)
        with torch.no_grad():
            next_q_values, _ = self.target_network(next_obs_batch)
            next_q_value = next_q_values.max(1)[0]
            target = reward_batch + (1 - done_batch.float()) * self.gamma * next_q_value
        
        # TD error
        td_error = target - q_value
        loss = (weights_batch * td_error.pow(2)).mean()
        
        # Update priorities
        priorities = (td_error.abs().cpu().numpy() + 1e-6)
        self.replay_buffer.update_priorities(indices, priorities)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.step_count += 1
        
        # Update target network
        if self.step_count % int(1.0 / self.tau) == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
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
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
        }, path)
    
    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step_count = checkpoint.get("step_count", 0)
