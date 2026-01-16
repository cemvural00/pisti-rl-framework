"""Opponent wrappers for self-play and league training."""

from typing import Dict, List, Optional
import numpy as np
import os
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecEnv

from agents.baselines import RandomValidAgent


class FrozenCheckpointOpponent:
    """Opponent that loads a frozen SB3 model checkpoint."""

    def __init__(self, checkpoint_path: str, algorithm: str = "PPO"):
        """
        Initialize frozen checkpoint opponent.
        
        Args:
            checkpoint_path: Path to saved model checkpoint
            algorithm: Algorithm type ("PPO" or "DQN")
        """
        self.checkpoint_path = checkpoint_path
        self.algorithm = algorithm
        
        if algorithm == "PPO":
            self.model = PPO.load(checkpoint_path)
        elif algorithm == "DQN":
            self.model = DQN.load(checkpoint_path)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.model.set_env(None)  # No training environment

    def predict(self, obs: Dict, action_mask: np.ndarray) -> int:
        """
        Predict action using frozen model.
        
        Args:
            obs: Observation dict
            action_mask: Boolean array of legal actions
        
        Returns:
            Card ID (0-51)
        """
        # Convert obs dict to format expected by SB3
        # For MaskablePPO, we need to pass action_mask
        if hasattr(self.model, "predict"):
            # Convert dict obs to vector if needed
            if isinstance(obs, dict):
                # Flatten dict to vector (or use appropriate format)
                # For now, assume model expects dict
                action, _ = self.model.predict(obs, deterministic=True)
            else:
                action, _ = self.model.predict(obs, deterministic=True)
            
            action = int(action)
            
            # Ensure action is legal
            legal_actions = np.where(action_mask)[0]
            if action in legal_actions:
                return action
            else:
                # Fallback to random legal action
                if len(legal_actions) > 0:
                    return int(np.random.choice(legal_actions))
                return 0
        
        return 0


class OpponentPool:
    """
    Opponent pool for league training.
    
    Samples from a pool of past checkpoints.
    """

    def __init__(self, pool_size: int = 5):
        """
        Initialize opponent pool.
        
        Args:
            pool_size: Maximum number of opponents in pool
        """
        self.pool_size = pool_size
        self.opponents: List[FrozenCheckpointOpponent] = []
        self.checkpoint_paths: List[str] = []

    def add_checkpoint(self, checkpoint_path: str, algorithm: str = "PPO"):
        """
        Add a checkpoint to the pool.
        
        Args:
            checkpoint_path: Path to checkpoint
            algorithm: Algorithm type
        """
        try:
            opponent = FrozenCheckpointOpponent(checkpoint_path, algorithm)
            self.opponents.append(opponent)
            self.checkpoint_paths.append(checkpoint_path)
            
            # Keep only most recent pool_size opponents
            if len(self.opponents) > self.pool_size:
                self.opponents.pop(0)
                self.checkpoint_paths.pop(0)
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {checkpoint_path}: {e}")

    def sample(self) -> Optional[FrozenCheckpointOpponent]:
        """
        Sample a random opponent from the pool.
        
        Returns:
            FrozenCheckpointOpponent or None if pool is empty
        """
        if len(self.opponents) == 0:
            return None
        return np.random.choice(self.opponents)

    def predict(self, obs: Dict, action_mask: np.ndarray) -> int:
        """
        Predict action using a sampled opponent.
        
        Args:
            obs: Observation dict
            action_mask: Boolean array of legal actions
        
        Returns:
            Card ID (0-51)
        """
        opponent = self.sample()
        if opponent is None:
            # Fallback to random agent
            random_agent = RandomValidAgent()
            return random_agent.predict(obs, action_mask)
        return opponent.predict(obs, action_mask)


class SelfPlayOpponent:
    """
    Self-play opponent that uses the current training policy.
    
    This is a wrapper that allows the training agent to play against itself.
    """

    def __init__(self, model=None):
        """
        Initialize self-play opponent.
        
        Args:
            model: Current training model (will be updated during training)
        """
        self.model = model

    def set_model(self, model):
        """Update the model used for self-play."""
        self.model = model

    def predict(self, obs: Dict, action_mask: np.ndarray) -> int:
        """
        Predict action using current training model.
        
        Args:
            obs: Observation dict
            action_mask: Boolean array of legal actions
        
        Returns:
            Card ID (0-51)
        """
        if self.model is None:
            # Fallback to random
            random_agent = RandomValidAgent()
            return random_agent.predict(obs, action_mask)
        
        # Use model to predict
        if hasattr(self.model, "predict"):
            action, _ = self.model.predict(obs, deterministic=False)
            action = int(action)
            
            # Ensure action is legal
            legal_actions = np.where(action_mask)[0]
            if action in legal_actions:
                return action
            else:
                # Fallback to random legal action
                if len(legal_actions) > 0:
                    return int(np.random.choice(legal_actions))
                return 0
        
        return 0
