"""Custom callbacks for SB3 training."""

import os
from typing import Any, Dict, Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from agents.opponents import OpponentPool
from training.metadata import ModelMetadata, save_metadata, update_metadata


class CheckpointCallback(BaseCallback):
    """Callback to save checkpoints periodically with metadata."""

    def __init__(
        self,
        save_path: str,
        save_freq: int,
        name_prefix: str = "pisti_model",
        metadata: Optional[ModelMetadata] = None,
        verbose: int = 0,
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            save_path: Directory to save checkpoints
            save_freq: Save every N steps
            name_prefix: Prefix for checkpoint filenames
            metadata: ModelMetadata instance to save with checkpoints
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.name_prefix = name_prefix
        self.metadata = metadata
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """Called at each step."""
        if self.n_calls % self.save_freq == 0:
            checkpoint_name = f"{self.name_prefix}_{self.num_timesteps}_steps"
            checkpoint_path = os.path.join(self.save_path, checkpoint_name)
            
            # Save model
            self.model.save(checkpoint_path)
            
            # Save metadata if available
            if self.metadata is not None:
                # Update metadata with current timestep
                updated_metadata = update_metadata(
                    self.metadata,
                    training_end_time=None,  # Keep original end time
                )
                save_metadata(updated_metadata, self.save_path, checkpoint_name)
            
            if self.verbose > 0:
                print(f"Saved checkpoint to {checkpoint_path}")
                if self.metadata is not None:
                    print(f"  Saved metadata to {checkpoint_name}_metadata.json")
        return True


class EvalCallback(BaseCallback):
    """Callback for periodic evaluation against baseline opponents."""

    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int = 10,
        eval_opponents: list = None,
        metadata: Optional[ModelMetadata] = None,
        verbose: int = 0,
    ):
        """
        Initialize evaluation callback.
        
        Args:
            eval_env: Environment for evaluation
            eval_freq: Evaluate every N steps
            n_eval_episodes: Number of episodes per evaluation
            eval_opponents: List of opponent names to evaluate against
            metadata: ModelMetadata instance to update with best scores
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_opponents = eval_opponents or ["random"]
        self.metadata = metadata
        self.best_score = float("-inf")
        self.best_step = 0

    def _on_step(self) -> bool:
        """Called at each step."""
        if self.n_calls % self.eval_freq == 0:
            # Run evaluation
            wins = 0
            total_score_diff = 0.0
            
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                episode_reward = 0.0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                
                # Check if won (positive final score diff)
                if "score_diff" in info:
                    score_diff = info["score_diff"]
                    total_score_diff += score_diff
                    if score_diff > 0:
                        wins += 1
            
            win_rate = wins / self.n_eval_episodes
            avg_score_diff = total_score_diff / self.n_eval_episodes
            
            # Update best score
            if avg_score_diff > self.best_score:
                self.best_score = avg_score_diff
                self.best_step = self.num_timesteps
                
                # Update metadata if available
                if self.metadata is not None:
                    update_metadata(
                        self.metadata,
                        best_eval_score=self.best_score,
                        best_eval_step=self.best_step,
                    )
            
            # Log metrics
            self.logger.record("eval/win_rate", win_rate)
            self.logger.record("eval/avg_score_diff", avg_score_diff)
            self.logger.record("eval/best_score", self.best_score)
            
            if self.verbose > 0:
                print(
                    f"Evaluation at step {self.num_timesteps}: "
                    f"win_rate={win_rate:.2f}, avg_score_diff={avg_score_diff:.2f} "
                    f"(best: {self.best_score:.2f} at step {self.best_step})"
                )
        
        return True


class LeagueCallback(BaseCallback):
    """Callback to manage opponent pool for self-play."""

    def __init__(
        self,
        opponent_pool: OpponentPool,
        snapshot_freq: int,
        checkpoint_path: str,
        algorithm: str = "PPO",
        verbose: int = 0,
    ):
        """
        Initialize league callback.
        
        Args:
            opponent_pool: OpponentPool instance
            snapshot_freq: Frequency of snapshots to add to pool
            checkpoint_path: Base path for checkpoints
            algorithm: Algorithm type ("PPO" or "DQN")
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.opponent_pool = opponent_pool
        self.snapshot_freq = snapshot_freq
        self.checkpoint_path = checkpoint_path
        self.algorithm = algorithm
        os.makedirs(checkpoint_path, exist_ok=True)

    def _on_step(self) -> bool:
        """Called at each step."""
        if self.n_calls % self.snapshot_freq == 0:
            # Save snapshot
            snapshot_path = os.path.join(
                self.checkpoint_path, f"snapshot_{self.num_timesteps}_steps"
            )
            self.model.save(snapshot_path)
            
            # Add to opponent pool
            self.opponent_pool.add_checkpoint(snapshot_path, self.algorithm)
            
            if self.verbose > 0:
                print(
                    f"Added snapshot to opponent pool: {snapshot_path} "
                    f"(pool size: {len(self.opponent_pool.opponents)})"
                )
        
        return True
