"""Custom callbacks for SB3 training."""

import os
from typing import Any, Dict, Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.vec_env import VecEnv
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
        self._unwrapped_env_cache = None  # Cache unwrapped env for faster access
    
    def _get_unwrapped_env(self):
        """Get unwrapped environment from vectorized env, with caching."""
        if self._unwrapped_env_cache is not None:
            return self._unwrapped_env_cache
        
        if hasattr(self.eval_env, 'envs') and len(self.eval_env.envs) > 0:
            env = self.eval_env.envs[0]
            # Handle ActionMasker wrapper if present
            if hasattr(env, 'env'):
                env = env.env
            if hasattr(env, 'unwrapped'):
                env = env.unwrapped
            if hasattr(env, 'engine'):
                self._unwrapped_env_cache = env
                return env
        return None

    def _on_step(self) -> bool:
        """Called at each step."""
        if self.n_calls % self.eval_freq == 0:
            # Run evaluation
            wins = 0
            total_score_diff = 0.0
            episodes_with_scores = 0  # Track how many episodes we successfully got scores from
            
            for episode_idx in range(self.n_eval_episodes):
                # Handle vectorized environment reset
                # DummyVecEnv.reset() returns (observations, infos) tuple
                reset_result = self.eval_env.reset()
                # Handle different return formats
                if isinstance(reset_result, tuple):
                    if len(reset_result) == 2:
                        obs_vec, _ = reset_result
                    else:
                        # Handle case where more than 2 values returned
                        obs_vec = reset_result[0]
                    # Extract first element from vectorized results
                    obs = obs_vec[0] if isinstance(obs_vec, (list, np.ndarray)) and len(obs_vec) > 0 else obs_vec
                else:
                    obs = reset_result
                
                done = False
                episode_reward = 0.0
                final_info = None  # Store info from final step
                score_diff = None  # Initialize score_diff
                max_steps = 1000  # Max steps per episode to prevent infinite loops
                step_count = 0
                episode_ended_with_invalid_action = False
                
                while not done and step_count < max_steps:
                    step_count += 1
                    
                    # Get action mask if available
                    action_mask = None
                    if isinstance(obs, dict) and "action_mask" in obs:
                        action_mask = obs["action_mask"]
                    
                    # Predict action (with action mask if available and model supports it)
                    if hasattr(self.model, 'predict') and action_mask is not None:
                        # For MaskablePPO, action mask should be handled by ActionMasker wrapper
                        # But we can still pass it if the model supports it
                        try:
                            action, _ = self.model.predict(obs, deterministic=True)
                        except:
                            # Fallback: use action mask manually
                            action, _ = self.model.predict(obs, deterministic=True)
                    else:
                        action, _ = self.model.predict(obs, deterministic=True)
                    
                    # Convert action to Python int (model.predict may return numpy array)
                    if isinstance(action, np.ndarray):
                        action = int(action.item() if action.size == 1 else action[0])
                    else:
                        action = int(action)
                    
                    # Handle vectorized environment step
                    step_result = self.eval_env.step([action])  # Vectorized envs expect list
                    if isinstance(step_result, tuple) and len(step_result) >= 4:
                        obs_vec, reward_vec, terminated_vec, truncated_vec = step_result[:4]
                        info_vec = step_result[4] if len(step_result) > 4 else [{}]
                        
                        # Extract from vectorized results with proper type conversion
                        obs = obs_vec[0] if isinstance(obs_vec, (list, np.ndarray)) and len(obs_vec) > 0 else obs_vec
                        reward = reward_vec[0] if isinstance(reward_vec, (list, np.ndarray)) else reward_vec
                        
                        # CRITICAL: Ensure terminated and truncated are booleans
                        term_val = terminated_vec[0] if isinstance(terminated_vec, (list, np.ndarray)) else terminated_vec
                        trunc_val = truncated_vec[0] if isinstance(truncated_vec, (list, np.ndarray)) else truncated_vec
                        
                        # Convert to boolean, handling edge cases
                        if isinstance(term_val, (dict, str)):
                            # If it's a dict or string, it's not a valid boolean - treat as False
                            terminated = False
                        else:
                            terminated = bool(term_val)
                        
                        if isinstance(trunc_val, (dict, str)):
                            truncated = False
                        else:
                            truncated = bool(trunc_val)
                        
                        info = info_vec[0] if isinstance(info_vec, list) and len(info_vec) > 0 else (info_vec if isinstance(info_vec, dict) else {})
                    else:
                        # Fallback for non-vectorized
                        obs, reward, term_val, trunc_val, info = step_result
                        # Ensure booleans
                        terminated = bool(term_val) if not isinstance(term_val, (dict, str)) else False
                        truncated = bool(trunc_val) if not isinstance(trunc_val, (dict, str)) else False
                    
                    # Check for invalid action in info
                    if isinstance(info, dict) and info.get("invalid_action", False):
                        episode_ended_with_invalid_action = True
                        # Don't mark as done - continue until natural completion or max steps
                        done = False
                        
                    else:
                        done = terminated or truncated
                    
                    episode_reward += reward
                    
                    # Store info from final step (only if actually done, not invalid action)
                    if done and not episode_ended_with_invalid_action:
                        final_info = info
                        # Try to extract score immediately when done (before loop ends)
                        # This ensures we get it while state is still accessible
                        if score_diff is None:  # Only if not already extracted
                            # ALWAYS extract from engine state - info dict is empty in vectorized envs
                            # Vectorized environments (DummyVecEnv) don't properly propagate info dict
                            unwrapped_env = self._get_unwrapped_env()
                            if unwrapped_env is not None and hasattr(unwrapped_env, 'engine'):
                                if unwrapped_env.engine.state is not None:
                                    try:
                                        if unwrapped_env.engine.state.is_terminal():
                                            scores = unwrapped_env.engine.state.get_final_scores()
                                            score_diff = scores[0] - scores[1]
                                    except Exception as e:
                                        if self.verbose > 0 and episode_idx == 0:
                                            print(f"[WARNING] Error extracting scores from engine: {e}")
                                        pass  # Will try again after loop
                
                # Handle max steps reached
                if step_count >= max_steps and not done:
                    if self.verbose > 0 and episode_idx == 0 and not hasattr(self, '_warned_max_steps'):
                        print(f"[WARNING] Episode {episode_idx} reached max steps ({max_steps}) without completing")
                        self._warned_max_steps = True
                    # Try to extract score anyway if state is terminal
                    unwrapped_env = self._get_unwrapped_env()
                    if unwrapped_env is not None and hasattr(unwrapped_env, 'engine'):
                        if unwrapped_env.engine.state is not None and unwrapped_env.engine.state.is_terminal():
                            try:
                                scores = unwrapped_env.engine.state.get_final_scores()
                                score_diff = scores[0] - scores[1]
                            except Exception:
                                pass
                
                # Extract score_diff if not already extracted during loop
                # Only extract if episode completed naturally (terminal state), not from invalid actions
                if score_diff is None:
                    unwrapped_env = self._get_unwrapped_env()
                    
                    # Only extract if state is actually terminal (natural completion)
                    if unwrapped_env is not None and hasattr(unwrapped_env, 'engine'):
                        if unwrapped_env.engine.state is not None:
                            try:
                                if unwrapped_env.engine.state.is_terminal():
                                    # State is terminal - extract scores
                                    scores = unwrapped_env.engine.state.get_final_scores()
                                    score_diff = scores[0] - scores[1]
                                elif episode_ended_with_invalid_action:
                                    # Episode ended with invalid action - skip this episode
                                    if self.verbose > 0 and episode_idx == 0 and not hasattr(self, '_warned_invalid_action'):
                                        print(f"[WARNING] Episode {episode_idx} ended with invalid action - skipping")
                                        self._warned_invalid_action = True
                                    score_diff = None  # Explicitly set to None to skip
                                else:
                                    # State not terminal and not invalid action - might be truncation
                                    if self.verbose > 0 and episode_idx == 0 and not hasattr(self, '_warned_truncation'):
                                        print(f"[WARNING] Episode {episode_idx} ended but state is not terminal (likely truncation) - skipping")
                                        self._warned_truncation = True
                                    score_diff = None  # Skip truncated episodes
                            except Exception as e:
                                if self.verbose > 0 and episode_idx == 0 and not hasattr(self, '_warned_score_extraction'):
                                    print(f"[WARNING] Error extracting scores from engine state: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    self._warned_score_extraction = True
                
                # Count episode only if we successfully extracted score (natural completion)
                if score_diff is not None:
                    episodes_with_scores += 1
                    total_score_diff += score_diff
                    if score_diff > 0:
                        wins += 1
                else:
                    # Log why episode was skipped (only first time to avoid spam)
                    if self.verbose > 0 and episode_idx == 0 and not hasattr(self, '_warned_episode_skip'):
                        reason = []
                        if episode_ended_with_invalid_action:
                            reason.append("invalid action")
                        if step_count >= max_steps:
                            reason.append("max steps reached")
                        unwrapped_env = self._get_unwrapped_env()
                        if unwrapped_env is not None and hasattr(unwrapped_env, 'engine'):
                            if unwrapped_env.engine.state is not None and not unwrapped_env.engine.state.is_terminal():
                                reason.append("state not terminal")
                        reason_str = ", ".join(reason) if reason else "unknown"
                        print(f"[INFO] Episode {episode_idx} skipped ({reason_str}) - only counting naturally completed episodes")
                        self._warned_episode_skip = True
            
            # Calculate win rate only from episodes with scores
            if episodes_with_scores > 0:
                win_rate = wins / episodes_with_scores
                avg_score_diff = total_score_diff / episodes_with_scores
            else:
                win_rate = 0.0
                avg_score_diff = 0.0
                if self.verbose > 0:
                    print(f"[ERROR] No episodes completed successfully! All {self.n_eval_episodes} episodes failed to extract scores.")
                    print(f"  This suggests an issue with the environment or evaluation setup.")
            
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
            checkpoint_path: Base path for snapshots (should be snapshots_dir)
            algorithm: Algorithm type ("PPO", "DQN", etc.)
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


class OpponentSwitchCallback(BaseCallback):
    """Callback to switch opponent at specified timestep."""
    
    def __init__(
        self,
        env: VecEnv,
        new_opponent,
        switch_at_timestep: int,
        verbose: int = 0,
    ):
        """
        Initialize opponent switch callback.
        
        Args:
            env: Vectorized environment
            new_opponent: New opponent to switch to
            switch_at_timestep: Timestep at which to switch
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.env = env
        self.new_opponent = new_opponent
        self.switch_at_timestep = switch_at_timestep
        self.switched = False
    
    def _on_step(self) -> bool:
        """Called at each step."""
        if not self.switched and self.num_timesteps >= self.switch_at_timestep:
            # Switch opponent in environment
            if hasattr(self.env, 'envs') and len(self.env.envs) > 0:
                env_unwrapped = self.env.envs[0]
                if hasattr(env_unwrapped, 'unwrapped'):
                    env_unwrapped = env_unwrapped.unwrapped
                if hasattr(env_unwrapped, 'opponent'):
                    env_unwrapped.opponent = self.new_opponent
                    self.switched = True
                    if self.verbose > 0:
                        print(f"Switched opponent at timestep {self.num_timesteps}")
        return True


class CurriculumCallback(BaseCallback):
    """Callback to manage curriculum learning with multiple opponent phases."""
    
    def __init__(
        self,
        env: VecEnv,
        curriculum_phases: list,
        opponent_factory: callable,
        verbose: int = 0,
    ):
        """
        Initialize curriculum callback.
        
        Args:
            env: Vectorized environment
            curriculum_phases: List of curriculum phases, each as a dict:
                - "timestep": int - Timestep when this phase starts
                - "opponent_type": str - Type of opponent ("random", "greedy", etc.)
                - "opponent_kwargs": dict - Keyword arguments for opponent factory
            opponent_factory: Function that creates opponents: (opponent_type, **kwargs) -> opponent
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.env = env
        self.curriculum_phases = sorted(curriculum_phases, key=lambda x: x.get("timestep", 0))
        self.opponent_factory = opponent_factory
        self.current_phase_idx = 0
        self.phases_switched = set()  # Track which phases we've already switched to
        
        if self.verbose > 0:
            print(f"Initialized curriculum with {len(self.curriculum_phases)} phases")
            for i, phase in enumerate(self.curriculum_phases):
                print(f"  Phase {i+1}: Start at {phase.get('timestep', 0)} steps, "
                      f"opponent={phase.get('opponent_type', 'unknown')}")
    
    def _get_unwrapped_env(self):
        """Get unwrapped environment from vectorized env."""
        if hasattr(self.env, 'envs') and len(self.env.envs) > 0:
            env = self.env.envs[0]
            # Handle ActionMasker wrapper if present
            if hasattr(env, 'env'):
                env = env.env
            if hasattr(env, 'unwrapped'):
                env = env.unwrapped
            return env
        return None
    
    def _switch_opponent(self, opponent):
        """Switch opponent in all environments."""
        env = self._get_unwrapped_env()
        if env is not None and hasattr(env, 'opponent'):
            env.opponent = opponent
            # Also switch in other vectorized environments if present
            if hasattr(self.env, 'envs'):
                for env_wrapper in self.env.envs:
                    unwrapped = env_wrapper
                    if hasattr(unwrapped, 'env'):
                        unwrapped = unwrapped.env
                    if hasattr(unwrapped, 'unwrapped'):
                        unwrapped = unwrapped.unwrapped
                    if hasattr(unwrapped, 'opponent'):
                        unwrapped.opponent = opponent
            return True
        return False
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Check if we need to switch to next phase
        for phase_idx, phase in enumerate(self.curriculum_phases):
            phase_timestep = phase.get("timestep", 0)
            
            # Skip if we've already switched to this phase
            if phase_idx in self.phases_switched:
                continue
            
            # Switch if we've reached the timestep for this phase
            if self.num_timesteps >= phase_timestep:
                opponent_type = phase.get("opponent_type", "random")
                opponent_kwargs = phase.get("opponent_kwargs", {})
                
                # Create new opponent using factory
                try:
                    new_opponent = self.opponent_factory(opponent_type, **opponent_kwargs)
                    
                    # Switch opponent in environment
                    if self._switch_opponent(new_opponent):
                        self.phases_switched.add(phase_idx)
                        self.current_phase_idx = phase_idx
                        
                        if self.verbose > 0:
                            print(
                                f"[Curriculum] Switched to Phase {phase_idx + 1} at timestep {self.num_timesteps}: "
                                f"opponent={opponent_type}"
                            )
                            if opponent_kwargs:
                                print(f"  Opponent kwargs: {opponent_kwargs}")
                    else:
                        if self.verbose > 0:
                            print(
                                f"[Curriculum] Warning: Failed to switch opponent at timestep {self.num_timesteps}"
                            )
                except Exception as e:
                    if self.verbose > 0:
                        print(
                            f"[Curriculum] Error creating opponent for phase {phase_idx + 1}: {e}"
                        )
        
        return True
