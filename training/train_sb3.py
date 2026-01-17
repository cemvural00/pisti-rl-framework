"""Training script for Pişti RL using Stable-Baselines3."""

import time
_start_time = time.time()

import argparse
import os
import yaml
import json
import traceback
import sys
from typing import Dict, Any
import numpy as np

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy

# Try to import sb3-contrib algorithms
# Import each class separately to handle cases where some may not be available
MaskablePPO = None
RecurrentPPO = None
Rainbow = None
ActionMasker = None

try:
    from sb3_contrib import MaskablePPO
except ImportError:
    pass

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    pass

try:
    from sb3_contrib import Rainbow
except ImportError:
    pass

try:
    from sb3_contrib.common.maskable.utils import ActionMasker
except ImportError:
    pass

SB3_CONTRIB_AVAILABLE = MaskablePPO is not None or RecurrentPPO is not None

from training.utils.network_architectures import get_network_arch

from envs.pisti_gym import PistiGymEnv
from encoding.encoders import (
    ObservationEncoder,
    MultiHotEncoder,
    CNNEncoder,
    FeatureEncoder,
    SequenceEncoder,
)
from agents.baselines import RandomValidAgent, GreedyCaptureAgent, PistiHunterAgent
from agents.probabilistic_agent import ProbabilisticOptimalAgent
from agents.opponents import OpponentPool, SelfPlayOpponent
from training.callbacks import (
    CheckpointCallback,
    EvalCallback,
    LeagueCallback,
    OpponentSwitchCallback,
    CurriculumCallback,
)
from training.metadata import create_metadata, save_metadata, update_metadata
from training.model_storage import (
    ensure_model_directories,
    get_checkpoint_path,
    get_snapshot_path,
)
from datetime import datetime

# Log import time
_import_time = time.time() - _start_time
if _import_time > 1.0:  # Only print if imports took more than 1 second
    print(f"[INFO] Module imports took {_import_time:.2f} seconds")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_encoder(config: Dict[str, Any]) -> ObservationEncoder:
    """Create observation encoder from config."""
    encoder_type = config.get("encoder_type", "MultiHotEncoder")
    encoder_config = config.copy()
    
    if encoder_type == "MultiHotEncoder":
        return MultiHotEncoder(encoder_config)
    elif encoder_type == "CNNEncoder":
        return CNNEncoder(encoder_config)
    elif encoder_type == "FeatureEncoder":
        return FeatureEncoder(encoder_config)
    elif encoder_type == "SequenceEncoder":
        return SequenceEncoder(encoder_config)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def create_opponent(
    opponent_type: str,
    opponent_pool: OpponentPool = None,
    seed: int = 42,
    **kwargs
):
    """
    Create opponent from type and kwargs.
    
    Args:
        opponent_type: Type of opponent ("random", "greedy", "pisti_hunter", "probabilistic", "self_play")
        opponent_pool: Opponent pool for self-play (required if opponent_type == "self_play")
        seed: Random seed for reproducibility
        **kwargs: Additional arguments for opponent (e.g., temperature, max_samples, depth for probabilistic)
    
    Returns:
        Opponent agent instance
    """
    if opponent_type == "probabilistic":
        return ProbabilisticOptimalAgent(
            max_samples=kwargs.get("max_samples", 50),
            depth=kwargs.get("depth", 1),
            temperature=kwargs.get("temperature", 0.0),
            seed=seed
        )
    elif opponent_type == "greedy":
        return GreedyCaptureAgent()
    elif opponent_type == "pisti_hunter":
        return PistiHunterAgent()
    elif opponent_type == "self_play":
        if opponent_pool is None:
            raise ValueError("OpponentPool required for self_play opponent")
        return opponent_pool
    else:  # default to random
        return RandomValidAgent()


def create_opponent_from_config(training_config: Dict[str, Any], opponent_pool: OpponentPool = None):
    """Create opponent from config (legacy function for backward compatibility)."""
    opponent_config = training_config.get("opponent", {})
    opponent_type = opponent_config.get("type", "random")
    seed = training_config.get("seed", 42)
    
    if opponent_type == "probabilistic":
        prob_config = opponent_config.get("probabilistic_config", {})
        kwargs = {
            "max_samples": prob_config.get("max_samples", 50),
            "depth": prob_config.get("depth", 1),
            "temperature": prob_config.get("temperature", 0.0),
        }
        return create_opponent(opponent_type, opponent_pool, seed, **kwargs)
    else:
        return create_opponent(opponent_type, opponent_pool, seed)


def make_env(config: Dict[str, Any], opponent=None, seed: int = None):
    """Create environment function for vec_env."""
    encoder = create_encoder(config.get("encoding", {}))
    reward_config = config.get("reward", {})
    game_config = config.get("game", {})
    
    def _init():
        env = PistiGymEnv(
            encoder=encoder,
            reward_config=reward_config,
            game_config=game_config,
            opponent=opponent,
            seed=seed,
        )
        return env
    
    return _init


def train(config_path: str):
    """Main training function."""
    print("[DEBUG] Starting train() function...")
    # Load config
    config = load_config(config_path)
    print("[DEBUG] Config loaded successfully")
    training_config = config.get("training", {})
    encoding_config = config.get("encoding", {})
    reward_config = config.get("reward", {})
    game_config = config.get("game", {})
    eval_config = config.get("evaluation", {})
    logging_config = config.get("logging", {})
    
    # Get algorithm FIRST (needed for model directories)
    algorithm = training_config.get("algorithm", "MaskablePPO")
    
    # Set up directories
    log_dir = logging_config.get("log_dir", "./logs")
    tensorboard_log = logging_config.get("tensorboard_log", "./logs/tensorboard")
    models_base_dir = logging_config.get("models_dir", logging_config.get("save_path", "./models"))
    
    # Ensure model directories exist (organized by algorithm)
    model_dirs = ensure_model_directories(models_base_dir, algorithm)
    checkpoints_dir = model_dirs["checkpoints_dir"]
    final_dir = model_dirs["final_dir"]
    snapshots_dir = model_dirs["snapshots_dir"]
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)
    
    # Set random seed
    seed = training_config.get("seed", 42)
    np.random.seed(seed)
    
    # Create opponent pool if self-play enabled OR if switching to self-play is configured
    opponent_pool = None
    opponent_config = training_config.get("opponent", {})
    switch_to_self_play_at = opponent_config.get("switch_to_self_play_at", 0)
    self_play_enabled = training_config.get("self_play", {}).get("enabled", False)
    
    # Store for later use in print statements
    save_freq = logging_config.get("save_freq", 50000)
    
    if self_play_enabled or switch_to_self_play_at > 0:
        pool_size = training_config.get("self_play", {}).get("opponent_pool_size", 5)
        opponent_pool = OpponentPool(pool_size=pool_size)
    
    # Create initial opponent (will be overridden by curriculum if enabled)
    curriculum_config = training_config.get("curriculum", {})
    curriculum_enabled = curriculum_config.get("enabled", False)
    
    if curriculum_enabled:
        # For curriculum, start with first phase opponent (or random if no phases)
        curriculum_phases = curriculum_config.get("phases", [])
        if curriculum_phases and len(curriculum_phases) > 0:
            first_phase = curriculum_phases[0]
            initial_opponent_type = first_phase.get("opponent_type", "random")
            initial_opponent_kwargs = first_phase.get("opponent_kwargs", {})
            opponent = create_opponent(initial_opponent_type, opponent_pool, seed, **initial_opponent_kwargs)
        else:
            opponent = create_opponent("random", opponent_pool, seed)
    else:
        # Legacy: use opponent config
        opponent = create_opponent_from_config(training_config, opponent_pool)
    
    env_fn = make_env(config, opponent=opponent, seed=seed)
    env = DummyVecEnv([env_fn])
    
    # Wrap with ActionMasker for MaskablePPO if needed
    # PistiGymEnv implements action_masks() method, but vectorized envs need wrapper
    if algorithm == "MaskablePPO" and ActionMasker is not None:
        def mask_fn(obs):
            """Extract action mask from observation dict."""
            if isinstance(obs, dict):
                return obs.get("action_mask", np.ones(52, dtype=bool))
            # For vectorized: obs might be a list/array
            elif hasattr(obs, '__iter__') and not isinstance(obs, (str, bytes)):
                # Try to get from first element if it's a sequence
                try:
                    first_obs = obs[0] if len(obs) > 0 else obs
                    if isinstance(first_obs, dict):
                        return first_obs.get("action_mask", np.ones(52, dtype=bool))
                except (TypeError, IndexError):
                    pass
            return np.ones(52, dtype=bool)
        
        env = ActionMasker(env, mask_fn)
    
    # Create model
    use_deep = training_config.get("use_deep_architecture", True)
    arch_type = "deep" if use_deep else "shallow"
    
    # Get network architecture
    net_arch_dict = get_network_arch(config, arch_type)
    
    if algorithm == "PPO":
        ppo_config = training_config.get("ppo", {})
        policy_kwargs = {}
        if "pi" in net_arch_dict:
            policy_kwargs["net_arch"] = {
                "pi": net_arch_dict["pi"],
                "vf": net_arch_dict["vf"],
            }
        
        model = PPO(
            "MultiInputPolicy",  # For Dict observation space
            env,
            learning_rate=ppo_config.get("learning_rate", 3e-4),
            n_steps=ppo_config.get("n_steps", 2048),
            batch_size=ppo_config.get("batch_size", 64),
            n_epochs=ppo_config.get("n_epochs", 10),
            gamma=ppo_config.get("gamma", 0.99),
            gae_lambda=ppo_config.get("gae_lambda", 0.95),
            clip_range=ppo_config.get("clip_range", 0.2),
            ent_coef=ppo_config.get("ent_coef", 0.01),
            vf_coef=ppo_config.get("vf_coef", 0.5),
            max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=1,
        )
    elif algorithm == "MaskablePPO":
        if not SB3_CONTRIB_AVAILABLE:
            raise ImportError("sb3-contrib is required for MaskablePPO. Install with: pip install sb3-contrib")
        
        maskable_ppo_config = training_config.get("maskable_ppo", training_config.get("ppo", {}))
        policy_kwargs = {}
        if "pi" in net_arch_dict:
            policy_kwargs["net_arch"] = {
                "pi": net_arch_dict["pi"],
                "vf": net_arch_dict["vf"],
            }
        
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=maskable_ppo_config.get("learning_rate", 3e-4),
            n_steps=maskable_ppo_config.get("n_steps", 2048),
            batch_size=maskable_ppo_config.get("batch_size", 64),
            n_epochs=maskable_ppo_config.get("n_epochs", 10),
            gamma=maskable_ppo_config.get("gamma", 0.99),
            gae_lambda=maskable_ppo_config.get("gae_lambda", 0.95),
            clip_range=maskable_ppo_config.get("clip_range", 0.2),
            ent_coef=maskable_ppo_config.get("ent_coef", 0.01),
            vf_coef=maskable_ppo_config.get("vf_coef", 0.5),
            max_grad_norm=maskable_ppo_config.get("max_grad_norm", 0.5),
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=1,
        )
    elif algorithm == "RecurrentPPO":
        if not SB3_CONTRIB_AVAILABLE:
            raise ImportError("sb3-contrib is required for RecurrentPPO. Install with: pip install sb3-contrib")
        
        recurrent_ppo_config = training_config.get("recurrent_ppo", {})
        recurrent_arch = get_network_arch(config, "recurrent")
        
        policy_kwargs = {
            "lstm_hidden_size": recurrent_arch.get("lstm_hidden_size", 256),
            "n_lstm_layers": recurrent_arch.get("lstm_layers", 2),
            "net_arch": recurrent_arch.get("mlp_layers", [128, 128]),
        }
        
        # Use SequenceEncoder for recurrent policies
        if encoding_config.get("encoder_type") != "SequenceEncoder":
            print("Warning: RecurrentPPO works best with SequenceEncoder. Consider using it.")
        
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=recurrent_ppo_config.get("learning_rate", 3e-4),
            n_steps=recurrent_ppo_config.get("n_steps", 2048),
            batch_size=recurrent_ppo_config.get("batch_size", 64),
            n_epochs=recurrent_ppo_config.get("n_epochs", 10),
            gamma=recurrent_ppo_config.get("gamma", 0.99),
            gae_lambda=recurrent_ppo_config.get("gae_lambda", 0.95),
            clip_range=recurrent_ppo_config.get("clip_range", 0.2),
            ent_coef=recurrent_ppo_config.get("ent_coef", 0.01),
            vf_coef=recurrent_ppo_config.get("vf_coef", 0.5),
            max_grad_norm=recurrent_ppo_config.get("max_grad_norm", 0.5),
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=1,
        )
    elif algorithm == "DQN":
        dqn_config = training_config.get("dqn", {})
        policy_kwargs = {}
        if "qf" in net_arch_dict:
            policy_kwargs["net_arch"] = net_arch_dict["qf"]
        
        model = DQN(
            "MultiInputPolicy",  # For Dict observation space
            env,
            learning_rate=dqn_config.get("learning_rate", 1e-4),
            buffer_size=dqn_config.get("buffer_size", 100000),
            learning_starts=dqn_config.get("learning_starts", 1000),
            batch_size=dqn_config.get("batch_size", 32),
            tau=dqn_config.get("tau", 1.0),
            gamma=dqn_config.get("gamma", 0.99),
            train_freq=dqn_config.get("train_freq", 4),
            gradient_steps=dqn_config.get("gradient_steps", 1),
            target_update_interval=dqn_config.get("target_update_interval", 1000),
            exploration_fraction=dqn_config.get("exploration_fraction", 0.1),
            exploration_initial_eps=dqn_config.get("exploration_initial_eps", 1.0),
            exploration_final_eps=dqn_config.get("exploration_final_eps", 0.05),
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=1,
        )
    elif algorithm == "RainbowDQN":
        if Rainbow is None:
            raise ImportError("Rainbow DQN is not available in this version of sb3-contrib. Install with: pip install sb3-contrib")
        
        rainbow_config = training_config.get("rainbow_dqn", training_config.get("dqn", {}))
        policy_kwargs = {}
        if "qf" in net_arch_dict:
            policy_kwargs["net_arch"] = net_arch_dict["qf"]
        
        model = Rainbow(
            "MultiInputPolicy",
            env,
            learning_rate=rainbow_config.get("learning_rate", 1e-4),
            buffer_size=rainbow_config.get("buffer_size", 100000),
            learning_starts=rainbow_config.get("learning_starts", 1000),
            batch_size=rainbow_config.get("batch_size", 32),
            tau=rainbow_config.get("tau", 1.0),
            gamma=rainbow_config.get("gamma", 0.99),
            train_freq=rainbow_config.get("train_freq", 4),
            gradient_steps=rainbow_config.get("gradient_steps", 1),
            target_update_interval=rainbow_config.get("target_update_interval", 1000),
            exploration_fraction=rainbow_config.get("exploration_fraction", 0.1),
            exploration_initial_eps=rainbow_config.get("exploration_initial_eps", 1.0),
            exploration_final_eps=rainbow_config.get("exploration_final_eps", 0.05),
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=1,
        )
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Available: PPO, MaskablePPO, RecurrentPPO, DQN, RainbowDQN"
        )
    
    # Create metadata
    training_start_time = datetime.now().isoformat()
    total_timesteps = training_config.get("total_timesteps", 1000000)
    metadata = create_metadata(
        config_path=config_path,
        config=config,
        algorithm=algorithm,
        model=model,
        total_timesteps=total_timesteps,
        training_start_time=training_start_time,
    )
    
    # Set up callbacks
    callbacks = []
    
    # Checkpoint callback with metadata
    save_freq = logging_config.get("save_freq", 50000)
    checkpoint_callback = CheckpointCallback(
        save_path=checkpoints_dir,  # Use checkpoints subdirectory
        save_freq=save_freq,
        name_prefix="pisti_model",
        metadata=metadata,
        verbose=1,
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback with metadata
    if eval_config.get("eval_freq", 0) > 0:
        eval_env_fn = make_env(config, opponent=RandomValidAgent(), seed=seed + 1000)
        eval_env = DummyVecEnv([eval_env_fn])
        
        # Wrap eval environment with ActionMasker if using MaskablePPO (same as training env)
        if algorithm == "MaskablePPO" and ActionMasker is not None:
            def eval_mask_fn(obs):
                """Extract action mask from observation dict for eval env."""
                if isinstance(obs, dict):
                    return obs.get("action_mask", np.ones(52, dtype=bool))
                # For vectorized: obs might be a list/array
                elif hasattr(obs, '__iter__') and not isinstance(obs, (str, bytes)):
                    try:
                        first_obs = obs[0] if len(obs) > 0 else obs
                        if isinstance(first_obs, dict):
                            return first_obs.get("action_mask", np.ones(52, dtype=bool))
                    except (TypeError, IndexError):
                        pass
                return np.ones(52, dtype=bool)
            
            eval_env = ActionMasker(eval_env, eval_mask_fn)
        
        eval_callback = EvalCallback(
            eval_env=eval_env,
            eval_freq=eval_config.get("eval_freq", 10000),
            n_eval_episodes=eval_config.get("n_eval_episodes", 10),
            eval_opponents=eval_config.get("eval_opponents", ["random"]),
            metadata=metadata,
            verbose=1,
        )
        callbacks.append(eval_callback)
    
    # Curriculum callback (if enabled)
    if curriculum_enabled:
        curriculum_phases = curriculum_config.get("phases", [])
        
        # Create opponent factory function for curriculum
        def opponent_factory(opponent_type: str, **kwargs):
            """Factory function for creating opponents during curriculum."""
            return create_opponent(opponent_type, opponent_pool, seed, **kwargs)
        
        curriculum_callback = CurriculumCallback(
            env=env,
            curriculum_phases=curriculum_phases,
            opponent_factory=opponent_factory,
            verbose=1,
        )
        callbacks.append(curriculum_callback)
    
    # Opponent switch callback (if configured) - deprecated: use curriculum instead
    if not curriculum_enabled and switch_to_self_play_at > 0 and opponent_pool is not None:
        switch_callback = OpponentSwitchCallback(
            env=env,
            new_opponent=opponent_pool,
            switch_at_timestep=switch_to_self_play_at,
            verbose=1,
        )
        callbacks.append(switch_callback)
    
    # League callback for self-play
    if training_config.get("self_play", {}).get("enabled", False):
        snapshot_freq = training_config.get("self_play", {}).get(
            "snapshot_frequency", 50000
        )
        league_callback = LeagueCallback(
            opponent_pool=opponent_pool,
            snapshot_freq=snapshot_freq,
            checkpoint_path=snapshots_dir,  # Use snapshots subdirectory
            algorithm=algorithm,
            verbose=1,
        )
        callbacks.append(league_callback)
        
        # Update opponent periodically to use latest model
        # This is a simplified approach - in practice, you might want to
        # update the opponent more frequently or use a different strategy
    
    callback_list = CallbackList(callbacks)
    
    # Print training information
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Algorithm: {algorithm}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Network architecture: {arch_type}")
    
    # Reward configuration
    reward_shaping_enabled = reward_config.get("shaping", {}).get("enabled", False)
    print(f"Reward shaping: {'Enabled' if reward_shaping_enabled else 'Disabled (sparse only)'}")
    
    # Curriculum or opponent configuration
    curriculum_config = training_config.get("curriculum", {})
    curriculum_enabled = curriculum_config.get("enabled", False)
    
    if curriculum_enabled:
        print(f"Curriculum learning: Enabled")
        curriculum_phases = curriculum_config.get("phases", [])
        print(f"  Number of phases: {len(curriculum_phases)}")
        for i, phase in enumerate(curriculum_phases, 1):
            timestep = phase.get("timestep", 0)
            opp_type = phase.get("opponent_type", "unknown")
            kwargs = phase.get("opponent_kwargs", {})
            temp_str = f" (temp={kwargs.get('temperature', 0.0)})" if "temperature" in kwargs else ""
            print(f"  Phase {i}: {opp_type}{temp_str} starting at {timestep:,} steps")
    else:
        opponent_config = training_config.get("opponent", {})
        opponent_type = opponent_config.get("type", "random")
        print(f"Initial opponent: {opponent_type}")
        if switch_to_self_play_at > 0:
            print(f"Switch to self-play at: {switch_to_self_play_at:,} timesteps")
    
    if self_play_enabled:
        print(f"Self-play enabled: Yes (pool size: {opponent_pool.pool_size if opponent_pool else 'N/A'})")
    print(f"Checkpoint frequency: {save_freq:,} steps")
    if eval_config.get("eval_freq", 0) > 0:
        print(f"Evaluation frequency: {eval_config.get('eval_freq')} steps")
    print(f"Model save directory: {final_dir}")
    print(f"TensorBoard logs: {tensorboard_log}")
    print("="*60 + "\n")
    
    # Train
    print("Starting training...\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        progress_bar=True,
        log_interval=1,  # Log every step for verbose output
    )
    
    # Update metadata with training end time and best scores
    training_end_time = datetime.now().isoformat()
    best_eval_score = None
    best_eval_step = None
    if eval_config.get("eval_freq", 0) > 0:
        best_eval_score = eval_callback.best_score
        best_eval_step = eval_callback.best_step
    
    metadata = update_metadata(
        metadata,
        best_eval_score=best_eval_score,
        best_eval_step=best_eval_step,
        training_end_time=training_end_time,
    )
    
    # Save final model and metadata (in final/ subdirectory)
    final_path = get_checkpoint_path(
        algorithm=algorithm,
        name_prefix="pisti_model",
        is_final=True,
        base_dir=models_base_dir,
    )
    model.save(final_path)
    save_metadata(metadata, final_dir, "pisti_model_final")
    print(f"Training complete. Final model saved to {final_path}")
    print(f"Metadata saved to {os.path.join(final_dir, 'pisti_model_final_metadata.json')}")


def main():
    """Main entry point."""
    _main_start = time.time()
    print(f"[DEBUG] main() called - script is starting (imports took {_import_time:.2f}s)")
    parser = argparse.ArgumentParser(description="Train Pişti RL agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()
    print(f"[DEBUG] Parsed arguments - config: {args.config}")
    
    train(args.config)


if __name__ == "__main__":
    main()
