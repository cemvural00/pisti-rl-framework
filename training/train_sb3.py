"""Training script for Pişti RL using Stable-Baselines3."""

import argparse
import os
import yaml
from typing import Dict, Any
import numpy as np

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy

# Try to import sb3-contrib algorithms
try:
    from sb3_contrib import MaskablePPO, RecurrentPPO, Rainbow
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False
    print("Warning: sb3-contrib not available. MaskablePPO, RecurrentPPO, and Rainbow DQN will not be available.")
    print("Install with: pip install sb3-contrib")

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
from agents.opponents import OpponentPool, SelfPlayOpponent
from training.callbacks import CheckpointCallback, EvalCallback, LeagueCallback
from training.metadata import create_metadata, save_metadata, update_metadata
from datetime import datetime


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


def create_opponent(config: Dict[str, Any], opponent_pool: OpponentPool = None):
    """Create opponent from config."""
    self_play_config = config.get("self_play", {})
    
    if self_play_config.get("enabled", False) and opponent_pool is not None:
        # Use opponent pool for self-play
        return opponent_pool
    else:
        # Default to random agent
        return RandomValidAgent()


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
    # Load config
    config = load_config(config_path)
    training_config = config.get("training", {})
    encoding_config = config.get("encoding", {})
    reward_config = config.get("reward", {})
    game_config = config.get("game", {})
    eval_config = config.get("evaluation", {})
    logging_config = config.get("logging", {})
    
    # Set up directories
    log_dir = logging_config.get("log_dir", "./logs")
    tensorboard_log = logging_config.get("tensorboard_log", "./logs/tensorboard")
    save_path = logging_config.get("save_path", "./checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    
    # Set random seed
    seed = training_config.get("seed", 42)
    np.random.seed(seed)
    
    # Create opponent pool if self-play enabled
    opponent_pool = None
    if training_config.get("self_play", {}).get("enabled", False):
        pool_size = training_config.get("self_play", {}).get("opponent_pool_size", 5)
        opponent_pool = OpponentPool(pool_size=pool_size)
    
    # Create environment
    opponent = create_opponent(training_config, opponent_pool)
    env_fn = make_env(config, opponent=opponent, seed=seed)
    env = DummyVecEnv([env_fn])
    
    # Create model
    algorithm = training_config.get("algorithm", "MaskablePPO")
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
        if not SB3_CONTRIB_AVAILABLE:
            raise ImportError("sb3-contrib is required for RainbowDQN. Install with: pip install sb3-contrib")
        
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
        save_path=save_path,
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
        eval_callback = EvalCallback(
            eval_env=eval_env,
            eval_freq=eval_config.get("eval_freq", 10000),
            n_eval_episodes=eval_config.get("n_eval_episodes", 10),
            eval_opponents=eval_config.get("eval_opponents", ["random"]),
            metadata=metadata,
            verbose=1,
        )
        callbacks.append(eval_callback)
    
    # League callback for self-play
    if training_config.get("self_play", {}).get("enabled", False):
        snapshot_freq = training_config.get("self_play", {}).get(
            "snapshot_frequency", 50000
        )
        league_callback = LeagueCallback(
            opponent_pool=opponent_pool,
            snapshot_freq=snapshot_freq,
            checkpoint_path=save_path,
            algorithm=algorithm,
            verbose=1,
        )
        callbacks.append(league_callback)
        
        # Update opponent periodically to use latest model
        # This is a simplified approach - in practice, you might want to
        # update the opponent more frequently or use a different strategy
    
    callback_list = CallbackList(callbacks)
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        progress_bar=True,
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
    
    # Save final model and metadata
    final_path = os.path.join(save_path, "pisti_model_final")
    model.save(final_path)
    save_metadata(metadata, save_path, "pisti_model_final")
    print(f"Training complete. Final model saved to {final_path}")
    print(f"Metadata saved to {final_path}_metadata.json")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Pişti RL agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()
    
    train(args.config)


if __name__ == "__main__":
    main()
