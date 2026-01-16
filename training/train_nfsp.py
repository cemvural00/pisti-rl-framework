"""Training script for Neural Fictitious Self-Play (NFSP)."""

import argparse
import os
import yaml
from typing import Dict, Any
import numpy as np
from datetime import datetime

from envs.pisti_gym import PistiGymEnv
from encoding.encoders import MultiHotEncoder
from agents.nfsp_agent import NFSPAgent
from agents.baselines import RandomValidAgent
from training.callbacks import CheckpointCallback
from training.metadata import create_metadata, save_metadata, update_metadata


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_encoder(config: Dict[str, Any]):
    """Create observation encoder from config."""
    encoder_type = config.get("encoder_type", "MultiHotEncoder")
    encoder_config = config.copy()
    
    if encoder_type == "MultiHotEncoder":
        return MultiHotEncoder(encoder_config)
    else:
        # NFSP works best with MultiHotEncoder
        return MultiHotEncoder(encoder_config)


def train_nfsp(config_path: str):
    """Main NFSP training function."""
    # Load config
    config = load_config(config_path)
    training_config = config.get("training", {})
    encoding_config = config.get("encoding", {})
    reward_config = config.get("reward", {})
    game_config = config.get("game", {})
    eval_config = config.get("evaluation", {})
    logging_config = config.get("logging", {})
    nfsp_config = training_config.get("nfsp", {})
    
    # Set up directories
    log_dir = logging_config.get("log_dir", "./logs")
    save_path = logging_config.get("save_path", "./checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    
    # Set random seed
    seed = training_config.get("seed", 42)
    np.random.seed(seed)
    
    # Create encoder
    encoder = create_encoder(encoding_config)
    
    # Create environment to get observation space
    test_env = PistiGymEnv(
        encoder=encoder,
        reward_config=reward_config,
        game_config=game_config,
        opponent=RandomValidAgent(),
        seed=seed,
    )
    obs, _ = test_env.reset()
    
    # Calculate observation dimension
    obs_dim = sum(v.size if hasattr(v, "size") else len(v.flatten()) for k, v in obs.items() if k != "action_mask")
    action_dim = 52  # Card space
    
    # Create NFSP agent
    nfsp_agent_config = {
        "anticipatory_param": nfsp_config.get("anticipatory_param", 0.1),
        "average_strategy_update_freq": nfsp_config.get("average_strategy_update_freq", 1000),
        "reservoir_buffer_size": nfsp_config.get("reservoir_buffer_size", 10000),
        "learning_rate": nfsp_config.get("learning_rate", 1e-4),
        **config,  # Include full config for network architectures
    }
    
    agent = NFSPAgent(
        observation_dim=obs_dim,
        action_dim=action_dim,
        config=nfsp_agent_config,
        device="cuda" if nfsp_config.get("use_gpu", False) else "cpu",
    )
    
    # Create training environment
    env = PistiGymEnv(
        encoder=encoder,
        reward_config=reward_config,
        game_config=game_config,
        opponent=RandomValidAgent(),  # NFSP uses self-play via reservoir buffer
        seed=seed,
    )
    
    # Training parameters
    total_timesteps = training_config.get("total_timesteps", 1000000)
    batch_size = nfsp_config.get("batch_size", 32)
    train_freq = nfsp_config.get("train_freq", 4)
    save_freq = logging_config.get("save_freq", 50000)
    
    # Training loop
    timestep = 0
    episode = 0
    
    print(f"Starting NFSP training for {total_timesteps} timesteps...")
    
    while timestep < total_timesteps:
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done and timestep < total_timesteps:
            # Get action mask
            action_mask = obs.get("action_mask", np.ones(52, dtype=bool))
            
            # Predict action
            action = agent.predict(obs, action_mask, deterministic=False)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Store experience
            agent.add_experience(obs, action, reward, next_obs, done, action_mask)
            
            # Train agent
            if timestep % train_freq == 0 and len(agent.best_response_buffer) >= batch_size:
                agent.train_step(batch_size)
            
            obs = next_obs
            timestep += 1
            
            # Save checkpoint
            if timestep % save_freq == 0:
                checkpoint_path = os.path.join(save_path, f"nfsp_model_{timestep}_steps")
                agent.save(checkpoint_path + ".pt")
                print(f"Saved checkpoint at timestep {timestep}")
        
        episode += 1
        if episode % 100 == 0:
            print(f"Episode {episode}, Timestep {timestep}, Reward: {episode_reward:.2f}")
    
    # Save final model
    final_path = os.path.join(save_path, "nfsp_model_final")
    agent.save(final_path + ".pt")
    print(f"Training complete. Final model saved to {final_path}.pt")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train NFSP agent for Pişti")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()
    
    train_nfsp(args.config)


if __name__ == "__main__":
    main()
