"""Training script for R2D2 (Recurrent Replay Distributed DQN)."""

import argparse
import os
import yaml
from typing import Dict, Any
import numpy as np
from datetime import datetime

from envs.pisti_gym import PistiGymEnv
from encoding.encoders import MultiHotEncoder, SequenceEncoder
from agents.r2d2_agent import R2D2Agent
from agents.baselines import RandomValidAgent


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_encoder(config: Dict[str, Any]):
    """Create observation encoder from config."""
    encoder_type = config.get("encoder_type", "SequenceEncoder")  # R2D2 works best with sequences
    encoder_config = config.copy()
    
    if encoder_type == "SequenceEncoder":
        return SequenceEncoder(encoder_config)
    else:
        # Fallback to MultiHotEncoder
        return MultiHotEncoder(encoder_config)


def train_r2d2(config_path: str):
    """Main R2D2 training function."""
    # Load config
    config = load_config(config_path)
    training_config = config.get("training", {})
    encoding_config = config.get("encoding", {})
    reward_config = config.get("reward", {})
    game_config = config.get("game", {})
    logging_config = config.get("logging", {})
    r2d2_config = training_config.get("r2d2", {})
    
    # Set up directories
    save_path = logging_config.get("save_path", "./checkpoints")
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
    obs_dim = sum(
        v.size if hasattr(v, "size") else len(v.flatten())
        for k, v in obs.items()
        if k != "action_mask"
    )
    action_dim = 52
    
    # Create R2D2 agent
    agent_config = {
        "learning_rate": r2d2_config.get("learning_rate", 1e-4),
        "gamma": r2d2_config.get("gamma", 0.99),
        "n_step": r2d2_config.get("n_step", 5),
        "tau": r2d2_config.get("tau", 1.0),
        "batch_size": r2d2_config.get("batch_size", 32),
        "buffer_size": r2d2_config.get("buffer_size", 100000),
        "replay_alpha": r2d2_config.get("replay_alpha", 0.6),
        "replay_beta": r2d2_config.get("replay_beta", 0.4),
        **config,
    }
    
    agent = R2D2Agent(
        observation_dim=obs_dim,
        action_dim=action_dim,
        config=agent_config,
        device="cuda" if r2d2_config.get("use_gpu", False) else "cpu",
    )
    
    # Create training environment
    env = PistiGymEnv(
        encoder=encoder,
        reward_config=reward_config,
        game_config=game_config,
        opponent=RandomValidAgent(),
        seed=seed,
    )
    
    # Training parameters
    total_timesteps = training_config.get("total_timesteps", 1000000)
    train_freq = r2d2_config.get("train_freq", 4)
    save_freq = logging_config.get("save_freq", 50000)
    learning_starts = r2d2_config.get("learning_starts", 1000)
    
    # Epsilon schedule
    exploration_initial_eps = r2d2_config.get("exploration_initial_eps", 1.0)
    exploration_final_eps = r2d2_config.get("exploration_final_eps", 0.05)
    exploration_fraction = r2d2_config.get("exploration_fraction", 0.1)
    exploration_steps = int(total_timesteps * exploration_fraction)
    
    print(f"Starting R2D2 training for {total_timesteps} timesteps...")
    
    timestep = 0
    episode = 0
    
    while timestep < total_timesteps:
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        hidden = None
        
        while not done and timestep < total_timesteps:
            # Compute epsilon
            epsilon = exploration_initial_eps
            if timestep < exploration_steps:
                epsilon = exploration_initial_eps - (
                    exploration_initial_eps - exploration_final_eps
                ) * (timestep / exploration_steps)
            
            # Get action mask
            action_mask = obs.get("action_mask", np.ones(52, dtype=bool))
            
            # Predict action
            if np.random.random() < epsilon:
                # Random action
                legal_actions = np.where(action_mask)[0]
                action = int(np.random.choice(legal_actions))
                _, hidden = agent.predict(obs, action_mask, deterministic=False, hidden=hidden)
            else:
                action, hidden = agent.predict(obs, action_mask, deterministic=False, hidden=hidden)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Get next hidden state
            _, next_hidden = agent.predict(next_obs, action_mask, deterministic=False, hidden=hidden)
            
            # Store experience
            agent.add_experience(obs, action, reward, next_obs, done, action_mask, hidden, next_hidden)
            
            # Train agent
            if timestep >= learning_starts and timestep % train_freq == 0:
                agent.train_step()
            
            obs = next_obs
            hidden = next_hidden
            timestep += 1
            
            # Save checkpoint
            if timestep % save_freq == 0:
                checkpoint_path = os.path.join(save_path, f"r2d2_model_{timestep}_steps")
                agent.save(checkpoint_path + ".pt")
                print(f"Saved checkpoint at timestep {timestep}")
        
        episode += 1
        if episode % 100 == 0:
            print(f"Episode {episode}, Timestep {timestep}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")
    
    # Save final model
    final_path = os.path.join(save_path, "r2d2_model_final")
    agent.save(final_path + ".pt")
    print(f"Training complete. Final model saved to {final_path}.pt")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train R2D2 agent for Pişti")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()
    
    train_r2d2(args.config)


if __name__ == "__main__":
    main()
