"""Training script for Deep Counterfactual Regret Minimization (Deep CFR)."""

import argparse
import os
import yaml
from typing import Dict, Any, List, Tuple
import numpy as np
from datetime import datetime

from envs.pisti_gym import PistiGymEnv
from encoding.encoders import MultiHotEncoder
from agents.deep_cfr_agent import DeepCFRAgent
from agents.baselines import RandomValidAgent
from engine.state import GameState


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
        return MultiHotEncoder(encoder_config)


def traverse_game_tree(
    agent: DeepCFRAgent,
    env: PistiGymEnv,
    player_id: int,
    reach_prob: float = 1.0,
) -> Tuple[float, List[Tuple]]:
    """
    Traverse game tree and compute counterfactual values.
    
    This is a simplified version - full Deep CFR requires recursive traversal.
    For efficiency, we use a single episode traversal.
    
    Returns:
        (counterfactual_value, traversal_data)
    """
    obs, _ = env.reset()
    done = False
    traversal_data = []
    cumulative_reward = 0.0
    
    current_player = 0
    
    while not done:
        action_mask = obs.get("action_mask", np.ones(52, dtype=bool))
        
        # Get information set
        info_set = agent.get_information_set(current_player, obs)
        info_set_key = info_set.key
        
        # Get strategy
        strategy = agent._get_strategy_from_regrets(info_set_key, current_player, action_mask)
        
        # Sample action
        action = int(np.random.choice(52, p=strategy))
        
        # Store traversal data
        obs_tensor = agent._obs_to_tensor(obs)
        traversal_data.append((info_set_key, current_player, obs_tensor, action, strategy))
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if current_player == player_id:
            cumulative_reward += reward
        
        obs = next_obs
        current_player = 1 - current_player
    
    # Get final reward (score difference)
    if env.engine.state and env.engine.state.is_terminal():
        scores = env.engine.state.get_final_scores()
        final_reward = scores[player_id] - scores[1 - player_id]
        cumulative_reward = final_reward
    
    return cumulative_reward, traversal_data


def train_deep_cfr(config_path: str):
    """Main Deep CFR training function."""
    # Load config
    config = load_config(config_path)
    training_config = config.get("training", {})
    encoding_config = config.get("encoding", {})
    reward_config = config.get("reward", {})
    game_config = config.get("game", {})
    logging_config = config.get("logging", {})
    deep_cfr_config = training_config.get("deep_cfr", {})
    
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
    
    # Create Deep CFR agent
    agent_config = {
        "regret_matching_epsilon": deep_cfr_config.get("regret_matching_epsilon", 0.001),
        "learning_rate": deep_cfr_config.get("learning_rate", 1e-4),
        "traversal_batch_size": deep_cfr_config.get("traversal_batch_size", 32),
        **config,
    }
    
    agent = DeepCFRAgent(
        observation_dim=obs_dim,
        action_dim=action_dim,
        config=agent_config,
        device="cuda" if deep_cfr_config.get("use_gpu", False) else "cpu",
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
    total_traversals = training_config.get("total_timesteps", 1000000) // 100  # Approximate
    save_freq = logging_config.get("save_freq", 50000) // 100
    
    print(f"Starting Deep CFR training for {total_traversals} traversals...")
    
    for traversal in range(total_traversals):
        # Alternate between players
        player_id = traversal % 2
        
        # Traverse game tree
        counterfactual_value, traversal_data = traverse_game_tree(agent, env, player_id)
        
        # Process traversal data
        info_sets_batch = []
        obs_batch = []
        cf_values_batch = []
        
        for info_set_key, p_id, obs_tensor, action, strategy in traversal_data:
            if p_id == player_id:
                info_sets_batch.append(info_set_key)
                obs_batch.append(obs_tensor)
                cf_values_batch.append(counterfactual_value)
                
                # Update strategy sum
                reach_prob = 1.0  # Simplified
                agent.update_strategy_sum(info_set_key, p_id, strategy, reach_prob)
        
        # Train counterfactual value network
        if len(obs_batch) > 0:
            agent.train_counterfactual_value(
                info_sets_batch, obs_batch, cf_values_batch, player_id
            )
        
        # Update regrets (simplified - full Deep CFR requires recursive updates)
        for info_set_key, p_id, obs_tensor, action, strategy in traversal_data:
            if p_id == player_id:
                # Simplified regret update
                action_values = {a: counterfactual_value for a in range(52)}
                agent.update_regrets(
                    info_set_key, p_id, action, counterfactual_value, action_values
                )
        
        agent.traversal_count += 1
        
        # Save checkpoint
        if traversal % save_freq == 0:
            checkpoint_path = os.path.join(save_path, f"deep_cfr_model_{traversal}_traversals")
            agent.save(checkpoint_path + ".pt")
            print(f"Saved checkpoint at traversal {traversal}")
        
        if traversal % 100 == 0:
            print(f"Traversal {traversal}/{total_traversals}")
    
    # Save final model
    final_path = os.path.join(save_path, "deep_cfr_model_final")
    agent.save(final_path + ".pt")
    print(f"Training complete. Final model saved to {final_path}.pt")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Deep CFR agent for Pişti")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()
    
    train_deep_cfr(args.config)


if __name__ == "__main__":
    main()
