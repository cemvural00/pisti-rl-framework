"""Evaluation script for Pişti RL agents."""

import argparse
import os
import yaml
from typing import Dict, Any, List
import numpy as np

from stable_baselines3 import PPO, DQN

from envs.pisti_gym import PistiGymEnv
from encoding.encoders import MultiHotEncoder
from agents.baselines import RandomValidAgent, GreedyCaptureAgent, PistiHunterAgent
from agents.opponents import FrozenCheckpointOpponent


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_opponent(opponent_name: str, checkpoint_path: str = None):
    """Create opponent by name."""
    if opponent_name == "random":
        return RandomValidAgent()
    elif opponent_name == "greedy":
        return GreedyCaptureAgent()
    elif opponent_name == "pisti_hunter":
        return PistiHunterAgent()
    elif opponent_name == "checkpoint" and checkpoint_path:
        # Try to determine algorithm from checkpoint
        # For now, assume PPO
        return FrozenCheckpointOpponent(checkpoint_path, algorithm="PPO")
    else:
        raise ValueError(f"Unknown opponent: {opponent_name}")


def evaluate(
    checkpoint_path: str,
    config_path: str,
    opponents: List[str],
    n_episodes: int = 100,
    seeds: List[int] = None,
):
    """
    Evaluate a trained model against baseline opponents.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config YAML
        opponents: List of opponent names
        n_episodes: Number of episodes per opponent
        seeds: List of random seeds (for stability)
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 999]
    
    # Load config
    config = load_config(config_path)
    encoding_config = config.get("encoding", {})
    reward_config = config.get("reward", {})
    game_config = config.get("game", {})
    
    # Load model
    # Try to determine algorithm from file or assume PPO
    try:
        model = PPO.load(checkpoint_path)
        algorithm = "PPO"
    except:
        try:
            model = DQN.load(checkpoint_path)
            algorithm = "DQN"
        except:
            raise ValueError(f"Could not load model from {checkpoint_path}")
    
    # Create encoder
    from training.train_sb3 import create_encoder
    encoder = create_encoder(encoding_config)
    
    results = {}
    
    for opponent_name in opponents:
        print(f"\nEvaluating against {opponent_name}...")
        
        opponent = create_opponent(opponent_name)
        
        # Metrics
        wins = 0
        total_score_diff = 0.0
        total_pistis = {0: 0, 1: 0}
        total_double_pistis = {0: 0, 1: 0}
        total_captures = {0: 0, 1: 0}
        
        for seed in seeds:
            np.random.seed(seed)
            
            # Create environment
            env = PistiGymEnv(
                encoder=encoder,
                reward_config=reward_config,
                game_config=game_config,
                opponent=opponent,
                seed=seed,
            )
            
            for episode in range(n_episodes // len(seeds)):
                obs, _ = env.reset(seed=seed + episode)
                done = False
                episode_reward = 0.0
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                
                # Extract metrics from final state
                if env.engine.state and env.engine.state.is_terminal():
                    scores = env.engine.state.get_final_scores()
                    score_diff = scores[0] - scores[1]
                    total_score_diff += score_diff
                    
                    if score_diff > 0:
                        wins += 1
                    
                    # Count pistis and captures
                    for player_id in [0, 1]:
                        total_pistis[player_id] += env.engine.state.score_breakdown[
                            player_id
                        ]["pistis"]
                        total_double_pistis[player_id] += env.engine.state.score_breakdown[
                            player_id
                        ]["double_pistis"]
                        total_captures[player_id] += len(
                            env.engine.state.captured[player_id]
                        )
        
        # Calculate averages
        n_total = n_episodes
        win_rate = wins / n_total
        avg_score_diff = total_score_diff / n_total
        avg_pistis = {
            player_id: count / n_total
            for player_id, count in total_pistis.items()
        }
        avg_double_pistis = {
            player_id: count / n_total
            for player_id, count in total_double_pistis.items()
        }
        avg_captures = {
            player_id: count / n_total
            for player_id, count in total_captures.items()
        }
        
        results[opponent_name] = {
            "win_rate": win_rate,
            "avg_score_diff": avg_score_diff,
            "avg_pistis": avg_pistis,
            "avg_double_pistis": avg_double_pistis,
            "avg_captures": avg_captures,
        }
        
        # Print results
        print(f"  Win rate: {win_rate:.2%}")
        print(f"  Avg score diff: {avg_score_diff:.2f}")
        print(f"  Avg pistis (player 0): {avg_pistis[0]:.2f}")
        print(f"  Avg double pistis (player 0): {avg_double_pistis[0]:.2f}")
        print(f"  Avg captures (player 0): {avg_captures[0]:.2f}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for opponent_name, metrics in results.items():
        print(f"\n{opponent_name.upper()}:")
        print(f"  Win rate: {metrics['win_rate']:.2%}")
        print(f"  Avg score diff: {metrics['avg_score_diff']:.2f}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate Pişti RL agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        default="random,greedy",
        help="Comma-separated list of opponents (random, greedy, pisti_hunter)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    args = parser.parse_args()
    
    opponents = [opp.strip() for opp in args.opponents.split(",")]
    
    evaluate(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        opponents=opponents,
        n_episodes=args.n_episodes,
    )


if __name__ == "__main__":
    main()
