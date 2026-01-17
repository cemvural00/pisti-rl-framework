"""Comprehensive evaluation script with statistical analysis."""

import argparse
import os
import yaml
from typing import Dict, Any, List, Optional
import numpy as np
from scipy import stats
import json
from datetime import datetime

from stable_baselines3 import PPO, DQN
try:
    from sb3_contrib import MaskablePPO, RecurrentPPO, Rainbow
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False

from envs.pisti_gym import PistiGymEnv
from encoding.encoders import MultiHotEncoder
from agents.baselines import RandomValidAgent, GreedyCaptureAgent, PistiHunterAgent
from agents.probabilistic_agent import ProbabilisticOptimalAgent
from agents.opponents import FrozenCheckpointOpponent
from agents.nfsp_agent import NFSPAgent
from agents.deep_cfr_agent import DeepCFRAgent
from agents.r2d2_agent import R2D2Agent
from training.metadata import load_metadata
from training.train_sb3 import create_encoder, load_config
from training.cleanup_results import cleanup_results


def create_opponent(opponent_name: str, checkpoint_path: str = None):
    """Create opponent by name."""
    if opponent_name == "random":
        return RandomValidAgent()
    elif opponent_name == "greedy":
        return GreedyCaptureAgent()
    elif opponent_name == "pisti_hunter":
        return PistiHunterAgent()
    elif opponent_name == "probabilistic":
        return ProbabilisticOptimalAgent(max_samples=50, depth=1, seed=42)
    elif opponent_name == "checkpoint" and checkpoint_path:
        return FrozenCheckpointOpponent(checkpoint_path, algorithm="PPO")
    else:
        raise ValueError(f"Unknown opponent: {opponent_name}")


def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple:
    """
    Calculate confidence interval for data.
    
    Args:
        data: Array of values
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        (mean, lower_bound, upper_bound, std)
    """
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample standard deviation
    n = len(data)
    
    # t-distribution for confidence interval
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin = t_critical * std / np.sqrt(n)
    
    return mean, mean - margin, mean + margin, std


def evaluate_episode(env, model, deterministic: bool = True, algorithm: str = None) -> Dict[str, Any]:
    """
    Evaluate a single episode and return metrics.
    
    Returns:
        Dict with episode metrics
    """
    obs, _ = env.reset()
    done = False
    episode_reward = 0.0
    steps = 0
    hidden = None  # For recurrent models
    
    while not done:
        action_mask = obs.get("action_mask", np.ones(52, dtype=bool))
        
        # Handle different model types
        if algorithm in ["NFSP", "DeepCFR", "R2D2"]:
            # Custom agents
            if algorithm == "R2D2":
                action, hidden = model.predict(obs, action_mask, deterministic=deterministic, hidden=hidden)
            else:
                action = model.predict(obs, action_mask, deterministic=deterministic)
        else:
            # SB3 models
            action, _ = model.predict(obs, deterministic=deterministic)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        steps += 1
    
    # Extract metrics from final state
    metrics = {
        "reward": episode_reward,
        "steps": steps,
        "score_diff": 0.0,
        "win": False,
        "pistis": {0: 0, 1: 0},
        "double_pistis": {0: 0, 1: 0},
        "captures": {0: 0, 1: 0},
        "final_scores": {0: 0, 1: 0},
    }
    
    if env.engine.state and env.engine.state.is_terminal():
        scores = env.engine.state.get_final_scores()
        metrics["final_scores"] = scores
        metrics["score_diff"] = scores[0] - scores[1]
        metrics["win"] = scores[0] > scores[1]
        
        for player_id in [0, 1]:
            metrics["pistis"][player_id] = env.engine.state.score_breakdown[
                player_id
            ]["pistis"]
            metrics["double_pistis"][player_id] = env.engine.state.score_breakdown[
                player_id
            ]["double_pistis"]
            metrics["captures"][player_id] = len(env.engine.state.captured[player_id])
    
    return metrics


def evaluate_comprehensive(
    checkpoint_path: str,
    config_path: str,
    opponents: List[str],
    n_episodes: int = 1000,
    n_seeds: int = 10,
    output_dir: Optional[str] = None,
    deterministic: bool = True,
    cleanup_old: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation with statistical analysis.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config YAML
        opponents: List of opponent names
        n_episodes: Total number of episodes (distributed across seeds)
        n_seeds: Number of random seeds
        output_dir: Directory to save results
        deterministic: Use deterministic policy
        cleanup_old: If specified, delete old results keeping only N most recent
    
    Returns:
        Dict with comprehensive results
    """
    # Cleanup old results if requested
    if cleanup_old is not None:
        base_results_dir = os.path.dirname(output_dir) if output_dir else "results"
        if not base_results_dir:
            base_results_dir = "results"
        print(f"Cleaning up old results (keeping {cleanup_old} most recent)...")
        cleanup_results(
            results_dir=base_results_dir,
            keep_recent=cleanup_old,
            dry_run=False,
        )
    
    if output_dir is None:
        output_dir = f"results/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    config = load_config(config_path)
    encoding_config = config.get("encoding", {})
    reward_config = config.get("reward", {})
    game_config = config.get("game", {})
    
    # Load model - try different formats
    model = None
    algorithm = None
    
    # Try SB3 models first
    try:
        model = PPO.load(checkpoint_path)
        algorithm = "PPO"
    except:
        try:
            model = DQN.load(checkpoint_path)
            algorithm = "DQN"
        except:
            if SB3_CONTRIB_AVAILABLE:
                try:
                    model = MaskablePPO.load(checkpoint_path)
                    algorithm = "MaskablePPO"
                except:
                    try:
                        model = RecurrentPPO.load(checkpoint_path)
                        algorithm = "RecurrentPPO"
                    except:
                        try:
                            model = Rainbow.load(checkpoint_path)
                            algorithm = "RainbowDQN"
                        except:
                            pass
    
    # Try custom agents (NFSP, Deep CFR, R2D2)
    if model is None:
        # Calculate observation dimension
        test_env = PistiGymEnv(
            encoder=encoder,
            reward_config=reward_config,
            game_config=game_config,
            opponent=RandomValidAgent(),
            seed=42,
        )
        test_obs, _ = test_env.reset()
        obs_dim = sum(
            v.size if hasattr(v, "size") else len(v.flatten())
            for k, v in test_obs.items()
            if k != "action_mask"
        )
        action_dim = 52
        
        # Try NFSP
        if checkpoint_path.endswith(".pt"):
            try:
                nfsp_agent = NFSPAgent(obs_dim, action_dim, config, device="cpu")
                nfsp_agent.load(checkpoint_path)
                model = nfsp_agent
                algorithm = "NFSP"
            except:
                try:
                    # Try Deep CFR
                    deep_cfr_agent = DeepCFRAgent(obs_dim, action_dim, config, device="cpu")
                    deep_cfr_agent.load(checkpoint_path)
                    model = deep_cfr_agent
                    algorithm = "DeepCFR"
                except:
                    try:
                        # Try R2D2
                        r2d2_agent = R2D2Agent(obs_dim, action_dim, config, device="cpu")
                        r2d2_agent.load(checkpoint_path)
                        model = r2d2_agent
                        algorithm = "R2D2"
                    except:
                        pass
    
    if model is None:
        raise ValueError(f"Could not load model from {checkpoint_path}")
    
    # Load metadata if available
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_name = os.path.basename(checkpoint_path).replace(".zip", "")
    metadata = load_metadata(checkpoint_dir, checkpoint_name)
    
    # Create encoder
    encoder = create_encoder(encoding_config)
    
    # Generate seeds
    seeds = list(range(42, 42 + n_seeds))
    episodes_per_seed = n_episodes // n_seeds
    
    all_results = {}
    
    for opponent_name in opponents:
        print(f"\n{'='*60}")
        print(f"Evaluating against {opponent_name}")
        print(f"{'='*60}")
        
        opponent = create_opponent(opponent_name)
        
        # Collect metrics across all seeds
        all_metrics = []
        
        for seed_idx, seed in enumerate(seeds):
            np.random.seed(seed)
            
            # Create environment
            env = PistiGymEnv(
                encoder=encoder,
                reward_config=reward_config,
                game_config=game_config,
                opponent=opponent,
                seed=seed,
            )
            
            # Run episodes
            for episode in range(episodes_per_seed):
                metrics = evaluate_episode(env, model, deterministic=deterministic, algorithm=algorithm)
                all_metrics.append(metrics)
            
            if (seed_idx + 1) % 5 == 0:
                print(f"  Completed {seed_idx + 1}/{n_seeds} seeds...")
        
        # Aggregate metrics
        score_diffs = np.array([m["score_diff"] for m in all_metrics])
        wins = np.array([m["win"] for m in all_metrics])
        rewards = np.array([m["reward"] for m in all_metrics])
        steps = np.array([m["steps"] for m in all_metrics])
        
        pistis_player0 = np.array([m["pistis"][0] for m in all_metrics])
        pistis_player1 = np.array([m["pistis"][1] for m in all_metrics])
        double_pistis_player0 = np.array([m["double_pistis"][0] for m in all_metrics])
        captures_player0 = np.array([m["captures"][0] for m in all_metrics])
        captures_player1 = np.array([m["captures"][1] for m in all_metrics])
        
        # Calculate statistics
        win_rate = np.mean(wins)
        win_rate_ci = calculate_confidence_interval(wins.astype(float))
        
        score_diff_mean, score_diff_lower, score_diff_upper, score_diff_std = (
            calculate_confidence_interval(score_diffs)
        )
        
        reward_mean, reward_lower, reward_upper, reward_std = (
            calculate_confidence_interval(rewards)
        )
        
        steps_mean, steps_lower, steps_upper, steps_std = (
            calculate_confidence_interval(steps)
        )
        
        pistis_mean = np.mean(pistis_player0)
        pistis_std = np.std(pistis_player0)
        double_pistis_mean = np.mean(double_pistis_player0)
        
        capture_efficiency = (
            np.mean(captures_player0) / (np.mean(captures_player0) + np.mean(captures_player1))
            if (np.mean(captures_player0) + np.mean(captures_player1)) > 0
            else 0.0
        )
        
        # Store results
        all_results[opponent_name] = {
            "n_episodes": len(all_metrics),
            "n_seeds": n_seeds,
            "win_rate": {
                "mean": float(win_rate),
                "ci_lower": float(win_rate_ci[1]),
                "ci_upper": float(win_rate_ci[2]),
                "std": float(win_rate_ci[3]),
            },
            "score_diff": {
                "mean": float(score_diff_mean),
                "ci_lower": float(score_diff_lower),
                "ci_upper": float(score_diff_upper),
                "std": float(score_diff_std),
            },
            "reward": {
                "mean": float(reward_mean),
                "ci_lower": float(reward_lower),
                "ci_upper": float(reward_upper),
                "std": float(reward_std),
            },
            "game_length": {
                "mean": float(steps_mean),
                "ci_lower": float(steps_lower),
                "ci_upper": float(steps_upper),
                "std": float(steps_std),
            },
            "pistis": {
                "mean": float(pistis_mean),
                "std": float(pistis_std),
            },
            "double_pistis": {
                "mean": float(double_pistis_mean),
            },
            "capture_efficiency": float(capture_efficiency),
            "raw_data": {
                "score_diffs": score_diffs.tolist(),
                "wins": wins.tolist(),
                "rewards": rewards.tolist(),
            },
        }
        
        # Print summary
        print(f"\nResults vs {opponent_name}:")
        print(f"  Win rate: {win_rate:.2%} (95% CI: [{win_rate_ci[1]:.2%}, {win_rate_ci[2]:.2%}])")
        print(f"  Score diff: {score_diff_mean:.2f} ± {score_diff_std:.2f} (95% CI: [{score_diff_lower:.2f}, {score_diff_upper:.2f}])")
        print(f"  Avg pistis: {pistis_mean:.2f} ± {pistis_std:.2f}")
        print(f"  Capture efficiency: {capture_efficiency:.2%}")
        print(f"  Avg game length: {steps_mean:.1f} steps")
    
    # Save results
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {results_file}")
    print(f"{'='*60}")
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive evaluation of Pişti RL agent")
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
        default="random,greedy,pisti_hunter,probabilistic",
        help="Comma-separated list of opponents",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=1000,
        help="Total number of evaluation episodes",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Number of random seeds for statistical robustness",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: auto-generated)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy (default: True)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (overrides --deterministic)",
    )
    parser.add_argument(
        "--cleanup-old",
        type=int,
        default=None,
        help="Delete old results, keeping only N most recent (default: don't cleanup)",
    )
    
    args = parser.parse_args()
    
    opponents = [opp.strip() for opp in args.opponents.split(",")]
    deterministic = args.deterministic and not args.stochastic
    
    evaluate_comprehensive(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        opponents=opponents,
        n_episodes=args.n_episodes,
        n_seeds=args.n_seeds,
        output_dir=args.output_dir,
        deterministic=deterministic,
        cleanup_old=args.cleanup_old,
    )


if __name__ == "__main__":
    main()
