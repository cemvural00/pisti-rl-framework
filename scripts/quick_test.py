#!/usr/bin/env python3
"""Quick test script to verify probabilistic agent works correctly."""

import sys
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, '.')

from agents.probabilistic_agent import ProbabilisticOptimalAgent
from agents.baselines import RandomValidAgent, GreedyCaptureAgent
from envs.pisti_gym import PistiGymEnv


def test_agent_prediction():
    """Test that agent can make predictions."""
    print("Testing agent prediction...")
    agent = ProbabilisticOptimalAgent(max_samples=5, depth=1, seed=42)
    
    # Create minimal observation
    obs = {
        "hand": np.zeros(52),
        "table_top": np.zeros(52),
        "seen_cards": np.zeros(52),
        "action_mask": np.zeros(52, dtype=bool),
        "table_count": np.array([0]),
        "opp_captured_count": np.array([0]),
        "my_captured_count": np.array([0]),
        "stock_remaining": np.array([44]),
    }
    
    # Set some cards
    obs["hand"][0] = 1.0
    obs["hand"][1] = 1.0
    obs["action_mask"][0] = True
    obs["action_mask"][1] = True
    
    start = time.time()
    action = agent.predict(obs, obs["action_mask"])
    elapsed = time.time() - start
    
    print(f"  ✓ Agent predicted action {action} in {elapsed:.3f}s")
    assert action in [0, 1], f"Invalid action: {action}"
    return True


def test_environment_step():
    """Test agent in environment for a few steps."""
    print("Testing environment integration...")
    agent = ProbabilisticOptimalAgent(max_samples=5, depth=1, seed=42)
    env = PistiGymEnv(opponent=RandomValidAgent(), seed=42)
    
    obs, _ = env.reset(seed=42)
    steps = 0
    max_steps = 5
    
    start = time.time()
    while steps < max_steps:
        action_mask = obs["action_mask"]
        legal_actions = np.where(action_mask)[0]
        
        if len(legal_actions) == 0:
            break
        
        action = int(legal_actions[0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
        
        steps += 1
    
    elapsed = time.time() - start
    print(f"  ✓ Completed {steps} steps in {elapsed:.3f}s")
    return True


def test_probabilistic_vs_baseline():
    """Test probabilistic agent as opponent."""
    print("Testing probabilistic agent as opponent...")
    prob_agent = ProbabilisticOptimalAgent(max_samples=5, depth=1, seed=42)
    env = PistiGymEnv(opponent=prob_agent, seed=42)
    
    obs, _ = env.reset(seed=42)
    steps = 0
    max_steps = 5
    
    start = time.time()
    while steps < max_steps:
        action_mask = obs["action_mask"]
        legal_actions = np.where(action_mask)[0]
        
        if len(legal_actions) == 0:
            break
        
        action = int(legal_actions[0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
        
        steps += 1
    
    elapsed = time.time() - start
    print(f"  ✓ Completed {steps} steps in {elapsed:.3f}s")
    return True


def test_multiple_agents():
    """Test with different agent configurations."""
    print("Testing different agent configurations...")
    
    configs = [
        {"max_samples": 3, "depth": 1},
        {"max_samples": 5, "depth": 1},
        {"max_samples": 10, "depth": 1},
    ]
    
    for config in configs:
        agent = ProbabilisticOptimalAgent(seed=42, **config)
        env = PistiGymEnv(opponent=agent, seed=42)
        
        obs, _ = env.reset(seed=42)
        
        # Play 2 steps
        for _ in range(2):
            action_mask = obs["action_mask"]
            legal_actions = np.where(action_mask)[0]
            
            if len(legal_actions) == 0:
                break
            
            action = int(legal_actions[0])
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        print(f"  ✓ Config {config} works")
    
    return True


def main():
    """Run all quick tests."""
    print("=" * 60)
    print("Quick Test Suite for Probabilistic Agent")
    print("=" * 60)
    print()
    
    tests = [
        ("Agent Prediction", test_agent_prediction),
        ("Environment Step", test_environment_step),
        ("Probabilistic vs Baseline", test_probabilistic_vs_baseline),
        ("Multiple Configurations", test_multiple_agents),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            print()
        except Exception as e:
            failed += 1
            print(f"  ✗ {test_name} FAILED: {e}")
            print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
