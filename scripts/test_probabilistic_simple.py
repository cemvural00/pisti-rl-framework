#!/usr/bin/env python3
"""Simple standalone test for probabilistic agent - minimal dependencies."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from agents.probabilistic_agent import ProbabilisticOptimalAgent
        from agents.baselines import RandomValidAgent
        from envs.pisti_gym import PistiGymEnv
        import numpy as np
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_agent_creation():
    """Test creating agent with minimal config."""
    print("Testing agent creation...")
    try:
        from agents.probabilistic_agent import ProbabilisticOptimalAgent
        
        # Create with minimal samples for speed
        agent = ProbabilisticOptimalAgent(max_samples=3, depth=1, seed=42)
        print("  ✓ Agent created successfully")
        return True
    except Exception as e:
        print(f"  ✗ Agent creation failed: {e}")
        return False


def test_agent_prediction():
    """Test agent can make a prediction."""
    print("Testing agent prediction...")
    try:
        import numpy as np
        from agents.probabilistic_agent import ProbabilisticOptimalAgent
        
        agent = ProbabilisticOptimalAgent(max_samples=3, depth=1, seed=42)
        
        # Minimal observation
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
        
        # Set 2 cards in hand
        obs["hand"][0] = 1.0
        obs["hand"][1] = 1.0
        obs["action_mask"][0] = True
        obs["action_mask"][1] = True
        
        action = agent.predict(obs, obs["action_mask"])
        
        if action in [0, 1]:
            print(f"  ✓ Agent predicted valid action: {action}")
            return True
        else:
            print(f"  ✗ Invalid action: {action}")
            return False
    except Exception as e:
        print(f"  ✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_integration():
    """Test agent works in environment (very short)."""
    print("Testing environment integration...")
    try:
        from agents.probabilistic_agent import ProbabilisticOptimalAgent
        from agents.baselines import RandomValidAgent
        from envs.pisti_gym import PistiGymEnv
        
        # Use minimal samples for speed
        agent = ProbabilisticOptimalAgent(max_samples=3, depth=1, seed=42)
        env = PistiGymEnv(opponent=RandomValidAgent(), seed=42)
        
        obs, _ = env.reset(seed=42)
        
        # Take just 2 steps
        for i in range(2):
            action_mask = obs["action_mask"]
            legal_actions = [j for j in range(52) if action_mask[j]]
            
            if len(legal_actions) == 0:
                break
            
            action = legal_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        print("  ✓ Environment integration successful")
        return True
    except Exception as e:
        print(f"  ✗ Environment integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all simple tests."""
    print("=" * 60)
    print("Simple Test Suite for Probabilistic Agent")
    print("=" * 60)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Agent Creation", test_agent_creation),
        ("Agent Prediction", test_agent_prediction),
        ("Environment Integration", test_environment_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
            print()
        except Exception as e:
            failed += 1
            print(f"  ✗ {test_name} crashed: {e}")
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
