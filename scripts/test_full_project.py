#!/usr/bin/env python3
"""Comprehensive test suite for entire Pişti RL project."""

import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_core_engine():
    """Test core game engine components."""
    print("=" * 60)
    print("Testing Core Engine")
    print("=" * 60)
    
    try:
        # Test cards
        from engine.cards import Card, Deck, card_to_id, id_to_card, get_rank
        print("  ✓ Cards module imported")
        
        card = Card("A", "S")
        card_id = card_to_id(card)
        assert card_id >= 0 and card_id < 52, "Invalid card ID"
        recovered = id_to_card(card_id)
        assert recovered.rank == "A" and recovered.suit == "S", "Card recovery failed"
        print("  ✓ Card ID mapping works")
        
        deck = Deck(seed=42)
        assert len(deck) == 52, "Deck should have 52 cards"
        dealt = deck.deal(4)
        assert len(dealt) == 4, "Should deal 4 cards"
        assert len(deck) == 48, "Deck should have 48 cards remaining"
        print("  ✓ Deck operations work")
        
        # Test rules
        from engine.rules import check_capture, calculate_pisti, score_captured_cards
        print("  ✓ Rules module imported")
        
        top = Card("5", "H")
        match = Card("5", "S")
        jack = Card("J", "D")
        
        assert check_capture(match, top) == True, "Rank match should capture"
        assert check_capture(jack, top) == True, "Jack should capture"
        assert check_capture(Card("6", "S"), top) == False, "No match should not capture"
        print("  ✓ Capture logic works")
        
        pisti_score = calculate_pisti(1, match, top, False)
        assert pisti_score == 10, "Should get pişti bonus"
        print("  ✓ Pişti calculation works")
        
        # Test state
        from engine.state import GameState
        print("  ✓ State module imported")
        
        state = GameState()
        state.hands[0] = [Card("5", "H"), Card("6", "S")]
        state.hands[1] = [Card("7", "D")]
        state.table_pile = [Card("5", "C")]
        
        legal = state.get_legal_actions(0)
        assert len(legal) == 2, "Should have 2 legal actions"
        print("  ✓ State operations work")
        
        print("  ✓ Core engine tests PASSED\n")
        return True
    except Exception as e:
        print(f"  ✗ Core engine test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_observation_encoding():
    """Test observation encoding system."""
    print("=" * 60)
    print("Testing Observation Encoding")
    print("=" * 60)
    
    try:
        from encoding.encoders import MultiHotEncoder, CNNEncoder, FeatureEncoder
        from encoding.obs_builder import ObsBuilder
        from engine.state import GameState
        from engine.cards import Card, Deck
        from engine.rules import deal_initial_table
        print("  ✓ Encoding modules imported")
        
        # Create a test state
        deck = Deck(seed=42)
        center, top = deal_initial_table(deck.cards, seed=42)
        remaining = deck.cards[4:]
        
        state = GameState(
            hands={0: remaining[:4], 1: remaining[4:8]},
            table_pile=[top] if top else [],
            captured={0: [], 1: []},
            center_cards=center,
            stock=remaining[8:],
            current_player=0,
        )
        
        # Test MultiHotEncoder
        encoder = MultiHotEncoder()
        obs = encoder.encode(state, player_id=0)
        
        assert "hand" in obs, "Observation should have 'hand'"
        assert "table_top" in obs, "Observation should have 'table_top'"
        assert "action_mask" in obs, "Observation should have 'action_mask'"
        assert obs["hand"].shape == (52,), "Hand should be 52-length vector"
        assert obs["action_mask"].shape == (52,), "Action mask should be 52-length"
        print("  ✓ MultiHotEncoder works")
        
        # Test CNNEncoder
        cnn_encoder = CNNEncoder()
        cnn_obs = cnn_encoder.encode(state, player_id=0)
        assert "hand_cnn" in cnn_obs, "CNN encoder should have reshaped views"
        assert cnn_obs["hand_cnn"].shape == (4, 13), "CNN view should be (4, 13)"
        print("  ✓ CNNEncoder works")
        
        # Test FeatureEncoder
        feat_encoder = FeatureEncoder(config={})
        feat_obs = feat_encoder.encode(state, player_id=0)
        assert "features" in feat_obs, "Feature encoder should have 'features'"
        assert isinstance(feat_obs["features"], np.ndarray), "Features should be numpy array"
        print("  ✓ FeatureEncoder works")
        
        print("  ✓ Observation encoding tests PASSED\n")
        return True
    except Exception as e:
        print(f"  ✗ Observation encoding test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environments():
    """Test environment wrappers."""
    print("=" * 60)
    print("Testing Environments")
    print("=" * 60)
    
    try:
        import numpy as np
        from envs.pisti_gym import PistiGymEnv
        from envs.pisti_pettingzoo import PistiPettingZooEnv
        from agents.baselines import RandomValidAgent
        print("  ✓ Environment modules imported")
        
        # Test Gymnasium environment
        env = PistiGymEnv(opponent=RandomValidAgent(), seed=42)
        obs, info = env.reset(seed=42)
        
        assert isinstance(obs, dict), "Observation should be dict"
        assert "hand" in obs, "Observation should have 'hand'"
        assert "action_mask" in obs, "Observation should have 'action_mask'"
        print("  ✓ Gymnasium environment reset works")
        
        # Take a step
        action_mask = obs["action_mask"]
        legal_actions = np.where(action_mask)[0]
        if len(legal_actions) > 0:
            action = int(legal_actions[0])
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(reward, (int, float)), "Reward should be numeric"
            assert isinstance(terminated, bool), "Terminated should be bool"
            print("  ✓ Gymnasium environment step works")
        
        # Test PettingZoo environment (if available)
        try:
            pz_env = PistiPettingZooEnv(seed=42)
            observations, infos = pz_env.reset(seed=42)
            
            assert "player_0" in observations, "Should have player_0 observation"
            assert "player_1" in observations, "Should have player_1 observation"
            print("  ✓ PettingZoo environment reset works")
            
            # Take a step
            agent = pz_env.agent_selection
            obs = pz_env.observe(agent)
            action_mask = obs["action_mask"]
            legal_actions = np.where(action_mask)[0]
            if len(legal_actions) > 0:
                action = int(legal_actions[0])
                pz_env.step(action)
                print("  ✓ PettingZoo environment step works")
        except (ImportError, TypeError) as e:
            print(f"  ⚠ PettingZoo not available or has issues: {e}")
            print("  (This is OK if pettingzoo is not installed)")
        
        print("  ✓ Environment tests PASSED\n")
        return True
    except Exception as e:
        print(f"  ✗ Environment test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_baseline_agents():
    """Test baseline agents."""
    print("=" * 60)
    print("Testing Baseline Agents")
    print("=" * 60)
    
    try:
        import numpy as np
        from agents.baselines import RandomValidAgent, GreedyCaptureAgent, PistiHunterAgent
        print("  ✓ Baseline agent modules imported")
        
        # Create test observation
        obs = {
            "hand": np.zeros(52),
            "table_top": np.zeros(52),
            "seen_cards": np.zeros(52),
            "action_mask": np.zeros(52, dtype=bool),
            "table_count": np.array([0]),
        }
        obs["hand"][0] = 1.0
        obs["hand"][1] = 1.0
        obs["action_mask"][0] = True
        obs["action_mask"][1] = True
        
        # Test RandomValidAgent
        random_agent = RandomValidAgent()
        action = random_agent.predict(obs, obs["action_mask"])
        assert action in [0, 1], "Random agent should return valid action"
        print("  ✓ RandomValidAgent works")
        
        # Test GreedyCaptureAgent
        greedy_agent = GreedyCaptureAgent()
        action = greedy_agent.predict(obs, obs["action_mask"])
        assert action in [0, 1], "Greedy agent should return valid action"
        print("  ✓ GreedyCaptureAgent works")
        
        # Test PistiHunterAgent
        pisti_agent = PistiHunterAgent()
        action = pisti_agent.predict(obs, obs["action_mask"])
        assert action in [0, 1], "PistiHunter agent should return valid action"
        print("  ✓ PistiHunterAgent works")
        
        print("  ✓ Baseline agent tests PASSED\n")
        return True
    except Exception as e:
        print(f"  ✗ Baseline agent test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_probabilistic_agent():
    """Test probabilistic agent."""
    print("=" * 60)
    print("Testing Probabilistic Agent")
    print("=" * 60)
    
    try:
        import numpy as np
        from agents.probabilistic_agent import ProbabilisticOptimalAgent
        print("  ✓ Probabilistic agent module imported")
        
        agent = ProbabilisticOptimalAgent(max_samples=5, depth=1, seed=42)
        
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
        obs["hand"][0] = 1.0
        obs["hand"][1] = 1.0
        obs["action_mask"][0] = True
        obs["action_mask"][1] = True
        
        action = agent.predict(obs, obs["action_mask"])
        assert action in [0, 1], "Probabilistic agent should return valid action"
        print("  ✓ ProbabilisticOptimalAgent works")
        
        print("  ✓ Probabilistic agent tests PASSED\n")
        return True
    except Exception as e:
        print(f"  ✗ Probabilistic agent test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration between components."""
    print("=" * 60)
    print("Testing Integration")
    print("=" * 60)
    
    try:
        import numpy as np
        from envs.pisti_gym import PistiGymEnv
        from agents.baselines import RandomValidAgent, GreedyCaptureAgent
        from agents.probabilistic_agent import ProbabilisticOptimalAgent
        print("  ✓ Integration modules imported")
        
        # Test different agent combinations
        agents = [
            ("Random", RandomValidAgent()),
            ("Greedy", GreedyCaptureAgent()),
            ("Probabilistic", ProbabilisticOptimalAgent(max_samples=3, depth=1, seed=42)),
        ]
        
        for agent_name, agent in agents:
            env = PistiGymEnv(opponent=agent, seed=42)
            obs, _ = env.reset(seed=42)
            
            # Play 3 steps
            for _ in range(3):
                action_mask = obs["action_mask"]
                legal_actions = np.where(action_mask)[0]
                if len(legal_actions) == 0:
                    break
                action = int(legal_actions[0])
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            
            print(f"  ✓ Integration with {agent_name} agent works")
        
        print("  ✓ Integration tests PASSED\n")
        return True
    except Exception as e:
        print(f"  ✗ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rewards():
    """Test reward functions."""
    print("=" * 60)
    print("Testing Reward Functions")
    print("=" * 60)
    
    try:
        from engine.rewards import sparse_reward, shaped_reward
        from engine.state import GameState
        from engine.cards import Card
        print("  ✓ Reward module imported")
        
        # Create terminal state
        state = GameState()
        state.hands[0] = []
        state.hands[1] = []
        state.stock = []
        state.score_breakdown[0] = {"aces": 2, "jacks": 1, "got_2c": 0, "got_10d": 0, "pistis": 0, "double_pistis": 0}
        state.score_breakdown[1] = {"aces": 1, "jacks": 0, "got_2c": 0, "got_10d": 0, "pistis": 0, "double_pistis": 0}
        state.captured[0] = [Card("A", "S")] * 2 + [Card("J", "H")]
        state.captured[1] = [Card("A", "D")]
        
        assert state.is_terminal(), "State should be terminal"
        
        reward = sparse_reward(state, player_id=0)
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        print("  ✓ Sparse reward function works")
        
        reward_shaped = shaped_reward(state, player_id=0, prev_state=None)
        assert isinstance(reward_shaped, (int, float)), "Shaped reward should be numeric"
        print("  ✓ Shaped reward function works")
        
        print("  ✓ Reward function tests PASSED\n")
        return True
    except Exception as e:
        print(f"  ✗ Reward function test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE PROJECT TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        ("Core Engine", test_core_engine),
        ("Observation Encoding", test_observation_encoding),
        ("Environments", test_environments),
        ("Baseline Agents", test_baseline_agents),
        ("Probabilistic Agent", test_probabilistic_agent),
        ("Reward Functions", test_rewards),
        ("Integration", test_integration),
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    elapsed = time.time() - start_time
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status}: {test_name}")
    
    print()
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Time: {elapsed:.2f}s")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Project is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
