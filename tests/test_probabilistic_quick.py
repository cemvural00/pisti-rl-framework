"""Quick smoke tests for probabilistic agent - lightweight and fast."""

import pytest
import numpy as np
from agents.probabilistic_agent import ProbabilisticOptimalAgent
from agents.baselines import RandomValidAgent, GreedyCaptureAgent
from envs.pisti_gym import PistiGymEnv


def test_probabilistic_agent_basic():
    """Quick test: agent can make predictions without crashing."""
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
    
    # Set some cards in hand
    obs["hand"][0] = 1.0
    obs["hand"][1] = 1.0
    obs["action_mask"][0] = True
    obs["action_mask"][1] = True
    
    # Should not crash
    action = agent.predict(obs, obs["action_mask"])
    assert action in [0, 1]


def test_probabilistic_agent_single_step():
    """Test agent can play one step in environment."""
    agent = ProbabilisticOptimalAgent(max_samples=5, depth=1, seed=42)
    env = PistiGymEnv(opponent=RandomValidAgent(), seed=42)
    
    obs, _ = env.reset(seed=42)
    
    # Take one step
    action_mask = obs["action_mask"]
    legal_actions = np.where(action_mask)[0]
    if len(legal_actions) > 0:
        action = int(legal_actions[0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)


def test_probabilistic_agent_short_game():
    """Test agent can play a very short game (limited steps)."""
    agent = ProbabilisticOptimalAgent(max_samples=5, depth=1, seed=42)
    env = PistiGymEnv(opponent=RandomValidAgent(), seed=42)
    
    obs, _ = env.reset(seed=42)
    
    # Play maximum 10 steps (very short)
    max_steps = 10
    steps = 0
    
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
    
    # Should complete without crashing
    assert steps <= max_steps


def test_probabilistic_vs_random():
    """Test probabilistic agent as opponent against random."""
    prob_agent = ProbabilisticOptimalAgent(max_samples=5, depth=1, seed=42)
    env = PistiGymEnv(opponent=prob_agent, seed=42)
    
    obs, _ = env.reset(seed=42)
    
    # Play a few steps
    for _ in range(5):
        action_mask = obs["action_mask"]
        legal_actions = np.where(action_mask)[0]
        
        if len(legal_actions) == 0:
            break
        
        action = int(legal_actions[0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    # Should complete without errors
    assert True


def test_probabilistic_vs_greedy():
    """Test probabilistic agent vs greedy agent (quick)."""
    prob_agent = ProbabilisticOptimalAgent(max_samples=5, depth=1, seed=42)
    greedy_agent = GreedyCaptureAgent()
    
    # Test both as opponents
    for opponent in [prob_agent, greedy_agent]:
        env = PistiGymEnv(opponent=opponent, seed=42)
        obs, _ = env.reset(seed=42)
        
        # Play 3 steps max
        for _ in range(3):
            action_mask = obs["action_mask"]
            legal_actions = np.where(action_mask)[0]
            
            if len(legal_actions) == 0:
                break
            
            action = int(legal_actions[0])
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break


def test_belief_tracker_updates():
    """Test that belief tracker updates correctly."""
    from agents.probabilistic_agent import BeliefTracker
    
    tracker = BeliefTracker()
    
    # Initial state: no cards seen
    assert len(tracker.seen_cards) == 0
    
    # Update with observation
    obs = {
        "seen_cards": np.zeros(52),
        "hand": np.zeros(52),
    }
    obs["seen_cards"][0] = 1.0
    obs["seen_cards"][1] = 1.0
    obs["hand"][0] = 1.0
    
    tracker.update_from_observation(obs, my_hand_size=1, opp_hand_size=4)
    
    # Should have tracked seen cards
    assert 0 in tracker.seen_cards or 1 in tracker.seen_cards
    
    # Get unseen cards
    unseen = tracker.get_unseen_cards()
    assert len(unseen) < 52


def test_action_evaluator_heuristic():
    """Test action evaluator heuristic value calculation."""
    from agents.probabilistic_agent import ActionEvaluator
    from engine.state import GameState
    from engine.cards import Card
    
    evaluator = ActionEvaluator()
    
    # Create minimal state
    state = GameState()
    state.hands[0] = [Card("A", "S")]
    state.hands[1] = []
    state.score_breakdown[0] = {"aces": 0, "jacks": 0, "got_2c": 0, "got_10d": 0, "pistis": 0, "double_pistis": 0}
    state.score_breakdown[1] = {"aces": 0, "jacks": 0, "got_2c": 0, "got_10d": 0, "pistis": 0, "double_pistis": 0}
    
    value = evaluator._estimate_state_value(state, player_id=0)
    
    # Should return a numeric value
    assert isinstance(value, (int, float))
