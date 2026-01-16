"""Unit tests for probabilistic optimal agent."""

import pytest
import numpy as np
from agents.probabilistic_agent import (
    BeliefTracker,
    ActionEvaluator,
    ProbabilisticOptimalAgent,
)
from engine.cards import Card, card_to_id
from engine.state import GameState


def test_belief_tracker_initialization():
    """Test BeliefTracker initialization."""
    tracker = BeliefTracker()
    assert len(tracker.seen_cards) == 0
    assert len(tracker.my_hand_history) == 0


def test_belief_tracker_update():
    """Test updating belief from observation."""
    tracker = BeliefTracker()
    
    # Create mock observation
    obs = {
        "seen_cards": np.zeros(52),
        "hand": np.zeros(52),
    }
    obs["seen_cards"][0] = 1.0  # Card 0 seen
    obs["seen_cards"][1] = 1.0  # Card 1 seen
    obs["hand"][0] = 1.0  # Card 0 in hand
    
    tracker.update_from_observation(obs, my_hand_size=1, opp_hand_size=4)
    
    assert 0 in tracker.seen_cards
    assert 1 in tracker.seen_cards
    assert len(tracker.my_hand_history) == 1


def test_belief_tracker_unseen_cards():
    """Test getting unseen cards."""
    tracker = BeliefTracker()
    tracker.seen_cards = {0, 1, 2, 3}
    
    unseen = tracker.get_unseen_cards()
    assert 0 not in unseen
    assert 1 not in unseen
    assert 4 in unseen
    assert 51 in unseen
    assert len(unseen) == 52 - 4


def test_belief_tracker_sample_opponent_hands():
    """Test sampling opponent hands."""
    tracker = BeliefTracker()
    unseen_cards = {10, 11, 12, 13, 14}
    
    samples = tracker.sample_opponent_hands(opp_hand_size=2, unseen_cards=unseen_cards, n_samples=5)
    
    assert len(samples) <= 5
    for hand in samples:
        assert len(hand) == 2
        assert all(card in unseen_cards for card in hand)


def test_action_evaluator_initialization():
    """Test ActionEvaluator initialization."""
    evaluator = ActionEvaluator()
    assert evaluator is not None


def test_action_evaluator_estimate_state_value():
    """Test state value estimation."""
    evaluator = ActionEvaluator()
    
    # Create a simple state
    state = GameState()
    state.hands[0] = [Card("A", "S"), Card("J", "H")]
    state.hands[1] = [Card("5", "D")]
    state.score_breakdown[0] = {"aces": 1, "jacks": 0, "got_2c": 0, "got_10d": 0, "pistis": 0, "double_pistis": 0}
    state.score_breakdown[1] = {"aces": 0, "jacks": 0, "got_2c": 0, "got_10d": 0, "pistis": 0, "double_pistis": 0}
    state.captured[0] = [Card("2", "C")]
    state.captured[1] = []
    
    value = evaluator._estimate_state_value(state, player_id=0)
    
    # Should be positive (player 0 has advantage)
    assert value > 0


def test_probabilistic_agent_initialization():
    """Test ProbabilisticOptimalAgent initialization."""
    agent = ProbabilisticOptimalAgent(max_samples=50, depth=1, seed=42)
    
    assert agent.max_samples == 50
    assert agent.depth == 1
    assert agent.belief_tracker is not None
    assert agent.action_evaluator is not None


def test_probabilistic_agent_predict():
    """Test agent prediction."""
    agent = ProbabilisticOptimalAgent(max_samples=10, depth=1, seed=42)
    
    # Create mock observation
    obs = {
        "hand": np.zeros(52),
        "table_top": np.zeros(52),
        "seen_cards": np.zeros(52),
        "action_mask": np.zeros(52, dtype=bool),
        "table_count": np.array([0]),
        "opp_captured_count": np.array([0]),
        "stock_remaining": np.array([44]),
    }
    
    # Set some cards in hand
    obs["hand"][0] = 1.0  # Card 0
    obs["hand"][1] = 1.0  # Card 1
    obs["action_mask"][0] = True
    obs["action_mask"][1] = True
    
    # Set seen cards
    obs["seen_cards"][0] = 1.0
    obs["seen_cards"][1] = 1.0
    obs["seen_cards"][10] = 1.0  # Some other seen card
    
    action = agent.predict(obs, obs["action_mask"])
    
    # Should return a valid action
    assert action in [0, 1]
    assert obs["action_mask"][action]


def test_probabilistic_agent_with_state():
    """Test agent with state access."""
    from engine.cards import Deck
    from engine.rules import deal_initial_table
    
    agent = ProbabilisticOptimalAgent(max_samples=10, depth=1, seed=42)
    
    # Create a real game state
    deck = Deck(seed=42)
    center_cards, top_card = deal_initial_table(deck.cards, seed=42)
    remaining = deck.cards[4:]
    
    state = GameState(
        hands={0: remaining[:4], 1: remaining[4:8]},
        table_pile=[top_card] if top_card else [],
        captured={0: [], 1: []},
        center_cards=center_cards,
        stock=remaining[8:],
        current_player=0,
    )
    
    # Update agent state
    agent.update_state(state)
    
    # Create observation
    from encoding.obs_builder import ObsBuilder
    from encoding.encoders import MultiHotEncoder
    
    encoder = MultiHotEncoder()
    obs = encoder.encode(state, player_id=0)
    
    # Predict action
    action = agent.predict(obs, obs["action_mask"])
    
    # Should be a valid action
    assert action in state.get_legal_actions(0)


def test_probabilistic_agent_heuristic_value():
    """Test heuristic action value calculation."""
    agent = ProbabilisticOptimalAgent(max_samples=10, depth=1, seed=42)
    
    # Test Jack value
    obs = {
        "table_top": np.zeros(52),
        "table_count": np.array([0]),
    }
    jack_card = Card("J", "H")
    jack_id = card_to_id(jack_card)
    value = agent._heuristic_action_value(jack_id, obs)
    assert value > 0  # Jack should have positive value
    
    # Test scoring card value
    ace_card = Card("A", "S")
    ace_id = card_to_id(ace_card)
    value = agent._heuristic_action_value(ace_id, obs)
    assert value > 0  # Ace should have positive value


def test_probabilistic_agent_integration():
    """Test agent integration with environment."""
    from envs.pisti_gym import PistiGymEnv
    from agents.baselines import RandomValidAgent
    
    # Create environment with probabilistic agent as opponent
    prob_agent = ProbabilisticOptimalAgent(max_samples=20, depth=1, seed=42)
    env = PistiGymEnv(opponent=prob_agent, seed=42)
    
    # Reset and take a few steps
    obs, info = env.reset(seed=42)
    
    # Take a step
    action_mask = obs["action_mask"]
    legal_actions = np.where(action_mask)[0]
    if len(legal_actions) > 0:
        action = int(legal_actions[0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Should complete without errors
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
