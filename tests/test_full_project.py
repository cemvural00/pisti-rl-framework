"""Comprehensive pytest test suite for entire Pişti RL project."""

import pytest
import numpy as np


def test_core_engine_cards():
    """Test core engine cards module."""
    from engine.cards import Card, Deck, card_to_id, id_to_card
    
    card = Card("A", "S")
    card_id = card_to_id(card)
    assert 0 <= card_id < 52
    recovered = id_to_card(card_id)
    assert recovered.rank == "A" and recovered.suit == "S"
    
    deck = Deck(seed=42)
    assert len(deck) == 52
    dealt = deck.deal(4)
    assert len(dealt) == 4
    assert len(deck) == 48


def test_core_engine_rules():
    """Test core engine rules module."""
    from engine.rules import check_capture, calculate_pisti
    from engine.cards import Card
    
    top = Card("5", "H")
    match = Card("5", "S")
    jack = Card("J", "D")
    
    assert check_capture(match, top) is True
    assert check_capture(jack, top) is True
    assert check_capture(Card("6", "S"), top) is False
    
    pisti_score = calculate_pisti(1, match, top, False)
    assert pisti_score == 10


def test_core_engine_state():
    """Test core engine state module."""
    from engine.state import GameState
    from engine.cards import Card
    
    state = GameState()
    state.hands[0] = [Card("5", "H"), Card("6", "S")]
    state.hands[1] = [Card("7", "D")]
    state.table_pile = [Card("5", "C")]
    
    legal = state.get_legal_actions(0)
    assert len(legal) == 2


def test_observation_encoding():
    """Test observation encoding system."""
    from encoding.encoders import MultiHotEncoder, CNNEncoder, FeatureEncoder
    from engine.state import GameState
    from engine.cards import Deck, Card
    from engine.rules import deal_initial_table
    
    # Create test state
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
    
    assert "hand" in obs
    assert "table_top" in obs
    assert "action_mask" in obs
    assert obs["hand"].shape == (52,)
    
    # Test CNNEncoder
    cnn_encoder = CNNEncoder()
    cnn_obs = cnn_encoder.encode(state, player_id=0)
    assert "hand_cnn" in cnn_obs
    assert cnn_obs["hand_cnn"].shape == (4, 13)
    
    # Test FeatureEncoder
    feat_encoder = FeatureEncoder(config={})
    feat_obs = feat_encoder.encode(state, player_id=0)
    assert "features" in feat_obs


def test_environments():
    """Test environment wrappers."""
    from envs.pisti_gym import PistiGymEnv
    from agents.baselines import RandomValidAgent
    
    # Test Gymnasium (always available)
    env = PistiGymEnv(opponent=RandomValidAgent(), seed=42)
    obs, info = env.reset(seed=42)
    
    assert isinstance(obs, dict)
    assert "hand" in obs
    assert "action_mask" in obs
    
    action_mask = obs["action_mask"]
    legal_actions = np.where(action_mask)[0]
    if len(legal_actions) > 0:
        action = int(legal_actions[0])
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (int, float))
    
    # Test PettingZoo (if available)
    try:
        from envs.pisti_pettingzoo import PistiPettingZooEnv
        pz_env = PistiPettingZooEnv(seed=42)
        observations, infos = pz_env.reset(seed=42)
        
        assert "player_0" in observations
        assert "player_1" in observations
    except ImportError:
        pytest.skip("PettingZoo not available")
    
    # Test Gymnasium
    env = PistiGymEnv(opponent=RandomValidAgent(), seed=42)
    obs, info = env.reset(seed=42)
    
    assert isinstance(obs, dict)
    assert "hand" in obs
    assert "action_mask" in obs
    
    action_mask = obs["action_mask"]
    legal_actions = np.where(action_mask)[0]
    if len(legal_actions) > 0:
        action = int(legal_actions[0])
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (int, float))
    
    # Test PettingZoo
    pz_env = PistiPettingZooEnv(seed=42)
    observations, infos = pz_env.reset(seed=42)
    
    assert "player_0" in observations
    assert "player_1" in observations


def test_baseline_agents():
    """Test baseline agents."""
    from agents.baselines import RandomValidAgent, GreedyCaptureAgent, PistiHunterAgent
    
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
    
    random_agent = RandomValidAgent()
    action = random_agent.predict(obs, obs["action_mask"])
    assert action in [0, 1]
    
    greedy_agent = GreedyCaptureAgent()
    action = greedy_agent.predict(obs, obs["action_mask"])
    assert action in [0, 1]
    
    pisti_agent = PistiHunterAgent()
    action = pisti_agent.predict(obs, obs["action_mask"])
    assert action in [0, 1]


def test_probabilistic_agent():
    """Test probabilistic agent."""
    from agents.probabilistic_agent import ProbabilisticOptimalAgent
    
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
    assert action in [0, 1]


def test_reward_functions():
    """Test reward functions."""
    from engine.rewards import sparse_reward, shaped_reward
    from engine.state import GameState
    from engine.cards import Card
    
    state = GameState()
    state.hands[0] = []
    state.hands[1] = []
    state.stock = []
    state.score_breakdown[0] = {"aces": 2, "jacks": 1, "got_2c": 0, "got_10d": 0, "pistis": 0, "double_pistis": 0}
    state.score_breakdown[1] = {"aces": 1, "jacks": 0, "got_2c": 0, "got_10d": 0, "pistis": 0, "double_pistis": 0}
    state.captured[0] = [Card("A", "S")] * 2 + [Card("J", "H")]
    state.captured[1] = [Card("A", "D")]
    
    assert state.is_terminal()
    
    reward = sparse_reward(state, player_id=0)
    assert isinstance(reward, (int, float))
    
    reward_shaped = shaped_reward(state, player_id=0, prev_state=None)
    assert isinstance(reward_shaped, (int, float))


def test_integration():
    """Test integration between components."""
    from envs.pisti_gym import PistiGymEnv
    from agents.baselines import RandomValidAgent, GreedyCaptureAgent
    from agents.probabilistic_agent import ProbabilisticOptimalAgent
    
    agents = [
        RandomValidAgent(),
        GreedyCaptureAgent(),
        ProbabilisticOptimalAgent(max_samples=3, depth=1, seed=42),
    ]
    
    for agent in agents:
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
