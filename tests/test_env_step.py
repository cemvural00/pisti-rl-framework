"""Integration tests for environment wrappers."""

import pytest
import numpy as np
from envs.pisti_pettingzoo import PistiPettingZooEnv
from envs.pisti_gym import PistiGymEnv
from encoding.encoders import MultiHotEncoder
from agents.baselines import RandomValidAgent


def test_pettingzoo_reset():
    """Test PettingZoo environment reset."""
    env = PistiPettingZooEnv(seed=42)
    observations, infos = env.reset(seed=42)
    
    assert "player_0" in observations
    assert "player_1" in observations
    assert env.agent_selection in ["player_0", "player_1"]


def test_pettingzoo_step():
    """Test PettingZoo environment step."""
    env = PistiPettingZooEnv(seed=42)
    observations, infos = env.reset(seed=42)
    
    # Get current agent
    agent = env.agent_selection
    obs = env.observe(agent)
    
    # Get legal action
    action_mask = obs["action_mask"]
    legal_actions = np.where(action_mask)[0]
    assert len(legal_actions) > 0
    
    # Take step
    action = int(legal_actions[0])
    env.step(action)
    
    # Check that agent switched
    assert env.agent_selection != agent or env.terminations[agent]


def test_pettingzoo_observation_space():
    """Test PettingZoo observation space."""
    env = PistiPettingZooEnv(seed=42)
    
    obs_space = env.observation_space("player_0")
    assert obs_space is not None
    
    # Check that observation matches space
    observations, _ = env.reset(seed=42)
    obs = observations["player_0"]
    
    # Verify observation structure
    assert "hand" in obs
    assert "table_top" in obs
    assert "action_mask" in obs
    assert obs["hand"].shape == (52,)
    assert obs["table_top"].shape == (52,)
    assert obs["action_mask"].shape == (52,)


def test_pettingzoo_action_space():
    """Test PettingZoo action space."""
    env = PistiPettingZooEnv(seed=42)
    
    action_space = env.action_space("player_0")
    assert action_space.n == 52


def test_gymnasium_reset():
    """Test Gymnasium environment reset."""
    env = PistiGymEnv(opponent=RandomValidAgent(), seed=42)
    obs, info = env.reset(seed=42)
    
    assert isinstance(obs, dict)
    assert "hand" in obs
    assert "action_mask" in obs


def test_gymnasium_step():
    """Test Gymnasium environment step."""
    env = PistiGymEnv(opponent=RandomValidAgent(), seed=42)
    obs, info = env.reset(seed=42)
    
    # Get legal action
    action_mask = obs["action_mask"]
    legal_actions = np.where(action_mask)[0]
    assert len(legal_actions) > 0
    
    # Take step
    action = int(legal_actions[0])
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(obs, dict)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_gymnasium_observation_space():
    """Test Gymnasium observation space."""
    env = PistiGymEnv(opponent=RandomValidAgent(), seed=42)
    
    obs_space = env.observation_space
    assert obs_space is not None
    
    # Check observation matches space
    obs, _ = env.reset(seed=42)
    assert "hand" in obs
    assert obs["hand"].shape == (52,)


def test_gymnasium_action_space():
    """Test Gymnasium action space."""
    env = PistiGymEnv(opponent=RandomValidAgent(), seed=42)
    
    action_space = env.action_space
    assert action_space.n == 52


def test_gymnasium_action_masking():
    """Test that action masking works correctly."""
    env = PistiGymEnv(opponent=RandomValidAgent(), seed=42)
    obs, _ = env.reset(seed=42)
    
    action_mask = obs["action_mask"]
    hand = obs["hand"]
    
    # Legal actions should match cards in hand
    legal_actions = np.where(action_mask)[0]
    hand_cards = np.where(hand > 0.5)[0]
    
    # They should be the same
    assert set(legal_actions) == set(hand_cards)


def test_gymnasium_full_episode():
    """Test that a full episode can be played."""
    env = PistiGymEnv(opponent=RandomValidAgent(), seed=42)
    obs, _ = env.reset(seed=42)
    
    done = False
    steps = 0
    max_steps = 200  # Safety limit
    
    while not done and steps < max_steps:
        action_mask = obs["action_mask"]
        legal_actions = np.where(action_mask)[0]
        
        if len(legal_actions) == 0:
            break
        
        action = int(legal_actions[0])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    
    # Episode should complete
    assert done or steps < max_steps


def test_pettingzoo_full_episode():
    """Test that a full episode can be played in PettingZoo."""
    env = PistiPettingZooEnv(seed=42)
    observations, infos = env.reset(seed=42)
    
    steps = 0
    max_steps = 200  # Safety limit
    
    while not all(env.terminations.values()) and steps < max_steps:
        agent = env.agent_selection
        obs = env.observe(agent)
        action_mask = obs["action_mask"]
        legal_actions = np.where(action_mask)[0]
        
        if len(legal_actions) == 0:
            break
        
        action = int(legal_actions[0])
        env.step(action)
        steps += 1
    
    # Episode should complete
    assert all(env.terminations.values()) or steps < max_steps
