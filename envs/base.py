"""Base game engine shared by environment wrappers."""

from typing import Dict, Optional
import numpy as np
from engine.cards import Deck, Card, id_to_card
from engine.state import GameState
from engine.rules import deal_initial_table
from engine.rewards import sparse_reward, shaped_reward
from encoding.encoders import ObservationEncoder, MultiHotEncoder


class PistiGameEngine:
    """Shared game engine for PettingZoo and Gymnasium wrappers."""

    def __init__(
        self,
        encoder: Optional[ObservationEncoder] = None,
        reward_config: Optional[Dict] = None,
        game_config: Optional[Dict] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize game engine.
        
        Args:
            encoder: Observation encoder (default: MultiHotEncoder)
            reward_config: Reward configuration dict
            game_config: Game configuration dict
            seed: Random seed
        """
        self.encoder = encoder or MultiHotEncoder()
        self.reward_config = reward_config or {}
        self.game_config = game_config or {}
        self.seed = seed
        self.state: Optional[GameState] = None
        self.prev_state: Optional[GameState] = None

    def reset(self, seed: Optional[int] = None) -> GameState:
        """
        Reset game to initial state.
        
        Args:
            seed: Optional random seed (overrides instance seed)
        
        Returns:
            Initial game state
        """
        if seed is not None:
            self.seed = seed
        
        # Create and shuffle deck
        deck = Deck(seed=self.seed)
        
        # Deal initial table cards
        center_cards, top_card = deal_initial_table(deck.cards, seed=self.seed)
        remaining_deck = deck.cards[4:]
        
        # Deal 4 cards to each player
        hands = {0: remaining_deck[:4], 1: remaining_deck[4:8]}
        stock = remaining_deck[8:]
        
        # Create initial state
        self.state = GameState(
            hands=hands,
            table_pile=[top_card] if top_card else [],
            captured={0: [], 1: []},
            center_cards=center_cards,
            stock=stock,
            current_player=0,
            first_capture_made=False,
        )
        
        self.prev_state = None
        return self.state

    def step(self, action: int) -> tuple[GameState, float, bool, Dict]:
        """
        Apply action and return new state, reward, done, info.
        
        Args:
            action: Card ID (0-51)
        
        Returns:
            (new_state, reward, done, info)
        """
        if self.state is None:
            raise ValueError("Game not initialized. Call reset() first.")
        
        # Convert action to card
        card = id_to_card(action)
        
        # Check if action is legal
        legal_actions = self.state.get_legal_actions(self.state.current_player)
        if action not in legal_actions:
            # Invalid action: return negative reward and don't advance
            return self.state, -10.0, False, {"invalid_action": True}
        
        # Store previous state for reward calculation
        self.prev_state = self.state
        
        # Apply action
        self.state = self.state.apply_action(card, config=self.game_config)
        
        # Calculate reward
        player_id = self.prev_state.current_player
        use_shaping = self.reward_config.get("shaping", {}).get("enabled", False)
        
        if use_shaping:
            reward = shaped_reward(
                self.state, player_id, self.prev_state, self.reward_config.get("shaping")
            )
        else:
            reward = sparse_reward(self.state, player_id, self.prev_state)
        
        # Check if terminal
        done = self.state.is_terminal()
        
        info = {}
        if done:
            scores = self.state.get_final_scores()
            info["final_scores"] = scores
            info["score_diff"] = scores[0] - scores[1]
        
        return self.state, reward, done, info

    def get_observation(self, player_id: int) -> Dict[str, np.ndarray]:
        """
        Get observation for a player.
        
        Args:
            player_id: Player ID (0 or 1)
        
        Returns:
            Observation dict
        """
        if self.state is None:
            raise ValueError("Game not initialized. Call reset() first.")
        
        return self.encoder.encode(self.state, player_id)

    def get_reward(self, player_id: int) -> float:
        """
        Get reward for a player (for current state).
        
        Args:
            player_id: Player ID (0 or 1)
        
        Returns:
            Reward value
        """
        if self.state is None:
            raise ValueError("Game not initialized. Call reset() first.")
        
        use_shaping = self.reward_config.get("shaping", {}).get("enabled", False)
        
        if use_shaping:
            return shaped_reward(
                self.state, player_id, self.prev_state, self.reward_config.get("shaping")
            )
        else:
            return sparse_reward(self.state, player_id, self.prev_state)
