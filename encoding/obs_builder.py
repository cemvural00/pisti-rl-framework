"""Observation builder: constructs legal observations from game state."""

import numpy as np
from typing import Dict
from engine.state import GameState
from engine.cards import card_to_id


class ObsBuilder:
    """Builds legal observations from GameState."""

    def __init__(self):
        """Initialize observation builder."""
        pass

    def build_hand(self, state: GameState, player_id: int) -> np.ndarray:
        """
        Build multi-hot vector for player's hand.
        
        Returns:
            (52,) array where 1 indicates card is in hand, 0 otherwise
        """
        hand_vector = np.zeros(52, dtype=np.float32)
        for card in state.hands[player_id]:
            card_id = card_to_id(card)
            hand_vector[card_id] = 1.0
        return hand_vector

    def build_table_top(self, state: GameState) -> np.ndarray:
        """
        Build one-hot vector for top card of table pile.
        
        Returns:
            (52,) array where 1 at top card position, all zeros if table empty
        """
        table_vector = np.zeros(52, dtype=np.float32)
        top_card = state.get_table_top_card()
        if top_card is not None:
            card_id = card_to_id(top_card)
            table_vector[card_id] = 1.0
        return table_vector

    def build_seen_cards(self, state: GameState, player_id: int) -> np.ndarray:
        """
        Build multi-hot vector for all cards seen by player.
        
        Includes:
        - Cards in player's hand
        - Cards in table pile (all visible)
        - Cards captured by either player (visible after capture)
        - Center cards (visible to player who made first capture)
        
        Returns:
            (52,) array where 1 indicates card has been revealed
        """
        seen_vector = np.zeros(52, dtype=np.float32)
        
        # Player's hand
        for card in state.hands[player_id]:
            card_id = card_to_id(card)
            seen_vector[card_id] = 1.0
        
        # Table pile (all visible)
        for card in state.table_pile:
            card_id = card_to_id(card)
            seen_vector[card_id] = 1.0
        
        # Captured cards (visible after capture)
        for player_captured in state.captured.values():
            for card in player_captured:
                card_id = card_to_id(card)
                seen_vector[card_id] = 1.0
        
        # Center cards (visible to player who made first capture)
        if state.first_capture_made:
            # Find who made first capture (player with center cards)
            for player_id_check in [0, 1]:
                if len(state.captured[player_id_check]) > 0:
                    # Check if they have more cards than expected (center cards included)
                    # Actually, center cards are in captured, so already counted above
                    pass
        
        return seen_vector

    def build_score_breakdown(self, state: GameState, player_id: int) -> np.ndarray:
        """
        Build score breakdown vector.
        
        Returns:
            Array with [aces, jacks, got_2c, got_10d, pistis, double_pistis]
        """
        breakdown = state.score_breakdown[player_id]
        return np.array(
            [
                breakdown["aces"],
                breakdown["jacks"],
                breakdown["got_2c"],
                breakdown["got_10d"],
                breakdown["pistis"],
                breakdown["double_pistis"],
            ],
            dtype=np.float32,
        )

    def build_action_mask(self, state: GameState, player_id: int) -> np.ndarray:
        """
        Build action mask for legal actions.
        
        Returns:
            (52,) bool array where 1 indicates legal action, 0 otherwise
        """
        mask = np.zeros(52, dtype=bool)
        legal_actions = state.get_legal_actions(player_id)
        for card_id in legal_actions:
            mask[card_id] = True
        return mask

    def build_scalar_features(
        self, state: GameState, player_id: int
    ) -> Dict[str, np.ndarray]:
        """
        Build scalar features (counts, etc.).
        
        Returns:
            Dict with scalar features
        """
        features = {
            "table_count": np.array([len(state.table_pile)], dtype=np.float32),
            "my_captured_count": np.array(
                [len(state.captured[player_id])], dtype=np.float32
            ),
            "opp_captured_count": np.array(
                [len(state.captured[1 - player_id])], dtype=np.float32
            ),
            "stock_remaining": np.array([len(state.stock)], dtype=np.float32),
            "hand_size": np.array([len(state.hands[player_id])], dtype=np.float32),
        }
        
        # Last capture indicator
        if state.move_history:
            last_player, _, _ = state.move_history[-1]
            if last_player == player_id:
                features["last_capture_by"] = np.array([1.0], dtype=np.float32)
            else:
                features["last_capture_by"] = np.array([-1.0], dtype=np.float32)
        else:
            features["last_capture_by"] = np.array([0.0], dtype=np.float32)
        
        # Last move card (one-hot)
        if state.move_history:
            _, last_card, _ = state.move_history[-1]
            last_card_vector = np.zeros(52, dtype=np.float32)
            last_card_id = card_to_id(last_card)
            last_card_vector[last_card_id] = 1.0
            features["last_move_card"] = last_card_vector
        else:
            features["last_move_card"] = np.zeros(52, dtype=np.float32)
        
        # Running score estimate (excluding majority until end)
        my_score = (
            state.score_breakdown[player_id]["aces"]
            + state.score_breakdown[player_id]["jacks"]
            + 2 * state.score_breakdown[player_id]["got_2c"]
            + 3 * state.score_breakdown[player_id]["got_10d"]
            + 10 * state.score_breakdown[player_id]["pistis"]
            + 20 * state.score_breakdown[player_id]["double_pistis"]
        )
        opp_score = (
            state.score_breakdown[1 - player_id]["aces"]
            + state.score_breakdown[1 - player_id]["jacks"]
            + 2 * state.score_breakdown[1 - player_id]["got_2c"]
            + 3 * state.score_breakdown[1 - player_id]["got_10d"]
            + 10 * state.score_breakdown[1 - player_id]["pistis"]
            + 20 * state.score_breakdown[1 - player_id]["double_pistis"]
        )
        features["running_score_estimate"] = np.array(
            [my_score - opp_score], dtype=np.float32
        )
        
        return features
