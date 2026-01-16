"""Baseline agent policies for Pişti."""

from typing import Dict
import numpy as np
from engine.cards import Card, card_to_id, id_to_card, get_rank, is_jack


class RandomValidAgent:
    """Random agent that plays a random legal card."""

    def predict(self, obs: Dict, action_mask: np.ndarray) -> int:
        """
        Predict action (random legal card).
        
        Args:
            obs: Observation dict
            action_mask: Boolean array of legal actions
        
        Returns:
            Card ID (0-51)
        """
        legal_actions = np.where(action_mask)[0]
        if len(legal_actions) == 0:
            return 0  # Fallback
        return int(np.random.choice(legal_actions))


class GreedyCaptureAgent:
    """Greedy agent that captures if possible, else plays low-value card."""

    def predict(self, obs: Dict, action_mask: np.ndarray) -> int:
        """
        Predict action (greedy capture strategy).
        
        Strategy:
        1. If can capture (match top card rank or have Jack), do so
        2. Prefer capturing with Jack if available
        3. Else, play lowest rank card
        
        Args:
            obs: Observation dict
            action_mask: Boolean array of legal actions
        
        Returns:
            Card ID (0-51)
        """
        legal_actions = np.where(action_mask)[0]
        if len(legal_actions) == 0:
            return 0
        
        # Get hand cards
        hand = obs.get("hand", np.zeros(52))
        hand_cards = [card_id for card_id in legal_actions if hand[card_id] > 0.5]
        
        # Get table top card
        table_top = obs.get("table_top", np.zeros(52))
        top_card_id = np.where(table_top > 0.5)[0]
        
        # If table has a card, try to capture
        if len(top_card_id) > 0:
            top_card_id = top_card_id[0]
            top_rank = get_rank(top_card_id)
            
            # Check for Jack in hand (can capture anything)
            jacks_in_hand = [
                card_id
                for card_id in hand_cards
                if is_jack(card_id)
            ]
            if jacks_in_hand:
                return int(jacks_in_hand[0])  # Play first Jack
            
            # Check for rank match
            matching_cards = [
                card_id
                for card_id in hand_cards
                if get_rank(card_id) == top_rank
            ]
            if matching_cards:
                return int(matching_cards[0])  # Play first matching card
        
        # No capture possible: play lowest rank card
        # Rank order: 2,3,4,5,6,7,8,9,10,J,Q,K,A
        # Lower rank_id (0-12) = lower card (2 is 0, A is 12)
        # So we want minimum rank_id
        if hand_cards:
            # Sort by rank_id (card_id % 13)
            hand_cards_sorted = sorted(hand_cards, key=lambda x: x % 13)
            return int(hand_cards_sorted[0])
        
        # Fallback
        return int(legal_actions[0])


class PistiHunterAgent:
    """
    Heuristic agent that tries to set up pişti opportunities.
    
    Strategy:
    1. If can capture, do so (especially if it's a pişti opportunity)
    2. If table has 1 card, try to match it for pişti
    3. If table is empty or has multiple cards, play a card that might set up pişti
    4. Prefer playing cards that opponent is less likely to match
    """

    def predict(self, obs: Dict, action_mask: np.ndarray) -> int:
        """
        Predict action (pişti hunting strategy).
        
        Args:
            obs: Observation dict
            action_mask: Boolean array of legal actions
        
        Returns:
            Card ID (0-51)
        """
        legal_actions = np.where(action_mask)[0]
        if len(legal_actions) == 0:
            return 0
        
        # Get hand cards
        hand = obs.get("hand", np.zeros(52))
        hand_cards = [card_id for card_id in legal_actions if hand[card_id] > 0.5]
        
        # Get table state
        table_top = obs.get("table_top", np.zeros(52))
        table_count = int(obs.get("table_count", [0])[0])
        top_card_id = np.where(table_top > 0.5)[0]
        
        # If table has exactly 1 card, try to match for pişti
        if table_count == 1 and len(top_card_id) > 0:
            top_card_id = top_card_id[0]
            top_rank = get_rank(top_card_id)
            
            # Prefer rank match (not Jack) for regular pişti
            matching_cards = [
                card_id
                for card_id in hand_cards
                if get_rank(card_id) == top_rank and not is_jack(card_id)
            ]
            if matching_cards:
                return int(matching_cards[0])
            
            # Or Jack for double pişti if top is Jack
            if is_jack(top_card_id):
                jacks_in_hand = [
                    card_id for card_id in hand_cards if is_jack(card_id)
                ]
                if jacks_in_hand:
                    return int(jacks_in_hand[0])
        
        # If table is empty or has multiple cards, try to set up pişti
        # Play a card that we have duplicates of (so we can match later)
        if table_count == 0 or table_count > 1:
            # Count cards by rank in hand
            rank_counts = {}
            for card_id in hand_cards:
                rank = get_rank(card_id)
                if rank not in rank_counts:
                    rank_counts[rank] = []
                rank_counts[rank].append(card_id)
            
            # Prefer playing a card where we have duplicates
            for rank, cards in rank_counts.items():
                if len(cards) > 1:
                    # Play one of the duplicates
                    return int(cards[0])
        
        # Fallback to greedy capture strategy
        greedy_agent = GreedyCaptureAgent()
        return greedy_agent.predict(obs, action_mask)
