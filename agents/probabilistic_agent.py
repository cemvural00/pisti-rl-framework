"""Probabilistic optimal agent using belief tracking and expectimax evaluation."""

from typing import Dict, Set, List, Tuple, Optional
import numpy as np
from itertools import combinations
import random
import math

from engine.cards import Card, card_to_id, id_to_card, get_rank, is_jack, is_scoring_card
from engine.state import GameState
from engine.rules import check_capture, calculate_pisti, get_majority_bonus


class BeliefTracker:
    """Tracks seen cards and infers opponent hand probabilities."""

    def __init__(self):
        """Initialize belief tracker."""
        self.seen_cards: Set[int] = set()  # Card IDs that have been seen
        self.my_hand_history: List[Set[int]] = []  # History of our hands
        self.opponent_hand_size_history: List[int] = []  # History of opponent hand sizes

    def update_from_observation(self, obs: Dict, my_hand_size: int, opp_hand_size: int):
        """
        Update belief from observation.
        
        Args:
            obs: Observation dict with seen_cards, hand, etc.
            my_hand_size: Number of cards in our hand
            opp_hand_size: Number of cards in opponent's hand
        """
        # Extract seen cards from observation
        seen_cards_vector = obs.get("seen_cards", np.zeros(52))
        seen_card_ids = set(np.where(seen_cards_vector > 0.5)[0])
        
        # Extract our hand
        hand_vector = obs.get("hand", np.zeros(52))
        my_hand_ids = set(np.where(hand_vector > 0.5)[0])
        
        # Update seen cards (union of all seen cards)
        self.seen_cards.update(seen_card_ids)
        
        # Track history
        self.my_hand_history.append(my_hand_ids.copy())
        self.opponent_hand_size_history.append(opp_hand_size)

    def get_unseen_cards(self) -> Set[int]:
        """
        Get set of card IDs that haven't been seen yet.
        
        Returns:
            Set of unseen card IDs (0-51)
        """
        all_cards = set(range(52))
        return all_cards - self.seen_cards

    def get_opponent_hand_probs(
        self, opp_hand_size: int, unseen_cards: Set[int]
    ) -> List[Tuple[Tuple[int, ...], float]]:
        """
        Get probability distribution over possible opponent hands.
        
        Uses uniform distribution over all possible hands of given size.
        For efficiency, returns a list of (hand_tuple, probability) pairs.
        
        Args:
            opp_hand_size: Size of opponent's hand
            unseen_cards: Set of unseen card IDs
        
        Returns:
            List of (hand_tuple, probability) pairs
        """
        unseen_list = list(unseen_cards)
        n_unseen = len(unseen_list)
        
        if opp_hand_size == 0 or n_unseen == 0:
            return [((), 1.0)]
        
        if opp_hand_size > n_unseen:
            return [((), 1.0)]  # Invalid state
        
        # Total number of possible hands (n choose k)
        def n_choose_k(n, k):
            """Calculate n choose k."""
            if k > n or k < 0:
                return 0
            if k == 0 or k == n:
                return 1
            k = min(k, n - k)  # Use symmetry
            result = 1
            for i in range(k):
                result = result * (n - i) // (i + 1)
            return result
        
        total_combinations = n_choose_k(n_unseen, opp_hand_size)
        prob = 1.0 / total_combinations if total_combinations > 0 else 0.0
        
        # For efficiency, we'll sample rather than enumerate all combinations
        # Return uniform probability for sampling
        return [(tuple(unseen_list), prob)]  # Simplified: return all unseen cards with uniform prob

    def sample_opponent_hands(
        self, opp_hand_size: int, unseen_cards: Set[int], n_samples: int
    ) -> List[Tuple[int, ...]]:
        """
        Sample possible opponent hands.
        
        Args:
            opp_hand_size: Size of opponent's hand
            unseen_cards: Set of unseen card IDs
            n_samples: Number of samples to generate
        
        Returns:
            List of sampled hand tuples
        """
        unseen_list = list(unseen_cards)
        n_unseen = len(unseen_list)
        
        if opp_hand_size == 0 or n_unseen == 0:
            return [()]
        
        if opp_hand_size > n_unseen:
            return [()]
        
        # Sample random combinations
        if n_unseen <= opp_hand_size:
            return [tuple(unseen_list)]
        
        # Generate samples
        def n_choose_k(n, k):
            """Calculate n choose k."""
            if k > n or k < 0:
                return 0
            if k == 0 or k == n:
                return 1
            k = min(k, n - k)  # Use symmetry
            result = 1
            for i in range(k):
                result = result * (n - i) // (i + 1)
            return result
        
        samples = []
        max_combinations = n_choose_k(n_unseen, opp_hand_size)
        n_samples = min(n_samples, max_combinations)
        
        if max_combinations <= n_samples:
            # Enumerate all if small enough
            for combo in combinations(unseen_list, opp_hand_size):
                samples.append(tuple(sorted(combo)))
        else:
            # Sample randomly
            seen_combos = set()
            while len(samples) < n_samples:
                combo = tuple(sorted(random.sample(unseen_list, opp_hand_size)))
                if combo not in seen_combos:
                    samples.append(combo)
                    seen_combos.add(combo)
        
        return samples


class ActionEvaluator:
    """Evaluates actions using expected value calculation."""

    def __init__(self):
        """Initialize action evaluator."""
        pass

    def evaluate_action(
        self,
        action: int,
        state: GameState,
        opponent_hands: List[Tuple[int, ...]],
        player_id: int,
        depth: int = 1,
    ) -> float:
        """
        Evaluate action using expectimax-style evaluation.
        
        Args:
            action: Card ID to evaluate
            state: Current game state
            opponent_hands: List of possible opponent hands (tuples of card IDs)
            player_id: Player making the action
            depth: Lookahead depth (currently 1 for efficiency)
        
        Returns:
            Expected value of the action
        """
        card = id_to_card(action)
        
        # Check if action is legal
        if action not in state.get_legal_actions(player_id):
            return -1000.0  # Very negative value for illegal actions
        
        # Simulate action
        new_state = state.apply_action(card)
        
        # If terminal, return final score differential
        if new_state.is_terminal():
            scores = new_state.get_final_scores()
            return float(scores[player_id] - scores[1 - player_id])
        
        # Evaluate against possible opponent responses
        if depth > 0 and len(opponent_hands) > 0:
            # Sample a few opponent hands for efficiency
            opp_hands_sample = opponent_hands[:min(10, len(opponent_hands))]
            
            opp_values = []
            for opp_hand in opp_hands_sample:
                # Evaluate opponent's best response (simplified)
                opp_value = self._evaluate_opponent_response(
                    new_state, opp_hand, 1 - player_id
                )
                opp_values.append(opp_value)
            
            # Average over opponent hands
            avg_opp_value = np.mean(opp_values) if opp_values else 0.0
            # Our value is negative of opponent's value (zero-sum)
            return -avg_opp_value
        else:
            # Use heuristic evaluation
            return self._estimate_state_value(new_state, player_id)

    def _evaluate_opponent_response(
        self, state: GameState, opp_hand: Tuple[int, ...], opp_id: int
    ) -> float:
        """
        Evaluate opponent's best response given a specific hand.
        
        Args:
            state: Current game state
            opp_hand: Opponent's hand (tuple of card IDs)
            opp_id: Opponent's player ID
        
        Returns:
            Estimated value for opponent
        """
        # Set opponent's hand (temporarily for evaluation)
        # Note: This modifies state, so we need to be careful
        # For now, just evaluate state heuristically
        return self._estimate_state_value(state, opp_id)

    def _estimate_state_value(self, state: GameState, player_id: int) -> float:
        """
        Estimate value of a state using heuristics.
        
        Args:
            state: Game state to evaluate
            player_id: Player to evaluate for
        
        Returns:
            Estimated value
        """
        # Current score differential
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
        score_diff = my_score - opp_score
        
        # Expected value of cards in hand
        hand_value = 0.0
        for card in state.hands[player_id]:
            if is_scoring_card(card):
                if card.rank == "A":
                    hand_value += 1.0
                elif card.rank == "J":
                    hand_value += 1.0
                elif card == Card("2", "C"):
                    hand_value += 2.0
                elif card == Card("10", "D"):
                    hand_value += 3.0
        
        # Pişti potential: if table has 1 card and we can match it
        pisti_potential = 0.0
        if len(state.table_pile) == 1:
            top_card = state.table_pile[0]
            for card in state.hands[player_id]:
                if card.rank == top_card.rank and card.rank != "J":
                    pisti_potential += 10.0  # Potential pişti
                elif card.rank == "J" and top_card.rank == "J":
                    pisti_potential += 20.0  # Potential double pişti
        
        # Card advantage: majority bonus potential
        my_captured = len(state.captured[player_id])
        opp_captured = len(state.captured[1 - player_id])
        card_advantage = 0.0
        if my_captured > opp_captured:
            card_advantage = 3.0  # Potential majority bonus
        elif my_captured == opp_captured:
            card_advantage = 0.0
        else:
            card_advantage = -3.0
        
        # Jack value: can capture anything
        jack_count = sum(1 for card in state.hands[player_id] if card.rank == "J")
        jack_value = jack_count * 2.0  # Jacks are valuable
        
        # Total value
        total_value = (
            score_diff
            + hand_value * 0.5  # Weight hand value
            + pisti_potential * 0.3  # Weight pişti potential
            + card_advantage * 0.5  # Weight card advantage
            + jack_value * 0.3  # Weight jack value
        )
        
        return total_value


class ProbabilisticOptimalAgent:
    """Agent that plays optimally using probabilistic information."""

    def __init__(
        self,
        max_samples: int = 50,
        depth: int = 1,
        seed: Optional[int] = None,
        temperature: float = 0.0,
    ):
        """
        Initialize probabilistic optimal agent.
        
        Args:
            max_samples: Maximum opponent hands to sample (for efficiency)
            depth: Lookahead depth for expectimax (currently limited to 1)
            seed: Random seed for reproducibility
            temperature: Temperature for softmax action selection (0.0=deterministic, >0=randomized)
        """
        self.belief_tracker = BeliefTracker()
        self.action_evaluator = ActionEvaluator()
        self.max_samples = max_samples
        self.depth = depth
        self.seed = seed
        self.temperature = max(0.0, temperature)  # Ensure non-negative
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Track state for reconstruction
        self.last_state: Optional[GameState] = None

    def _reconstruct_state_from_obs(self, obs: Dict) -> Optional[GameState]:
        """
        Attempt to reconstruct minimal state from observation.
        
        This is challenging since we don't have full state info.
        For now, we'll work with what we have in the observation.
        
        Args:
            obs: Observation dict
        
        Returns:
            Partial state info or None
        """
        # We can't fully reconstruct state, but we can use the last state
        # if available, or work with observation directly
        return self.last_state

    def predict(self, obs: Dict, action_mask: np.ndarray) -> int:
        """
        Predict optimal action using probabilistic evaluation.
        
        Args:
            obs: Observation dict
            action_mask: Boolean array of legal actions
        
        Returns:
            Card ID (0-51) of best action
        """
        # Extract information from observation
        hand_vector = obs.get("hand", np.zeros(52))
        my_hand_ids = set(np.where(hand_vector > 0.5)[0])
        my_hand_size = len(my_hand_ids)
        
        # Estimate opponent hand size from observation
        stock_remaining = int(obs.get("stock_remaining", [0])[0])
        table_count = int(obs.get("table_count", [0])[0])
        my_captured_count = int(obs.get("my_captured_count", [0])[0])
        opp_captured_count = int(obs.get("opp_captured_count", [0])[0])
        
        # Calculate total cards accounted for
        seen_cards_vector = obs.get("seen_cards", np.zeros(52))
        n_seen = np.sum(seen_cards_vector > 0.5)
        
        # Estimate opponent hand size
        # Total cards = 52
        # Accounted: my_hand + table_pile + captured + center_cards (3) + stock
        # Opponent hand = 52 - (my_hand + table + captured + center + stock)
        # But we don't know center cards exactly, so estimate
        center_cards = 3  # Initially 3, then 0 after first capture
        if n_seen > 4:  # If we've seen more than initial 4, center cards are captured
            center_cards = 0
        
        total_accounted = my_hand_size + table_count + my_captured_count + opp_captured_count + stock_remaining + center_cards
        opp_hand_size = max(0, 52 - total_accounted)
        
        # Clamp to reasonable values
        if stock_remaining == 0 and my_hand_size == 0:
            opp_hand_size = 0
        elif stock_remaining > 0:
            # During dealing phase, hands are typically 4 cards
            opp_hand_size = min(4, opp_hand_size) if my_hand_size > 0 else 4
        
        # Update belief tracker
        self.belief_tracker.update_from_observation(obs, my_hand_size, opp_hand_size)
        
        # Get unseen cards
        unseen_cards = self.belief_tracker.get_unseen_cards()
        
        # Sample possible opponent hands
        if opp_hand_size > 0 and len(unseen_cards) >= opp_hand_size:
            opponent_hands = self.belief_tracker.sample_opponent_hands(
                opp_hand_size, unseen_cards, self.max_samples
            )
        else:
            opponent_hands = [()]
        
        # Get legal actions
        legal_actions = np.where(action_mask)[0]
        if len(legal_actions) == 0:
            return 0
        
        # If we have access to state, use it; otherwise use observation-based evaluation
        if self.last_state is not None:
            # Evaluate each legal action
            action_values = []
            for action in legal_actions:
                try:
                    value = self.action_evaluator.evaluate_action(
                        action, self.last_state, opponent_hands, 0, self.depth
                    )
                    action_values.append((action, value))
                except Exception as e:
                    # Fallback: use simple heuristic
                    action_values.append((action, self._heuristic_action_value(action, obs)))
        else:
            # Use observation-based heuristic evaluation
            action_values = [
                (action, self._heuristic_action_value(action, obs))
                for action in legal_actions
            ]
        
        # Select action based on temperature
        if not action_values:
            return int(legal_actions[0])
        
        if self.temperature > 0.0:
            # Softmax selection with temperature
            values_array = np.array([v for _, v in action_values])
            
            # Apply temperature scaling: divide values by temperature
            scaled_values = values_array / self.temperature
            
            # Compute softmax probabilities
            # Subtract max for numerical stability
            exp_values = np.exp(scaled_values - np.max(scaled_values))
            probs = exp_values / np.sum(exp_values)
            
            # Sample action from distribution
            action_idx = np.random.choice(len(action_values), p=probs)
            return int(action_values[action_idx][0])
        else:
            # Deterministic: select best action (current behavior)
            best_action = max(action_values, key=lambda x: x[1])[0]
            return int(best_action)

    def _heuristic_action_value(self, action: int, obs: Dict) -> float:
        """
        Heuristic evaluation of action when full state is unavailable.
        
        Args:
            action: Card ID
            obs: Observation dict
        
        Returns:
            Heuristic value
        """
        card = id_to_card(action)
        value = 0.0
        
        # Check if can capture
        table_top = obs.get("table_top", np.zeros(52))
        top_card_id = np.where(table_top > 0.5)[0]
        table_count = int(obs.get("table_count", [0])[0])
        
        if len(top_card_id) > 0:
            top_card_id = top_card_id[0]
            top_rank = get_rank(top_card_id)
            
            # Can capture?
            if card.rank == "J" or card.rank == top_rank:
                value += 5.0  # Capture bonus
                
                # Pişti potential
                if table_count == 1:
                    if card.rank == top_rank and card.rank != "J":
                        value += 10.0  # Pişti
                    elif card.rank == "J" and top_rank == "J":
                        value += 20.0  # Double pişti
        
        # Scoring card value
        if is_scoring_card(card):
            if card.rank == "A":
                value += 1.0
            elif card.rank == "J":
                value += 1.0
            elif card == Card("2", "C"):
                value += 2.0
            elif card == Card("10", "D"):
                value += 3.0
        
        # Jack is always valuable
        if card.rank == "J":
            value += 2.0
        
        # Prefer lower cards if can't capture (dump strategy)
        if value == 0.0:
            rank_id = action % 13
            value = -rank_id * 0.1  # Lower rank = slightly better to dump
        
        return value

    def update_state(self, state: GameState):
        """
        Update internal state reference (called by environment if available).
        
        Args:
            state: Current game state
        """
        self.last_state = state
