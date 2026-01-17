"""Reward functions for Pişti RL environment."""

from typing import Dict, Optional
from engine.state import GameState
from engine.rules import get_majority_bonus


def sparse_reward(
    state: GameState, player_id: int, prev_state: Optional[GameState] = None
) -> float:
    """
    Sparse reward: only terminal reward based on final score differential.
    
    Args:
        state: Current game state
        player_id: Player to calculate reward for
        prev_state: Previous state (unused for sparse, but kept for interface consistency)
    
    Returns:
        Final score differential (player_score - opponent_score) if terminal, else 0.0
    """
    if not state.is_terminal():
        return 0.0
    
    scores = state.get_final_scores()
    my_score = scores[player_id]
    opp_score = scores[1 - player_id]
    
    return float(my_score - opp_score)


def shaped_reward(
    state: GameState,
    player_id: int,
    prev_state: Optional[GameState],
    config: Optional[Dict] = None,
) -> float:
    """
    Shaped reward with immediate rewards for captures and scoring cards.
    
    Args:
        state: Current game state
        player_id: Player to calculate reward for
        prev_state: Previous state (to detect changes)
        config: Reward shaping configuration with weights
    
    Returns:
        Shaped reward value
    """
    if config is None:
        config = {
            "capture_bonus": 0.1,
            "ace_bonus": 1.0,
            "jack_bonus": 1.0,
            "two_clubs_bonus": 2.0,
            "ten_diamonds_bonus": 3.0,
            "pisti_bonus": 10.0,
            "capture_pile_size_bonus": 0.01,
            "majority_bonus": 0.5,
        }
    
    from engine.cards import Card
    
    reward = 0.0
    
    # If terminal, add final score differential
    if state.is_terminal():
        scores = state.get_final_scores()
        my_score = scores[player_id]
        opp_score = scores[1 - player_id]
        reward += float(my_score - opp_score)
        return reward
    
    # Detect immediate rewards from move history
    if prev_state is not None and state.move_history:
        last_move = state.move_history[-1]
        move_player, card, captured = last_move
        
        # Only reward the player who made the move
        if move_player == player_id:
            # Capture bonus
            if captured:
                reward += config.get("capture_bonus", 0.1)
                
                # Get cards that were just captured (in table pile before)
                if prev_state.table_pile:
                    captured_cards = prev_state.table_pile + [card]
                    
                    # Individual card type rewards (matches game scoring)
                    for c in captured_cards:
                        if c.rank == "A":
                            reward += config.get("ace_bonus", 1.0)
                        elif c.rank == "J":
                            reward += config.get("jack_bonus", 1.0)
                        elif c == Card("2", "C"):
                            reward += config.get("two_clubs_bonus", 2.0)
                        elif c == Card("10", "D"):
                            reward += config.get("ten_diamonds_bonus", 3.0)
                    
                    # Capture pile size bonus (reward proportional to cards captured)
                    pile_size = len(captured_cards)
                    reward += config.get("capture_pile_size_bonus", 0.01) * pile_size
            
            # Pişti bonus (already in score breakdown, but add immediate reward)
            prev_pistis = prev_state.score_breakdown[player_id]["pistis"]
            prev_double_pistis = prev_state.score_breakdown[player_id]["double_pistis"]
            curr_pistis = state.score_breakdown[player_id]["pistis"]
            curr_double_pistis = state.score_breakdown[player_id]["double_pistis"]
            
            if curr_pistis > prev_pistis:
                reward += config.get("pisti_bonus", 10.0)
            if curr_double_pistis > prev_double_pistis:
                reward += 2 * config.get("pisti_bonus", 10.0)
            
            # Majority bonus: intermediate reward when agent first achieves majority advantage
            my_captured = len(state.captured[player_id])
            opp_captured = len(state.captured[1 - player_id])
            prev_my_captured = len(prev_state.captured[player_id])
            prev_opp_captured = len(prev_state.captured[1 - player_id])
            
            # Check if agent just achieved majority advantage (didn't have it before)
            had_majority_before = prev_my_captured > prev_opp_captured
            has_majority_now = my_captured > opp_captured
            total_cards_captured = my_captured + opp_captured
            
            if has_majority_now and not had_majority_before and total_cards_captured > 0:
                # Give reward proportional to advantage
                advantage = (my_captured - opp_captured) / max(total_cards_captured, 1)
                reward += config.get("majority_bonus", 0.5) * advantage
    
    return reward
