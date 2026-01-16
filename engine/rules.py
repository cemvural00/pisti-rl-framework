"""Game rules for Pişti: capture logic, pişti detection, and scoring."""

from typing import List, Dict, Tuple, Optional
from engine.cards import Card, is_jack, card_to_id, SCORING_CARDS


def check_capture(played_card: Card, top_card: Card) -> bool:
    """
    Check if a played card captures the table pile.
    
    Capture occurs if:
    - The played card matches the rank of the top card, OR
    - The played card is a Jack (captures regardless of top card)
    """
    if played_card.rank == "J":
        return True
    return played_card.rank == top_card.rank


def calculate_pisti(
    pile_size: int,
    played_card: Card,
    top_card: Card,
    is_jack_capture: bool,
    config: Optional[Dict] = None,
) -> int:
    """
    Calculate pişti bonus points.
    
    Returns:
        - 10 for regular pişti (single card captured by rank match)
        - 20 for double pişti (Jack captures single Jack)
        - 0 otherwise
    
    Pişti occurs when:
    - Table pile has exactly 1 card
    - Capture is by rank match (NOT by Jack), OR
    - Both cards are Jacks (double pişti)
    
    Config can specify exceptions (e.g., no pişti on first capture).
    """
    if config is None:
        config = {}
    
    # Pişti requires exactly 1 card in pile
    if pile_size != 1:
        return 0
    
    # Double pişti: Jack captures Jack
    if played_card.rank == "J" and top_card.rank == "J":
        return 20
    
    # Regular pişti: rank match (not Jack capture)
    if not is_jack_capture and played_card.rank == top_card.rank:
        return 10
    
    return 0


def score_captured_cards(captured: List[Card]) -> Dict[str, int]:
    """
    Calculate scoring breakdown for captured cards.
    
    Returns dict with:
    - aces: count of Aces
    - jacks: count of Jacks
    - got_2c: 1 if 2♣ captured, else 0
    - got_10d: 1 if 10♦ captured, else 0
    - pistis: count of pişti bonuses (handled separately in game state)
    - double_pistis: count of double pişti bonuses
    """
    breakdown = {
        "aces": 0,
        "jacks": 0,
        "got_2c": 0,
        "got_10d": 0,
        "pistis": 0,  # Will be updated by game state
        "double_pistis": 0,  # Will be updated by game state
    }
    
    for card in captured:
        if card.rank == "A":
            breakdown["aces"] += 1
        elif card.rank == "J":
            breakdown["jacks"] += 1
        elif card == Card("2", "C"):
            breakdown["got_2c"] = 1
        elif card == Card("10", "D"):
            breakdown["got_10d"] = 1
    
    return breakdown


def calculate_final_score(
    score_breakdown: Dict[str, int], card_counts: Dict[int, int]
) -> int:
    """
    Calculate final score including majority bonus.
    
    Args:
        score_breakdown: Dict with aces, jacks, got_2c, got_10d, pistis, double_pistis
        card_counts: Dict mapping player_id -> number of captured cards
    
    Returns:
        Total score including:
        - 1 point per Ace
        - 1 point per Jack
        - 2 points for 2♣
        - 3 points for 10♦
        - 10 points per pişti
        - 20 points per double pişti
        - 3 points for majority (if tie, +0)
    """
    score = (
        score_breakdown["aces"]
        + score_breakdown["jacks"]
        + 2 * score_breakdown["got_2c"]
        + 3 * score_breakdown["got_10d"]
        + 10 * score_breakdown["pistis"]
        + 20 * score_breakdown["double_pistis"]
    )
    
    # Majority bonus: +3 to player with more cards (tie = +0)
    if len(card_counts) == 2:
        counts = list(card_counts.values())
        if counts[0] > counts[1]:
            # Player 0 gets majority (handled per-player in game logic)
            pass
        elif counts[1] > counts[0]:
            # Player 1 gets majority (handled per-player in game logic)
            pass
        # Tie: no bonus
    
    return score


def get_majority_bonus(card_counts: Dict[int, int], player_id: int) -> int:
    """
    Get majority bonus for a specific player.
    
    Returns 3 if player has more cards than opponent, 0 otherwise (including ties).
    """
    if len(card_counts) != 2:
        return 0
    
    my_count = card_counts.get(player_id, 0)
    opp_count = card_counts.get(1 - player_id, 0)
    
    if my_count > opp_count:
        return 3
    return 0


def deal_initial_table(deck: List[Card], seed: int = None) -> Tuple[List[Card], Card]:
    """
    Deal initial 4 cards to table center, flip one face-up.
    
    Rules:
    - Deal 4 cards face-down to center
    - Flip one face-up to start discard pile
    - If flipped card is Jack, flip additional until non-Jack appears
    - If all 4 are Jacks, redeal (raise error or handle in caller)
    
    Returns:
        (center_cards, top_card) where:
        - center_cards: 3 face-down cards (not visible to players)
        - top_card: 1 face-up card to start pile
    """
    import random
    
    if seed is not None:
        random.seed(seed)
    
    # Deal 4 cards
    table_cards = deck[:4]
    remaining_deck = deck[4:]
    
    # Find first non-Jack to flip
    top_idx = None
    for i, card in enumerate(table_cards):
        if card.rank != "J":
            top_idx = i
            break
    
    # If all are Jacks, we need to handle this (redeal or use first Jack)
    if top_idx is None:
        # Use first Jack as fallback (caller should handle redeal)
        top_idx = 0
    
    top_card = table_cards[top_idx]
    center_cards = [card for i, card in enumerate(table_cards) if i != top_idx]
    
    return center_cards, top_card
