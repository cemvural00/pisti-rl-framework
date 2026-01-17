"""Game state management for Pişti with immutable state transitions."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import copy
from engine.cards import Card, card_to_id, id_to_card
from engine.rules import (
    check_capture,
    calculate_pisti,
    score_captured_cards,
    get_majority_bonus,
)


@dataclass(frozen=False)  # We'll use copy for immutability
class GameState:
    """
    Represents the complete game state of a Pişti game.
    
    State is conceptually immutable: apply_action() returns a new state.
    """

    hands: Dict[int, List[Card]] = field(default_factory=dict)
    table_pile: List[Card] = field(default_factory=list)
    captured: Dict[int, List[Card]] = field(default_factory=dict)
    center_cards: List[Card] = field(default_factory=list)  # 3 face-down initial cards
    stock: List[Card] = field(default_factory=list)
    current_player: int = 0
    first_capture_made: bool = False
    score_breakdown: Dict[int, Dict[str, int]] = field(default_factory=dict)
    move_history: List[Tuple[int, Card, bool]] = field(
        default_factory=list
    )  # (player, card, captured)

    def __post_init__(self):
        """Initialize default values for dicts."""
        if not self.hands:
            self.hands = {0: [], 1: []}
        if not self.captured:
            self.captured = {0: [], 1: []}
        if not self.score_breakdown:
            self.score_breakdown = {
                0: {
                    "aces": 0,
                    "jacks": 0,
                    "got_2c": 0,
                    "got_10d": 0,
                    "pistis": 0,
                    "double_pistis": 0,
                },
                1: {
                    "aces": 0,
                    "jacks": 0,
                    "got_2c": 0,
                    "got_10d": 0,
                    "pistis": 0,
                    "double_pistis": 0,
                },
            }

    def apply_action(self, card: Card, config: Optional[Dict] = None) -> "GameState":
        """
        Apply an action (play a card) and return a new game state.
        
        Args:
            card: Card to play
            config: Optional config dict for pişti exceptions, etc.
        
        Returns:
            New GameState after action is applied
        """
        if config is None:
            config = {}
        
        # Create deep copy for new state
        new_state = copy.deepcopy(self)
        
        # Remove card from hand
        if card not in new_state.hands[new_state.current_player]:
            raise ValueError(
                f"Card {card} not in player {new_state.current_player}'s hand"
            )
        new_state.hands[new_state.current_player].remove(card)
        
        # Check if capture occurs
        captured = False
        is_jack_capture = card.rank == "J"
        
        if new_state.table_pile:
            top_card = new_state.table_pile[-1]
            captured = check_capture(card, top_card)
        else:
            # Empty table: card becomes new pile
            captured = False
        
        # Handle capture or add to pile
        if captured:
            # Capture: move entire pile to captured
            pile_to_capture = new_state.table_pile.copy()
            
            # Check for pişti bonus
            pile_size = len(pile_to_capture)
            if pile_size > 0:
                top_card = pile_to_capture[-1]
                pisti_bonus = calculate_pisti(
                    pile_size, card, top_card, is_jack_capture, config
                )
                
                if pisti_bonus == 20:
                    new_state.score_breakdown[new_state.current_player][
                        "double_pistis"
                    ] += 1
                elif pisti_bonus == 10:
                    new_state.score_breakdown[new_state.current_player]["pistis"] += 1
            
            # Add captured pile and played card to captured
            new_state.captured[new_state.current_player].extend(pile_to_capture)
            new_state.captured[new_state.current_player].append(card)
            
            # If first capture, add center cards
            if not new_state.first_capture_made:
                new_state.captured[new_state.current_player].extend(
                    new_state.center_cards
                )
                new_state.center_cards = []
                new_state.first_capture_made = True
            
            # Update score breakdown for newly captured cards
            newly_captured = pile_to_capture + [card]
            new_breakdown = score_captured_cards(newly_captured)
            for key in ["aces", "jacks", "got_2c", "got_10d"]:
                new_state.score_breakdown[new_state.current_player][
                    key
                ] += new_breakdown[key]
            
            # Clear table pile
            new_state.table_pile = []
        else:
            # No capture: add card to table pile
            new_state.table_pile.append(card)
        
        # Record move
        new_state.move_history.append((new_state.current_player, card, captured))
        
        # Switch to next player
        new_state.current_player = 1 - new_state.current_player
        
        # Check if need to deal new cards
        if (
            len(new_state.hands[0]) == 0
            and len(new_state.hands[1]) == 0
            and len(new_state.stock) > 0
        ):
            # Deal cards from stock: try to give 4 to each, but handle remaining cards
            # If stock has odd number or less than 8 cards, distribute as evenly as possible
            stock_size = len(new_state.stock)
            if stock_size >= 8:
                # Normal case: deal 4 to each
                new_state.hands[0] = new_state.stock[:4]
                new_state.hands[1] = new_state.stock[4:8]
                new_state.stock = new_state.stock[8:]
            elif stock_size >= 2:
                # Less than 8 cards: distribute evenly
                cards_per_player = stock_size // 2
                new_state.hands[0] = new_state.stock[:cards_per_player]
                new_state.hands[1] = new_state.stock[cards_per_player:2 * cards_per_player]
                new_state.stock = new_state.stock[2 * cards_per_player:]
            else:
                # Only 1 card left: give it to current player (or player 0 if no current player)
                # Actually, if there's only 1 card, we should give it to player 0 to keep game going
                # But according to rules, if stock is exhausted and hands are empty, game should end
                # So we should NOT deal the last card - let the game end naturally
                # The remaining card will be handled by terminal check
                pass
        
        return new_state

    def is_terminal(self) -> bool:
        """
        Check if game is in terminal state.
        
        Game ends when:
        - Both players have no cards in hand
        - Stock is exhausted OR has fewer than 2 cards (can't be dealt evenly)
        - Last cards have been played
        """
        hands_empty = (
            len(self.hands[0]) == 0 and len(self.hands[1]) == 0
        )
        stock_empty = len(self.stock) == 0
        stock_insufficient = len(self.stock) < 2  # Can't deal evenly to both players
        
        return hands_empty and (stock_empty or stock_insufficient)

    def get_legal_actions(self, player_id: int) -> List[int]:
        """
        Get list of legal action IDs (card IDs) for a player.
        
        Returns:
            List of card IDs (0-51) representing cards in player's hand
        """
        return [card_to_id(card) for card in self.hands[player_id]]

    def get_final_scores(self) -> Dict[int, int]:
        """
        Calculate final scores for both players including majority bonus.
        
        Returns:
            Dict mapping player_id -> final score
        """
        card_counts = {
            0: len(self.captured[0]),
            1: len(self.captured[1]),
        }
        
        scores = {}
        for player_id in [0, 1]:
            base_score = (
                self.score_breakdown[player_id]["aces"]
                + self.score_breakdown[player_id]["jacks"]
                + 2 * self.score_breakdown[player_id]["got_2c"]
                + 3 * self.score_breakdown[player_id]["got_10d"]
                + 10 * self.score_breakdown[player_id]["pistis"]
                + 20 * self.score_breakdown[player_id]["double_pistis"]
            )
            majority_bonus = get_majority_bonus(card_counts, player_id)
            scores[player_id] = base_score + majority_bonus
        
        return scores

    def get_table_top_card(self) -> Optional[Card]:
        """Get the top card of the table pile, or None if empty."""
        if self.table_pile:
            return self.table_pile[-1]
        return None
