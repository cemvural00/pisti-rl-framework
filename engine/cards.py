"""Card representation and deck management for Pişti."""

from dataclasses import dataclass
from typing import List
import random


@dataclass(frozen=True)
class Card:
    """Represents a playing card with rank and suit."""

    rank: str  # 'A', '2', '3', ..., 'K'
    suit: str  # 'S' (Spades), 'H' (Hearts), 'D' (Diamonds), 'C' (Clubs)

    def __str__(self) -> str:
        suit_symbols = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
        return f"{self.rank}{suit_symbols.get(self.suit, self.suit)}"

    def __repr__(self) -> str:
        return f"Card(rank='{self.rank}', suit='{self.suit}')"


# Constants
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["S", "H", "D", "C"]  # Spades, Hearts, Diamonds, Clubs

# Scoring cards: Aces, Jacks, 2♣, 10♦
SCORING_CARDS = {
    Card("A", "S"),
    Card("A", "H"),
    Card("A", "D"),
    Card("A", "C"),
    Card("J", "S"),
    Card("J", "H"),
    Card("J", "D"),
    Card("J", "C"),
    Card("2", "C"),  # 2♣ worth +2
    Card("10", "D"),  # 10♦ worth +3
}


def card_to_id(card: Card) -> int:
    """
    Maps a Card to its canonical 0-51 action ID.
    
    Mapping: card_id = suit_id * 13 + rank_id
    - suit_id: 0=S, 1=H, 2=D, 3=C
    - rank_id: 0=2, 1=3, ..., 9=10, 10=J, 11=Q, 12=K, 13=A (but A is rank 12 in RANKS)
    
    Actually, RANKS[0]='2', RANKS[12]='A', so:
    - rank_id = index in RANKS (0-12)
    - suit_id = index in SUITS (0-3)
    """
    rank_id = RANKS.index(card.rank)
    suit_id = SUITS.index(card.suit)
    return suit_id * 13 + rank_id


def id_to_card(card_id: int) -> Card:
    """
    Inverse mapping: converts 0-51 card ID to Card.
    
    Recoverable via divmod(card_id, 13) -> (suit_id, rank_id)
    """
    suit_id, rank_id = divmod(card_id, 13)
    return Card(rank=RANKS[rank_id], suit=SUITS[suit_id])


def get_rank(card_id: int) -> str:
    """Extract rank string from card ID for matching purposes."""
    rank_id = card_id % 13
    return RANKS[rank_id]


def is_jack(card_id: int) -> bool:
    """Check if card ID represents a Jack."""
    return get_rank(card_id) == "J"


def is_scoring_card(card: Card) -> bool:
    """Check if card is a scoring card (A, J, 2♣, 10♦)."""
    return card in SCORING_CARDS


class Deck:
    """Standard 52-card deck with shuffling."""

    def __init__(self, seed: int = None):
        """Initialize a full deck of 52 cards."""
        self.cards: List[Card] = [
            Card(rank=rank, suit=suit) for suit in SUITS for rank in RANKS
        ]
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.cards)

    def deal(self, n: int) -> List[Card]:
        """Deal n cards from the top of the deck."""
        if n > len(self.cards):
            raise ValueError(f"Cannot deal {n} cards, only {len(self.cards)} remaining")
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt

    def __len__(self) -> int:
        """Return number of cards remaining in deck."""
        return len(self.cards)

    def shuffle(self, seed: int = None):
        """Shuffle the remaining cards in the deck."""
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.cards)
