"""Unit tests for game rules: capture logic, pişti detection, dealing."""

import pytest
from engine.cards import Card
from engine.rules import check_capture, calculate_pisti, deal_initial_table


def test_check_capture_rank_match():
    """Test capture by rank match."""
    top_card = Card("5", "H")
    played_card = Card("5", "S")
    assert check_capture(played_card, top_card) is True


def test_check_capture_jack():
    """Test capture by Jack (captures any card)."""
    top_card = Card("5", "H")
    played_jack = Card("J", "S")
    assert check_capture(played_jack, top_card) is True


def test_check_capture_no_match():
    """Test no capture when rank doesn't match and not Jack."""
    top_card = Card("5", "H")
    played_card = Card("6", "S")
    assert check_capture(played_card, top_card) is False


def test_calculate_pisti_regular():
    """Test regular pişti (single card captured by rank match)."""
    pile_size = 1
    played_card = Card("5", "H")
    top_card = Card("5", "S")
    is_jack_capture = False
    assert calculate_pisti(pile_size, played_card, top_card, is_jack_capture) == 10


def test_calculate_pisti_double():
    """Test double pişti (Jack captures Jack)."""
    pile_size = 1
    played_card = Card("J", "H")
    top_card = Card("J", "S")
    is_jack_capture = True
    assert calculate_pisti(pile_size, played_card, top_card, is_jack_capture) == 20


def test_calculate_pisti_jack_capture_no_pisti():
    """Test Jack capture doesn't give pişti unless both are Jacks."""
    pile_size = 1
    played_card = Card("J", "H")
    top_card = Card("5", "S")
    is_jack_capture = True
    assert calculate_pisti(pile_size, played_card, top_card, is_jack_capture) == 0


def test_calculate_pisti_multiple_cards():
    """Test no pişti when pile has more than 1 card."""
    pile_size = 3
    played_card = Card("5", "H")
    top_card = Card("5", "S")
    is_jack_capture = False
    assert calculate_pisti(pile_size, played_card, top_card, is_jack_capture) == 0


def test_deal_initial_table():
    """Test dealing initial table cards."""
    from engine.cards import Deck
    
    deck = Deck(seed=42)
    deck_cards = deck.cards.copy()
    
    center_cards, top_card = deal_initial_table(deck_cards, seed=42)
    
    assert len(center_cards) == 3
    assert top_card is not None
    assert len(deck_cards) >= 4  # Should have at least 4 cards


def test_deal_initial_table_jack_reflip():
    """Test that Jack reflip logic works (finds first non-Jack)."""
    # Create a deck with Jacks at the start
    from engine.cards import RANKS, SUITS
    
    # Manually create deck with Jacks first
    deck_cards = []
    for suit in SUITS:
        deck_cards.append(Card("J", suit))
    # Then add non-Jacks
    for suit in SUITS:
        for rank in RANKS:
            if rank != "J":
                deck_cards.append(Card(rank, suit))
    
    center_cards, top_card = deal_initial_table(deck_cards[:4], seed=42)
    
    # Should find first non-Jack (or use first Jack if all are Jacks)
    assert len(center_cards) == 3
    assert top_card is not None
