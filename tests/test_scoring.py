"""Unit tests for scoring: breakdown, final score calculation."""

import pytest
from engine.cards import Card
from engine.rules import score_captured_cards, calculate_final_score, get_majority_bonus


def test_score_captured_cards_aces():
    """Test scoring for Aces."""
    captured = [Card("A", "S"), Card("A", "H"), Card("5", "D")]
    breakdown = score_captured_cards(captured)
    assert breakdown["aces"] == 2
    assert breakdown["jacks"] == 0
    assert breakdown["got_2c"] == 0
    assert breakdown["got_10d"] == 0


def test_score_captured_cards_jacks():
    """Test scoring for Jacks."""
    captured = [Card("J", "S"), Card("J", "H"), Card("5", "D")]
    breakdown = score_captured_cards(captured)
    assert breakdown["aces"] == 0
    assert breakdown["jacks"] == 2
    assert breakdown["got_2c"] == 0
    assert breakdown["got_10d"] == 0


def test_score_captured_cards_2c():
    """Test scoring for 2♣."""
    captured = [Card("2", "C"), Card("5", "D")]
    breakdown = score_captured_cards(captured)
    assert breakdown["got_2c"] == 1
    assert breakdown["got_10d"] == 0


def test_score_captured_cards_10d():
    """Test scoring for 10♦."""
    captured = [Card("10", "D"), Card("5", "D")]
    breakdown = score_captured_cards(captured)
    assert breakdown["got_2c"] == 0
    assert breakdown["got_10d"] == 1


def test_score_captured_cards_all_scoring():
    """Test scoring for all scoring cards."""
    captured = [
        Card("A", "S"),
        Card("J", "H"),
        Card("2", "C"),
        Card("10", "D"),
    ]
    breakdown = score_captured_cards(captured)
    assert breakdown["aces"] == 1
    assert breakdown["jacks"] == 1
    assert breakdown["got_2c"] == 1
    assert breakdown["got_10d"] == 1


def test_calculate_final_score_base():
    """Test base score calculation without majority."""
    score_breakdown = {
        "aces": 2,
        "jacks": 1,
        "got_2c": 1,
        "got_10d": 1,
        "pistis": 1,
        "double_pistis": 0,
    }
    card_counts = {0: 20, 1: 20}  # Tie, no majority bonus
    
    score = calculate_final_score(score_breakdown, card_counts)
    expected = 2 + 1 + 2 + 3 + 10  # aces + jacks + 2c + 10d + pisti
    assert score == expected


def test_calculate_final_score_with_pistis():
    """Test score calculation with pişti bonuses."""
    score_breakdown = {
        "aces": 1,
        "jacks": 0,
        "got_2c": 0,
        "got_10d": 0,
        "pistis": 2,
        "double_pistis": 1,
    }
    card_counts = {0: 20, 1: 20}
    
    score = calculate_final_score(score_breakdown, card_counts)
    expected = 1 + 20 + 20  # ace + 2 pistis + 1 double pisti
    assert score == expected


def test_get_majority_bonus_winner():
    """Test majority bonus for winner."""
    card_counts = {0: 30, 1: 22}
    assert get_majority_bonus(card_counts, 0) == 3
    assert get_majority_bonus(card_counts, 1) == 0


def test_get_majority_bonus_tie():
    """Test no majority bonus on tie."""
    card_counts = {0: 26, 1: 26}
    assert get_majority_bonus(card_counts, 0) == 0
    assert get_majority_bonus(card_counts, 1) == 0


def test_get_majority_bonus_loser():
    """Test no majority bonus for loser."""
    card_counts = {0: 20, 1: 32}
    assert get_majority_bonus(card_counts, 0) == 0
    assert get_majority_bonus(card_counts, 1) == 3
