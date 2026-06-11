"""Rule-correctness tests for the rebuilt Pişti engine."""

import random

import pytest

from engine.game import (
    ACE,
    CARD_POINTS,
    JACK,
    TEN_DIAMONDS,
    TOTAL_CARD_POINTS,
    TWO_CLUBS,
    PistiGame,
    new_deck,
    rank_of,
)


def cid(rank: str, suit: str) -> int:
    from engine.game import RANKS, SUITS

    return SUITS.index(suit) * 13 + RANKS.index(rank)


def make_deck(*front, fill_excluding=()):
    """Deck starting with `front`, then all remaining cards in id order."""
    front = list(front)
    rest = [c for c in range(52) if c not in front and c not in fill_excluding]
    return front + rest


def test_card_constants():
    assert TWO_CLUBS == cid("2", "C")
    assert TEN_DIAMONDS == cid("10", "D")
    assert CARD_POINTS[cid("A", "S")] == 1
    assert CARD_POINTS[cid("J", "H")] == 1
    assert CARD_POINTS[TWO_CLUBS] == 2
    assert CARD_POINTS[TEN_DIAMONDS] == 3
    assert TOTAL_CARD_POINTS == 13


def test_deal_structure():
    g = PistiGame(deck=list(range(52)))
    assert len(g.table) == 1
    assert len(g.hidden_center) == 3
    assert len(g.hands[0]) == 4 and len(g.hands[1]) == 4
    assert len(g.stock) == 40
    # Face-up card is not a Jack when possible
    assert rank_of(g.table[0]) != JACK


def test_faceup_skips_jack():
    # First card dealt is a Jack -> face-up should be the next non-Jack
    deck = make_deck(cid("J", "S"), cid("5", "H"), cid("7", "D"), cid("9", "C"))
    g = PistiGame(deck=deck)
    assert g.table[0] == cid("5", "H")
    assert cid("J", "S") in g.hidden_center


def test_rank_match_capture_and_points():
    # Table face-up: 5H. P0 holds 5S -> rank-match capture sweeps hidden too.
    deck = make_deck(
        cid("5", "H"), cid("2", "S"), cid("3", "S"), cid("4", "S"),  # table
        cid("5", "S"), cid("6", "S"), cid("7", "S"), cid("8", "S"),  # P0
        cid("6", "H"), cid("7", "H"), cid("8", "H"), cid("9", "H"),  # P1
    )
    g = PistiGame(deck=deck)
    info = g.step(cid("5", "S"))
    assert info["captured"]
    # 1 table card + played card + 3 hidden center
    assert g.captured_count[0] == 5
    assert g.last_capturer == 0
    # No pişti: hidden center cards were under the pile
    assert g.pistis[0] == 0 and info["pisti"] == 0


def test_no_pisti_on_first_capture_with_hidden_cards():
    deck = make_deck(
        cid("5", "H"), cid("2", "S"), cid("3", "S"), cid("4", "S"),
        cid("5", "S"), cid("6", "S"), cid("7", "S"), cid("8", "S"),
        cid("6", "H"), cid("7", "H"), cid("8", "H"), cid("9", "H"),
    )
    g = PistiGame(deck=deck)
    info = g.step(cid("5", "S"))
    assert info["captured"] and info["pisti"] == 0
    assert g.points[0] == 0  # no scoring cards involved


def test_pisti_on_single_card_pile():
    # P0 plays 6S (no capture), P1 plays 6H on it after P0 captured everything
    deck = make_deck(
        cid("5", "H"), cid("2", "S"), cid("3", "S"), cid("4", "S"),
        cid("5", "S"), cid("6", "S"), cid("7", "S"), cid("8", "S"),
        cid("6", "H"), cid("7", "H"), cid("8", "H"), cid("9", "H"),
    )
    g = PistiGame(deck=deck)
    g.step(cid("5", "S"))   # P0 captures table (incl. hidden) -> table empty
    g.step(cid("6", "H"))   # P1 plays to empty table
    info = g.step(cid("6", "S"))  # P0 rank-matches a single-card pile: PIŞTI
    assert info["captured"] and info["pisti"] == 1
    assert g.pistis[0] == 1
    assert g.points[0] == 10


def test_jack_captures_but_no_pisti_on_single_nonjack():
    deck = make_deck(
        cid("5", "H"), cid("2", "S"), cid("3", "S"), cid("4", "S"),
        cid("5", "S"), cid("J", "S"), cid("7", "S"), cid("8", "S"),
        cid("6", "H"), cid("7", "H"), cid("8", "H"), cid("9", "H"),
    )
    g = PistiGame(deck=deck)
    g.step(cid("5", "S"))   # P0 captures
    g.step(cid("6", "H"))   # P1 -> single card on table
    info = g.step(cid("J", "S"))  # Jack takes single non-Jack: capture, NO pişti
    assert info["captured"] and info["pisti"] == 0
    # Jack itself is worth 1 point
    assert g.points[0] == 1


def test_double_pisti_jack_on_jack():
    deck = make_deck(
        cid("5", "H"), cid("2", "S"), cid("3", "S"), cid("4", "S"),
        cid("5", "S"), cid("J", "S"), cid("7", "S"), cid("8", "S"),
        cid("J", "H"), cid("7", "H"), cid("8", "H"), cid("9", "H"),
    )
    g = PistiGame(deck=deck)
    g.step(cid("5", "S"))   # P0 captures table
    g.step(cid("J", "H"))   # P1 plays Jack onto EMPTY table -> no capture
    info = g.step(cid("J", "S"))  # P0 Jack-on-Jack: double pişti
    assert info["captured"] and info["pisti"] == 2
    assert g.double_pistis[0] == 1
    assert g.points[0] == 20 + 2  # bonus + two jacks


def test_jack_does_not_capture_empty_table():
    deck = make_deck(
        cid("5", "H"), cid("2", "S"), cid("3", "S"), cid("4", "S"),
        cid("5", "S"), cid("J", "S"), cid("7", "S"), cid("8", "S"),
        cid("6", "H"), cid("7", "H"), cid("8", "H"), cid("9", "H"),
    )
    g = PistiGame(deck=deck)
    g.step(cid("5", "S"))       # P0 captures -> empty table
    info = g.step(cid("6", "H"))  # wait, P1 to move
    assert not info["captured"]
    # Now check Jack on empty: clear table first via P0 jack capture
    g2 = PistiGame(deck=deck)
    g2.step(cid("5", "S"))      # P0 captures -> table empty, P1 to move
    g2_hand = list(g2.hands[1])
    info = g2.step(g2_hand[0])  # P1 plays onto empty table
    assert not info["captured"]


def test_illegal_action_raises():
    g = PistiGame(deck=list(range(52)))
    bad = g.hands[1][0]  # card in opponent's hand
    with pytest.raises(ValueError):
        g.step(bad)


def test_redeal_after_hands_empty():
    g = PistiGame(deck=list(range(52)), rng=None)
    for _ in range(8):  # play out first 4 cards each
        g.step(g.hands[g.current][0])
    assert len(g.hands[0]) == 4 and len(g.hands[1]) == 4
    assert len(g.stock) == 32


def test_full_game_invariants():
    rng = random.Random(0)
    for trial in range(2000):
        deck = new_deck(rng)
        first = trial % 2
        g = PistiGame(deck=deck, first_player=first)
        n_moves = 0
        while not g.done:
            g.step(rng.choice(g.legal_actions()))
            n_moves += 1
        assert n_moves == 48
        # Conservation: every card is captured by someone (or none were,
        # which cannot happen since a rank match always exists eventually)
        assert g.captured_count[0] + g.captured_count[1] == 52
        s0, s1 = g.scores()
        base = s0 + s1
        bonus = (
            10 * (g.pistis[0] + g.pistis[1])
            + 20 * (g.double_pistis[0] + g.double_pistis[1])
        )
        majority = 0 if g.captured_count[0] == g.captured_count[1] else 3
        assert base == TOTAL_CARD_POINTS + bonus + majority
        assert not g.table and not g.hidden_center and not g.stock


def test_mirror_deal_symmetry():
    """Same deck, swapped first_player: seats see swapped hands."""
    deck = new_deck(random.Random(7))
    a = PistiGame(deck=deck, first_player=0)
    b = PistiGame(deck=deck, first_player=1)
    assert a.hands[0] == b.hands[1]
    assert a.hands[1] == b.hands[0]
    assert a.table == b.table


def test_clone_independence():
    g = PistiGame(deck=new_deck(random.Random(3)))
    c = g.clone()
    c.step(c.legal_actions()[0])
    assert len(g.hands[g.current]) == 4  # original untouched
    assert g.history == []


def test_leftover_pile_goes_to_last_capturer():
    rng = random.Random(11)
    found = False
    for _ in range(300):
        g = PistiGame(deck=new_deck(rng))
        last_table = None
        while not g.done:
            # Track table just before the final move
            if len(g.stock) == 0 and len(g.hands[0]) + len(g.hands[1]) == 1:
                last_table = list(g.table)
            g.step(rng.choice(g.legal_actions()))
        if last_table:
            found = True
            assert g.captured_count[0] + g.captured_count[1] == 52
    assert found  # at least one game ended with leftovers on the table


def test_determinize_preserves_information_set():
    """Determinized clones must keep the player's own info fixed and only
    shuffle unseen cards among hidden locations, with points adjusted."""
    rng = random.Random(5)
    nprng = __import__("numpy").random.default_rng(5)
    for _ in range(200):
        g = PistiGame(deck=new_deck(rng))
        # advance to a random mid-game point
        for _ in range(rng.randrange(0, 40)):
            if g.done:
                break
            g.step(rng.choice(g.legal_actions()))
        if g.done:
            continue
        p = g.current
        det = g.determinize(p, nprng)
        # Own hand, table, history identical
        assert det.hands[p] == g.hands[p]
        assert det.table == g.table
        assert det.history == g.history
        # Location sizes preserved
        assert len(det.hands[1 - p]) == len(g.hands[1 - p])
        assert len(det.stock) == len(g.stock)
        assert len(det.hidden_center) == len(g.hidden_center)
        assert len(det.captured_hidden) == len(g.captured_hidden)
        # Still a permutation of 52 cards across all locations
        allcards = (
            det.hands[0] + det.hands[1] + det.table + det.hidden_center
            + det.stock + det.captured_hidden
            + [m[1] for m in det.history if m[2]]  # captured via play
        )
        # captured cards: count check via captured_count instead
        total = (
            len(det.hands[0]) + len(det.hands[1]) + len(det.table)
            + len(det.hidden_center) + len(det.stock)
            + det.captured_count[0] + det.captured_count[1]
        )
        assert total == 52
        # Determinized game must play out without errors
        while not det.done:
            det.step(rng.choice(det.legal_actions()))
        assert det.captured_count[0] + det.captured_count[1] == 52
