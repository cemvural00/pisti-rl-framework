"""Fast, correct 2-player Pişti game engine.

Cards are integers 0-51: card_id = suit_id * 13 + rank_id.
  - suit_id: 0=Spades, 1=Hearts, 2=Diamonds, 3=Clubs
  - rank_id: 0='2', 1='3', ..., 8='10', 9='J', 10='Q', 11='K', 12='A'

Rules implemented (standard Turkish 2-player Pişti):
  - Deal: 4 cards to the table (3 face-down "hidden center" + 1 face-up,
    chosen as the first non-Jack among the four), then 4 cards to each
    player. Re-deal 4 each when both hands are empty (6 rounds total).
  - Capture: played card matches rank of the table's top card, or played
    card is a Jack (Jack does not capture an empty table).
  - Pişti: capturing a pile of exactly one card by rank match scores 10
    (Jack-on-Jack scores 20). A Jack capturing a single non-Jack is NOT a
    pişti. A capture that sweeps the hidden center cards and a capture by
    the final card of the deal are never a pişti.
  - First capture also takes the 3 hidden center cards.
  - Game end: when all cards are played, the leftover table pile (and any
    never-captured hidden cards) goes to the player who made the last
    capture.
  - Scoring: each Ace +1, each Jack +1, 2♣ +2, 10♦ +3, pişti +10,
    double pişti +20, majority of captured cards (>26) +3.
"""

from typing import List, Optional, Tuple
import random

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["S", "H", "D", "C"]
SUIT_SYMBOLS = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}

JACK = 9  # rank_id of Jack
ACE = 12  # rank_id of Ace
TWO_CLUBS = 3 * 13 + 0  # 2♣ = 39
TEN_DIAMONDS = 2 * 13 + 8  # 10♦ = 34

PISTI_POINTS = 10
DOUBLE_PISTI_POINTS = 20
MAJORITY_POINTS = 3

# Point value of each card id (A=1, J=1, 2♣=2, 10♦=3)
CARD_POINTS = [0] * 52
for _suit in range(4):
    CARD_POINTS[_suit * 13 + ACE] = 1
    CARD_POINTS[_suit * 13 + JACK] = 1
CARD_POINTS[TWO_CLUBS] = 2
CARD_POINTS[TEN_DIAMONDS] = 3

# Total card points in the deck (4 aces + 4 jacks + 2♣ + 10♦ = 13);
# with majority bonus, a hand is worth 16 points plus pişti bonuses.
TOTAL_CARD_POINTS = sum(CARD_POINTS)


def rank_of(card: int) -> int:
    return card % 13


def card_name(card: int) -> str:
    suit, rank = divmod(card, 13)
    return f"{RANKS[rank]}{SUIT_SYMBOLS[SUITS[suit]]}"


def new_deck(rng: Optional[random.Random] = None) -> List[int]:
    """Return a shuffled deck with a valid non-Jack opening center."""
    shuffler = rng or random
    while True:
        deck = list(range(52))
        shuffler.shuffle(deck)
        if any(rank_of(card) != JACK for card in deck[:4]):
            return deck


class PistiGame:
    """Mutable, fast Pişti game. Use clone() for search.

    Players are 0 and 1. `first_player` leads the first trick and receives
    the first 4-card packet of every deal round (mirror evaluation swaps
    this to give the other seat the same cards).
    """

    __slots__ = (
        "hands",
        "table",
        "hidden_center",
        "stock",
        "current",
        "first_player",
        "captured_count",
        "points",
        "pistis",
        "double_pistis",
        "last_capturer",
        "done",
        "initial_upcard",
        "history",
        "captured_hidden",
        "captured_hidden_by",
    )

    def __init__(
        self,
        deck: Optional[List[int]] = None,
        rng: Optional[random.Random] = None,
        first_player: int = 0,
    ):
        if deck is None:
            deck = new_deck(rng)
        if len(deck) != 52 or len(set(deck)) != 52:
            raise ValueError("deck must be a permutation of 0..51")

        # Table: face-up card is the first non-Jack among the first 4;
        # the other three stay hidden. An all-Jack center requires a redeal.
        first4 = deck[:4]
        if all(rank_of(card) == JACK for card in first4):
            raise ValueError("all-Jack opening center requires a redeal")
        up_idx = next(i for i, card in enumerate(first4) if rank_of(card) != JACK)
        self.initial_upcard = first4[up_idx]
        self.table: List[int] = [self.initial_upcard]
        self.hidden_center: List[int] = [c for i, c in enumerate(first4) if i != up_idx]

        self.first_player = first_player
        self.hands: List[List[int]] = [[], []]
        self.hands[first_player] = list(deck[4:8])
        self.hands[1 - first_player] = list(deck[8:12])
        self.stock: List[int] = list(deck[12:])

        self.current = first_player
        self.captured_count = [0, 0]
        self.points = [0, 0]  # card points + pişti bonuses (no majority)
        self.pistis = [0, 0]
        self.double_pistis = [0, 0]
        self.last_capturer: Optional[int] = None
        self.done = False
        # Hidden center cards after they are swept face-down into a pile:
        # their location is public, their identity is not.
        self.captured_hidden: List[int] = []
        self.captured_hidden_by: Optional[int] = None
        # (player, card, captured, pisti_kind) per move; pisti_kind: 0/1/2
        self.history: List[Tuple[int, int, bool, int]] = []

    # ------------------------------------------------------------------
    def legal_actions(self) -> List[int]:
        """Any card in the current player's hand may be played."""
        return list(self.hands[self.current])

    def step(self, card: int) -> dict:
        """Play `card` for the current player. Returns an info dict."""
        if self.done:
            raise RuntimeError("game is over")
        player = self.current
        try:
            self.hands[player].remove(card)
        except ValueError:
            raise ValueError(f"illegal action: {card_name(card)} not in hand of player {player}")

        captured = False
        pisti_kind = 0  # 0=no, 1=pişti, 2=double pişti
        gained = 0
        is_final_play = not self.stock and not self.hands[0] and not self.hands[1]

        if self.table:
            top = self.table[-1]
            if rank_of(card) == rank_of(top) or rank_of(card) == JACK:
                captured = True
                # Pişti requires a single-card pile with no hidden cards under it
                if len(self.table) == 1 and not self.hidden_center and not is_final_play:
                    if rank_of(card) == rank_of(top):
                        if rank_of(card) == JACK:
                            pisti_kind = 2
                            self.double_pistis[player] += 1
                            gained += DOUBLE_PISTI_POINTS
                        else:
                            pisti_kind = 1
                            self.pistis[player] += 1
                            gained += PISTI_POINTS
                swept = self.table + [card]
                if self.hidden_center:
                    swept += self.hidden_center
                    self.captured_hidden = self.hidden_center
                    self.captured_hidden_by = player
                    self.hidden_center = []
                gained += sum(CARD_POINTS[c] for c in swept)
                self.captured_count[player] += len(swept)
                self.points[player] += gained
                self.table = []
                self.last_capturer = player
        if not captured:
            self.table.append(card)

        self.history.append((player, card, captured, pisti_kind))
        self.current = 1 - player

        # Re-deal or finish
        if not self.hands[0] and not self.hands[1]:
            if self.stock:
                f, s = self.first_player, 1 - self.first_player
                self.hands[f] = list(self.stock[:4])
                self.hands[s] = list(self.stock[4:8])
                self.stock = self.stock[8:]
            else:
                self._finish()

        return {"captured": captured, "pisti": pisti_kind, "points_gained": gained}

    def _finish(self) -> None:
        """Sweep leftovers to the last capturer and mark the game done."""
        leftovers = self.table + self.hidden_center
        if leftovers and self.last_capturer is not None:
            p = self.last_capturer
            self.captured_count[p] += len(leftovers)
            self.points[p] += sum(CARD_POINTS[c] for c in leftovers)
        self.table = []
        self.hidden_center = []
        self.done = True

    # ------------------------------------------------------------------
    def scores(self) -> Tuple[int, int]:
        """Current scores; includes the majority bonus only when done."""
        s0, s1 = self.points[0], self.points[1]
        if self.done:
            if self.captured_count[0] > self.captured_count[1]:
                s0 += MAJORITY_POINTS
            elif self.captured_count[1] > self.captured_count[0]:
                s1 += MAJORITY_POINTS
        return s0, s1

    def score_diff(self, player: int) -> int:
        s0, s1 = self.scores()
        return (s0 - s1) if player == 0 else (s1 - s0)

    def winner(self) -> Optional[int]:
        """0, 1, or None for a tie. Only meaningful when done."""
        s0, s1 = self.scores()
        if s0 > s1:
            return 0
        if s1 > s0:
            return 1
        return None

    # ------------------------------------------------------------------
    def seen_cards(self, player: int) -> List[int]:
        """Cards `player` has observed: every card played face-up plus
        their own current hand. The first capturer also privately observes
        the three initially face-down center cards."""
        seen = [move[1] for move in self.history]
        seen.append(self.initial_upcard)
        seen.extend(self.hands[player])
        if self.captured_hidden_by == player:
            seen.extend(self.captured_hidden)
        return seen

    def clone(self) -> "PistiGame":
        g = object.__new__(PistiGame)
        g.hands = [list(self.hands[0]), list(self.hands[1])]
        g.table = list(self.table)
        g.hidden_center = list(self.hidden_center)
        g.stock = list(self.stock)
        g.current = self.current
        g.first_player = self.first_player
        g.captured_count = list(self.captured_count)
        g.points = list(self.points)
        g.pistis = list(self.pistis)
        g.double_pistis = list(self.double_pistis)
        g.last_capturer = self.last_capturer
        g.done = self.done
        g.initial_upcard = self.initial_upcard
        g.captured_hidden = list(self.captured_hidden)
        g.captured_hidden_by = self.captured_hidden_by
        g.history = list(self.history)
        return g

    def determinize(self, player: int, rng) -> "PistiGame":
        """Clone with all information hidden from `player` resampled.

        Unseen cards (opponent hand, stock, unrevealed center, and center
        cards privately held by the opponent) are redistributed uniformly
        among those same locations. The holder's points are adjusted for
        resampled captured-center cards, so a search over determinized clones
        uses exactly the information set of `player` — no private info leaks.

        `rng` needs shuffle(); both random.Random and numpy Generators work.
        """
        g = self.clone()
        opp = 1 - player
        seen = set(self.seen_cards(player))
        pool = [c for c in range(52) if c not in seen]
        rng.shuffle(pool)

        n_opp = len(g.hands[opp])
        n_hc = len(g.hidden_center)
        # The first capturer privately knows the center cards it collected;
        # only the opponent must resample their identities.
        n_ch = len(g.captured_hidden) if g.captured_hidden_by != player else 0
        if len(pool) != n_opp + n_hc + n_ch + len(g.stock):
            raise AssertionError("information-set accounting is broken")

        g.hands[opp] = pool[:n_opp]
        g.hidden_center = pool[n_opp : n_opp + n_hc]
        new_captured_hidden = pool[n_opp + n_hc : n_opp + n_hc + n_ch]
        g.stock = pool[n_opp + n_hc + n_ch :]

        if n_ch:
            holder = g.captured_hidden_by
            g.points[holder] += sum(CARD_POINTS[c] for c in new_captured_hidden)
            g.points[holder] -= sum(CARD_POINTS[c] for c in g.captured_hidden)
            g.captured_hidden = new_captured_hidden
        return g

    def render(self) -> str:
        lines = [
            f"table: {' '.join(card_name(c) for c in self.table) or '(empty)'}"
            + (f"  (+{len(self.hidden_center)} hidden)" if self.hidden_center else ""),
            f"hand P0: {' '.join(card_name(c) for c in sorted(self.hands[0]))}",
            f"hand P1: {' '.join(card_name(c) for c in sorted(self.hands[1]))}",
            f"stock: {len(self.stock)}  captured: {self.captured_count}"
            f"  points: {self.points}  to move: P{self.current}",
        ]
        return "\n".join(lines)
