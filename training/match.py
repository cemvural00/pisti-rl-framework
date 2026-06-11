"""Match and tournament driver with mirrored deals.

Mirrored ("duplicate") evaluation: every deal is played twice with the
seats swapped — agent A first gets the leading seat and its 4-card
packets, then B does, on the identical deck order. Deal luck cancels in
the paired difference, which slashes the variance of skill estimates.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from encoding.obs import Observer
from engine.game import PistiGame, new_deck


@dataclass
class GameRecord:
    deal: int
    a_leads: bool
    score_a: int
    score_b: int
    pistis_a: int
    pistis_b: int
    captured_a: int
    captured_b: int

    @property
    def diff(self) -> int:
        return self.score_a - self.score_b


@dataclass
class MatchResult:
    agent_a: str
    agent_b: str
    records: List[GameRecord] = field(default_factory=list)

    @property
    def n_games(self) -> int:
        return len(self.records)

    @property
    def wins_a(self) -> int:
        return sum(r.diff > 0 for r in self.records)

    @property
    def wins_b(self) -> int:
        return sum(r.diff < 0 for r in self.records)

    @property
    def ties(self) -> int:
        return sum(r.diff == 0 for r in self.records)

    @property
    def win_rate_a(self) -> float:
        """Ties count as half a win for each side."""
        return (self.wins_a + 0.5 * self.ties) / max(self.n_games, 1)

    def mean_diff(self) -> float:
        return float(np.mean([r.diff for r in self.records]))

    def paired_diffs(self) -> np.ndarray:
        """Per-deal mean diff across the two seatings (mirror-paired)."""
        by_deal: Dict[int, List[int]] = {}
        for r in self.records:
            by_deal.setdefault(r.deal, []).append(r.diff)
        return np.array([np.mean(v) for v in by_deal.values()])

    def diff_ci95(self) -> Tuple[float, float]:
        """Mean score diff with a 95% CI from mirror-paired deals."""
        d = self.paired_diffs()
        m = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        return m, 1.96 * se

    def summary(self) -> Dict:
        m, ci = self.diff_ci95()
        return {
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "n_games": self.n_games,
            "win_rate_a": round(self.win_rate_a, 4),
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "ties": self.ties,
            "mean_diff": round(m, 3),
            "diff_ci95": round(ci, 3),
            "pistis_per_game": (
                round(np.mean([r.pistis_a for r in self.records]), 3),
                round(np.mean([r.pistis_b for r in self.records]), 3),
            ),
        }


def play_game(
    agent_a,
    agent_b,
    deck: List[int],
    a_leads: bool,
    deal: int = 0,
    observer_a: Optional[Observer] = None,
    observer_b: Optional[Observer] = None,
) -> GameRecord:
    """Play one game; agent_a is player 0, agent_b is player 1."""
    obs_a = observer_a or Observer()
    obs_b = observer_b or Observer()
    game = PistiGame(deck=list(deck), first_player=0 if a_leads else 1)
    for agent in (agent_a, agent_b):
        if hasattr(agent, "reset"):
            agent.reset()

    while not game.done:
        p = game.current
        agent = agent_a if p == 0 else agent_b
        observer = obs_a if p == 0 else obs_b
        obs = observer.encode(game, p)
        if getattr(agent, "wants_game", False):
            action = agent.predict(obs, obs["action_mask"], game=game, player=p)
        else:
            action = agent.predict(obs, obs["action_mask"])
        game.step(int(action))

    s0, s1 = game.scores()
    return GameRecord(
        deal=deal,
        a_leads=a_leads,
        score_a=s0,
        score_b=s1,
        pistis_a=game.pistis[0] + game.double_pistis[0],
        pistis_b=game.pistis[1] + game.double_pistis[1],
        captured_a=game.captured_count[0],
        captured_b=game.captured_count[1],
    )


def play_match(
    agent_a,
    agent_b,
    n_deals: int = 200,
    seed: int = 0,
    mirrored: bool = True,
    name_a: str = "A",
    name_b: str = "B",
    observer_a: Optional[Observer] = None,
    observer_b: Optional[Observer] = None,
) -> MatchResult:
    """Play n_deals decks; with mirrored=True each deck is played twice
    (seats swapped), so the match has 2*n_deals games."""
    rng = random.Random(seed)
    result = MatchResult(agent_a=name_a, agent_b=name_b)
    for deal in range(n_deals):
        deck = new_deck(rng)
        seatings = (True, False) if mirrored else (deal % 2 == 0,)
        for a_leads in seatings:
            result.records.append(
                play_game(
                    agent_a,
                    agent_b,
                    deck,
                    a_leads,
                    deal=deal,
                    observer_a=observer_a,
                    observer_b=observer_b,
                )
            )
    return result
