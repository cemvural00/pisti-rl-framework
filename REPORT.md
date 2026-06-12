# Learning Pişti: a self-play RL and game-theory study

*June 12, 2026 — single overnight run on an Apple M1 (8 cores, no GPU).*

## Summary

We train reinforcement learning agents to play the Turkish card game **Pişti** (2-player, zero-sum, imperfect information) and analyze the result with game-theoretic instruments. Main findings:

1. **Self-play PPO converges to equilibrium-grade robustness.** A dedicated best response wins +11.0 points/game against the early policy, but only **+0.5 ±0.9 (statistically zero)** against the final one. The policy crosses below the scripted heuristics' exploitability (~+3.2) at ~2M steps and matches determinized search (~+0.1) by 4M.
2. **The RL agent and Perfect-Information-Monte-Carlo expectimax tie at the top of the ladder**, both ~2–3.5 pts/game ahead of greedy/heuristic play (Bradley–Terry: 1550 vs 1542; head-to-head −0.4 ±1.2).
3. **Card counting is worth ≈ 2.5 points per game.** Ablating the agent's memory of seen cards costs 2.1–2.9 pts/game against its memory-equipped twins — roughly the entire gap between a greedy heuristic and the best agents. In Pişti, memory *is* the skill.
4. **Luck is large but not overwhelming:** between equally strong agents the deal explains ~30–46% of outcome variance, and there is a consistent **first-mover advantage of ≈ +1.6 pts/game**.
5. The RL agent independently discovers human-recognizable tactics: jack discipline (saving Jacks for piles of 4.2 cards on average vs greedy's 3.6) and a higher capture tempo (5.0 captures/game).

## 1. Setup

### Game and engine

Standard 2-player Pişti: 4 table cards (3 face-down, 1 face-up), 4-card hands re-dealt over 6 rounds; capture by rank match or Jack; pişti (capturing a single-card pile by rank match) scores 10, Jack-on-Jack 20; A/J = 1 pt each, 2♣ = 2, 10♦ = 3, card majority +3; the end-of-game leftover pile goes to the last capturer. Hidden information: opponent hand, stock order, and the face-down center cards (which we keep hidden even after they are captured).

The engine (`engine/game.py`) runs at ~1M moves/s and exposes `determinize(player, rng)` — a uniform resampling of everything outside the player's information set, with captured-hidden-card points adjusted, so search agents can plan honestly without leaking private state. Rule correctness is enforced by golden-scenario tests plus invariants (card conservation, score accounting) over thousands of random games.

### Agents

| agent | description |
|---|---|
| `random` | uniform random legal card |
| `greedy` | always captures if possible; discards cheapest card |
| `hunter` | pişti-focused heuristic (takes pişti, baits with duplicate ranks, hoards Jacks) |
| `expectimax` | PIMC: 16 determinizations × greedy rollouts per legal action |
| `ppo_main/s1/s2` | MaskablePPO (256×256), 6M steps, 3 seeds |
| `ppo_nomem` | identical, but the `seen` (card-memory) observation is zeroed |

**Training:** the env's per-step reward is the change in score differential between the agent's decision points, which telescopes exactly to the final score differential — the true game objective, densely distributed, with no shaping hyperparameters. Opponents follow a curriculum of scripted baselines that transitions into a self-play league (snapshot pool + live mirror) from 1M steps. One 6M-step run ≈ 67 min on the M1.

**Evaluation:** all matches use **mirrored deals** (each deck played twice with seats/hands swapped) and report 95% CIs over deal-paired differences. This matters: deal luck is large, and pairing roughly halves the sample size needed for a given precision.

## 2. The ladder

Round-robin tournament, 250 mirrored deals (500 games) per pair, deterministic play for trained policies:

| rank | agent | Bradley–Terry (Elo-like) |
|---|---|---|
| 1 | ppo_main | 1549.5 |
| 2 | ppo_s2 | 1547.4 |
| 3 | expectimax | 1542.2 |
| 4 | ppo_s1 | 1536.0 |
| 5 | ppo_nomem | 1509.7 |
| 6 | hunter | 1509.1 |
| 7 | greedy | 1497.4 |
| 8 | random | 1308.8 |

Selected head-to-heads (mean score diff ± 95% CI, positive = row wins):

| matchup | win rate | pts/game |
|---|---|---|
| ppo_main vs greedy | 0.588 | **+2.36 ±1.31** |
| ppo_main vs hunter | 0.571 | **+2.12 ±1.30** |
| ppo_main vs expectimax | 0.521 | −0.39 ±1.18 (tie) |
| expectimax vs greedy | 0.595 | **+3.59 ±1.25** |
| ppo_main vs ppo_nomem | 0.548 | **+2.49 ±1.31** |
| greedy vs hunter | 0.487 | −0.28 ±0.65 (tie) |

The three RL seeds finish within ~14 Elo of each other and are mutually statistically tied — self-play training is reproducible in strength, not just in curve shape. See `plots/ratings.png`, `plots/tournament_heatmap.png`, `plots/training_curves.png`.

## 3. Exploitability

True Nash-distance is intractable here, so we report a lower bound: train a **best response** (MaskablePPO warm-started from the strongest agent, 1.5M specialization steps) against each frozen *stochastic* target, then measure its edge over 400 mirrored deals.

| target | BR edge (pts/game) | BR win rate |
|---|---|---|
| ppo @ 250k steps | +11.04 ±0.97 | 0.78 |
| ppo @ 500k | +9.95 ±0.99 | 0.74 |
| ppo @ 1M | +6.58 ±0.97 | 0.67 |
| ppo @ 2M | +2.96 ±0.98 | 0.59 |
| ppo @ 4M | **+0.59 ±1.00** | 0.53 |
| ppo @ 6M (final) | **+0.48 ±0.93** | 0.50 |
| greedy | +3.20 ±1.01 | 0.59 |
| hunter | +3.25 ±0.99 | 0.59 |
| expectimax (light) | **+0.06 ±0.96** | 0.49 |

Three observations (`plots/exploitability.png`):

- The curve drops steeply exactly when the self-play league enters the opponent mixture (1M steps). Training only against scripted opponents left the policy highly exploitable; playing against its own past selves is what produced robustness.
- Deterministic heuristics sit at ~+3.2: predictability is itself a vulnerability.
- PIMC expectimax is unexploitable *without any training* — its determinization sampling acts as a naturally mixed strategy. Self-play RL needs ~4M steps to earn the same property, but then also plays stronger overall.

*Caveat: these are lower bounds under a fixed BR protocol; a stronger/longer BR could raise all numbers, but the comparisons across targets share the protocol.*

## 4. What is memory worth?

`ppo_nomem` trains identically but cannot see which cards have been played (the `seen` vector is zeroed; opponents keep full memory).

- Head-to-head vs its three memory-equipped twins: **−2.49 ±1.31, −2.87 ±1.19, −2.11 ±1.26 pts/game**.
- On the ladder it falls from the top group (~1545) to heuristic level (1510), statistically tied with `hunter`.
- Against greedy it manages only −0.09 ±1.25 (a tie), where memory-equipped seeds score +2.4 to +2.5.

So perfect card memory is worth **≈ 2.5 points per game** — almost exactly the entire skill gap between a greedy baseline and the strongest agents. This quantifies the folk wisdom that counting cards is the core skill of Pişti.

## 5. Luck, skill, and the seat

Mirror-pairing lets us decompose each deal's outcome into a **deal effect** (mean of the two seatings — pure card luck), and a seat/noise component. Across matchups (`results/luck_vs_skill.json`, `plots/luck_share.png`):

- Between near-equal strong agents (e.g. ppo seeds), the deal explains **~28–30%** of outcome variance; the median across all matchups is ~36%, and it never exceeds ~46% (vs random it's diluted by random's own noise). The rest is policy stochasticity and within-game interaction — Pişti rewards skill more than its reputation suggests, *if* you average over enough games.
- **First-mover advantage: ≈ +1.6 pts/game** averaged across matchups (individually ±1.5–1.9 CI). Leading gives first crack at the face-up card and pile control.
- A skill gap of +2.5 pts/game (memory vs no memory) corresponds to a ~55% win rate: in a single game, the better player is only modestly favored; over a 151-point match (~10 hands), they are heavily favored.

## 6. Emergent tactics

Behavioral statistics over 400 games vs greedy (`results/behavior.json`):

| metric | greedy | hunter | expectimax | **ppo_main** |
|---|---|---|---|---|
| captures/game | 4.65 | 4.52 | 4.83 | **5.01** |
| pişti/game | 0.71 | 0.66 | 0.82 | 0.72 |
| avg pile size on Jack captures | 3.56 | 4.32 | 4.41 | **4.18** |
| risky discards/game | 2.29 | 2.39 | 2.44 | 2.44 |
| bait rate (duplicate-rank discards on empty table) | 0.10 | 0.05 | 0.08 | 0.08 |

The RL agent learned **jack discipline** (holding Jacks until piles are ~17% larger than greedy tolerates) and the highest capture tempo, without ever being told these concepts exist — its only signal was the final score differential.

## 7. Reproducibility

- Engine invariants and rule tests: `venv/bin/python -m pytest` (22 tests).
- Each run directory carries its resolved config, git hash, eval history, checkpoints.
- Tournament/exploitability JSONs include seeds and protocols; figures regenerate via `scripts/run_analysis.py`.
- Total compute: ~4.5 h wall on one M1 (4 training runs, 9 best responses, 14k tournament games).

## 8. Limitations & next steps

- Exploitability numbers are lower bounds under a 1.5M-step warm-started BR protocol.
- Deterministic tournament play slightly flatters predictable policies; stochastic-policy tournaments are one flag away.
- Natural extensions: NFSP/regret-based methods as a principled equilibrium comparison; LSTM policies vs the explicit `seen` vector; 4-player partnership Pişti; opponent modeling (exploiting *weak* opponents on purpose — the current agent plays safe, not maximally punishing).
