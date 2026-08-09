# Learning Pişti: a self-play RL and game-theory study

> **Historical exploratory report.** These results predate the confirmatory rule and information-set corrections documented in `research/experiment_preregistration.md`. They describe the earlier game variant and are not pooled with the seed-replicated study in `paper/main.tex`.

*June 12, 2026 — single overnight run on an Apple M1 (8 cores, no GPU).*

## Summary

We train reinforcement learning agents to play the Turkish card game **Pişti** (2-player, zero-sum, imperfect information) and analyze the result with game-theoretic instruments. Main findings:

1. **Self-play PPO converges to equilibrium-grade robustness.** A dedicated best response wins +11.0 points/game against the early policy, but only **+0.5 ±0.9 (statistically zero)** against the final one. The policy crosses below the scripted heuristics' exploitability (~+3.2) at ~2M steps and matches determinized search (~+0.1) by 4M.
2. **The RL agent and Perfect-Information-Monte-Carlo expectimax tie at the top of the ladder**, both ~2–3.5 pts/game ahead of greedy/heuristic play (Bradley–Terry: 1550 vs 1542; head-to-head −0.4 ±1.2, p = 1.0). All cross-tier gaps are significant after Holm–Bonferroni correction; all within-tier orderings are statistical ties (§2).
3. **Card counting is worth ≈ 2.5 points per game.** Ablating the agent's memory of seen cards costs 2.1–2.9 pts/game against its memory-equipped twins — roughly the entire gap between a greedy heuristic and the best agents. In Pişti, memory *is* the skill.
4. **Luck is large but not overwhelming:** between equally strong agents the deal explains ~30–46% of outcome variance, and there is a consistent **first-mover advantage of ≈ +1.6 pts/game**.
5. The RL agent independently discovers human-recognizable tactics: jack discipline (saving Jacks for piles of 4.2 cards on average vs greedy's 3.6) and a higher capture tempo (5.0 captures/game).
6. **A masked DQN trained in the same league reaches the same top tier (BT 1529 [1522, 1537]) and is equally unexploitable (+0.23 ±0.98)** — falsifying our pre-registered guess that its deterministic policy would pay the "predictability tax." Private information already supplies the mixing; what matters is rich conditioning + self-play, not a stochastic action rule. DQN was also far more sample-efficient early (parity with greedy by ~50k steps vs PPO's ~1.5M).
7. **NFSP — the method with the Nash convergence story — matches the league agents' robustness (+0.74 ±0.99 external; +1.15 ±0.90 against its own internal best response) but lands ~25 Elo below them on the ladder** at a larger step budget (10M). The internal-vs-external gap also calibrates the whole exploitability table: read "statistically zero" as "≤ ~1–2 pts/game."

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
| `dqn_main` | masked DQN (Q-values masked inside the network forward pass; ε-greedy and warm-up sample legal actions only), same 6M-step curriculum + self-play league |
| `nfsp_main` | NFSP (Heinrich & Silver 2016): DQN best response + reservoir-averaged policy Π, η = 0.1, pure self-play with shared parameters, no curriculum, 10M steps. Π (stochastic) is what plays. |

**Training:** the undiscounted sum of per-step score-differential changes telescopes exactly to the final score differential. The historical PPO configuration used `gamma=0.999`, so it added slight temporal weighting. Opponents follow a curriculum of scripted baselines that transitions into a self-play league (snapshot pool + live mirror) from 1M steps. One 6M-step run ≈ 67 min on the M1.

**Evaluation:** all matches use **mirrored deals** (each deck played twice with seats/hands swapped) and report 95% CIs over deal-paired differences. This matters: deal luck is large, and pairing roughly halves the sample size needed for a given precision.

## 2. The ladder

Round-robin tournament, 250 mirrored deals (500 games) per pair, deterministic play for trained policies. Ratings carry deal-level bootstrap 95% CIs (1000 resamples):

| rank | agent | Bradley–Terry (Elo-like) | 95% CI |
|---|---|---|---|
| 1 | ppo_main | 1542.3 | [1534, 1550] |
| 2 | ppo_s2 | 1541.3 | [1533, 1549] |
| 3 | expectimax | 1537.4 | [1530, 1545] |
| 4 | ppo_s1 | 1533.9 | [1526, 1542] |
| 5 | dqn_main | 1529.4 | [1522, 1537] |
| 6 | nfsp_main | 1517.0 | [1509, 1525] |
| 7 | ppo_nomem | 1503.7 | [1496, 1512] |
| 8 | hunter | 1502.3 | [1495, 1510] |
| 9 | greedy | 1492.2 | [1485, 1500] |
| 10 | random | 1300.5 | [1291, 1310] |

*(10-agent tournament, `results/tournament_v3.json`; shared matchups replicate the earlier tournaments exactly — same seeds and deals — so only the ratings refit.)*

Three tiers emerge, separated by non-overlapping CIs: **{ppo seeds, expectimax, dqn} > {ppo_nomem, hunter, greedy} > {random}** — with NFSP occupying its own intermediate rung (1517 [1509, 1525]): significantly above greedy, significantly below the top PPO seeds, marginally overlapping DQN. Its individual head-to-heads are all statistical ties (it loses ~1 pt/game to each league agent and beats heuristics by <1; nothing survives Holm) — the rating separation comes from aggregating across all nine of its matches. Differences *within* a tier are not statistically distinguishable.

Head-to-heads (mean pts/game ± 95% CI; p-values are paired t-tests on mirror-paired deal differences, Holm–Bonferroni corrected across all 28 matches; \* = significant at α=0.05):

| matchup | win rate | pts/game | p (Holm) | |
|---|---|---|---|---|
| ppo_main vs greedy | 0.588 | +2.36 ±1.31 | 0.007 | \* |
| ppo_main vs hunter | 0.571 | +2.12 ±1.30 | 0.018 | \* |
| expectimax vs greedy | 0.595 | +3.59 ±1.25 | 1.2e-06 | \* |
| ppo_main vs ppo_nomem | 0.548 | +2.49 ±1.31 | 0.004 | \* |
| ppo_s1 vs ppo_nomem | 0.564 | +2.87 ±1.19 | 7e-05 | \* |
| ppo_s2 vs ppo_nomem | 0.544 | +2.11 ±1.26 | 0.014 | \* |
| dqn_main vs greedy | 0.567 | +2.59 ±1.29 | 0.002 | \* |
| dqn_main vs hunter | 0.547 | +1.90 ±1.22 | 0.037 | \* |
| dqn_main vs ppo_nomem | 0.561 | +2.68 ±1.26 | 0.001 | \* |
| dqn_main vs ppo_main | 0.510 | +0.43 ±1.16 | 1.0 | tie |
| ppo_main vs expectimax | 0.521 | −0.39 ±1.18 | 1.0 | tie |
| ppo_main vs ppo_s1 | 0.504 | +0.05 ±1.11 | 1.0 | tie |
| ppo_main vs ppo_s2 | 0.509 | +0.41 ±1.29 | 1.0 | tie |
| greedy vs hunter | 0.487 | −0.28 ±0.65 | 1.0 | tie |
| greedy vs ppo_nomem | 0.504 | −0.09 ±1.25 | 1.0 | tie |

Full table: `results/tournament_significance.json` (Wilcoxon signed-rank p-values agree with the t-tests throughout).

**What is significant and what is noise.** All claims of the form "trained agents beat heuristics" and "memory beats no-memory" survive multiple-comparison correction comfortably. The *ordering within the top tier* (ppo_main > ppo_s2 > expectimax > ppo_s1, a 14-Elo span) is **noise** — these four are statistically indistinguishable with 500 games per pair; detecting a 0.4 pts/game gap (if it exists) would need roughly 10× more deals. The same is true for nomem vs hunter vs greedy. See `plots/ratings.png`, `plots/tournament_heatmap.png`, `plots/training_curves.png`.

## 3. Exploitability

True Nash-distance is intractable here, so we report a lower bound: train a **best response** (MaskablePPO warm-started from the strongest agent, 1.5M specialization steps) against each frozen *stochastic* target, then measure its edge over 400 mirrored deals.

| target | BR edge (pts/game) | BR win rate | p (edge ≠ 0) |
|---|---|---|---|
| ppo @ 250k steps | +11.04 ±0.97 | 0.78 | <1e-15 |
| ppo @ 500k | +9.95 ±0.99 | 0.74 | <1e-15 |
| ppo @ 1M | +6.58 ±0.97 | 0.67 | <1e-15 |
| ppo @ 2M | +2.96 ±0.98 | 0.59 | 3.5e-09 |
| ppo @ 4M | **+0.59 ±1.00** | 0.53 | 0.24 |
| ppo @ 6M (final) | **+0.48 ±0.93** | 0.50 | 0.31 |
| greedy | +3.20 ±1.01 | 0.59 | 4.3e-10 |
| hunter | +3.25 ±0.99 | 0.59 | 1.4e-10 |
| expectimax (light) | **+0.06 ±0.96** | 0.49 | 0.90 |
| dqn @ 6M (deterministic) | **+0.23 ±0.98** | 0.50 | 0.64 |
| nfsp Π @ 10M | **+0.74 ±0.99** | 0.51 | 0.14 |

The exploitability of the 4M and final policies is **not statistically distinguishable from zero** (p = 0.24, 0.31), while every checkpoint up to 2M and both scripted heuristics are exploitable at overwhelming significance. Read the final numbers as "≤ ~1.4 pts/game with 95% confidence" (upper end of the CI), not as exactly zero — and as lower bounds given the fixed BR protocol.

**NFSP: theory vs the league.** NFSP is the method with an actual convergence story — its averaged policy Π should approach Nash. After 10M self-play steps (no curriculum, no league): external exploitability **+0.74 ±0.99** — statistically indistinguishable from zero and from the league agents' numbers. NFSP also carries an *internal* exploitability estimate: its own Q-network is a continuously-trained best response to Π, and it beats Π by **+1.15 ±0.90** — slightly larger than what our external 1.5M-step assassin finds (+0.74), which is expected: the internal attacker trained against Π ~7× longer. Two implications: (a) all the "statistically zero" exploitability numbers in this table should be read as *≤ ~1–2 pts/game*, since a sufficiently trained attacker finds at least +1.15 against NFSP; (b) the pragmatic self-play league matched the principled method's robustness while finishing ~25 Elo stronger on the ladder at a smaller step budget — in this game, fictitious-play averaging buys theoretical reassurance, not measurable extra safety.

**A falsified hypothesis (and what it taught us).** We added the DQN specifically to test the prediction that its *deterministic* greedy policy would be exploitable like the scripted heuristics (~+3), since equilibrium play in imperfect-information games requires mixing. The data said no: **+0.23 ±0.98 (p = 0.64)**. The resolution is that mixing does not require a stochastic action rule — a deterministic function of *private* information (your hidden hand, your card memory) is already unpredictable from the opponent's seat, because the deal supplies the randomness. What actually separates the exploitable from the unexploitable in our table is **conditioning richness and self-play training**: greedy/hunter are predictable *from public information alone* and have systematic habits; both league-trained agents and belief-sampling search are not. Determinism per se carries no measurable tax.

Three observations (`plots/exploitability.png`):

- The curve drops steeply exactly when the self-play league enters the opponent mixture (1M steps). Training only against scripted opponents left the policy highly exploitable; playing against its own past selves is what produced robustness.
- Deterministic heuristics sit at ~+3.2: predictability is itself a vulnerability.
- PIMC expectimax is unexploitable *without any training* — its determinization sampling acts as a naturally mixed strategy. Self-play RL needs ~4M steps to earn the same property, but then also plays stronger overall.

*Caveat: these are lower bounds under a fixed BR protocol; a stronger/longer BR could raise all numbers, but the comparisons across targets share the protocol.*

**Policy + decision-time search adds nothing (the "Beast" experiment).** Following AlphaZero's recipe, `BeastAgent` wraps the trained PPO policy in determinized flat Monte-Carlo search: for each legal action, sample 32 determinizations, play the action, roll every game to terminal with the stochastic policy on both seats, and pick the best mean score differential. Over 150 mirrored deals it is statistically identical to its own base policy (**−0.05 ±1.48** vs `ppo_main`) and to `expectimax32` (−0.10 ±1.46) — despite *overriding* the policy's own choice on **52%** of multi-action decisions. Search changes half the moves and none of the outcome: the rollout value estimates are noise around actions whose true values are nearly equal. Together with the statistically-zero exploitability of the final policies, this is converging evidence that the top agents sit at (or within measurement noise of) the game's effective skill ceiling — there is little value left on the table for one-step lookahead to recover. (`results/beast_benchmark.json`)

## 4. What is memory worth?

`ppo_nomem` trains identically but cannot see which cards have been played (the `seen` vector is zeroed; opponents keep full memory).

- Head-to-head vs its three memory-equipped twins: **−2.49 ±1.31, −2.87 ±1.19, −2.11 ±1.26 pts/game** (all three significant after Holm correction: p = 0.004, 7e-05, 0.014).
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

*(These behavioral counts are descriptive, from 400 games each without significance tests; treat differences smaller than ~10% as suggestive rather than established.)*

## 7. Reproducibility

- Engine invariants and rule tests: `venv/bin/python -m pytest` (22 tests).
- Each run directory carries its resolved config, git hash, eval history, checkpoints.
- Tournament/exploitability JSONs include seeds and protocols; figures regenerate via `scripts/run_analysis.py`.
- Total compute: ~4.5 h wall on one M1 (4 training runs, 9 best responses, 14k tournament games).

## 8. Limitations & next steps

- Exploitability numbers are lower bounds under a 1.5M-step warm-started BR protocol.
- Deterministic tournament play slightly flatters predictable policies; stochastic-policy tournaments are one flag away.
- Natural extensions: regret-based methods (Deep CFR) to complete the equilibrium-method comparison; a history-conditioned best response (the sharpest test of the private-information-mixing hypothesis); LSTM policies vs the explicit `seen` vector; 4-player partnership Pişti; opponent modeling (exploiting *weak* opponents on purpose — the current agents play safe, not maximally punishing).
