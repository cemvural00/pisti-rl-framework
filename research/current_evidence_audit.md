# Current Evidence and Reproducibility Audit

**Audit date:** 2026-08-08
**Audited revision:** `465fe24`

> **Post-audit status:** The audit intentionally records the starting state. The definitive study subsequently corrected the four rule/information issues documented in the protocol, expanded the suite to 32 passing tests, reached 95% coverage across the engine/encoding/environment/match core, restored editable installation and clean formatting/lint checks, and regenerated all confirmatory artifacts. See `final_research_summary.md` for the completed evidence.

## Artifact Status

- The rule/environment suite passes: 22 tests in 0.91 seconds on Python 3.12.2.
- Coverage is concentrated in the engine (97%), environment (90%), and mirrored-match code (94%); total measured coverage is 32%. Training entry points and most learned/search agents have no direct test coverage.
- `pip install -e ".[dev]"` fails because setuptools discovers artifact directories as top-level packages. Bootstrapping from `requirements.txt` works, but dependencies are lower-bounded rather than locked.
- `black --check .` reports 22 files requiring formatting. The documented formatting command therefore does not pass at the audited revision.
- Run metadata preserves resolved configurations for PPO seeds 0--2, one no-memory PPO run (seed 3), DQN seed 0, and NFSP seed 0. Runs were produced at multiple historical commits; there is no single frozen environment manifest that recreates every result.
- Tournament JSON retains deal-level records and pairing; exploitability files retain aggregate protocol metadata. Existing plots and reports are generated from checked-in results.
- A cross-check against independent rules descriptions found that the audited engine awarded pişti on the final card, while standard descriptions disallow it. Confirmatory runs started under that implementation were stopped before completion. The rule and a regression test were corrected before restarting; historical results are explicitly treated as results for the prior variant.

## Claims-to-Evidence Inventory

| Claim | Current evidence | Status | Main threat |
|---|---|---|---|
| Memory of seen cards is worth about 2.5 points/game | One no-memory training run compared with three memory-enabled PPO policies over mirrored deals | Suggestive, replicated across opponents | Treatment is confounded with training seed; no paired full/no-memory runs by seed |
| PPO, DQN, expectimax, and other top agents have similar playing strength | 250 mirrored deals per tournament pairing, deal bootstrap ratings, paired tests with Holm correction | Established for evaluated artifacts/protocol | Training uncertainty is represented only for PPO; opponent set is endogenous |
| Later trained policies resist the implemented best-response attack | Warm-started PPO attacker and 400 mirrored evaluation deals per target | Established only for this attacker | Approximate BR is a lower bound; attacker seed/architecture/budget uncertainty absent |
| PPO approaches robustness after league self-play begins | Checkpoint attack curve with league introduced at 1M steps | Suggestive | Time and curriculum stage are confounded; no no-league control |
| Deterministic DQN suffers no measurable "predictability tax" | DQN attack estimate `+0.23 ± 0.98` under one protocol | Suggestive negative result | One target seed and one attacker family; private-history explanation not directly tested |
| NFSP matches league robustness but is weaker on the ladder | One 10M-step NFSP run; external and internal attackers; tournament | Suggestive | One training seed; implementation and budget differ from PPO/DQN |
| Card deal explains roughly 30--46% of equal-agent variance | Decomposition of mirrored tournament records | Descriptive for tested matchups | Policy stochasticity and interaction remain in residual; scope is not all Pişti play |
| First player has about a 1.6-point advantage | Seat-swapped mirrored records across matchups | Descriptive and well controlled | Dependence across repeated agents/matchups is not modelled hierarchically |
| The policy learned human-recognizable tactics | 400-game behavioral summaries against greedy | Exploratory | Most behavioral differences lack uncertainty tests; no human expert validation |
| Decision-time rollout search adds no strength | 150 mirrored deals for Beast vs base policy/search; high action override rate | Suggestive negative result | Search configuration is one point in a broad design space; CI remains wide |

## Highest-Priority Unknowns Before Literature Synthesis

1. Whether a Pişti-specific domain contribution is genuinely new.
2. Whether venue-relevant application papers require multiple independent training seeds or primarily game-level replication.
3. Whether the memory result survives seed-matched training.
4. Whether approximate exploitability should be strengthened through attacker replication/diversity rather than another equilibrium-learning algorithm.
5. Whether the most defensible paper is a Pişti case study, an evaluation-method paper, or a focused memory/private-information empirical paper.
