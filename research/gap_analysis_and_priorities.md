# Gap Analysis and Experiment Priorities

## Candidate Paper Narratives

1. **Memory in Pişti:** quantify the causal value of explicit played-card memory and separate training adaptation from acute information removal.
2. **Pişti benchmark:** introduce a correct, fast, reproducible environment with learning, search, heuristic, and robustness baselines.
3. **Self-play comparison:** compare league PPO/DQN/NFSP and fixed-opponent training.
4. **Near-equilibrium claim:** argue that learned agents approach a Pişti equilibrium.

The review favors narratives 1+2. Narrative 3 overlaps recent Big 2 and broad imperfect-information results. Narrative 4 cannot be supported without exact exploitability or much stronger approximate attacks.

## Predefined Ranking Rule

Each candidate receives 1--5 for validity necessity (`V`), novelty gained (`N`), ability to discriminate explanations (`D`), and reuse across paper/blog/portfolio/video (`R`). Effort/risk (`C`) is 1--5. Priority is `3V + 2N + 2D + R - C`. Correctness and reproducibility defects can be mandatory regardless of score.

| Candidate | V | N | D | R | C | Score | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Seed-matched full/no-memory PPO training | 5 | 5 | 5 | 5 | 3 | 37 | Run |
| Acute inference-time memory removal on trained policies | 4 | 4 | 5 | 5 | 1 | 34 | Run |
| Replicate approximate-BR training across attacker seeds | 5 | 3 | 5 | 4 | 4 | 31 | Run for the principal target |
| Fixed-opponent control for the league-transition claim | 4 | 3 | 5 | 4 | 3 | 29 | Run as secondary ablation if pilot cost is acceptable |
| Add uncertainty tests for behavioral metrics | 3 | 3 | 4 | 5 | 1 | 27 | Run on regenerated records where available |
| Human-expert evaluation/interviews | 3 | 4 | 4 | 5 | 5 | 25 | Defer: requires external participants and ethics/consent decisions |
| Fix installation and freeze dependencies/results manifest | 5 | 1 | 1 | 5 | 1 | 23 | Mandatory artifact work |
| Implement tuned ISMCTS | 3 | 2 | 3 | 4 | 4 | 19 | Defer unless learned agents lack a fair search baseline |
| Port to OpenSpiel and implement Deep CFR | 2 | 2 | 3 | 2 | 5 | 13 | Reject for this paper; high cost, low question alignment |

## Decision

The primary experiment is a seed-matched memory study. A pre-outcome timing pilot showed that ten fully retrained seed pairs were practical, so memory-on and memory-off policies are trained from scratch with identical seeds and schedules for seeds 0--9. Each policy is evaluated both with its training-time observation and under acute inference-time memory removal. This distinguishes (a) the benefit of access to history, (b) adaptation during training, and (c) training-run variation.

A five-seed fixed-opponent control tests whether the observed post-1M robustness change can be attributed to self-play rather than elapsed learning alone. This is secondary because a similar curriculum conclusion now exists in Big 2.

For robustness, three independently trained warm-started PPO attackers target the principal stochastic PPO policy under a frozen budget. Results will be described as **attack returns/lower bounds**, not exact exploitability. No new equilibrium algorithm is justified by the selected research question.

## Go/No-Go Rules

- The memory claim is retained only if the seed-paired effect has a consistent sign and its seed-level interval is practically meaningful; deal-only significance is insufficient.
- The acute ablation is interpreted separately from retraining. A large acute loss and small retrained loss means adaptation, not that memory is intrinsically worth the acute estimate.
- The league claim is retained only if the between-condition effect exceeds training variation; otherwise it becomes exploratory.
- Failure of replicated attackers to win supports "no exploit found under three attacks," never equilibrium convergence.
- New results supersede old prose where they conflict; they are not selected for agreement with the existing report.
