# Pişti RL — a reinforcement learning & game theory study

**Author:** Cem Vural

Teaching RL agents the Turkish card game **Pişti** (2-player, zero-sum, imperfect information), and measuring what they learn with game-theoretic tools: mirrored-deal tournaments, Bradley–Terry ratings, approximate exploitability via best-response training, and a card-counting ablation.

## The game

Pişti is played with a standard 52-card deck. Players take turns playing one card onto a shared pile; you **capture** the pile by matching the top card's rank (or playing a Jack). Capturing a *single-card* pile by rank match is a **pişti** — worth 10 points (Jack-on-Jack: 20). Scoring: each Ace and Jack 1, 2♣ 2, 10♦ 3, majority of captured cards +3 → 16 base points per hand plus pişti bonuses. Hidden information: the opponent's hand, the stock order, and 3 face-down center cards.

## Why it's interesting

- **Imperfect information**: pure self-play PPO has no equilibrium guarantee here — how close does it get? We measure approximate exploitability by training best responses against frozen policies.
- **Memory matters (or does it?)**: card counting is the classic human skill in Pişti. We ablate the agent's `seen` vector to put a number on what perfect memory is worth.
- **Luck vs skill**: every deal is played twice with seats swapped (duplicate-bridge style), which both slashes evaluation variance and lets us decompose outcomes into deal luck, seat advantage, and skill.

## Quick start

```bash
python3 -m venv venv && venv/bin/pip install -e ".[dev]"
venv/bin/python -m pytest                  # rule-correctness suite

# Train the main agent (MaskablePPO + curriculum + self-play league)
venv/bin/python -m training.train --config configs/default.yaml

# Tournament with mirrored deals + Bradley–Terry ratings
venv/bin/python -m training.evaluate \
    --agents random greedy hunter expectimax "ppo:runs/ppo_main/final_model" \
    --n-deals 300 --out results/tournament.json

# Approximate exploitability of a trained policy
venv/bin/python -m training.exploitability \
    --target ppo-stoch:runs/ppo_main/final_model.zip \
    --init-from runs/ppo_main/final_model.zip

# Play against the agents — browser GUI (recommended) or terminal
venv/bin/python scripts/gui.py     # -> http://localhost:8777
venv/bin/python scripts/play.py
```

## Architecture

```
engine/game.py      int-card Pişti engine (~1M moves/s), clone() for search,
                    determinize() for honest information-set sampling
encoding/obs.py     Observer -> Dict obs (hand/table_top/seen + stats + mask);
                    memory=False ablates card counting
envs/pisti_env.py   Gymnasium env; episode return ≡ final score differential
agents/             baselines (random/greedy/pisti-hunter), PIMC expectimax,
                    frozen-policy self-play league
training/           train.py (curriculum+league), match.py (mirrored deals),
                    evaluate.py (tournament+ratings), exploitability.py (BR)
analysis/           plots, luck-vs-skill decomposition, behavior stats
```

Design choices that matter:

- **Auditable dense reward.** The undiscounted per-step reward sum telescopes to the scaled final score difference (a tested invariant). Training uses `gamma=0.999`, which adds slight temporal weighting and is held constant across study conditions.
- **Honest search baselines.** The expectimax agent samples completions of its *information set* (`PistiGame.determinize`): it preserves cards privately known to the observer and resamples only hidden hands, stock, and opponent-private center cards.
- **Mirrored evaluation everywhere.** Every reported matchup replays each deal with seats swapped. Confirmatory uncertainty is computed over independent training seeds, with a crossed seed/deal bootstrap for the primary estimate.

## Results

The corrected confirmatory study trained ten seed-paired memory-on/off policies for six million steps each. In 500 mirrored deals per pair, explicit played-card memory improved deterministic head-to-head score by **2.656 points per game** (95% seed-level CI 2.413--2.899); all 10 pairs were positive and the exact sign-flip test gave `p=0.001953`. Acute removal produced a similar 2.635-point loss. Fixed-budget attacks did not support the separate hypothesis that league targets were harder to attack than fixed-opponent controls.

- Full manuscript: [`paper/main.pdf`](paper/main.pdf)
- Research report: [`research/final_research_summary.md`](research/final_research_summary.md)
- Protocol and amendments: [`research/experiment_preregistration.md`](research/experiment_preregistration.md)
- Literature evidence matrix: [`research/literature_matrix.md`](research/literature_matrix.md)
- Blog outline: [`BLOG_OUTLINE.md`](BLOG_OUTLINE.md)

Historical exploratory plots remain available below, but are not pooled with the corrected confirmatory study:

- Agent ladder & ratings: `plots/ratings.png`, `plots/tournament_heatmap.png`
- Training curves: `plots/training_curves.png`
- Exploitability over self-play training: `plots/exploitability.png`
- Luck-vs-skill decomposition: `plots/luck_share.png`
- Historical write-up: [REPORT.md](REPORT.md)

## Reproducibility

Every run directory (`runs/<name>/`) contains the resolved config, git hash, package versions, eval history, and checkpoints. Tournament and exploitability JSONs include seeds and protocols. Engine correctness is enforced by `tests/test_game.py` (golden rule scenarios + invariants over thousands of random games).

Run the confirmatory pipeline with `scripts/run_memory_study.sh`, `scripts/run_fixed_controls.sh`, `scripts/run_study_evaluation.sh`, and `scripts/run_robustness_study.sh`. Validate completed targets with `venv/bin/python scripts/validate_study_runs.py`; regenerate manuscript numbers with `venv/bin/python -m analysis.paper_assets`.

## License

MIT; see [LICENSE](LICENSE).
