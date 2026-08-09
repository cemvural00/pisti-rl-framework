# Final Research Summary

## Executive verdict

This repository now supports a defensible empirical research paper, not only a blog post. Its strongest scientific contribution is narrow: under a corrected two-player Pişti implementation and a fixed feed-forward Maskable PPO protocol, explicit memory of observed card identities causes a large, seed-consistent performance advantage. It is a strong portfolio project and a plausible workshop, game-AI, or empirical case-study submission. It is not an algorithmic contribution, an equilibrium result, evidence of human-level play, or yet a broad claim about memory architectures.

The confirmatory estimate is **2.656 points per game** in favor of memory (95% seed-level CI 2.413--2.899; crossed seed/deal bootstrap CI 2.165--3.149). Every one of ten independently trained seed pairs was positive, and the exact paired sign-flip test gave `p=0.001953`. This closely reproduces the original exploratory magnitude while replacing its single-run comparison with independent training replication.

## How the question was selected

The work followed a discovery sequence rather than implementing the first appealing extension:

1. Audit existing claims, code, reports, result files, and model provenance.
2. Run a structured scoping review across Pişti, related fishing games, imperfect-information self-play, information-set search, exploitability, and empirical RL methodology.
3. Map 26 retained works to the repository's possible contributions.
4. Rank gaps by novelty, scientific value, feasibility, and ability to falsify the existing narrative.
5. Freeze hypotheses, interventions, independent units, exclusions, and interpretation rules before completed confirmatory outcomes.
6. Correct rule and information-set defects, restart all definitive training, and retain raw deal-level records.

The review found established prior work for PPO/self-play, NFSP, CFR and Deep CFR, PSRO, ReBeL, information-set MCTS, duplicate tournaments, RL evaluation, and approximate exploitability. It found no directly relevant Pişti AI study under the documented protocol. That is a bounded search result, not proof of universal absence. Consequently, the study does not claim algorithmic novelty. The review protocol and evidence are in [literature_review_protocol.md](literature_review_protocol.md) and [literature_matrix.md](literature_matrix.md); the decision is recorded in [gap_analysis_and_priorities.md](gap_analysis_and_priorities.md).

## Pre-outcome audit and corrections

Four implementation issues were found before definitive outcomes were inspected:

- the last played card could incorrectly receive a pişti bonus;
- the frozen self-play opponent in a no-memory run would receive a full-memory observation;
- the first capturer neither retained the three privately revealed center cards nor protected their point value from the opponent's scalar observation;
- an all-Jack opening center failed to trigger a redeal.

The engine, observation layer, determinization, and training observer assignment were corrected and regression-tested. Observation statistics were also clipped to the declared space. Incomplete affected runs were stopped, never pooled, and moved recoverably to four timestamped directories under `/tmp/pisti-invalid-*-runs-20260808/` (about 1.2 GB total). The complete chronology is in [experiment_preregistration.md](experiment_preregistration.md).

## Confirmatory design

The primary treatment used ten paired training seeds. For each seed, memory-on and memory-off agents shared the model architecture, seed, curriculum, opponent league, reward, and six-million-step budget; only the 52-dimensional `seen` vector differed. Twenty main targets and five fixed-opponent controls completed training. All 25 passed resolved-configuration, completion, final-evaluation, finite-parameter, and model-checksum validation. Target training consumed about 23.5 CPU-hours; the robustness attackers added about 2.6 CPU-hours.

Each matchup used 500 generated deals and replayed every deal with seats swapped, giving 1,000 games per comparison. The training seed—not an individual game—is the independent unit. The same deals were reused across seeds as common random numbers, so uncertainty includes a seed-level Student-t interval, an exact paired sign-flip test, and a crossed bootstrap that resamples both seed and shared-deal factors.

Four hypotheses were frozen:

- **H1, primary:** retrained memory-on policies outperform paired memory-off policies.
- **H2, secondary:** acute removal of memory harms a memory-trained policy.
- **H3, secondary:** acute loss is at least as large as the retrained difference because no-memory training may adapt.
- **H4, secondary:** league-trained targets are harder for a fixed-budget attacker to exploit than fixed-opponent controls.

## Results

### H1: explicit memory improves trained play — supported

The ten seed effects ranged from 1.958 to 3.048 points per game and averaged 2.656 (median 2.761, SD 0.340). All ten were positive. The effect is “large” under the frozen descriptive threshold of more than 1.5 points per game. Across the 10,000 underlying direct games, memory-on had a tie-adjusted win rate of 0.564, produced 0.180 additional pişti events, and captured 3.355 additional cards per game. These secondary decompositions are descriptive.

### H2: trained policies rely on the feature — supported

Zeroing `seen` only at inference cost 2.635 points per game (95% seed CI 2.289--2.980; crossed bootstrap CI 2.113--3.145). All ten seed effects were positive and the exact sign-flip `p` was 0.001953. This is an intervention on the trained mapping, not a claim that the network internally performs human-like counting.

### H3: no-memory training adapts — not supported

The acute-minus-retrained effect was -0.022 points per game, with four of ten seed contrasts positive and exact `p=0.8828`. Acute and retrained losses were effectively the same. Within this model and budget, there is no detectable evidence that retraining compensated for missing identity history.

### Sensitivity analyses

Stochastic action sampling retained a 2.381-point effect (95% seed CI 2.192--2.571; all ten positive; `p=0.001953`). Averaging performance against greedy, hunter, and honest-determinization search anchors yielded a 2.414-point memory advantage, again positive for all ten seeds (`p=0.001953`). In the complete 20-policy cross-play tournament (190 pairings and 190,000 games), memory-on ratings averaged 42.6 Elo-like points above their paired memory-off ratings. Every memory-on policy ranked above every memory-off policy. Ratings are descriptive because policies and evaluation deals are reused across pairings.

### H4: the league is harder to attack — not supported

Five paired one-million-step attacker comparisons produced a fixed-minus-league difference of -0.188 points per game (95% seed CI -0.973--0.598; two of five positive; exact `p=0.625`). The paired effects changed sign and did not show that fixed-opponent targets were easier. Three attacks against principal league seed 0 discovered edges of 0.983, 0.075, and -0.076 points per game; the maximum was 0.983. These are optimizer- and budget-dependent lower bounds, never exact exploitability. Neither a positive attack nor failure to find one establishes equilibrium distance.

## Interpretation and limits

The causal claim is specific and well supported: giving exact observed-card identities to this feed-forward PPO system materially improves its play relative to paired agents that lack the vector. Agreement across retrained, acute, stochastic, anchor, and cross-play views makes a seed accident or deterministic-action artifact implausible under the tested protocol.

Important boundaries remain. The study uses one algorithm, network family, budget, curriculum, and rule variant. Exact engineered recall is not equivalent to recurrent memory, a learned belief state, or human cognition. The undiscounted dense-reward sum equals scaled terminal score, but `gamma=0.999` adds a small temporal weighting during training. The search anchor is independent-determinization lookahead, not ISMCTS or an equilibrium solver. Approximate attacks are lower bounds. Ten seeds precisely identify the observed large effect but would be inadequate for subtle interactions. No human participants or expert validation were used.

## Recommended next research

The systematic next step is a representation ladder, not another grab bag of agents: compare exact card identities, compressed rank counts, a recurrent policy receiving events, and a learned belief representation under the same paired-seed protocol. This separates information content from input dimension and inductive bias. A second factorial study should cross representation with opponent population to test whether memory value depends on the league. Stronger game-theoretic calibration would require an OpenSpiel port, exact reduced subgames, or ISMCTS, followed by larger and initialization-diverse response-oracle budgets. Human comparison should wait until rule variants and computational hypotheses are fixed.

## Deliverables and reproducibility

- Full paper source and PDF: [`paper/main.tex`](../paper/main.tex) and [`paper/main.pdf`](../paper/main.pdf)
- Blog-only outline: [`BLOG_OUTLINE.md`](../BLOG_OUTLINE.md)
- Preregistered protocol and amendments: [experiment_preregistration.md](experiment_preregistration.md)
- Raw and summarized study records: [`results/`](../results/)
- Generated figures: [`plots/`](../plots/)
- Exact direct dependencies: [`requirements-research.txt`](../requirements-research.txt)
- Study validation: [`results/study_validation.json`](../results/study_validation.json)
- Final checksums and byte sizes: [artifact_manifest.json](artifact_manifest.json)

The paper should lead any scientific submission; the blog should tell the audit-and-replication story; the repository and optional video can demonstrate the engineering and visual intuition. The defensible headline is not “we solved Pişti.” It is: **in a corrected, replicated Pişti benchmark, exact played-card memory was worth about 2.7 points per game, and the result survived independent seeds and multiple evaluation views.**
