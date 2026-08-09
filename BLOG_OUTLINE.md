# Blog Outline: We Thought Card Counting Was Worth 2.5 Points. Then We Audited the Experiment.

## Audience and promise

- **Audience:** technically curious developers, RL practitioners, card-game players, and hiring managers; no game-theory background assumed.
- **Promise:** show how an interesting portfolio experiment became a defensible research result by testing the claim that explicit played-card memory improves Pişti agents.
- **Core thesis:** Exact played-card memory was worth about 2.7 points per game in this system—but the more transferable result is the audit-and-replication process that made that number defensible.
- **Target length:** 1,800--2,300 words, with the paper carrying methodological detail.

## 1. Cold open: the result that looked too clean

- Open on the original exploratory headline: card counting appeared to be worth about 2.5 points per game.
- Reveal the problem: one no-memory run was being compared with differently seeded memory runs.
- Frame the story as an investigation, not a victory lap: “Was this memory, training luck, or our implementation?”
- Visual: old single-run estimate beside the final ten seed-paired estimate, clearly labeling the old result as exploratory.

## 2. Pişti in sixty seconds

- Explain rank captures, Jacks, the one-card pişti bonus, the three hidden center cards, and why played-card identities matter.
- Use one illustrated three-turn example rather than a complete rules dump.
- Establish the measurement unit: points per game; every deck is replayed with seats swapped.
- Visual: compact tabletop diagram with “known to both,” “known only to capturer,” and “unknown” regions.

## 3. Before adding ideas, build an evidence map

- Summarize the structured review: 26 retained works; no directly relevant Pişti AI paper found under the protocol.
- Explain what adjacent work ruled out as novelty: PPO, self-play leagues, NFSP, determinization, and approximate best responses already exist.
- Show why the focused contribution became “the value of an explicit memory representation,” with an open Pişti benchmark as a secondary contribution.
- Link the review protocol, evidence matrix, and ranked gap analysis.

## 4. The audit changed the code before it changed the prose

- Present the four pre-outcome corrections as a concise forensic sequence:
  1. the final card could incorrectly score pişti;
  2. no-memory self-play opponents would receive full-memory inputs;
  3. the first capturer could not observe its privately revealed center cards, while the opponent leaked their point value;
  4. an all-Jack center did not trigger a redeal.
- Emphasize that incomplete runs were stopped, preserved outside the repository, and never pooled with final results.
- Takeaway: test information ownership, not only card conservation and legal actions.
- Visual: “information flow before/after” diagram.

## 5. The experiment that could answer the question

- Ten paired seeds, memory on versus the same 52-dimensional input zeroed; identical architecture and six-million-step budget.
- Separate two interventions:
  - **Retrained ablation:** can learning adapt when memory is absent?
  - **Acute ablation:** what happens when a memory-trained policy suddenly loses it?
- Explain 500 mirrored deals per matchup and why 10,000 games are still only ten independent training comparisons.
- Explain the seed interval, exact sign-flip test, and crossed seed/deal bootstrap in plain language.
- Visual: experiment flow from 20 training runs to three evaluation lanes.

## 6. The answer

- Lead with the direct paired estimate: memory-on beat memory-off by 2.656 points per game across 10 seed-matched policy pairs.
- State seed consistency and interval before any deal-level numbers: all 10 pairs favored memory; the seed-level 95% CI was 2.413--2.899 and the exact sign-flip `p` was 0.001953.
- Compare acute removal: zeroing memory at inference cost 2.635 points per game, again positive in all 10 seeds.
- Explain adaptation using the acute-minus-retrained contrast: -0.022 points (`p=0.8828`), so retraining without memory showed no detectable compensation relative to sudden removal.
- Confirm the result's scope using stochastic policies (2.381 points), common anchors (2.414 points), and full cross-play: memory-on averaged 42.6 Elo-like points higher, and every memory-on policy ranked above every memory-off policy.
- Visual: `plots/memory_study_effects.png`; annotate the distinction between dots (training seeds) and thousands of games.

## 7. Did the self-play league make policies harder to attack?

- Explain the five fixed-opponent controls and paired one-million-step attackers.
- Report discovered edges as lower bounds, never “true exploitability”: the fixed-minus-league difference averaged -0.188 points per game (95% seed CI -0.973--0.598; 2/5 positive), so league targets were not consistently harder to attack. Three attackers found at most a 0.983-point edge against the principal league target.
- Show the three attacks on the principal target and state exactly what was or was not found.
- Visual: `plots/robustness_study.png`.

## 8. What this result does—and does not—mean

- Defensible claim: under the corrected rules, feed-forward Maskable PPO architecture, curriculum, and six-million-step budget, providing exact observed-card identities caused a large, seed-consistent advantage over otherwise matched policies without that feature.
- Non-claims: human-level play, equilibrium convergence, universal benefit across architectures, or transfer to every regional Pişti rule.
- Explain that no-memory policies still receive the hand, table top, and scalar public state; the study removes explicit identity history, not all state.
- Note that a null or smaller replication is still useful because it corrects the original portfolio narrative.

## 9. The reusable lesson

- End with the discovery loop: audit claims → review literature → rank gaps → freeze discriminating tests → run independent seeds → preserve raw records → write conclusions last.
- Point readers to the complete paper PDF, preregistration/amendments, exact dependencies, checksum manifest, and reproducible scripts.
- Closing line: the most valuable output was not a stronger agent; it was learning which sentence the evidence actually permits.

## Supporting package

- **Hero graphic:** memory-on and memory-off observations diverging from the same hand.
- **Inline figures:** information-ownership diagram, experiment flow, memory effects, robustness pairs, learning curves.
- **Interactive/portfolio idea:** let readers zero the seen-card vector mid-game and compare policy choices.
- **Optional video adaptation:** use sections 1, 2, 4, 6, and 8 as a 7--9 minute arc; animate one mirrored deal and one audit bug rather than showing terminal footage.
- **CTA:** “Read the paper,” “inspect the raw JSON,” and “try the browser game”—not a generic newsletter prompt.
