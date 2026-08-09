# Experiment Preregistration

**Frozen before new scientific runs:** 2026-08-08

**Code baseline:** `465fe24` plus packaging/test-only changes, if required

**Primary question:** How much does explicit memory of previously observed cards improve learned Pişti play?

## Pre-outcome Protocol Amendment

During the timing/training phase, an independent rules-source audit found that the engine allowed a pişti bonus on the final played card. Standard descriptions explicitly disallow that bonus. All incomplete confirmatory runs were stopped before any terminal study outcome was inspected, the generated directories were moved out of the repository, and `test_final_card_capture_is_not_pisti` was added. The study restarts from scratch on the corrected rule. Historical repository results remain valid only for the previously documented variant and are not pooled with the confirmatory results.

A subsequent pre-league code audit found that the no-memory learner received `seen=0`, but frozen self-play policies acting as opponents would have received the environment's default full-memory observation. The no-memory runs were stopped during the scripted-opponent phase, before this path became active and before their outcome evaluations were inspected. `make_vec_env` now gives both seats the condition's observer, a regression test enforces this, and all no-memory runs restart from scratch. Memory-on and fixed-control runs are unaffected and continue.

Before that restart completed, an information-set audit found that the first capturer's three center cards remained hidden from the capturer in `seen_cards` and were resampled by determinization. Standard rules allow the capturer to inspect those cards. All active study runs were stopped without inspecting trained outcomes. The capturer now privately observes and remembers these cards, its determinizations preserve them, the opponent's determinizations resample them, and an asymmetric-knowledge regression test covers the behavior. Because this change affects the treatment itself, every confirmatory and control run restarts from scratch at the unified corrected source state.

The same audit found that scalar point features exposed the private center-card point total to the non-capturing player and that rare high-scoring states could exceed the declared observation bounds. Point features now show only publicly inferable points plus privately known points for the capturer, and normalized statistics are clipped to their declared range. Tests cover both privacy and observation-space validity. No confirmatory run was active when this final pre-outcome correction was made.

A final rare-branch audit found that an all-Jack four-card center selected a Jack instead of requiring a redeal. Although the event has probability below four per million deals, the total training budget makes it reachable. Generated decks now reshuffle until the opening is valid, explicit all-Jack decks fail loudly, and a regression test covers the rule. The incomplete matrix was again discarded without outcome inspection and restarted. This is the last rule/observation amendment; the definitive run set uses the frozen implementation after all amendments in this section.

## Hypotheses

- **H1 (training-time memory):** seed-matched PPO agents trained with the `seen` vector outperform agents trained with it zeroed.
- **H2 (policy reliance):** replacing the `seen` vector with zeros at inference reduces the performance of memory-trained policies.
- **H3 (adaptation):** the acute inference loss is at least as large as the difference between separately trained memory/no-memory policies, because no-memory training can adapt to missing information.
- **H4 (secondary, self-play):** league-trained policies are harder for a fixed-budget attacker to exploit than fixed-opponent controls trained for the same steps.

H1 is primary. H2--H4 are secondary. The prior exploratory report motivated these hypotheses, so the study is a confirmatory replication/extension, not a claim of pristine preregistration.

## Training Design

Use `configs/default.yaml`, 6,000,000 environment steps, and identical settings across memory conditions except `observer.memory`. Train paired seeds from scratch at one code revision and dependency environment. The seed count is selected solely from a timing pilot performed before looking at any new outcome: target 10 pairs when the projected wall/compute budget is practical, otherwise a minimum of 5 pairs. Runs use unique `runs/study_mem_{on|off}_s<seed>/` directories.

The fixed-opponent control uses the same model, reward, initial curriculum, and seeds 0--4. At each later phase, `pool` and `latest` are removed and the remaining scripted weights are renormalized: `{greedy: .5, hunter: .5}` at 1M and equal greedy/hunter/light-expectimax weights at 2.5M. This tests the league claim but is not required for H1. Each fixed target and its seed-matched league target receive a 1M-step PPO attack, initialized from the same historical `ppo_main` model and using attacker seed `100 + target_seed`; evaluation uses 500 mirrored deals. The principal league target (training seed 0) receives two additional attacks with seeds 201 and 202, making three attackers including seed 100.

## Evaluation Design

All main evaluations use deterministic policies, 500 independently generated deals per matchup, and both seat assignments per deal. The same evaluation deal seeds are reused across conditions.

1. Seed-matched direct matches: memory-on seed `s` vs memory-off seed `s`.
2. Common anchor panel: greedy, hunter, and honest determinization-based expectimax.
3. Cross-seed tournament among all newly trained policies.
4. Acute ablation: each memory-on model plays with its ordinary observer and with the same weights receiving `seen=0`.
5. Sensitivity: stochastic policy sampling for the seed-matched direct matches.

The independent unit for H1 is the training seed. Deal pairs are repeated evaluation observations, not additional training replications. Because the same evaluation deals are reused across seeds as common random numbers, training seed and deal are crossed factors.

## Metrics and Inference

Primary outcome: seed-paired mean score differential per game in direct memory-on vs memory-off matches. Report all seed effects, their mean/median, a seed-level t interval, a sign-flip permutation p-value, and a crossed bootstrap interval that resamples training seeds and shared deal indices as two factors. Secondary outcomes are win rate, Bradley--Terry rating, anchor-panel aggregate, pişti count, and captured-card count.

No minimum practically important effect was established externally. For interpretation, effects below 0.5 points/game are called small, 0.5--1.5 moderate, and above 1.5 large; these labels are declared conventions, not validated human thresholds.

For attacker replication, report each attacker separately and the maximum discovered edge. Approximate best-response returns are lower bounds on exploitability. Failure to reject a zero edge is not evidence that true exploitability is zero.

## Exclusions and Failure Handling

No completed seed is removed for poor performance. A run is excluded only for a logged crash, non-finite parameters, configuration mismatch, or failed post-training load/smoke test, and is rerun with the same seed after documenting the cause. Evaluation deals are never filtered. Any deviation from this file is recorded before examining the affected results where possible.
