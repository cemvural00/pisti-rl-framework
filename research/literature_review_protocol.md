# Literature Review Protocol

**Protocol frozen:** 2026-08-08

**Review type:** structured scoping review with backward/forward snowballing

**Scope:** learning, search, evaluation, and reproducibility for traditional imperfect-information card-game AI, with special attention to Pişti and related capture/fishing games.

## Review Questions

1. Has Pişti already been used as an AI or reinforcement-learning research domain?
2. What constitutes a publishable contribution in a single traditional card-game study?
3. Which learning and search baselines are expected for two-player zero-sum imperfect-information games?
4. How are strength, robustness/exploitability, stochastic deal effects, and training uncertainty evaluated?
5. Which gaps in the current repository threaten validity, novelty, or reproducibility?

## Sources and Search Families

Searches cover Google-indexed scholarly pages, arXiv, ACM, IEEE, Springer, IJCAI, PMLR, JMLR, OpenReview, and Semantic Scholar/OpenAlex metadata. English and Turkish terms are used. Pilot searches performed before freezing this protocol refined terminology but do not determine inclusion.

Formal search families:

```text
("Pişti" OR "Pisti") AND (AI OR "artificial intelligence" OR
  "reinforcement learning" OR search OR agent)
(Scopa OR Cassino OR Basra OR "fishing card game") AND
  (AI OR learning OR MCTS OR agent)
"traditional card game" AND (AI OR "reinforcement learning")
"imperfect-information card game" AND (self-play OR PPO OR DQN OR NFSP)
(exploitability OR "best response") AND (card game OR
  "imperfect-information game")
(determinization OR PIMC OR ISMCTS) AND "card game"
("deep reinforcement learning" AND evaluation AND seeds) OR
  "empirical design in reinforcement learning"
```

## Inclusion and Exclusion

Include peer-reviewed papers, credible preprints, and theses that contribute at least one of: a relevant game environment, learning/search method, robustness metric, statistical protocol, or reproducibility standard. Include foundational method papers even when evaluated on poker or toy games. Prefer primary sources.

Exclude tutorials, news, unsourced blog posts, unrelated collectible-card deck construction, perfect-information-only work without methodological relevance, and papers that mention cards or self-play only incidentally. Non-peer-reviewed application papers remain in the matrix but are labelled accordingly.

## Screening and Extraction

Screen title/abstract first, then full text or extended metadata. Record: publication status, game and information structure, contribution type, algorithms, baselines, training runs/seeds, evaluation unit, uncertainty treatment, exploitability method, artifacts, closest relevance, and limitations. A paper can support more than one category.

Backward references and forward citations are followed for the closest application and method papers. Searching stops after two consecutive query/snowball rounds add no new contribution, method, or evaluation category. Absence of a discoverable Pişti paper is reported as "none found under this protocol," never proof that none exists.

## Synthesis Rules

Novelty is assessed at three levels: new algorithm, new empirical finding, and new domain/benchmark. Claims are separated from evidence. Deal-level replication does not substitute for independent training runs. Approximate best-response results are described as lower bounds on exploitability (or failed attacks), not exact Nash-distance estimates. Candidate improvements must trace to a literature gap, validity threat, or reproducibility defect before entering the experiment backlog.
