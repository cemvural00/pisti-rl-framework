# Literature Evidence Matrix

The structured search retained 26 primary works after title/abstract screening and snowballing. Searches in English and Turkish found no directly relevant Pişti AI study; this means **none was found under the documented protocol**, not that none exists. Scopone/Scopa is the closest studied capture/fishing family.

| Work | Status/domain | Main contribution and protocol | Relevance or limitation |
|---|---|---|---|
| [Yu & Cowan (1995)](https://doi.org/10.1111/j.1467-842X.1995.tb00868.x) | Journal; duplicate tournaments | Statistical model for replaying random deals across contestants | Direct foundation for deal pairing |
| [Zinkevich et al. (2007)](https://proceedings.neurips.cc/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html) | NeurIPS; extensive-form games | Counterfactual regret minimization and equilibrium convergence | Canonical equilibrium baseline; full traversal is likely infeasible here |
| [Buro et al. (2009)](https://www.ijcai.org/Proceedings/09/Papers/233.pdf) | IJCAI; Skat | Improves inference, sampling, and state evaluation for trick-based search | Shows history-consistent belief sampling matters |
| [Cowling et al. (2012)](https://doi.org/10.1109/TCIAIG.2012.2200894) | IEEE; several hidden-information games | Information Set MCTS avoids independent determinization trees | Stronger search family than the repository's PIMC expectimax |
| [Heinrich & Silver (2016)](https://arxiv.org/abs/1603.01121) | ICML; poker | NFSP combines average-policy learning with approximate best response | Already implemented; one Pişti training seed |
| [Lanctot et al. (2017)](https://arxiv.org/abs/1711.00832) | NeurIPS workshop/arXiv; general games | Policy-Space Response Oracles formalize population/league learning | Frames the snapshot league, but current mixture is heuristic |
| [Henderson et al. (2018)](https://arxiv.org/abs/1709.06560) | AAAI; deep RL | Demonstrates sensitivity to seeds, implementations, and reporting choices | Training-run uncertainty is a current weakness |
| [Burch et al. (2018)](https://doi.org/10.1609/aaai.v32i1.11481) | AAAI; poker | AIVAT gives unbiased low-variance agent evaluation | Confirms chance variance is a first-class design issue |
| [Di Palma & Lanzi (2018)](https://arxiv.org/abs/1807.06813) | Conference/preprint; Scopone | Compares expert rules, MCTS, and ISMCTS in a traditional capture game | Closest mechanics; motivates an ISMCTS baseline, not another weak heuristic |
| [Charlesworth (2018)](https://arxiv.org/abs/1808.10442) | Preprint; Big 2 | PPO self-play agent and environment; evaluation vs random, past selves, humans | Early single-game application with limited robustness evidence |
| [Brown et al. (2019)](https://proceedings.mlr.press/v97/brown19b.html) | ICML; poker | Deep CFR replaces hand-built abstraction with function approximation | Important reference; expensive new implementation for a case study |
| [Lanctot et al. (2019)](https://arxiv.org/abs/1908.09453) | Scientific Reports; multiple games | OpenSpiel standardizes game-theoretic environments and evaluation | Porting would improve interoperability but not directly test the main claim |
| [Zha et al. (2020)](https://arxiv.org/abs/1910.04376) | Workshop/toolkit; card games | RLCard environments and NFSP/DQN/CFR comparisons; three timing seeds | Explicitly warns that beating random is insufficient |
| [Brown et al. (2020)](https://papers.nips.cc/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html) | NeurIPS; poker/Liar's Dice | ReBeL combines belief-state learning and sound search | Establishes why PIMC strength is not an equilibrium guarantee |
| [Zha et al. (2021)](https://proceedings.mlr.press/v139/zha21a.html) | ICML; DouDizhu | Deep Monte Carlo self-play at large card-game scale | Demonstrates a competitive value-learning alternative |
| [Agarwal et al. (2021)](https://papers.nips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html) | NeurIPS; deep RL | Interval estimates and robust aggregate metrics for few-run evaluation | Deal CIs cannot replace uncertainty over training runs |
| [Timbers et al. (2022)](https://www.ijcai.org/proceedings/2022/484) | IJCAI; several games | ISMCTS-BR approximates worst-case performance in large games | Validates approximate BR, while emphasizing it is not exact exploitability |
| [Demirdöver et al. (2022)](https://doi.org/10.55730/1300-0632.3940) | Journal; Hearts | Decomposed learning agent evaluated against rules and humans | Relevant regional application; task decomposition is game-specific |
| [Cutajar & Bajada (2023)](https://www.um.edu.mt/library/oar/handle/123456789/131779) | AIxIA/LNCS; Jaipur | PPO/A2C/DQN/DDQN self-play with action masking | Close two-player score-based case; limited game-theoretic evaluation |
| [Patterson et al. (2024)](https://www.jmlr.org/papers/v25/23-0183.html) | JMLR; empirical RL | Comprehensive guidance on variation, tests, baselines, and experimenter bias | Basis for preregistration and seed-level inference here |
| [Malla (2025)](https://arxiv.org/abs/2510.11736) | Non-peer-reviewed preprint; Dhumbal | Cultural-game comparison of rules, ISMCTS, DQN, and PPO with CIs | Close publication archetype; weak learning performance and limited replication |
| [Rudolph et al. (2026)](https://arxiv.org/abs/2502.08938) | ICLR; five imperfect-information games | 5,600 runs find generic policy gradients competitive using exact exploitability | Makes "PPO can work" non-novel; exact/replicated evaluation is the standard to emulate |
| [Patwa (2026)](https://arxiv.org/abs/2605.28863) | Preprint; Big 2 | Controlled PPO/value-method and self-play-regime comparison | Self-play curriculum result is no longer independently novel |
| [Giacomelli (2026)](https://arxiv.org/abs/2605.17043) | Preprint; Briscola | Preregistered million-game test of folklore, strategy, and deal luck | Strong model for a focused traditional-game empirical paper |
| [Goadrich et al. (2026)](https://arxiv.org/abs/2603.03252) | Accepted CG; 21 card games | Valet/RECYCLE benchmark; includes Scopa and stresses cross-game generality | Raises the bar for claims of general algorithmic relevance |
| [DTCard (2026)](https://doi.org/10.3390/app16073117) | Journal; trick-taking games | Cross-game sequence model; explicitly encodes played cards | Closest support for memory/history representation, but no quantified Pişti-style ablation |

## Saturation and Synthesis

Two final searches on memory representations and Pişti-specific theses added DTCard but no new Pişti work or new methodological category. The review therefore reached the protocol's category-saturation rule.

The literature does **not** support an algorithmic-novelty paper: PPO, DQN, NFSP, self-play leagues, determinizations, and approximate best responses are established. It does support a focused domain/empirical contribution. The strongest underexplored result is the controlled value of explicit played-card memory in a compact traditional capture game, evaluated with duplicate deals and adversarial robustness checks. A secondary contribution is the open, tested Pişti environment and benchmark. General claims about self-play or imperfect-information RL must remain narrow because Valet and recent multi-game work explicitly expose the risk of single-game generalization.
