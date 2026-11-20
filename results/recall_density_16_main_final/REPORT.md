# Recall density: final main-intervention checkpoints

[Combined graph](interventions_by_task.png) · [PDF](interventions_by_task.pdf) · [16-bin averages](averages_16.csv) · [Per-checkpoint samples](manifest.json) · [Checkpoint selection](selection.json)

48 final checkpoints (98,304 training steps), one per task/mode/condition/seed. Plain LRU uses seeds 0–2 for CartPoleHard and seed 0 for the other tasks; interventions use seeds 0–2. Uses the established three-seed intervention run lineages; alternate KL configurations and intermediate snapshots are excluded. Each checkpoint averages its valid trajectory distributions, then each intervention averages checkpoint means with equal weight. Whiskers show training-seed SD, not evaluation SE; they are omitted for single-seed LRU groups.

Paper action-probability gradient definition, 16 bins, eight trajectories/checkpoint, ten observations per window. Shared requested endpoints: [23, 27, 25, 21, 18, 15, 13, 22]. If a trajectory ends early with at least ten observations, use its last pre-terminal observation as the endpoint; actual endpoints are saved separately. **29 samples used this fallback; 64 samples had fewer than ten observations and remain unavailable.** No resampling or extra trajectories. Pre-window recurrent context is preserved and held fixed during differentiation; mass is normalized within the window.

| Task | Mode | Intervention | Checkpoints | Seeds | Valid trajectories per seed |
|---|---|---|---:|---|---|
| CartPoleHard | MDP | Plain LRU | 3 | [0, 1, 2] | [8, 8, 8] |
| CartPoleHard | MDP | KL | 3 | [0, 1, 2] | [8, 8, 8] |
| CartPoleHard | MDP | Prediction | 3 | [0, 1, 2] | [8, 8, 8] |
| CartPoleHard | MDP | FSQ (5 levels) | 3 | [0, 1, 2] | [8, 8, 8] |
| CartPoleHard | POMDP | Plain LRU | 3 | [0, 1, 2] | [8, 8, 8] |
| CartPoleHard | POMDP | KL | 3 | [0, 1, 2] | [8, 8, 8] |
| CartPoleHard | POMDP | Prediction | 3 | [0, 1, 2] | [8, 8, 8] |
| CartPoleHard | POMDP | FSQ (5 levels) | 3 | [0, 1, 2] | [8, 8, 8] |
| NavigatorEasy | MDP | Plain LRU | 1 | [0] | [2] |
| NavigatorEasy | MDP | KL | 3 | [0, 1, 2] | [2, 6, 3] |
| NavigatorEasy | POMDP | Plain LRU | 1 | [0] | [3] |
| NavigatorEasy | POMDP | KL | 3 | [0, 1, 2] | [2, 2, 6] |
| MineSweeperEasy | MDP | Plain LRU | 1 | [0] | [6] |
| MineSweeperEasy | MDP | KL | 3 | [0, 1, 2] | [5, 3, 7] |
| MineSweeperEasy | POMDP | Plain LRU | 1 | [0] | [4] |
| MineSweeperEasy | POMDP | KL | 3 | [0, 1, 2] | [4, 5, 4] |
| BreakoutEasy | MDP | Plain LRU | 1 | [0] | [8] |
| BreakoutEasy | MDP | KL | 3 | [0, 1, 2] | [8, 8, 8] |
| BreakoutEasy | POMDP | Plain LRU | 1 | [0] | [8] |
| BreakoutEasy | POMDP | KL | 3 | [0, 1, 2] | [8, 8, 8] |

Prediction and five-level post-activation FSQ are available only for CartPoleHard; missing interventions are labeled, not plotted as zero.

[CartPoleHard](CartPoleHard.png)

[NavigatorEasy](NavigatorEasy.png)

[MineSweeperEasy](MineSweeperEasy.png)

[BreakoutEasy](BreakoutEasy.png)


Checkpoint storage was pruned to the 36 final policies of the three main interventions, plus their matching resumable states. Plain LRU remains in these saved diagnostics and plots, but its weights have been removed. See [checkpoint inventory](../CHECKPOINTS.json). Older recall runs and superseded job records are preserved in [the diagnostic archive](../_archive/README.md).
