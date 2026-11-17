> **Default IB residual ablation:** [Eight-run comparison](../default_ib_no_residual_20261109/RESULTS.md).

# Vanilla LRU baselines: MDP and POMDP

**8/8 runs complete and audited.** Four tasks × two observation modes. All runs use seed 0, 100,000 requested / 98,304 actual training steps, and 64 held-out stochastic-policy evaluation episodes.

[Summary CSV](summary.csv) · [All 128 IB comparisons](vs_corrected_ib.csv) · [Audit](audit.json) · [Artifact checksums](provenance/artifact_manifest.json) · [MDP import provenance](provenance/mdp_imports.json)

## Scores and corrected IB comparisons

± denotes episode SE conditional on one trained seed. Best IB scores select among 16 configurations and are exploratory.

| Task | Mode | Vanilla LRU ± SE | Best corrected IB ± SE | Best IB − LRU | IB configs above LRU |
|---|---|---:|---:|---:|---:|
| CartPoleHard | MDP | [0.6007 ± 0.0210](CartPoleHard/mdp/lru_seed0/result.json) | 0.5324 ± 0.0192 | -0.0683 | 0/16 |
| CartPoleHard | POMDP | [0.2624 ± 0.0106](CartPoleHard/pomdp/lru_seed0/result.json) | 0.3460 ± 0.0164 | +0.0835 | 15/16 |
| NavigatorEasy | MDP | [0.4049 ± 0.0879](NavigatorEasy/mdp/lru_seed0/result.json) | 0.4965 ± 0.0839 | +0.0916 | 3/16 |
| NavigatorEasy | POMDP | [0.4050 ± 0.0879](NavigatorEasy/pomdp/lru_seed0/result.json) | 0.4045 ± 0.0890 | -0.0005 | 0/16 |
| MineSweeperEasy | MDP | [0.5187 ± 0.0371](MineSweeperEasy/mdp/lru_seed0/result.json) | 0.5059 ± 0.0348 | -0.0128 | 0/16 |
| MineSweeperEasy | POMDP | [0.3530 ± 0.0311](MineSweeperEasy/pomdp/lru_seed0/result.json) | 0.3742 ± 0.0299 | +0.0212 | 1/16 |
| BreakoutEasy | MDP | [-0.7034 ± 0.0167](BreakoutEasy/mdp/lru_seed0/result.json) | -0.6086 ± 0.0241 | +0.0948 | 6/16 |
| BreakoutEasy | POMDP | [-0.7541 ± 0.0090](BreakoutEasy/pomdp/lru_seed0/result.json) | -0.6607 ± 0.0151 | +0.0934 | 4/16 |

## Model and comparison details

Vanilla LRU uses two 512-coordinate complex LRU layers and current input plus a 256-dimensional projected LRU output at both RL heads. There is no added inter-layer residual connection, Gaussian bottleneck, RMS normalization, weight decay, or state dropout. PPO settings match the corrected IB sweep. The prior/posterior head bug did not affect the vanilla branch.

IB models enable added residual connections and larger posterior-statistics heads, so these comparisons do not isolate regularization. Device provenance appears in the CSV. Episode uncertainty does not measure training-seed variability.

## Self-contained artifact layout

```text
Task/
  mdp/lru_seed0/    # historical MDP reference, copied byte-for-byte
  pomdp/lru_seed0/  # new POMDP reference
provenance/        # source snapshot, original execution logs/manifests, import records
```

Each run contains its checkpoint, evaluation returns, config, result metadata, and full training log. All eight baseline runs are physically stored here; older result directories are not needed to retain or audit them. Original paths inside archived logs/scripts or `reused_from` metadata are historical provenance. The copied CartPole MDP result is one training run, not an additional seed; MineSweeper MDP evaluation was recovered from its checkpoint.

The comparison CSV stores all IB scores, SEs, devices, and deltas, so it remains readable after the source IB results are removed. Its `ib_result` column records historical source locations, not bundled IB checkpoints. Report regeneration uses those IB artifacts when present and falls back to the cached comparison CSV when they are unavailable.

Regenerate:

```bash
JAX_PLATFORMS=cpu uv run python -m popgym_arcade.experiments.report_vanilla_lru results/vanilla_lru_baselines_20261109
```

Training and sync scripts under `provenance/` are archived execution records and retain their original paths; they are not active launchers.
