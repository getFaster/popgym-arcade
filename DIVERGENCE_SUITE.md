## Current default IB preset

The repository's provisional IB default is **`ib_reverse_prior20`**:
reverse KL `KL(posterior || prior)`, posterior memory input `output`, and
prior/posterior auxiliary gradient weights **0.4/1.6** (shares 20%/80%).
Both RL heads receive current input plus posterior mean/log variance, without
sampling. Added inter-layer LRU residuals are enabled for IB; vanilla controls
retain their existing defaults. The auxiliary objective remains coefficient 1,
with a one-bit full-distribution free region and frozen branch-specific RMS
normalization of the excess above that region.

`memory_bias` replaces the historical IB entry in its default five-condition
comparison with this preset. For the preset, it automatically calibrates all
four divergences on a shared 2,048-sample random-policy trajectory at default
rollout sizes, saves the calibration under `<output>/_calibration/`, and freezes
its scales. Calibration is specific to task, observation mode, seed, model
size, posterior input, residual setting, observation size, and rollout shape.
A supplied `--rms-calibration FILE` is validated against those settings.

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python -m popgym_arcade.experiments.memory_bias \
  --variants ib_reverse_prior20 --timesteps 100000 --seeds 0 \
  --output results/default_ib_new_run
```

`--posterior-input state`, `--no-lru-residual`, and
`--ib-normalization none` explicitly override the preset. Other named
divergence/balancing variants remain available; use `--ib-normalization
fixed_rms` to request automatic calibration for them. The explicitly named
`information_bottleneck` variant retains historical forward-KL/unbalanced
semantics; its name is not silently remapped. Fixed-condition suite runners
continue to specify their own architectures. Use fresh output directories
when changing settings. This is a provisional configuration choice from the
single-seed sweep, not a claim of consistent superiority over vanilla LRU.

---

> **Superseded implementation:** historical IB results below used prior statistics at the RL heads and are invalid for the intended posterior-head experiment. Existing artifacts are retained for provenance.

# Multi-task Gaussian divergence pilot

This extends the initial CartPole experiment to four task families: fully observed `CartPoleHard` (control), `NavigatorEasy` (navigation), `MineSweeperEasy` (grid reasoning), and `BreakoutEasy` (arcade control). Difficulty is fixed within each family; the experiment is not a full benchmark sweep. `partial_obs=False` selects the repository's fully observed mode.

There are six conditions per task: an MLP reference needed to compute memory bias, the unregularized LRU baseline, and four otherwise identical Gaussian bottlenecks. The four bottlenecks use the same architecture, parameter initialization, coefficient, and free region; only the divergence changes. Weight decay and recurrent dropout are excluded from this comparison.

Let `P = prior(input_t)` and `Q = posterior(input_t, current LRU state)`. Both are diagonal multivariate Gaussians. The losses are:

| Condition | Divergence before conversion to bits |
|---|---|
| Forward KL | `KL(P ∥ Q)` — original specification |
| Reverse KL | `KL(Q ∥ P)` |
| Symmetric KL | `(KL(P ∥ Q) + KL(Q ∥ P)) / 2` |
| Rényi-½ | `-2 log ∫ sqrt(p(z) q(z)) dz` — twice Bhattacharyya distance |

All expressions are exact and summed across the complete Gaussian event. For Rényi-½, with diagonal variances `vP, vQ`, and `s=(vP+vQ)/2`, the natural-log expression is `sum((muP-muQ)^2/(4s) + log(s) - (log(vP)+log(vQ))/2)`. The implementation computes `log(s)` with `logaddexp` and clips only tiny negative results from floating-point roundoff. Numerical quadrature independently checks this formula.

Divide each divergence by `log(2)` and apply `max(1, divergence_bits)` separately per sample and actor/critic branch. Average across training samples, then add the actor and critic losses with coefficient 1. The floor is per full multivariate distribution, not per coordinate. Both branches reaching the free region gives a constant total auxiliary loss of 2. A common bit unit and coefficient do not equate the divergences' local curvature or regularization strength; this pilot performs no strength calibration or coefficient sweep. Bounded losses such as Jensen–Shannon would require reconsidering the one-bit floor, which would otherwise make their loss constant.

Every agent uses `input_t = (a_{t-1}, r_{t-1}, o_t)` in rollout, replay, bootstrapping, and evaluation. Both RL heads receive `(input_t, mean(P), log_variance(P))` directly. Neither Gaussian is sampled. The LRU has two layers of 512 complex coordinates per branch; flattening real and imaginary parts gives 2,048 real scalar dimensions and `floor(2/3 × 2048) = 1,365` Gaussian coordinates. Log variances are clipped to [-10, 10]. The policy has no recurrent-state path at inference; the LRU influences the IB condition through auxiliary training. The MLP, LRU, and IB architectures differ in parameter count.

The pilot preserves upstream `train.py` PPO hyperparameters and the previously approved budget: 100,000 requested steps, rounded to 48 complete rollouts = 98,304 actual steps per model. It uses seed 0 and 64 held-out stochastic-policy episodes per model. Task-specific episode horizons, plus a terminal-step allowance for tasks that check the old timestep before incrementing, ensure evaluation collects the requested episode count on every task. Evaluation stops as soon as all environments have supplied their requested episodes. A completed training checkpoint can be recovered for evaluation without retraining. The earlier matching CartPole MLP, LRU, and forward-KL runs are reused; copied results retain their original code/lock hashes and explicit provenance. This leaves 21 fresh training runs out of 24 total task/condition pairs. Existing original pilot artifacts remain intact.

Scores and memory biases are reported per task. They are not averaged across tasks with different reward scales. Episode standard errors are conditional on the trained model; one seed cannot establish a reproducible advantage across training seeds. Both the short budget and fixed divergence coefficient limit conclusions.

```bash
uv sync --extra baselines --extra cuda --dev
JAX_PLATFORMS=cpu uv run pytest -q tests/test_memory_bias.py
uv run ruff check popgym_arcade/experiments tests/test_memory_bias.py tests/check_lru_cuda.py
uv run ruff format --check popgym_arcade/experiments tests/test_memory_bias.py tests/check_lru_cuda.py
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python -m popgym_arcade.experiments.divergence_suite \
  --output results/divergence_suite
```

The suite refreshes a cross-task `RESULTS.md` and `summary.csv` after each completed run. A final chart is generated at completion. Weights, per-episode scores, metadata, and training logs remain in task/condition directories. Completed runs are skipped when resuming the exact same suite. Use a fresh output directory for a different task list, budget, seed set, or model configuration. Reports can be regenerated without CUDA:

```bash
JAX_PLATFORMS=cpu uv run python -m popgym_arcade.experiments.divergence_suite \
  --output results/divergence_suite --report-only
```

Sources: [upstream training defaults](https://github.com/bolt-research/popgym-arcade/blob/d061b611718ae55d095791b4ea7046b5266cafd4/popgym_arcade/train.py), [Rényi divergence definitions](https://arxiv.org/abs/1206.2459). See [initial design](MEMORY_BIAS.md) for the controlled LRU implementation and CNN details.

## KL-balancing follow-up

The balancing pilot keeps the architecture, PPO settings, coefficient 1, full-event
one-bit floor, and prior-only RL head inputs above. It tests both `KL(P || Q)` and
`KL(Q || P)` with prior gradient shares `alpha = 0.2, 0.8`. For either direction
`D(P,Q)`, gradients come from
`2 alpha D(P, stop(Q)) + 2 (1-alpha) D(stop(P), Q)`.
The reported value remains `D(P,Q)` and the free-bit gate uses that value.
Thus alpha=0.5 exactly reproduces the original unbalanced gradient; alpha=0.8
weights prior/posterior gradients 1.6/0.4, and alpha=0.2 weights them 0.4/1.6.
This normalization avoids confounding asymmetry with halving both gradients.
Stop gradients include both means and log variances. Shared CNNs receive both
paths; PPO gradients are unchanged. These are auxiliary-gradient weights, not
separate Adam learning rates (adaptive moments and global clipping matter).
The posterior/LRU has no independent prediction objective in this architecture,
so benefits reported for world models need not transfer.

Inspired by [DreamerV2 KL balancing](https://danijar.com/asset/dreamerv2/poster.pdf).
Start with seed 0, 100,000 requested / 98,304 actual steps and 64 evaluation
episodes on CartPoleHard; compare with the existing matching forward/reverse
baselines in `results/divergence_suite`. Keep new artifacts separate:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python -m popgym_arcade.experiments.memory_bias \
  --env CartPoleHard --timesteps 100000 --seeds 0 \
  --variants ib_forward_prior80 ib_forward_prior20 ib_reverse_prior80 ib_reverse_prior20 \
  --output results/kl_balance_pilot/CartPoleHard
```

Repeat with the other task names and distinct task directories to extend the pilot.
A single-seed comparison is exploratory; episode SEs do not establish a benefit
across training seeds. The original divergence-suite condition list is unchanged.

Regenerate the balancing comparison table and chart with:

```bash
uv run python -m popgym_arcade.experiments.report_kl_balance results/kl_balance_pilot
```

## Final-LRU-output posterior comparison

`memory_bias --posterior-input output` replaces the posterior's flattened complex
state input with the real, activated output of the last LRU layer. At the default
size, the posterior is a linear map from 518 + 512 inputs to 2 × 1,365 Gaussian
parameters. The latent dimension, prior-only RL heads, and all PPO settings stay
fixed. The default `--posterior-input state` retains the original architecture.
`--partial-obs` selects POMDP rendering for both training and evaluation.

The CartPoleHard comparison uses `ib_forward_prior20`, the highest-return previous
KL-balancing condition (0.7171): forward KL with prior/posterior gradient weights
0.4/1.6. Seed 0, 100,000 requested (98,304 actual) steps and 64 held-out evaluation
episodes match that pilot. New results live in
`results/lru_output_pilot/CartPoleHard/{mdp,pomdp}`. Episode uncertainty remains
conditional on one training seed; selecting the previous best is exploratory.

`--lru-residual` additionally applies `y_i = activation(C_i h_i + D_i y_{i-1})
+ y_{i-1}` for layers after the first. The first layer remains unchanged because
its input and output dimensions differ. The residual changes the layer output,
not the complex carried state. Both serial rollout and parallel replay use the
same rule. The residual comparison uses the output posterior and the same
forward-KL configuration in `results/lru_output_residual_pilot/CartPoleHard`.

## Corrected posterior-head factorial sweep (2026-11-09)

Both actor and critic receive `(input_t, posterior_mean, posterior_log_variance)` without sampling in rollout, replay, bootstrapping, and evaluation. Residual LRU connections are enabled for both posterior inputs: final layer output or flattened full complex state. Four divergences (forward KL P||Q, reverse KL Q||P, averaged symmetric KL, Renyi half) cross prior gradient shares 0.2/0.8 with normalized branch weights 2 alpha and 2(1-alpha). All four tasks run in MDP and POMDP modes: 128 runs, seed 0, 100000 requested/98304 actual steps, 64 evaluation episodes. PPO settings and latent dimensions remain unchanged.

Each task/mode/posterior-input group calibrates all four divergences on identical Gaussian pairs from an untrained seed-0 model and a 128-step, 16-environment uniform-random-policy trajectory (2048 samples). Calibration uses seed 900000, resets at episode boundaries, and the same previous-action/reward/current-observation inputs. Branch-specific RMS scales in bits are frozen, floored at 1e-8, and shared across balancing choices. The auxiliary objective is the sum of branch means of `max(D_bits - 1, 0)/stop_gradient(scale_bits)`, coefficient 1. The free region remains one bit per full distribution. Calibration values, trajectory hash, raw training divergences, normalized losses, source hashes, and device metadata are saved. Calibration values do not guarantee matched gradient norms.

`posterior_sweep --worker local|colab --output <fresh-directory>` runs a sequential 64-run shard. Local handles MDP groups and Colab handles POMDP groups, each with both posterior inputs. Each configuration uses a separate subprocess; completed results are resumable. Results belong under `results/posterior_rms_suite_20261109`. Episode uncertainty remains conditional on one training seed. Historical prior-head checkpoints are not reused.

The completed corrected sweep is organized in
`results/posterior_rms_suite_20261109/RESULTS.md`, with all 128 cells in
`summary.csv`, per-task matrices and plots in `reports/`, calibration tables,
and an artifact audit. Regenerate and validate without CUDA:

```bash
JAX_PLATFORMS=cpu uv run python -m popgym_arcade.experiments.report_posterior_sweep \
  results/posterior_rms_suite_20261109
```

The report preserves raw artifacts and excludes superseded prior-head scores.
Selected maxima and factorial contrasts are exploratory, conditional on seed 0.

The organized report also links `reports/vs_vanilla_lru.md`, comparing the 64
corrected MDP cells against the previous `results/divergence_suite/*/lru_seed0`
references. It validates matching PPO settings, budget, seed, episode counts,
and raw return statistics, retaining baseline provenance. Historical vanilla
LRU lacks the added inter-layer residual and has a smaller non-Gaussian head;
the comparison therefore does not isolate regularization. No previous POMDP
vanilla baseline exists, so those 64 comparison rows have missing-reference
labels and blank deltas. The posterior-head correction does not affect the
historical vanilla LRU branch.

## Vanilla LRU POMDP follow-up (2026-11-09)

Four new vanilla LRU references run on all four tasks with `partial_obs=True`,
seed 0, 100000 requested/98304 actual steps, and 64 held-out evaluation episodes.
These use the historical vanilla architecture: no added inter-layer residual,
current-input skip plus 256-dimensional projected LRU output at each RL head,
no Gaussian bottleneck, RMS scaling, weight decay, or state dropout. PPO settings
match the corrected posterior sweep. Fresh outputs are in
`results/vanilla_lru_baselines_20261109`; local RTX 2060 runs Navigator and CartPole,
while Colab T4 runs MineSweeper and Breakout. Each device executes sequentially.
The training runner now labels non-IB head inputs accurately in metadata; model
behavior is unchanged. Source snapshots and per-run hashes retain provenance.

Regenerate the vanilla POMDP score summary, artifact audit, and all 64 IB-versus-LRU
comparisons without CUDA:

```bash
JAX_PLATFORMS=cpu uv run python -m popgym_arcade.experiments.report_vanilla_lru \
  results/vanilla_lru_baselines_20261109
```

This comparison still changes residual connections, head architecture, and
parameter counts between vanilla LRU and the IB models. Device metadata records
any hardware differences. A single training seed and selected best IB scores
remain exploratory.


## Consolidated vanilla LRU archive

`results/vanilla_lru_baselines_20261109` now contains all eight vanilla LRU
references in `Task/{mdp,pomdp}/lru_seed0/`. Historical MDP artifacts were copied
byte-for-byte with SHA-256 verification; POMDP artifacts were moved from the old
POMDP-only directory. Checkpoints, episode returns, full logs, configs, and
provenance are self-contained. Summary tables and the cached comparison CSV
cover both modes. Archived execution scripts retain original paths as historical
records. Older result directories are not required to preserve these baselines.

## Default-IB residual ablation

The eight-run local-GPU comparison is organized in
`results/default_ib_no_residual_20261109/RESULTS.md`: the default reverse-KL,
output, prior20 IB with residuals disabled, compared with the saved
residual-enabled configuration and vanilla LRU within each task/mode. New runs
retain seed 0, 98304 actual steps, and 64 evaluation episodes; each architecture
uses freshly calibrated frozen RMS scales. Score tables, episode SEs, the
`LRU mean - LRU episode SE` criterion, training diagnostics, calibration tables,
and an artifact audit are included. Reference metadata, per-episode returns,
and training logs are bundled for regeneration after older suites are removed.
The provisional residual-enabled repository default is unchanged.

```bash
JAX_PLATFORMS=cpu uv run python -m popgym_arcade.experiments.report_residual_ablation \
  results/default_ib_no_residual_20261109
```
