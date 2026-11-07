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
