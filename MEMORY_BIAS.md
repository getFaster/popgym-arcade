# Memory-bias regularization experiment

The approved pilot compares one MLP reference with an LRU baseline and three separate variants on fully observed `CartPoleHard`. All agents use `input_t = (a_{t-1}, r_{t-1}, o_t)`. The action is one-hot encoded, reward is scalar, and the RGB observation passes through the upstream CNN. Episode starts use zero action/reward sentinels and reset both recurrent states.

The runner reads PPO_RNN defaults directly from `popgym_arcade/train.py`: learning rate 1e-4, 16 environments, 128 steps per rollout, four epochs, 16 recurrent minibatches, gamma .99, GAE lambda .95, clipping .2, entropy coefficient .01, value coefficient .5, gradient norm .5, Adam epsilon 1e-5, learning-rate annealing, and 128×128 images. The user-approved pilot changes only the training budget to 100,000 requested steps (48 complete rollouts = 98,304 actual steps). The learning-rate schedule spans this pilot budget.

The model uses the upstream separate actor/critic CNNs, two complex diagonal LRU layers with 512 states each per branch, and a 256-unit normalized MLP head. The LRU is implemented explicitly to control state dropout and expose the full state. Training evaluates the LRU with an associative affine scan; rollouts evaluate it step by step. Tests compare the states and RL outputs from both paths, including episode resets and replayed dropout. This is a controlled LRU experiment, not a bit-for-bit reproduction of the upstream memax residual wrapper. Both heads have a current-input skip connection. The MLP reference uses the same CNN and current-input head without recurrent features. Model parameter counts differ; there is no parameter-matched MLP control.

Conditions:

- `lru`: unregularized LRU + MLP.
- `weight_decay`: the same model with AdamW weight decay 1e-4, applied to all trainable parameters.
- `state_dropout`: the same model with 10% dropout on the carried recurrent state at every step and layer, before the recurrence. Real and imaginary parts share each mask. No inverted scaling is applied, preserving contractive dynamics. Masks are sampled during rollout, stored as PRNG keys, and replayed during PPO optimization. Evaluation disables dropout.
- `information_bottleneck`: diagonal multivariate Gaussian prior `Z' = prior(input)` and posterior `Z = posterior(input, current LRU state)`. Both RL heads receive `(input, mean(Z'), log_variance(Z'))`, with no sampling. Each branch's flattened LRU state has 2 layers × 512 complex entries × 2 real coordinates = 2,048 real scalars. Its Gaussian dimension is `floor(2/3 × 2048) = 1365`. Log variances are clipped to [-10, 10] for numerical stability. KL is `KL(prior || posterior)`, summed over latent coordinates and converted to bits. The auxiliary loss is `max(1, KL_bits)` per state, averaged over training samples, separately for actor and critic, then summed with coefficient 1. The one-bit floor is per full distribution, not per coordinate. Gradients flow through both Gaussian networks and the LRU.

The literal IB specification gives the policy no recurrent-state path at inference. The LRU affects learning through the auxiliary loss. Thus this condition measures auxiliary-memory regularization rather than preserving a memory-dependent policy.

Evaluation uses 64 held-out episodes per condition, the same evaluation seed, and stochastic categorical actions. It collects exactly four episodes from each of 16 environments. Scores are raw episode returns. Bias is `MLP score − condition score`; reduction relative to the LRU baseline equals `condition score − LRU score`. Evaluation standard errors describe episode variability, not variability across training seeds. A single-seed 100K pilot is insufficient to establish a regularization benefit.

```bash
uv sync --extra baselines --extra cuda --dev
uv run ruff format --check popgym_arcade/experiments tests/test_memory_bias.py
uv run ruff check popgym_arcade/experiments tests/test_memory_bias.py
JAX_PLATFORMS=cpu uv run pytest -q tests/test_memory_bias.py
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python -m popgym_arcade.experiments.memory_bias \
  --timesteps 100000 --output results/memory_bias_pilot
```

The runner requires CUDA and refuses CPU fallback. It writes per-run configuration, training logs, model weights, individual evaluation returns, and a summary table. Existing completed results are skipped; use a fresh output directory when changing any settings. For a follow-up, use `--timesteps 10000000 --seeds 0 1 2 3 4` and a new output directory.

Source: https://github.com/bolt-research/popgym-arcade/tree/d061b611718ae55d095791b4ea7046b5266cafd4

Render the saved comparison chart with:

```bash
JAX_PLATFORMS=cpu uv run python -m popgym_arcade.experiments.plot_memory_bias results/memory_bias_pilot
JAX_PLATFORMS=cpu uv run python -m popgym_arcade.experiments.report_memory_bias results/memory_bias_pilot
```

The full-size CUDA replay check (512 states × 2 layers × 128 timesteps) can be reproduced with:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python tests/check_lru_cuda.py
```
