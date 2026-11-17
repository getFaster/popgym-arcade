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

## Action-conditioned LRU prediction auxiliary

`--variants lru lru_prediction` compares vanilla LRU with linear prediction
heads on `(input_t, final activated LRU output)` of each actor/critic branch.
`ib_reverse_prior20_prediction` adds the same targets to the default IB,
with prediction inputs `(input_t, posterior_mean, posterior_log_variance)`;
its reverse KL, 0.4/1.6 balancing, residuals, one-bit free region and frozen
RMS normalization match `ib_reverse_prior20`. Each head
outputs `[5, 513]`: a 512-dimensional next CNN embedding plus scalar next reward
for each action. Only the executed action is supervised. Embedding MSE averages
over coordinates and nonterminal transitions; reward MSE includes terminal
rewards. The two losses sum across branches with
`--prediction-coefficient 1` by default. Episode-end next images are auto-reset
observations and are excluded. Both current and target CNN paths have stopped
auxiliary gradients; PPO continues to train the CNN normally. No divergence,
free-bit floor, or RMS scaling applies to the prediction MSE itself. Shared parameter initialization and RL
heads match vanilla LRU; auxiliary heads add parameters. Use a fresh output
directory and compare evaluation returns across training seeds before claiming
a reliable benefit. The first pilot is documented under
`results/lru_input_prediction_20261111/EXPERIMENT.md`.

`mlp_prediction` adds the same selected-action next-embedding/reward losses to
plain MLP, with linear prediction heads on each shared 256-dimensional MLP
hidden layer. Auxiliary replay blocks both CNN paths but trains the MLP trunk;
PPO output layers are unchanged. The shared trunk and CNN initialize identically
to plain MLP. Compare both modes with fresh output directories. In addition to
plain-MLP-relative bias, the pilot report uses MLP-plus-prediction as a matched
reference for LRU/IB prediction variants.
