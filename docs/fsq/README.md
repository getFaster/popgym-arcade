# FSQ LRU experiment

Use `memory_bias --variants lru` with `--recurrent-state-quantization none|fsq|output`,
`--fsq-levels 3|5|7` (any odd integer >=3 supported), and `--fsq-scale 1.0`.
Defaults preserve continuous LRU behavior, dimensions, initialization, and PPO.
The initial integration restricts quantization to the vanilla `lru` variant.
`run_comparison.sh` provides matched baseline, three state FSQ and three output
FSQ configurations in both observation modes; it is never invoked automatically.
Use a fresh results directory. The runner records configuration and elapsed time.

State FSQ quantizes each complex component after every layer update, before C
readout and before carrying the state. Replay switches to `jax.lax.scan` over the
same step function because the nonlinear recurrence is not associative. Baseline
keeps its associative scan. Output FSQ means `C Q(h) + D x`, before activation,
in every layer, with continuous carried h; its affine recurrence stays parallel.
This is not quantization of the final activated policy feature.
Reset clears old state before processing the first new input. No burn-in exists
in this runner; PPO starts from the rollout's saved initial memory. Termination,
truncation handling, previous action/reward inputs and evaluation use existing paths.

Training logs contain `fsq_{actor|critic}_{real|imag}_*` statistics from the complete
pre-optimization rollout, computed on device and transferred once per PPO update:
error is mean squared L2 norm summed over all layer/state coordinates; saturation
is endpoint frequency; utilization is the mean fraction of scalar levels used per
feature; entropy is empirical per-feature entropy averaged over features, in nats.
Change is the fraction of codes changing from the previous step, excluding reset
steps. Output mode compares Q(h_t) with Q(h_{t-1}). Histograms pool time and
environments within the rollout. Diagnostics retain temporary raw/previous states,
so enabled runs have some memory and computation overhead. No auxiliary loss is
added. Existing cumulative `seconds` includes compilation and evaluation in final
results; difference successive training records for steady update timing and
report compilation separately when comparing runtime.

Validation:
```
JAX_PLATFORMS=cpu uv run pytest -q tests/test_fsq.py tests/test_memory_bias.py
JAX_PLATFORMS=cpu uv run python tests/smoke_fsq.py
```
The smoke performs two real CartPole POMDP environment steps and one PPO update
per mode with tiny memory dimensions; it does not establish learning performance.
Compare MDP/POMDP returns and their difference within each seed, then summarize
across training seeds. Episode standard errors are not training-seed uncertainty.
Quantization can erase weak signals, accumulate error, or saturate. Hundreds of
scalar codes still permit enormous state spaces. This tests an inductive bias,
not guaranteed memory improvement; no performance claim follows from unit tests.

## Post-activation ablations

`activation` applies `Q(leaky_relu(Re(C h) + D x))` in each LRU layer,
with continuous state. `state_activation` additionally quantizes the complex
state after every update. Both use the same levels and scale at both locations.
The former keeps layerwise associative replay; the latter uses sequential replay.
FSQ is before any optional residual addition (these experiments disable residuals).
This does not change the historical `output` mode, which quantizes before C.
Activation diagnostics use `fsq_{actor|critic}_activation_real_*`; code-change
statistics exclude resets and the first step of each rollout, since no previous
activation is stored across rollouts. State diagnostics are also logged for
`state_activation`. The memory-gap definition is **MLP minus LRU-variant return
on fully observed MDP**; a shared MLP reference cancels in relative gap changes.
