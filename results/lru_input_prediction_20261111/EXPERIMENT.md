# Input-conditioned next-embedding/reward prediction pilot

CartPoleHard MDP, seed 0, 100000 requested / 98304 actual steps per condition,
64 held-out evaluation episodes. Upstream PPO settings are unchanged. Inputs
remain (previous action, previous reward, current observation), with the usual
reset sentinels. Two 512-state LRU layers per actor/critic branch.

Four conditions:

- `lru`: vanilla LRU, no added inter-layer residuals.
- `lru_prediction`: vanilla LRU plus linear heads on `(input_t, final_LRU_output)`.
- `ib_reverse_prior20`: repository default IB, inter-layer residuals enabled.
- `ib_reverse_prior20_prediction`: default IB plus linear heads on
  `(input_t, posterior_mean, posterior_log_variance)`.

Each predictive branch outputs [5, 513]: a 512-dimensional next embedding from
its own CNN and one scalar next reward, for every action. Only the executed
action is supervised. Embedding MSE averages over coordinates and nonterminal
transitions; reward MSE includes terminals. Losses sum over branches with
prediction coefficient 1. Both current and target CNN paths have stopped
auxiliary gradients. PPO still trains CNNs normally; IB divergence gradients
are unchanged. Prediction gradients reach LRU and (for IB) posterior parameters.
No sampling. Terminal embedding targets are masked because step auto-resets.

IB is reverse KL KL(posterior || prior), with 0.4/1.6 prior/posterior gradient
weights, output posterior input, one-bit full-distribution free region, frozen
branch RMS normalization and coefficient 1. Both IB conditions share calibration.
Residuals add earlier activated LRU outputs to later layer outputs. Residual
settings are fixed within each prediction/no-prediction pair, but differ between
vanilla and IB. Shared parameters initialize identically within each pair;
independently initialized prediction heads add parameters. The two losses are
tested together; no separate observation/reward ablations or coefficient sweep.

The completed vanilla reference is copied byte-for-byte from the preceding run;
see `provenance/reused_baseline.json` and retained source hashes. The abandoned
final-output-only prediction run is excluded. New conditions use:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python \
  -m popgym_arcade.experiments.memory_bias --env CartPoleHard \
  --variants lru_prediction ib_reverse_prior20 ib_reverse_prior20_prediction \
  --ib-normalization fixed_rms --timesteps 100000 --seeds 0 \
  --output results/lru_input_prediction_20261111/CartPoleHard/mdp
```

Episode uncertainty is conditional on each trained model. This single training
seed cannot establish a reproducible advantage. Regenerate the report and plot
with `uv run python results/lru_input_prediction_20261111/report.py`.

## Extension to both observation modes

Both MDP and POMDP now include an MLP reference to calculate within-mode memory
bias (MLP score minus condition score). The report separately labels the
MDP-to-POMDP return gap. The matching existing MDP MLP, POMDP vanilla LRU and
POMDP residual-enabled default IB were copied byte-for-byte with hashes and
provenance in `provenance/reused_references.json`. Their original metadata and
source hashes are retained. POMDP IB prediction uses the saved reference's exact
calibration, validated against the task, mode, model and rollout settings.
No existing POMDP MLP was found. `bash run_pomdp.sh` trains that reference and
the two POMDP prediction variants. It preserves the matched 98,304 actual step
budget, seed 0, and 64 evaluation episodes. Configurations and residual settings
match the respective MDP conditions. Regenerate both-mode reports with report.py.

## MLP auxiliary control

`mlp_prediction` is included in both modes, using the user-selected shared
256-dimensional normalized and activated MLP hidden layer as prediction input.
Each branch's linear auxiliary head predicts the same [5, 513] targets and uses
the same losses/coefficient/masks as the recurrent variants. CNN gradients are
blocked for auxiliary replay and targets. The auxiliary loss trains the shared
MLP trunk; PPO output layers keep their original inputs and initialization.
`bash run_mlp_prediction.sh` runs the two controls. The report presents both
plain-MLP-relative bias and matched-auxiliary bias (MLP-prediction return minus
LRU/IB-prediction return), and separately labels MDP-to-POMDP gaps. Source v1
snapshots cover the initial runs; v2 adds MLP prediction without changing the
LRU or IB numerical formulas. The audit matches every run's source hashes to
its preserved snapshot or verifies copied reference artifact hashes.

Hardware caveat: the reused POMDP default-IB baseline used Tesla T4, while the
new POMDP prediction model uses RTX 2060. Both use JAX 0.6.2, matching PPO,
architecture and calibration. Device-dependent numerics remain a confound;
the small POMDP IB difference should not be treated as a clean hardware-matched
auxiliary ablation. All other reused references and new runs use RTX 2060.
