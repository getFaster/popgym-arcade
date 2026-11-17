# LRU next-embedding and reward prediction pilot

Fresh paired CartPoleHard MDP runs, seed 0, 100000 requested / 98304 actual
steps, 64 evaluation episodes. All upstream PPO settings, two 512-state LRU
layers per branch, current-input skips, and disabled inter-layer residuals
are shared. The prediction condition adds a linear head on each branch's final
activated LRU output: five actions × (512 next-CNN-embedding coordinates +
one next reward). Only the executed action receives supervised gradients.
Each branch predicts its own encoder's next embedding.

The loss is PPO + sum over actor/critic of (embedding MSE averaged over 512
coordinates and nonterminal transitions + scalar reward MSE over all
transitions), coefficient 1. No free bits, divergence or RMS normalization.
Targets use stopped gradients. Prediction replay also stops gradients on the
current CNN encoding, so neither prediction objective trains CNN parameters;
PPO still does. LRU states and heads receive prediction gradients. Targets
therefore move as PPO trains the CNN. Terminal embedding targets are excluded
because env.step auto-resets; terminal rewards remain supervised. Targets are
stored for the final rollout transition too. Inputs remain previous action,
previous reward, current observation with the existing reset rules.

The heads use an independent folded-in initialization key; all shared initial
parameters match vanilla LRU. Added heads increase the parameter count. This
single-seed pilot measures a joint objective, not separate observation/reward
ablations; evaluation episode SE does not measure training-seed variability.

Command:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python \
  -m popgym_arcade.experiments.memory_bias --env CartPoleHard \
  --variants lru lru_prediction --timesteps 100000 --seeds 0 \
  --output results/lru_prediction_20261111/CartPoleHard/mdp
```

SUPERSEDED: User revised prediction-head inputs to include input_t and requested
an additional IB combination. The final-output-only prediction run was stopped
before completion; do not treat it as a result. The completed vanilla baseline
is reused byte-for-byte in ../lru_input_prediction_20261111 with provenance.
