#!/usr/bin/env bash
set -euo pipefail
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run python -m popgym_arcade.experiments.memory_bias \
  --env CartPoleHard --partial-obs --variants mlp lru_prediction \
  --timesteps 100000 --seeds 0 \
  --output results/lru_input_prediction_20261111/CartPoleHard/pomdp
uv run python -m popgym_arcade.experiments.memory_bias \
  --env CartPoleHard --partial-obs --variants ib_reverse_prior20_prediction \
  --ib-normalization fixed_rms \
  --rms-calibration results/lru_input_prediction_20261111/provenance/pomdp_calibration.json \
  --timesteps 100000 --seeds 0 \
  --output results/lru_input_prediction_20261111/CartPoleHard/pomdp
