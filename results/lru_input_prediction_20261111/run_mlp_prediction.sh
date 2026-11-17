#!/usr/bin/env bash
set -euo pipefail
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run python -m popgym_arcade.experiments.memory_bias \
  --env CartPoleHard --variants mlp_prediction --timesteps 100000 --seeds 0 \
  --output results/lru_input_prediction_20261111/CartPoleHard/mdp
uv run python -m popgym_arcade.experiments.memory_bias \
  --env CartPoleHard --partial-obs --variants mlp_prediction --timesteps 100000 --seeds 0 \
  --output results/lru_input_prediction_20261111/CartPoleHard/pomdp
