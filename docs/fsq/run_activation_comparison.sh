#!/usr/bin/env bash
set -euo pipefail
root=${1:?Supply a fresh results directory}
for obs in mdp pomdp; do
  partial=()
  if [[ "$obs" == pomdp ]]; then partial=(--partial-obs); fi
  for mode in activation state_activation; do
    for levels in 3 5 7; do
      JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python -m popgym_arcade.experiments.memory_bias \
        --variants lru --env CartPoleHard --seeds 0 1 2 --timesteps 100000 \
        --recurrent-state-quantization "$mode" --fsq-levels "$levels" --fsq-scale 1.0 \
        "${partial[@]}" --output "$root/$obs/${mode}_${levels}"
    done
  done
done
