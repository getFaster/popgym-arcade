#!/usr/bin/env bash
# Explicitly invoke this script to launch experiments; not part of validation.
set -euo pipefail
root=${1:?Supply a fresh results directory}
for mode in mdp pomdp; do
  partial=()
  if [[ "$mode" == pomdp ]]; then partial=(--partial-obs); fi
  for spec in none:5 fsq:3 fsq:5 fsq:7 output:3 output:5 output:7; do
    quant=${spec%:*}
    levels=${spec#*:}
    JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python -m popgym_arcade.experiments.memory_bias \
      --variants lru --env CartPoleHard --seeds 0 1 2 --timesteps 100000 \
      --recurrent-state-quantization "$quant" --fsq-levels "$levels" --fsq-scale 1.0 \
      "${partial[@]}" --output "$root/$mode/${quant}_${levels}"
  done
done
