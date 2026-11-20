#!/usr/bin/env bash
set -euo pipefail
root=$1
shard=$2
export JAX_PLATFORMS=cuda
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export UV_CACHE_DIR=/tmp/uv-cache
if [[ $shard == local ]]; then seed=1; else seed=2; fi
run() {
  local task=$1 mode=$2
  shift 2
  local partial=()
  if [[ $mode == pomdp ]]; then partial=(--partial-obs); fi
  uv run --no-sync python -m popgym_arcade.experiments.memory_bias \
    --env "$task" "${partial[@]}" --output "$root/$shard/$task/$mode" \
    --timesteps 100000 --checkpoint-interval 100000 "$@"
}
run CartPoleHard mdp --seeds "$seed" --variants mlp
if [[ $shard == local ]]; then
  for task in NavigatorEasy MineSweeperEasy BreakoutEasy; do
    run "$task" mdp --seeds 0 --variants mlp
  done
fi
for mode in mdp pomdp; do
  run CartPoleHard "$mode" --seeds "$seed" --variants mlp lru lru_prediction mlp_prediction ib_reverse_prior20 ib_reverse_prior20_prediction
  for task in NavigatorEasy MineSweeperEasy BreakoutEasy; do
    run "$task" "$mode" --seeds "$seed" --variants ib_reverse_prior20
    if [[ $mode == mdp ]]; then run "$task" mdp --seeds "$seed" --variants mlp; fi
  done
done
touch "$root/$shard/complete"
