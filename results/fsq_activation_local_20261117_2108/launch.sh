#!/usr/bin/env bash
cd /home/user/popgym_arcade || exit 1
export UV_CACHE_DIR=/tmp/uv-cache
export MPLCONFIGDIR=/tmp/fsq-matplotlib
bash results/fsq_activation_local_20261117_2108/source/run_activation_comparison.sh results/fsq_activation_local_20261117_2108 > results/fsq_activation_local_20261117_2108/console.log 2>&1
status=$?
printf '%s\n' "$status" > results/fsq_activation_local_20261117_2108/exit_code
exit "$status"
