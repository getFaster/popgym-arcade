#!/usr/bin/env bash
cd /home/user/popgym_arcade || exit 1
export UV_CACHE_DIR=/tmp/uv-cache
export MPLCONFIGDIR=/tmp/fsq-matplotlib
bash results/fsq_lru_local_20261117_1420/source/run_comparison.sh results/fsq_lru_local_20261117_1420 > results/fsq_lru_local_20261117_1420/console.log 2>&1
status=$?
printf '%s\n' "$status" > results/fsq_lru_local_20261117_1420/exit_code
exit "$status"
