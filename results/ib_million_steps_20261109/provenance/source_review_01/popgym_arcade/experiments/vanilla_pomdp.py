"""Run one sequential device shard of vanilla LRU POMDP reference experiments."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ASSIGNMENTS = {
    "local": ("NavigatorEasy", "CartPoleHard"),
    "colab": ("MineSweeperEasy", "BreakoutEasy"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=ASSIGNMENTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "worker": args.worker,
        "tasks": ASSIGNMENTS[args.worker],
        "variant": "lru",
        "partial_obs": True,
        "lru_residual": False,
        "head_inputs": "input_t, projected_lru_output",
        "seed": 0,
        "requested_steps": 100000,
        "actual_steps": 98304,
        "evaluation_episodes": 64,
        "ib_coefficient": 0,
        "rms_calibration": None,
    }
    (args.output / f"{args.worker}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    env = os.environ | {
        "JAX_PLATFORMS": "cuda",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "MPLBACKEND": "Agg",
    }
    failures = []
    for task in ASSIGNMENTS[args.worker]:
        directory = args.output / task / "pomdp"
        directory.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "popgym_arcade.experiments.memory_bias",
            "--env",
            task,
            "--partial-obs",
            "--variants",
            "lru",
            "--seeds",
            "0",
            "--timesteps",
            "100000",
            "--eval-episodes",
            "64",
            "--ib-coefficient",
            "0",
            "--output",
            str(directory),
        ]
        started = time.time()
        with (directory / "worker.log").open("a") as stream:
            result = subprocess.run(
                command, check=False, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
        record = {
            "task": task,
            "returncode": result.returncode,
            "seconds": time.time() - started,
        }
        with (args.output / f"{args.worker}_progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if result.returncode:
            failures.append(record)
    (args.output / f"{args.worker}_done.json").write_text(
        json.dumps({"failures": failures}, indent=2) + "\n"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
