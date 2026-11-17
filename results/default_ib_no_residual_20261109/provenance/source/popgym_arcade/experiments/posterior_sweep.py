"""Sequential, resumable device shard of the corrected 128-run factorial pilot."""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TASKS = ("CartPoleHard", "NavigatorEasy", "MineSweeperEasy", "BreakoutEasy")
VARIANTS = [
    f"ib_{kind}_prior{share}"
    for kind in ("forward", "reverse", "symmetric", "renyi_half")
    for share in (20, 80)
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["local", "colab"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ | {
        "JAX_PLATFORMS": "cuda",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    groups = list(itertools.product(TASKS, (False, True), ("output", "state")))
    failures = []
    manifest = {
        "worker": args.worker,
        "total_runs": 64,
        "all_suite_runs": 128,
        "timesteps": 100000,
        "actual_steps": 98304,
        "seed": 0,
        "eval_episodes": 64,
        "residual": True,
        "head_inputs": "input_t, posterior_mean, posterior_log_variance",
        "groups": [],
    }
    # Alternate complete calibration groups; each device receives both posterior inputs.
    selected = [
        (i, g)
        for i, g in enumerate(groups)
        if (i // 2) % 2 == (0 if args.worker == "local" else 1)
    ]
    manifest["groups"] = [
        {
            "env": g[0],
            "partial_obs": g[1],
            "posterior_input": g[2],
            "variants": VARIANTS,
        }
        for _, g in selected
    ]
    (args.output / f"{args.worker}_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    for _, (task, partial, posterior) in selected:
        group = args.output / task / ("pomdp" if partial else "mdp") / posterior
        group.mkdir(parents=True, exist_ok=True)
        calibration = group / "calibration.json"
        common = ["--env", task, "--posterior-input", posterior] + (
            ["--partial-obs"] if partial else []
        )
        if not calibration.exists():
            with (group / "calibration.log").open("a") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "popgym_arcade.experiments.calibrate_divergences",
                        *common,
                        "--output",
                        str(calibration),
                    ],
                    check=False,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            if result.returncode:
                failures.append(str(calibration))
                continue
        for variant in VARIANTS:
            started = time.time()
            command = [
                sys.executable,
                "-m",
                "popgym_arcade.experiments.memory_bias",
                *common,
                "--lru-residual",
                "--rms-calibration",
                str(calibration),
                "--timesteps",
                "100000",
                "--seeds",
                "0",
                "--variants",
                variant,
                "--output",
                str(group),
            ]
            with (group / f"{variant}.log").open("a") as log:
                result = subprocess.run(
                    command, check=False, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            record = {
                "task": task,
                "partial_obs": partial,
                "posterior": posterior,
                "variant": variant,
                "returncode": result.returncode,
                "seconds": time.time() - started,
            }
            with (args.output / f"{args.worker}_progress.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")
            if result.returncode:
                failures.append(record)
    (args.output / f"{args.worker}_done.json").write_text(
        json.dumps({"failures": failures}, indent=2)
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
