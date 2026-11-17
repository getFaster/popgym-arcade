"""Colab-only million-step comparison of vanilla LRU and default IB ± residuals."""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TASKS = ("CartPoleHard", "NavigatorEasy", "MineSweeperEasy", "BreakoutEasy")
CONDITIONS = ("vanilla_lru", "ib_residual", "ib_no_residual")


def report(root):
    rows = []
    for task, mode, condition in itertools.product(TASKS, ("mdp", "pomdp"), CONDITIONS):
        variant = "lru" if condition == "vanilla_lru" else "ib_reverse_prior20"
        directory = root / task / mode / condition / f"{variant}_seed0"
        if not (directory / "result.json").exists():
            continue
        data = json.loads((directory / "result.json").read_text())
        returns = np.load(directory / "evaluation_returns.npy")
        records = [
            json.loads(line)
            for line in (directory / "training.jsonl").read_text().splitlines()
        ]
        if (
            returns.shape != (64,)
            or not np.isfinite(returns).all()
            or not np.isclose(returns.mean(), data["evaluation_mean"])
            or not np.isclose(
                returns.std(ddof=1) / 8, data["evaluation_standard_error"]
            )
        ):
            raise ValueError(f"Invalid evaluation: {directory}")
        if (
            data["actual_steps"] != 999424
            or len(records) != 488
            or records[-1]["steps"] != 999424
            or data["checkpoint_interval"] != 100000
        ):
            raise ValueError(
                f"Incomplete budget or incorrect checkpoint setting: {directory}"
            )
        if data["lru_residual"] != (condition == "ib_residual"):
            raise ValueError(f"Incorrect residual setting: {directory}")
        if not all(np.isfinite(r["loss"]) for r in records):
            raise ValueError(f"Nonfinite training: {directory}")
        rows.append(
            {
                "task": task,
                "mode": mode,
                "condition": condition,
                "mean_return": data["evaluation_mean"],
                "episode_se": data["evaluation_standard_error"],
                "actual_steps": data["actual_steps"],
                "requested_steps": data["requested_steps"],
                "seed": data["seed"],
                "seconds": data["seconds"],
                "device": data["device_kind"],
                "checkpoint_count": len(
                    list((directory / "checkpoints").glob("step_*/checkpoint.json"))
                ),
                "run_directory": str(directory.relative_to(root)),
            }
        )
    if rows:
        with (root / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Million-step LRU and default IB comparison",
        "",
        f"**{len(rows)}/24 runs complete.** Colab GPU; actual accelerator is recorded per run. Four tasks × MDP/POMDP × vanilla LRU / default IB with residuals / default IB without residuals. Seed 0, 1,000,000 requested / 999,424 actual training steps, 64 stochastic-policy evaluation episodes. PPO settings are unchanged; the learning-rate schedule spans this longer budget.",
        "",
        "IB uses reverse KL, output posterior input, prior20 gradient weights 0.4/1.6, coefficient 1, one free bit per Gaussian event, and fresh frozen branch RMS scales. Vanilla LRU has no added residual or bottleneck. Every condition receives `(previous_action, previous_reward, current_observation)`; IB heads receive posterior statistics.",
        "",
        "Resumable checkpoints retain model, optimizer/schedule state, environment and recurrent state, and RNG. Periodic checkpoints occur at the first complete 2,048-step rollout crossing 100K-step thresholds, with an additional final checkpoint at 999,424 steps. Completed checkpoints are synced locally. A resumed run continues its original 1M schedule and optimizer state.",
        "",
        "| Task | Mode | Condition | Return ± episode SE | Checkpoints |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['mode'].upper()} | {row['condition']} | [{row['mean_return']:.4f} ± {row['episode_se']:.4f}]({row['run_directory']}/result.json) | {row['checkpoint_count']} |"
        )
    lines += [
        "",
        "The three conditions are compared at the same training budget. One seed and episode SE do not establish training-seed variability. Calibration is repeated for each IB architecture; normalization scales therefore differ. Historical 100K results are not reused for training.",
    ]
    if rows:
        lines += ["", "[Summary CSV](summary.csv)"]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report(args.output)
        return
    expected = {
        "tasks": TASKS,
        "modes": ["mdp", "pomdp"],
        "conditions": CONDITIONS,
        "seed": 0,
        "requested_steps": 1000000,
        "actual_steps": 999424,
        "checkpoint_interval": 100000,
        "gradient_log_interval": 10000,
        "evaluation_episodes": 64,
        "device": "Colab GPU; record actual accelerator per run",
    }
    path = args.output / "suite.json"
    serialized = json.loads(json.dumps(expected))
    if path.exists() and json.loads(path.read_text()) != serialized:
        raise ValueError("Suite settings differ; use a fresh output directory")
    path.write_text(json.dumps(expected, indent=2) + "\n")
    env = os.environ | {
        "JAX_PLATFORMS": "cuda",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "MPLBACKEND": "Agg",
    }
    failures = []
    report(args.output)
    for index, (task, mode, condition) in enumerate(
        itertools.product(TASKS, ("mdp", "pomdp"), CONDITIONS), start=1
    ):
        directory = args.output / task / mode / condition
        directory.mkdir(parents=True, exist_ok=True)
        variant = "lru" if condition == "vanilla_lru" else "ib_reverse_prior20"
        (args.output / "status.json").write_text(
            json.dumps(
                {
                    "status": "running",
                    "task": task,
                    "mode": mode,
                    "condition": condition,
                    "run": index,
                    "total": 24,
                    "time": time.time(),
                },
                indent=2,
            )
        )
        command = [
            sys.executable,
            "-m",
            "popgym_arcade.experiments.memory_bias",
            "--env",
            task,
            "--variants",
            variant,
            "--timesteps",
            "1000000",
            "--seeds",
            "0",
            "--eval-episodes",
            "64",
            "--checkpoint-interval",
            "100000",
            "--gradient-log-interval",
            "10000",
            "--output",
            str(directory),
        ]
        if mode == "pomdp":
            command.append("--partial-obs")
        if condition == "vanilla_lru":
            command += [
                "--no-lru-residual",
                "--ib-coefficient",
                "0",
                "--ib-normalization",
                "none",
            ]
        else:
            command += [
                "--posterior-input",
                "output",
                "--ib-normalization",
                "fixed_rms",
                "--lru-residual" if condition == "ib_residual" else "--no-lru-residual",
            ]
        started = time.monotonic()
        with (directory / "worker.log").open("a") as log:
            process = subprocess.run(
                command, env=env, check=False, stdout=log, stderr=subprocess.STDOUT
            )
        record = {
            "task": task,
            "mode": mode,
            "condition": condition,
            "returncode": process.returncode,
            "seconds_this_attempt": time.monotonic() - started,
        }
        with (args.output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if process.returncode:
            failures.append(record)
        report(args.output)
    (args.output / "done.json").write_text(
        json.dumps(
            {"failures": failures, "completed": len(report(args.output))}, indent=2
        )
        + "\n"
    )
    (args.output / "status.json").write_text(
        json.dumps(
            {"status": "failed" if failures else "complete", "failures": failures},
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
