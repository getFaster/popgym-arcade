"""Local-only, snapshot-isolated residual ablation of the provisional default IB."""

import csv
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
SOURCE = ROOT / "provenance/source"
TASKS = ("CartPoleHard", "NavigatorEasy", "MineSweeperEasy", "BreakoutEasy")
VARIANT = "ib_reverse_prior20"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def result(path):
    data = json.loads(path.read_text())
    scores = np.load(path.with_name("evaluation_returns.npy"))
    if (
        scores.shape != (64,)
        or not np.isfinite(scores).all()
        or not np.isclose(scores.mean(), data["evaluation_mean"])
        or not np.isclose(scores.std(ddof=1) / 8, data["evaluation_standard_error"])
    ):
        raise ValueError(f"Invalid evaluation: {path}")
    return data


def report():
    rows = []
    for task, mode in itertools.product(TASKS, ("mdp", "pomdp")):
        path = ROOT / task / mode / f"{VARIANT}_seed0/result.json"
        if not path.exists():
            continue
        current = result(path)
        reference_path = (
            REPO
            / "results/posterior_rms_suite_20261109"
            / task
            / mode
            / "output"
            / f"{VARIANT}_seed0/result.json"
        )
        baseline_path = (
            REPO
            / "results/vanilla_lru_baselines_20261109"
            / task
            / mode
            / "lru_seed0/result.json"
        )
        reference, baseline = result(reference_path), result(baseline_path)
        for field in (
            "config",
            "seed",
            "actual_steps",
            "requested_steps",
            "hidden",
            "layers",
            "posterior_input",
            "head_inputs",
            "kl_prior_gradient_weight",
            "kl_posterior_gradient_weight",
            "ib_coefficient",
            "evaluation_episodes",
            "normalization",
            "parameter_count",
        ):
            if current[field] != reference[field]:
                raise ValueError(
                    f"Unmatched residual comparison {field}: {task}/{mode}"
                )
        if (
            current["lru_residual"] is not False
            or reference["lru_residual"] is not True
            or current["rms_calibration"]["residual"] is not False
        ):
            raise ValueError("Incorrect residual setting or calibration")
        logs = [
            json.loads(line)
            for line in path.with_name("training.jsonl").read_text().splitlines()
        ]
        if (
            len(logs) != 48
            or logs[-1]["steps"] != 98304
            or not all(
                np.isfinite([r[k] for k in ("loss", "ib_loss_normalized")]).all()
                for r in logs
            )
        ):
            raise ValueError(f"Incomplete or nonfinite training: {path}")
        # Keep reference evidence here so the comparison survives future directory cleanup.
        saved = ROOT / "provenance/references" / task / mode
        saved.mkdir(parents=True, exist_ok=True)
        for label, value, original in (
            ("residual_ib", reference, reference_path),
            ("vanilla_lru", baseline, baseline_path),
        ):
            write_json(
                saved / f"{label}_result.json",
                {**value, "reference_original_path": str(original)},
            )
        rows.append(
            {
                "task": task,
                "mode": mode,
                "no_residual_return": current["evaluation_mean"],
                "no_residual_episode_se": current["evaluation_standard_error"],
                "residual_return": reference["evaluation_mean"],
                "residual_episode_se": reference["evaluation_standard_error"],
                "no_residual_minus_residual": current["evaluation_mean"]
                - reference["evaluation_mean"],
                "vanilla_lru_return": baseline["evaluation_mean"],
                "vanilla_lru_episode_se": baseline["evaluation_standard_error"],
                "no_residual_minus_lru": current["evaluation_mean"]
                - baseline["evaluation_mean"],
                "at_least_lru_minus_se": current["evaluation_mean"]
                >= baseline["evaluation_mean"] - baseline["evaluation_standard_error"],
                "actor_rms_bits": current["rms_scales_bits"][0],
                "critic_rms_bits": current["rms_scales_bits"][1],
                "residual_actor_rms_bits": reference["rms_scales_bits"][0],
                "residual_critic_rms_bits": reference["rms_scales_bits"][1],
                "calibration_trajectory_matches_reference": current["rms_calibration"][
                    "batch_sha256"
                ]
                == reference["rms_calibration"]["batch_sha256"],
                "device": current["device_kind"],
                "reference_device": reference["device_kind"],
                "actual_steps": current["actual_steps"],
                "seed": current["seed"],
                "run_directory": str(path.parent.relative_to(ROOT)),
            }
        )
    if rows:
        with (ROOT / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Default IB without residual connections",
        "",
        f"**{len(rows)}/8 runs complete.** Local RTX 2060 only. Reverse KL (posterior || prior), output posterior input, prior20 (gradient weights 0.4/1.6), residuals disabled. Seed 0, 100,000 requested / 98,304 actual steps, 64 evaluation episodes. Fresh frozen RMS calibration per task/mode uses the no-residual architecture.",
        "",
        "Residual-enabled default IB references and vanilla LRU references are compared within the same task/mode. ± is episode SE, not training-seed variability. Calibration is recomputed for each architecture, so the comparison includes its changed frozen scale. POMDP residual references ran on T4; new runs use the local RTX 2060.",
        "",
        "| Task | Mode | No residual ± SE | Residual ± SE | Change | Vanilla LRU ± SE | ≥ LRU − SE |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['mode'].upper()} | {row['no_residual_return']:.4f} ± {row['no_residual_episode_se']:.4f} | {row['residual_return']:.4f} ± {row['residual_episode_se']:.4f} | {row['no_residual_minus_residual']:+.4f} | {row['vanilla_lru_return']:.4f} ± {row['vanilla_lru_episode_se']:.4f} | {'yes' if row['at_least_lru_minus_se'] else 'no'} |"
        )
    lines += ["", "[Run manifest](manifest.json) · [Progress](progress.jsonl)"]
    if rows:
        lines += ["", "[Detailed comparison CSV](summary.csv)"]
    lines += [
        "",
        "Training uses the frozen source snapshot in `provenance/source/`; original run metadata and per-episode returns remain in each task/mode directory. Reference result metadata is copied into `provenance/references/`. These single-seed comparisons are exploratory.",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    write_json(
        ROOT / "audit.json",
        {
            "completed": len(rows),
            "expected": 8,
            "validated": "scores, training steps, finite losses, residual settings, and matching reference configuration",
        },
    )


def main():
    env = os.environ | {
        "JAX_PLATFORMS": "cuda",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "MPLBACKEND": "Agg",
    }
    failures = []
    (ROOT / "progress.jsonl").touch(exist_ok=True)
    report()
    for index, (task, mode) in enumerate(
        itertools.product(TASKS, ("mdp", "pomdp")), start=1
    ):
        directory = ROOT / task / mode
        directory.mkdir(parents=True, exist_ok=True)
        write_json(
            ROOT / "status.json",
            {
                "status": "running",
                "task": task,
                "mode": mode,
                "run": index,
                "total": 8,
                "updated_at": time.time(),
            },
        )
        command = [
            sys.executable,
            "-m",
            "popgym_arcade.experiments.memory_bias",
            "--env",
            task,
            "--variants",
            VARIANT,
            "--posterior-input",
            "output",
            "--no-lru-residual",
            "--ib-normalization",
            "fixed_rms",
            "--timesteps",
            "100000",
            "--seeds",
            "0",
            "--eval-episodes",
            "64",
            "--output",
            str(directory),
        ]
        if mode == "pomdp":
            command.append("--partial-obs")
        started = time.monotonic()
        with (directory / "worker.log").open("a") as log:
            process = subprocess.run(
                command,
                cwd=SOURCE,
                env=env,
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        record = {
            "task": task,
            "mode": mode,
            "returncode": process.returncode,
            "seconds": time.monotonic() - started,
        }
        with (ROOT / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if process.returncode:
            failures.append(record)
        report()
    write_json(ROOT / "done.json", {"failures": failures})
    write_json(
        ROOT / "status.json",
        {
            "status": "failed" if failures else "complete",
            "failures": failures,
            "updated_at": time.time(),
        },
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
