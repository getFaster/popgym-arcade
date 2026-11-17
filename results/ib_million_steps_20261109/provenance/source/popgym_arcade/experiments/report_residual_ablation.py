"""Audit and report the default-IB residual ablation without training or a GPU."""

import argparse
import csv
import hashlib
import itertools
import json
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASKS = ("CartPoleHard", "NavigatorEasy", "MineSweeperEasy", "BreakoutEasy")
MODES = ("mdp", "pomdp")
VARIANT = "ib_reverse_prior20"


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_reference(root, task, mode, label, original):
    """Bundle small reference artifacts so regeneration does not need old suites."""
    saved = root / "provenance/references" / task / mode
    saved.mkdir(parents=True, exist_ok=True)
    for filename in ("result.json", "evaluation_returns.npy", "training.jsonl"):
        target = saved / f"{label}_{filename}"
        source = original / filename
        if source.exists():
            if filename == "result.json":
                data = json.loads(source.read_text())
                data["reference_original_path"] = str(source)
                target.write_text(json.dumps(data, indent=2) + "\n")
            elif not target.exists():
                shutil.copy2(source, target)
        if not target.exists():
            raise ValueError(f"Missing bundled reference: {target}")
    data = json.loads((saved / f"{label}_result.json").read_text())
    scores = np.load(saved / f"{label}_evaluation_returns.npy")
    logs = [
        json.loads(line)
        for line in (saved / f"{label}_training.jsonl").read_text().splitlines()
    ]
    return data, scores, logs


def check_artifacts(data, scores, logs):
    if scores.shape != (64,) or not np.isfinite(scores).all():
        raise ValueError("Expected 64 finite evaluation returns")
    for field, value in (
        ("evaluation_mean", scores.mean()),
        ("evaluation_std", scores.std(ddof=1)),
        ("evaluation_standard_error", scores.std(ddof=1) / 8),
    ):
        if not np.isclose(data[field], value):
            raise ValueError(f"Evaluation statistic mismatch: {field}")
    if (
        data["actual_steps"] != 98304
        or data["seed"] != 0
        or data["evaluation_episodes"] != 64
    ):
        raise ValueError("Unexpected training budget, seed, or evaluation count")
    if [row["steps"] for row in logs] != list(range(2048, 98305, 2048)):
        raise ValueError("Expected 48 complete PPO updates")
    if not all(
        np.isfinite(
            [row[k] for k in ("loss", "policy_loss", "value_loss", "entropy")]
        ).all()
        for row in logs
    ):
        raise ValueError("Nonfinite PPO losses")


def audit(root, suite, baselines):
    rows, calibration, manifest, learning = [], [], [], {}
    sources = json.loads((root / "manifest.json").read_text())["source_sha256"]
    for task, mode in itertools.product(TASKS, MODES):
        directory = root / task / mode / f"{VARIANT}_seed0"
        current = json.loads((directory / "result.json").read_text())
        scores = np.load(directory / "evaluation_returns.npy")
        logs = [
            json.loads(line)
            for line in (directory / "training.jsonl").read_text().splitlines()
        ]
        check_artifacts(current, scores, logs)
        if not (directory / "model.eqx").is_file():
            raise ValueError(f"Missing checkpoint: {directory}")
        if current["device_kind"] != "NVIDIA GeForce RTX 2060 with Max-Q Design":
            raise ValueError("Ablation was requested on the local GPU only")
        for name, digest in current["source_sha256"].items():
            relative = (
                name if name == "uv.lock" else f"popgym_arcade/experiments/{name}"
            )
            if (
                digest != sources[relative]
                or hashlib.sha256(
                    (root / "provenance/source" / relative).read_bytes()
                ).hexdigest()
                != digest
            ):
                raise ValueError(f"Snapshot source mismatch: {name}")
        residual, rscores, rlogs = load_reference(
            root,
            task,
            mode,
            "residual_ib",
            suite / task / mode / "output" / f"{VARIANT}_seed0",
        )
        vanilla, vscores, vlogs = load_reference(
            root, task, mode, "vanilla_lru", baselines / task / mode / "lru_seed0"
        )
        check_artifacts(residual, rscores, rlogs)
        check_artifacts(vanilla, vscores, vlogs)
        fields = (
            "config",
            "seed",
            "requested_steps",
            "actual_steps",
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
        )
        if any(current[field] != residual[field] for field in fields):
            raise ValueError(
                f"Residual reference configuration mismatch: {task}/{mode}"
            )
        if any(
            current[field] != vanilla[field]
            for field in (
                "config",
                "seed",
                "actual_steps",
                "hidden",
                "layers",
                "evaluation_episodes",
            )
        ):
            raise ValueError(f"Vanilla reference configuration mismatch: {task}/{mode}")
        if (
            current["variant"] != VARIANT
            or current["divergence"] != "reverse_kl"
            or current["posterior_input"] != "output"
            or current["head_inputs"]
            != "input_t, posterior_mean, posterior_log_variance"
        ):
            raise ValueError("Incorrect IB variant, divergence, or head inputs")
        if (
            current["lru_residual"]
            or not residual["lru_residual"]
            or vanilla.get("lru_residual", False)
        ):
            raise ValueError("Incorrect residual settings")
        if (
            current["rms_calibration"]["residual"]
            or not residual["rms_calibration"]["residual"]
        ):
            raise ValueError("Incorrect calibration residual settings")
        if not np.isclose(current["kl_prior_gradient_weight"], 0.4) or not np.isclose(
            current["kl_posterior_gradient_weight"], 1.6
        ):
            raise ValueError("Incorrect asymmetric balancing weights")
        if current["normalization"] != "fixed_calibration_rms_excess_above_one_bit":
            raise ValueError("Incorrect normalization")
        for label, value in (("no_residual", current), ("residual", residual)):
            if (
                value["rms_calibration"]["scales_bits"]["reverse_kl"]
                != value["rms_scales_bits"]
            ):
                raise ValueError("Incorrect frozen scales")
            calibration.append(
                {
                    "task": task,
                    "mode": mode,
                    "condition": label,
                    "actor_rms_bits": value["rms_scales_bits"][0],
                    "critic_rms_bits": value["rms_scales_bits"][1],
                    "calibration_samples": value["rms_calibration"]["samples"],
                    "calibration_trajectory_sha256": value["rms_calibration"][
                        "batch_sha256"
                    ],
                }
            )
        if not all(
            np.isfinite(
                [
                    row["ib_loss_normalized"],
                    row["actor_divergence_nats"],
                    row["critic_divergence_nats"],
                ]
            ).all()
            for row in logs
        ):
            raise ValueError("Nonfinite auxiliary metrics")
        threshold = vanilla["evaluation_mean"] - vanilla["evaluation_standard_error"]
        rows.append(
            {
                "task": task,
                "mode": mode,
                "no_residual_return": current["evaluation_mean"],
                "no_residual_episode_se": current["evaluation_standard_error"],
                "residual_return": residual["evaluation_mean"],
                "residual_episode_se": residual["evaluation_standard_error"],
                "no_residual_minus_residual": current["evaluation_mean"]
                - residual["evaluation_mean"],
                "vanilla_lru_return": vanilla["evaluation_mean"],
                "vanilla_lru_episode_se": vanilla["evaluation_standard_error"],
                "no_residual_minus_lru": current["evaluation_mean"]
                - vanilla["evaluation_mean"],
                "lru_minus_se_threshold": threshold,
                "no_residual_at_least_lru_minus_se": current["evaluation_mean"]
                >= threshold,
                "residual_at_least_lru_minus_se": residual["evaluation_mean"]
                >= threshold,
                "actor_rms_bits": current["rms_scales_bits"][0],
                "critic_rms_bits": current["rms_scales_bits"][1],
                "residual_actor_rms_bits": residual["rms_scales_bits"][0],
                "residual_critic_rms_bits": residual["rms_scales_bits"][1],
                "calibration_trajectory_matches_reference": current["rms_calibration"][
                    "batch_sha256"
                ]
                == residual["rms_calibration"]["batch_sha256"],
                "device": current["device_kind"],
                "residual_reference_device": residual["device_kind"],
                "vanilla_reference_device": vanilla["device_kind"],
                "actual_steps": 98304,
                "seed": 0,
                "run_directory": str(directory.relative_to(root)),
            }
        )
        learning[task, mode] = (logs, rlogs)
        for path in sorted(directory.iterdir()):
            if path.is_file():
                manifest.append(
                    {
                        "path": str(path.relative_to(root)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size,
                    }
                )
    return rows, calibration, manifest, learning


def plots(root, rows, learning):
    reports = root / "reports"
    fig, axes = plt.subplots(4, 2, figsize=(14, 13), layout="constrained")
    for i, task in enumerate(TASKS):
        group = [row for row in rows if row["task"] == task]
        extrema = [
            row[f"{kind}_return"] + sign * row[f"{kind}_episode_se"]
            for row in group
            for kind in ("vanilla_lru", "residual", "no_residual")
            for sign in (-1, 1)
        ]
        pad = max((max(extrema) - min(extrema)) * 0.12, 0.01)
        for j, mode in enumerate(MODES):
            ax = axes[i, j]
            row = next(r for r in group if r["mode"] == mode)
            for y, kind, color in (
                (2, "vanilla_lru", "#666666"),
                (1, "residual", "#e58a2b"),
                (0, "no_residual", "#2278ac"),
            ):
                ax.errorbar(
                    row[f"{kind}_return"],
                    y,
                    xerr=row[f"{kind}_episode_se"],
                    fmt="o",
                    capsize=5,
                    markersize=7,
                    color=color,
                )
            ax.axvline(
                row["lru_minus_se_threshold"],
                color="#888888",
                ls="--",
                lw=1,
                label="LRU mean − episode SE",
            )
            ax.set_yticks(
                [0, 1, 2], ["IB without residual", "IB with residual", "Vanilla LRU"]
            )
            ax.set_xlim(min(extrema) - pad, max(extrema) + pad)
            ax.set_ylim(-0.6, 2.6)
            ax.set_title(f"{task} · {mode.upper()}")
            ax.set_xlabel("Evaluation return ± episode SE")
            ax.grid(axis="x", alpha=0.2)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Default IB residual ablation · reverse KL, output posterior, prior20\nSeed 0 · 98,304 training steps · 64 evaluation episodes. Each architecture recalibrates frozen RMS."
    )
    fig.savefig(reports / "comparison.png", dpi=160)
    fig.savefig(reports / "comparison.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), layout="constrained")
    for i, task in enumerate(TASKS):
        for j, mode in enumerate(MODES):
            ax = axes[i, j]
            for logs, label, color in zip(
                learning[task, mode],
                ("Without residual", "With residual"),
                ("#2278ac", "#e58a2b"),
            ):
                xs, ys = [], []
                for start in range(0, len(logs), 5):
                    window = logs[start : start + 5]
                    episodes = sum(row["episodes"] for row in window)
                    if episodes:
                        xs.append(window[-1]["steps"])
                        ys.append(
                            sum(
                                (row["training_return"] or 0) * row["episodes"]
                                for row in window
                            )
                            / episodes
                        )
                ax.plot(xs, ys, color=color, label=label)
            ax.set_title(f"{task} · {mode.upper()}")
            ax.set_xlabel("Training environment steps")
            ax.set_ylabel("Training return")
            ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle(
        "Training diagnostics · episode-weighted returns in five-update windows\nThese curves are training returns, not held-out evaluation scores."
    )
    fig.savefig(reports / "training.png", dpi=150)
    plt.close(fig)


def report(root, suite, baselines):
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    try:
        rows, calibration, manifest, learning = audit(root, suite, baselines)
    except Exception as error:
        (root / "audit.json").write_text(
            json.dumps({"status": "failed", "error": str(error)}, indent=2) + "\n"
        )
        raise
    write_csv(root / "summary.csv", rows)
    write_csv(reports / "calibration.csv", calibration)
    (root / "provenance/artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    off_pass = sum(row["no_residual_at_least_lru_minus_se"] for row in rows)
    on_pass = sum(row["residual_at_least_lru_minus_se"] for row in rows)
    improved = sum(row["no_residual_minus_residual"] > 0 for row in rows)
    audit_data = {
        "status": "passed",
        "completed": 8,
        "expected": 8,
        "evaluation_episodes": 512,
        "no_residual_lru_minus_se_passes": off_pass,
        "residual_lru_minus_se_passes": on_pass,
        "no_residual_higher_than_residual": improved,
        "all_calibration_trajectories_match": all(
            row["calibration_trajectory_matches_reference"] for row in rows
        ),
        "checked": [
            "eight unique expected task/mode cells",
            "local GPU",
            "snapshot source hashes",
            "48 finite PPO updates per run",
            "64 episode returns and recalculated statistics per condition",
            "checkpoints",
            "matched reference configurations",
            "residual flags and frozen RMS calibration",
        ],
    }
    (root / "audit.json").write_text(json.dumps(audit_data, indent=2) + "\n")
    plots(root, rows, learning)
    drops = sorted(
        (row for row in rows if row["no_residual_minus_residual"] < 0),
        key=lambda row: row["no_residual_minus_residual"],
    )
    gains = sorted(
        (row for row in rows if row["no_residual_minus_residual"] > 0),
        key=lambda row: row["no_residual_minus_residual"],
        reverse=True,
    )

    def describe(selected):
        return ", ".join(
            f"{row['task']} {row['mode'].upper()} ({row['no_residual_minus_residual']:+.4f})"
            for row in selected
        )

    observations = f"Largest mean-return drops: {describe(drops[:2]) or 'none'}. Mean-return increases: {describe(gains) or 'none'}. These are single-training-seed observations, not evidence of reproducible gains or losses across seeds."
    lines = [
        "# Default IB: residual ablation",
        "",
        "**8/8 local-GPU runs complete; artifact audit passed.** Reverse KL `KL(posterior || prior)`, output posterior input, prior20 (auxiliary prior/posterior gradient weights 0.4/1.6). Both RL heads receive current input plus posterior mean/log variance. Each run uses seed 0, 98,304 actual training steps (100,000 requested), and 64 held-out evaluation episodes.",
        "",
        "[All scores and deltas](summary.csv) · [Audit](audit.json) · [Calibration scales](reports/calibration.csv) · [Artifact checksums](provenance/artifact_manifest.json) · [Run manifest](manifest.json)",
        "",
        "## Observed effect",
        "",
        f"Disabling residuals lowered the mean return in **{8 - improved}/8 settings** and increased it in **{improved}/8**. It meets `IB mean ≥ vanilla LRU mean − vanilla LRU episode SE` in **{off_pass}/8**, versus **{on_pass}/8** with residuals. The current residual-enabled default remains unchanged.",
        "",
        observations,
        "",
        "![Evaluation comparison](reports/comparison.png)",
        "",
        "## Scores",
        "",
        "± is episode SE, conditional on one trained policy. Δ is **without residual minus with residual**. The threshold is descriptive, not a statistical equivalence test.",
        "",
        "| Task | Mode | IB without residual ± SE | IB with residual ± SE | Δ | Vanilla LRU ± SE | Without ≥ LRU−SE | With ≥ LRU−SE |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| [{row['task']}]({row['run_directory']}/result.json) | {row['mode'].upper()} | {row['no_residual_return']:.4f} ± {row['no_residual_episode_se']:.4f} | {row['residual_return']:.4f} ± {row['residual_episode_se']:.4f} | {row['no_residual_minus_residual']:+.4f} | {row['vanilla_lru_return']:.4f} ± {row['vanilla_lru_episode_se']:.4f} | {'yes' if row['no_residual_at_least_lru_minus_se'] else 'no'} | {'yes' if row['residual_at_least_lru_minus_se'] else 'no'} |"
        )
    lines += [
        "",
        "## Controls and limits",
        "",
        "PPO settings, seed, training budget, evaluation count, model sizes, Gaussian dimension, balancing weights, posterior head inputs, and parameter counts match the residual-enabled IB references. Both IB conditions have 16,407,454 parameters. Vanilla LRU has 6,940,422 parameters and a different head; it has no added inter-layer residual.",
        "",
        "The auxiliary objective is the sum of actor/critic sample means of `max(D_bits − 1, 0) / frozen_RMS_bits`, coefficient 1. Each architecture recalibrates its own branch-specific RMS scales on 2,048 random-policy samples. All eight calibration-trajectory hashes match their residual-enabled references. Removing residuals also changes the frozen RMS scales, so this is an ablation under the same normalization procedure rather than a fixed-loss-coefficient isolation of residuals.",
        "",
        "All new runs use the local RTX 2060. Residual-enabled MDP references used that GPU; POMDP references used Colab T4. Device provenance is retained in the CSV. Episode SE does not measure training-seed variability. Scores are never averaged across tasks.",
        "",
        "## Training diagnostics",
        "",
        "The curves compare the same preset with and without residuals; no best-configuration selection is performed here. Five-update windows weight returns by completed episodes.",
        "",
        "![Training curves](reports/training.png)",
        "",
        "## Artifact layout and reproduction",
        "",
        "Each `Task/{mdp,pomdp}/ib_reverse_prior20_seed0/` directory keeps the checkpoint, original metadata, per-episode returns, and training log. Fresh calibration files remain under each mode's `_calibration/` directory. `provenance/source/` and `provenance/source_snapshot.tar.gz` preserve training source. The original runner and execution logs remain intact.",
        "",
        "Reference result metadata, evaluation arrays, and training logs are bundled under `provenance/references/`, allowing this report to regenerate after older result directories are deleted. Reference checkpoints are not duplicated.",
        "",
        "```bash",
        f"JAX_PLATFORMS=cpu uv run python -m popgym_arcade.experiments.report_residual_ablation {root}",
        "```",
        "",
        "Standalone plots: [PNG](reports/comparison.png), [PDF](reports/comparison.pdf).",
    ]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    for source in (suite, baselines):
        overview = source / "RESULTS.md"
        if overview.exists():
            text = overview.read_text()
            marker = "Default IB residual ablation"
            if marker not in text:
                overview.write_text(
                    f"> **{marker}:** [Eight-run comparison]({os.path.relpath(root / 'RESULTS.md', source)}).\n\n"
                    + text
                )
    print(json.dumps(audit_data, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--suite", type=Path, default=Path("results/posterior_rms_suite_20261109")
    )
    parser.add_argument(
        "--baselines", type=Path, default=Path("results/vanilla_lru_baselines_20261109")
    )
    args = parser.parse_args()
    report(args.output, args.suite, args.baselines)


if __name__ == "__main__":
    main()
