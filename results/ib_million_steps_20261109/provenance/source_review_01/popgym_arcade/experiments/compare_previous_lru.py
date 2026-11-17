"""Compare corrected bottlenecks with historical vanilla LRU evaluations."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compare_previous_lru(root, rows, reference=None):
    consolidated = root.parent / "vanilla_lru_baselines_20261109"
    reference = reference or (
        consolidated if consolidated.exists() else root.parent / "divergence_suite"
    )

    def mdp_path(task):
        combined = reference / task / "mdp" / "lru_seed0" / "result.json"
        return (
            combined
            if combined.exists()
            else reference / task / "lru_seed0" / "result.json"
        )

    pomdp_baselines = {}
    for task in ("CartPoleHard", "NavigatorEasy", "MineSweeperEasy", "BreakoutEasy"):
        candidate = consolidated / task / "pomdp" / "lru_seed0" / "result.json"
        if candidate.exists():
            pomdp_baselines[task] = json.loads(candidate.read_text())
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    tasks = ("CartPoleHard", "NavigatorEasy", "MineSweeperEasy", "BreakoutEasy")
    baselines, provenance, comparisons, summary = {}, [], [], []
    for task in tasks:
        path = mdp_path(task)
        baseline = json.loads(path.read_text())
        returns = np.load(path.with_name("evaluation_returns.npy"))
        if baseline["variant"] != "lru" or baseline["config"]["PARTIAL"]:
            raise ValueError(f"Expected fully observed vanilla LRU: {path}")
        if returns.shape != (64,) or not np.isfinite(returns).all():
            raise ValueError(f"Invalid baseline returns: {path}")
        for field, value in (
            ("evaluation_mean", returns.mean()),
            ("evaluation_std", returns.std(ddof=1)),
            ("evaluation_standard_error", returns.std(ddof=1) / 8),
        ):
            if not np.isclose(baseline[field], value):
                raise ValueError(f"Baseline {field} disagrees with returns: {path}")
        baselines[task] = baseline
        provenance.append(
            {
                "task": task,
                "mode": "mdp",
                "result_path": str(path),
                "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_sha256": baseline["source_sha256"],
                "reused_from": baseline.get("reused_from"),
                "evaluation_recovered_from_checkpoint": baseline.get(
                    "evaluation_recovered_from_checkpoint", False
                ),
                "evaluation_code_sha256": baseline.get("evaluation_code_sha256"),
                "baseline_interlayer_residual": False,
                "baseline_parameters": baseline["parameter_count"],
            }
        )
        selected = [r for r in rows if r["task"] == task and r["mode"] == "mdp"]
        for row in selected:
            current = json.loads(
                (root / row["run_directory"] / "result.json").read_text()
            )
            for field in (
                "config",
                "seed",
                "requested_steps",
                "actual_steps",
                "hidden",
                "layers",
                "evaluation_episodes",
                "jax_version",
                "upstream_commit",
                "device_kind",
            ):
                if current[field] != baseline[field]:
                    raise ValueError(f"Unmatched baseline {field}: {task}")
            if (
                current["source_sha256"]["uv.lock"]
                != baseline["source_sha256"]["uv.lock"]
            ):
                raise ValueError(f"Unmatched dependency lock: {task}")
            if (
                baseline["weight_decay"] != 0
                or baseline["state_dropout"] != 0
                or baseline["ib_coefficient"] != 0
            ):
                raise ValueError(f"Not an unregularized vanilla baseline: {task}")
        best = max(selected, key=lambda r: r["mean_return"])
        summary.append(
            {
                "task": task,
                "baseline_mean": baseline["evaluation_mean"],
                "baseline_episode_se": baseline["evaluation_standard_error"],
                "best_mean": best["mean_return"],
                "best_episode_se": best["episode_se"],
                "best_divergence": best["divergence"],
                "best_posterior_input": best["posterior_input"],
                "best_prior_share": best["prior_share"],
                "best_delta": best["mean_return"] - baseline["evaluation_mean"],
                "configurations_above_baseline": sum(
                    r["mean_return"] > baseline["evaluation_mean"] for r in selected
                ),
                "configurations_total": len(selected),
            }
        )
    for row in rows:
        baseline = (
            baselines[row["task"]]
            if row["mode"] == "mdp"
            else pomdp_baselines.get(row["task"])
        )
        comparisons.append(
            {
                **row,
                "baseline_available": baseline is not None,
                "baseline_mean": baseline["evaluation_mean"] if baseline else None,
                "baseline_episode_se": baseline["evaluation_standard_error"]
                if baseline
                else None,
                "delta_vs_vanilla_lru": row["mean_return"] - baseline["evaluation_mean"]
                if baseline
                else None,
                "comparison_status": (
                    "historical MDP; residual and head architectures differ"
                    if row["mode"] == "mdp"
                    else "new POMDP; residual/head architectures and possibly device differ"
                )
                if baseline
                else "POMDP baseline unavailable",
            }
        )
    for filename, values in (
        ("vs_vanilla_lru.csv", comparisons),
        ("vs_vanilla_lru_best.csv", summary),
    ):
        with (reports / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (reports / "vanilla_lru_provenance.json").write_text(
        json.dumps(
            {
                "references": provenance,
                "matched": [
                    "fully observed task",
                    "PPO configuration",
                    "seed 0",
                    "98304 training steps",
                    "64 evaluation episodes",
                    "hidden/layers",
                    "JAX and dependency lock",
                    "local GPU",
                ],
                "limitations": [
                    "POMDP baselines were run separately; consult the consolidated archive for provenance",
                    "Historical vanilla LRU has no added inter-layer residual; all corrected IB runs enable it",
                    "LRU head and posterior-statistics head differ in architecture and parameter count",
                    "Training code hashes differ and are retained",
                    "Single training seed; highest-scoring configurations selected after evaluation",
                    "CartPole reference is a copy of the initial pilot, not a second training seed",
                    "MineSweeper baseline evaluation was recovered from its completed checkpoint",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# Comparison with previous vanilla LRU",
        "",
        "[Suite overview](../RESULTS.md) · [All 128 rows with baseline availability and deltas](vs_vanilla_lru.csv) · [Baseline provenance](vanilla_lru_provenance.json)",
        "",
        "Previous vanilla LRU runs exist for the four **MDP** tasks only. The prior/posterior head error was confined to IB models: vanilla LRU uses current input plus projected LRU output and has no Gaussian head. Its historical scores remain usable as vanilla LRU references.",
        "",
        "PPO configuration, task/mode, seed 0, 98,304 actual training steps, hidden/layers, 64 held-out evaluation episodes, JAX/dependency lock, and local GPU match. Raw baseline episode returns were rechecked against the stored mean, standard deviation, and SE. Sources retain their original hashes; the CartPole copy is counted once and MineSweeper's recovered evaluation is identified in the provenance file.",
        "",
        "**Architecture difference:** historical vanilla LRU has no added inter-layer residual connection, whereas all corrected IB runs enable it. Vanilla heads consume current input plus a 256-dimensional projection of LRU output; IB heads consume current input plus posterior mean/log variance. Parameter counts are 6,940,422 for vanilla LRU versus 16,407,454 (output IB) or 24,794,014 (state IB). These comparisons do not isolate the divergence regularizer alone.",
        "",
        "## Highest observed MDP scores",
        "",
        "Highest IB scores are selected from 16 configurations per task. ± is episode SE for one trained seed, not training-seed uncertainty. Positive deltas mean higher raw return; no statistical-significance claim is made.",
        "",
        "| Task | Vanilla LRU ± SE | Best corrected IB ± SE | IB − LRU | Best configuration | IB configs above LRU |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in summary:
        path = mdp_path(row["task"])
        link = os.path.relpath(path, reports)
        label = f"{row['best_divergence']}, {row['best_posterior_input']}, prior{round(row['best_prior_share'] * 100)}"
        lines.append(
            f"| {row['task']} | [{row['baseline_mean']:.4f} ± {row['baseline_episode_se']:.4f}]({link}) | {row['best_mean']:.4f} ± {row['best_episode_se']:.4f} | {row['best_delta']:+.4f} | {label} | {row['configurations_above_baseline']}/16 |"
        )
    lines += [
        "",
        "![All MDP configurations versus vanilla LRU](vs_vanilla_lru.png)",
        "",
        "## All MDP deltas",
        "",
        "Values are corrected IB mean return minus the same-task vanilla LRU mean. All runs have residuals enabled except the historical vanilla reference.",
    ]
    kinds = ("forward_kl", "reverse_kl", "symmetric_kl", "renyi_half")
    columns = [(p, s) for p in ("output", "state") for s in (20, 80)]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), layout="constrained")
    for task, ax in zip(tasks, axes.ravel()):
        group = [r for r in comparisons if r["task"] == task and r["mode"] == "mdp"]
        lookup = {
            (r["divergence"], r["posterior_input"], round(r["prior_share"] * 100)): r
            for r in group
        }
        values = np.array(
            [
                [
                    lookup[kind, posterior, share]["delta_vs_vanilla_lru"]
                    for posterior, share in columns
                ]
                for kind in kinds
            ]
        )
        limit = max(np.abs(values).max(), 1e-6)
        img = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit)
        for y in range(4):
            for x in range(4):
                ax.text(
                    x,
                    y,
                    f"{values[y, x]:+.3f}",
                    ha="center",
                    va="center",
                    color="white" if abs(values[y, x]) > 0.65 * limit else "black",
                )
        ax.set_xticks(range(4), [f"{p}\nprior {s}%" for p, s in columns])
        ax.set_yticks(range(4), ["Forward KL", "Reverse KL", "Symmetric KL", "Rényi-½"])
        ax.set_title(f"{task} · vanilla LRU = {baselines[task]['evaluation_mean']:.4f}")
        fig.colorbar(img, ax=ax, shrink=0.8, label="IB − vanilla LRU")
        lines += [
            "",
            f"### {task}",
            "",
            "| Divergence | Output, prior20 | Output, prior80 | State, prior20 | State, prior80 |",
            "|---|---:|---:|---:|---:|",
        ]
        for kind, row_values in zip(kinds, values):
            lines.append(
                "| " + " | ".join([kind, *(f"{v:+.4f}" for v in row_values)]) + " |"
            )
    fig.suptitle(
        "Corrected posterior-head IB versus historical vanilla LRU · MDP only\nSeed 0; 98,304 steps. IB enables residuals; historical vanilla LRU does not.\nColor scales differ by task; blue = higher return, red = lower return."
    )
    fig.savefig(reports / "vs_vanilla_lru.png", dpi=160)
    fig.savefig(reports / "vs_vanilla_lru.pdf")
    plt.close(fig)
    lines += [
        "",
        "## POMDP coverage",
        "",
        (
            "New POMDP baselines are available in the [consolidated vanilla LRU archive](../../vanilla_lru_baselines_20261109/RESULTS.md). The comparison CSV now includes both observation modes with their own baselines. No MDP baseline is substituted for POMDP."
            if len(pomdp_baselines) == 4
            else "POMDP baseline coverage is incomplete; missing references remain blank in the CSV."
        ),
    ]
    (reports / "vs_vanilla_lru.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    from popgym_arcade.experiments.report_posterior_sweep import load_and_audit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    rows, _, audit = load_and_audit(args.output)
    if audit["errors"]:
        raise ValueError(audit["errors"])
    summary = compare_previous_lru(args.output, rows, args.reference)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
