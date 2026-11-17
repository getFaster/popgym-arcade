"""Audit and organize the corrected posterior-head factorial sweep without retraining."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASKS = ("CartPoleHard", "NavigatorEasy", "MineSweeperEasy", "BreakoutEasy")
KINDS = ("forward_kl", "reverse_kl", "symmetric_kl", "renyi_half")
LABELS = ("Forward KL", "Reverse KL", "Symmetric KL", "Rényi-½")
COLUMNS = tuple(itertools.product(("output", "state"), (20, 80)))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_and_audit(root):
    rows, errors, sources, configs = [], [], set(), set()
    calibration_rows = {}
    for path in sorted(root.glob("*/*/*/*/result.json")):
        result = json.loads(path.read_text())
        task, mode, posterior, run = path.relative_to(root).parts[:4]
        kind = result["divergence"]
        prior = round(result["kl_prior_gradient_weight"] * 50)
        scores = np.load(path.with_name("evaluation_returns.npy"))
        records = [
            json.loads(line)
            for line in path.with_name("training.jsonl").read_text().splitlines()
        ]
        calibration = result["rms_calibration"]
        stored_calibration = json.loads(
            path.parent.parent.joinpath("calibration.json").read_text()
        )
        checks = {
            "expected task/mode/input": task in TASKS
            and mode in ("mdp", "pomdp")
            and posterior in ("output", "state"),
            "path matches metadata": task == result["config"]["ENV_NAME"]
            and (mode == "pomdp") == result["config"]["PARTIAL"]
            and posterior == result["posterior_input"]
            and run == f"{result['variant']}_seed0",
            "posterior head": result["head_inputs"]
            == "input_t, posterior_mean, posterior_log_variance",
            "fixed RMS normalization": result["normalization"]
            == "fixed_calibration_rms_excess_above_one_bit",
            "residual enabled": result["lru_residual"] is True,
            "pilot settings": result["seed"] == 0
            and result["requested_steps"] == 100000
            and result["actual_steps"] == 98304
            and result["evaluation_episodes"] == 64
            and result["ib_free_bits"] == 1
            and result["ib_coefficient"] == 1,
            "model dimensions": result["hidden"] == 512
            and result["layers"] == 2
            and result["gaussian_dimension_per_branch"] == 1365,
            "balancing": prior in (20, 80)
            and np.isclose(
                result["kl_posterior_gradient_weight"],
                2 - result["kl_prior_gradient_weight"],
            ),
            "evaluation samples": scores.shape == (64,) and np.isfinite(scores).all(),
            "evaluation summaries": np.isclose(scores.mean(), result["evaluation_mean"])
            and np.isclose(scores.std(ddof=1), result["evaluation_std"])
            and np.isclose(scores.std(ddof=1) / 8, result["evaluation_standard_error"]),
            "complete training": len(records) == 48
            and [x["steps"] for x in records] == list(range(2048, 98305, 2048)),
            "finite training": all(
                np.isfinite(
                    [
                        x[k]
                        for k in (
                            "loss",
                            "policy_loss",
                            "value_loss",
                            "entropy",
                            "ib_loss_normalized",
                            "actor_divergence_nats",
                            "critic_divergence_nats",
                        )
                    ]
                ).all()
                for x in records
            ),
            "checkpoint present": path.with_name("model.eqx").is_file(),
            "shared calibration": calibration == stored_calibration
            and result["rms_scales_bits"] == calibration["scales_bits"][kind],
            "calibration settings": calibration["env"] == task
            and calibration["partial_obs"] == (mode == "pomdp")
            and calibration["posterior_input"] == posterior
            and calibration["seed"] == 0
            and calibration["residual"]
            and calibration["samples"] == 2048,
        }
        for check, passed in checks.items():
            if not passed:
                errors.append(f"{path.relative_to(root)}: {check}")
        sources.add(json.dumps(result["source_sha256"], sort_keys=True))
        configs.add(
            json.dumps(
                {
                    k: v
                    for k, v in result["config"].items()
                    if k not in ("ENV_NAME", "PARTIAL")
                },
                sort_keys=True,
            )
        )
        for divergence, scales in calibration["scales_bits"].items():
            if not np.isfinite(scales).all() or min(scales) <= 0:
                errors.append(f"Invalid calibration scales: {path}")
            calibration_rows[(task, mode, posterior, divergence)] = {
                "task": task,
                "mode": mode,
                "posterior_input": posterior,
                "divergence": divergence,
                "actor_rms_bits": scales[0],
                "critic_rms_bits": scales[1],
                "samples": calibration["samples"],
                "batch_sha256": calibration["batch_sha256"],
            }
        rows.append(
            {
                "task": task,
                "mode": mode,
                "posterior_input": posterior,
                "divergence": kind,
                "prior_share": prior / 100,
                "posterior_share": 1 - prior / 100,
                "seed": result["seed"],
                "mean_return": result["evaluation_mean"],
                "episode_std": result["evaluation_std"],
                "episode_se": result["evaluation_standard_error"],
                "evaluation_episodes": 64,
                "actual_steps": result["actual_steps"],
                "seconds": result["seconds"],
                "device": result["device_kind"],
                "parameters": result["parameter_count"],
                "actor_rms_bits": result["rms_scales_bits"][0],
                "critic_rms_bits": result["rms_scales_bits"][1],
                "final_normalized_ib_loss": records[-1]["ib_loss_normalized"],
                "final_actor_divergence_nats": records[-1]["actor_divergence_nats"],
                "final_critic_divergence_nats": records[-1]["critic_divergence_nats"],
                "run_directory": str(path.parent.relative_to(root)),
            }
        )
    expected = set(
        itertools.product(TASKS, ("mdp", "pomdp"), ("output", "state"), KINDS, (20, 80))
    )
    observed = [
        (
            r["task"],
            r["mode"],
            r["posterior_input"],
            r["divergence"],
            round(r["prior_share"] * 100),
        )
        for r in rows
    ]
    if len(observed) != len(set(observed)):
        errors.append("Duplicate factorial cells")
    if set(observed) != expected:
        errors.append(
            f"Missing cells: {sorted(expected - set(observed))}; unexpected cells: {sorted(set(observed) - expected)}"
        )
    if len(sources) != 1 or len(configs) != 1:
        errors.append("Inconsistent training sources/lock or PPO configurations")
    audit = {
        "status": "passed" if not errors else "failed",
        "completed_runs": len(rows),
        "expected_runs": 128,
        "evaluation_episodes_total": sum(r["evaluation_episodes"] for r in rows),
        "training_steps_total": sum(r["actual_steps"] for r in rows),
        "errors": errors,
        "source_hashes": [json.loads(s) for s in sorted(sources)],
        "checked": [
            "complete unique factorial matrix",
            "posterior head metadata",
            "residual and normalization settings",
            "identical training source/lock hashes and PPO settings",
            "48 finite PPO updates per run",
            "64 finite evaluation returns and recalculated statistics per run",
            "checkpoint presence",
            "shared frozen calibration across divergence/balancing choices",
        ],
    }
    return rows, list(calibration_rows.values()), audit


def select(rows, task, mode, posterior=None):
    return [
        r
        for r in rows
        if r["task"] == task
        and r["mode"] == mode
        and (posterior is None or r["posterior_input"] == posterior)
    ]


def heatmaps(rows, reports):
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), layout="constrained")
    for i, task in enumerate(TASKS):
        task_values = [r["mean_return"] for r in rows if r["task"] == task]
        for j, mode in enumerate(("mdp", "pomdp")):
            ax = axes[i, j]
            group = select(rows, task, mode)
            lookup = {
                (
                    r["divergence"],
                    r["posterior_input"],
                    round(r["prior_share"] * 100),
                ): r
                for r in group
            }
            values = np.array(
                [
                    [
                        lookup[kind, posterior, share]["mean_return"]
                        for posterior, share in COLUMNS
                    ]
                    for kind in KINDS
                ]
            )
            image = ax.imshow(
                values, cmap="viridis", vmin=min(task_values), vmax=max(task_values)
            )
            for y, x in itertools.product(range(4), repeat=2):
                rgba = image.cmap(image.norm(values[y, x]))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(
                    x,
                    y,
                    f"{values[y, x]:.3f}",
                    ha="center",
                    va="center",
                    color="black" if luminance > 0.55 else "white",
                )
            ax.set_xticks(range(4), [f"{p}\nprior {s}%" for p, s in COLUMNS])
            ax.set_yticks(range(4), LABELS)
            ax.set_title(f"{task} · {mode.upper()}")
            fig.colorbar(image, ax=ax, shrink=0.8, label="Mean evaluation return")
    fig.suptitle(
        "Posterior-head sweep · seed 0 · 98,304 training steps\nResiduals in both inputs; frozen RMS normalization. Colors shared within each task."
    )
    fig.savefig(reports / "comparison.png", dpi=160)
    fig.savefig(reports / "comparison.pdf")
    plt.close(fig)


def task_report(root, reports, task, rows):
    lines = [
        f"# {task}: corrected posterior-head sweep",
        "",
        "[Suite overview](../RESULTS.md) · [Heatmap](comparison.png) · [Vanilla LRU comparison](vs_vanilla_lru.md)",
        "",
        "Scores are mean evaluation return ± episode SE over 64 episodes, conditional on training seed 0. Prior shares 20%/80% give auxiliary prior/posterior gradient weights 0.4/1.6 and 1.6/0.4. Both memory inputs enable residuals. Highest values are descriptive selections.",
    ]
    for mode in ("mdp", "pomdp"):
        lines += [
            "",
            f"## {mode.upper()}",
            "",
            "| Divergence | Output, prior20 | Output, prior80 | State, prior20 | State, prior80 |",
            "|---|---:|---:|---:|---:|",
        ]
        group = select(rows, task, mode)
        for kind, label in zip(KINDS, LABELS):
            cells = []
            for posterior, share in COLUMNS:
                row = next(
                    r
                    for r in group
                    if r["divergence"] == kind
                    and r["posterior_input"] == posterior
                    and round(r["prior_share"] * 100) == share
                )
                cells.append(
                    f"[{row['mean_return']:.4f} ± {row['episode_se']:.4f}](../{row['run_directory']}/result.json)"
                )
            lines.append("| " + " | ".join([label, *cells]) + " |")
    lines += [
        "",
        "## Training diagnostics",
        "",
        "Each curve is the highest-evaluation-return configuration selected within one posterior-input/mode group. Selection uses the held-out evaluation and is exploratory. Training returns use five-update windows weighted by completed episodes; these are not held-out evaluation scores.",
        "",
        f"![Training diagnostics]({task}_training.png)",
    ]
    (reports / f"{task}.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), layout="constrained")
    for mode, posterior in itertools.product(("mdp", "pomdp"), ("output", "state")):
        row = max(select(rows, task, mode, posterior), key=lambda r: r["mean_return"])
        records = [
            json.loads(line)
            for line in (root / row["run_directory"] / "training.jsonl")
            .read_text()
            .splitlines()
        ]
        xs, ys = [], []
        for start in range(0, len(records), 5):
            group = records[start : start + 5]
            episodes = sum(r["episodes"] for r in group)
            if episodes:
                xs.append(group[-1]["steps"])
                ys.append(
                    sum((r["training_return"] or 0) * r["episodes"] for r in group)
                    / episodes
                )
        label = f"{mode}/{posterior}: {row['divergence']}, prior{round(row['prior_share'] * 100)}"
        axes[0].plot(xs, ys, label=label)
        steps = [r["steps"] for r in records]
        axes[1].plot(steps, [r["ib_loss_normalized"] for r in records])
        axes[2].plot(steps, [r["actor_divergence_nats"] / np.log(2) for r in records])
    for ax, title in zip(
        axes,
        (
            "Training return (episode weighted)",
            "Normalized auxiliary loss",
            "Actor divergence (raw bits)",
        ),
    ):
        ax.set_title(title)
        ax.set_xlabel("Training environment steps")
        ax.grid(alpha=0.2)
    axes[1].set_yscale("symlog", linthresh=1e-4)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{task} · selected configuration diagnostics")
    fig.savefig(reports / f"{task}_training.png", dpi=150)
    plt.close(fig)


def report(root):
    from popgym_arcade.experiments.compare_previous_lru import compare_previous_lru

    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    rows, calibration, audit = load_and_audit(root)
    (reports / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["errors"]:
        raise ValueError(f"Audit failed: {audit['errors']}")
    write_csv(root / "summary.csv", rows)
    write_csv(reports / "calibration.csv", calibration)
    best = [
        max(select(rows, task, mode, posterior), key=lambda r: r["mean_return"])
        for task, mode, posterior in itertools.product(
            TASKS, ("mdp", "pomdp"), ("output", "state")
        )
    ]
    write_csv(reports / "best_by_input.csv", best)
    contrasts = []
    for task, mode in itertools.product(TASKS, ("mdp", "pomdp")):
        group = select(rows, task, mode)
        for factor, positive, negative in (
            ("posterior_input", "state", "output"),
            ("prior_share", 0.8, 0.2),
        ):
            a = [r["mean_return"] for r in group if r[factor] == positive]
            b = [r["mean_return"] for r in group if r[factor] == negative]
            contrasts.append(
                {
                    "task": task,
                    "mode": mode,
                    "contrast": f"{positive} minus {negative}",
                    "factor": factor,
                    "mean_positive": float(np.mean(a)),
                    "mean_negative": float(np.mean(b)),
                    "difference_of_configuration_means": float(np.mean(a) - np.mean(b)),
                    "configurations_per_level": len(a),
                }
            )
    write_csv(reports / "factor_contrasts.csv", contrasts)
    heatmaps(rows, reports)
    compare_previous_lru(root, rows)
    lines = [
        "# Corrected posterior-head sweep",
        "",
        "**128/128 runs complete; artifact audit passed.** Four tasks × two observation modes × four divergences × two asymmetric balances × two posterior memory inputs. All runs use seed 0, 98,304 actual training steps (100,000 requested), and 64 held-out stochastic-policy evaluation episodes.",
        "",
        "[All 128 runs (CSV)](summary.csv) · [Consolidated vanilla LRU baselines](../vanilla_lru_baselines_20261109/RESULTS.md) · [MDP comparison plots](reports/vs_vanilla_lru.md) · [Audit](reports/audit.json) · [Calibration scales](reports/calibration.csv) · [Factor contrasts](reports/factor_contrasts.csv) · [Source snapshot](source_snapshot.tar.gz)",
        "",
        "![Evaluation returns](reports/comparison.png)",
        "",
        "## Highest observed scores by memory input",
        "",
        "These are selected maxima across eight configurations per input, not independently validated winners. ± denotes episode SE; it does not measure training-seed variability.",
        "",
        "| Task | Mode | Posterior input | Divergence | Prior/posterior shares | Return ± episode SE |",
        "|---|---|---|---|---|---:|",
    ]
    for r in best:
        label = LABELS[KINDS.index(r["divergence"])]
        lines.append(
            f"| [{r['task']}](reports/{r['task']}.md) | {r['mode'].upper()} | {r['posterior_input']} | {label} | {round(r['prior_share'] * 100)}/{round(r['posterior_share'] * 100)} | {r['mean_return']:.4f} ± {r['episode_se']:.4f} |"
        )
    lines += [
        "",
        "## Descriptive factor comparisons",
        "",
        "These differences average all matched configurations within each task/mode. They are descriptive factorial contrasts, not seed-level estimates or significance tests. Scores are never averaged across tasks.",
        "",
        "| Task | Mode | Full state − output | Prior80 − prior20 |",
        "|---|---|---:|---:|",
    ]
    for task, mode in itertools.product(TASKS, ("mdp", "pomdp")):
        selected = [c for c in contrasts if c["task"] == task and c["mode"] == mode]
        lines.append(
            f"| {task} | {mode.upper()} | {selected[0]['difference_of_configuration_means']:+.4f} | {selected[1]['difference_of_configuration_means']:+.4f} |"
        )
    lines += [
        "",
        "## Configuration and interpretation",
        "",
        "Actor and critic both consume `(previous_action, previous_reward, current_observation, posterior_mean, posterior_log_variance)` through the existing current-input encoding, without latent sampling. Residual LRU connections are enabled for both posterior memory inputs. Forward KL is KL(P‖Q), reverse KL is KL(Q‖P), symmetric KL is their average, and Rényi-½ is twice Bhattacharyya distance; P is prior and Q posterior.",
        "",
        "Prior shares 0.2/0.8 correspond to auxiliary prior/posterior gradient weights 0.4/1.6 and 1.6/0.4. Branch-specific RMS scales are calibrated on 2,048 shared random-policy samples per task/mode/memory-input group, frozen, and reused across all balancing choices. The auxiliary loss sums actor/critic sample means of `max(D_bits − 1, 0) / RMS_bits`, coefficient 1. Raw bits retain the full-event one-bit free region; normalized loss is dimensionless. Calibration equalizes initial value scales, not gradient norms.",
        "",
        "Both models use two 512-coordinate complex LRU layers and 1,365 Gaussian coordinates per branch. Output-input models have 16,407,454 parameters; full-state models have 24,794,014. Thus the input comparison also changes posterior parameter count. PPO hyperparameters, seeds, budgets, and evaluation counts are otherwise identical.",
        "",
        "MDP runs used the local RTX 2060; POMDP runs used Colab T4. Observation-mode comparisons also differ in hardware. All 128 runs share identical training-source and lock-file hashes, recorded in the audit. No historical prior-head scores or checkpoints are reused. A separate linked report compares the 64 MDP scores with previous vanilla LRU baselines; those references have no added inter-layer residuals. New POMDP vanilla LRU baselines are included in the consolidated archive. No corrected MLP reference runs are included, so this report makes no memory-bias estimates.",
        "",
        "## Artifacts and reproduction",
        "",
        "Raw checkpoints, per-episode returns, metadata, calibration, and training logs remain in `Task/{mdp,pomdp}/{output,state}/variant_seed0/`. Nothing has been moved or deleted. Per-task reports provide every score and links to the original metadata.",
        "",
        "Regenerate this report without GPU allocation:",
        "",
        "```bash",
        f"JAX_PLATFORMS=cpu uv run python -m popgym_arcade.experiments.report_posterior_sweep {root}",
        "```",
        "",
        "Plots are available as [PNG](reports/comparison.png) and [PDF](reports/comparison.pdf). Selected-configuration training diagnostics are linked in each task report. Individual evaluation episodes are not independent training seeds; the single-seed short-budget sweep supports exploratory comparisons only.",
    ]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    for task in TASKS:
        task_report(root, reports, task, rows)
    print(
        json.dumps(
            {
                "audit": audit["status"],
                "runs": len(rows),
                "report": str(root / "RESULTS.md"),
                "best": [
                    {
                        k: r[k]
                        for k in (
                            "task",
                            "mode",
                            "posterior_input",
                            "divergence",
                            "prior_share",
                            "mean_return",
                        )
                    }
                    for r in best
                ],
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report(args.output)


if __name__ == "__main__":
    main()
