"""Multi-task CUDA comparison of LRU and Gaussian divergence bottlenecks."""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

from popgym_arcade.experiments.memory_bias import run_one, upstream_config
from popgym_arcade.experiments.memory_models import IB_DIVERGENCES, KL_BALANCES

# Keep the original suite and its reports unchanged.
IB_DIVERGENCES = {k: v for k, v in IB_DIVERGENCES.items() if k not in KL_BALANCES}

TASKS = ("CartPoleHard", "NavigatorEasy", "MineSweeperEasy", "BreakoutEasy")
CONDITIONS = ("mlp", "lru", *IB_DIVERGENCES)
LABELS = ("MLP", "LRU", "Forward KL", "Reverse KL", "Symmetric KL", "Rényi ½")


def summarize(output, tasks, seeds, *, plot=False):
    rows = []
    for task in tasks:
        for seed in seeds:
            reference_path = output / task / f"mlp_seed{seed}" / "result.json"
            baseline_path = output / task / f"lru_seed{seed}" / "result.json"
            reference = (
                json.loads(reference_path.read_text())["evaluation_mean"]
                if reference_path.exists()
                else None
            )
            baseline = (
                json.loads(baseline_path.read_text())["evaluation_mean"]
                if baseline_path.exists()
                else None
            )
            for variant in CONDITIONS:
                path = output / task / f"{variant}_seed{seed}" / "result.json"
                if not path.exists():
                    continue
                result = json.loads(path.read_text())
                returns = np.load(path.with_name("evaluation_returns.npy"))
                if (
                    returns.shape != (result["evaluation_episodes"],)
                    or not np.isfinite(returns).all()
                    or not np.isclose(returns.mean(), result["evaluation_mean"])
                ):
                    raise ValueError(f"Invalid evaluation artifact: {path}")
                score = result["evaluation_mean"]
                rows.append(
                    {
                        "task": task,
                        "variant": variant,
                        "seed": seed,
                        "steps": result["actual_steps"],
                        "score": score,
                        "episode_se": result["evaluation_standard_error"],
                        "memory_bias": reference - score
                        if reference is not None
                        else None,
                        "bias_reduction_vs_lru": score - baseline
                        if baseline is not None
                        else None,
                        "divergence": IB_DIVERGENCES.get(variant),
                        "reused": result.get("reused_from", ""),
                    }
                )
    if not rows:
        return
    with (output / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Multi-task divergence pilot",
        "",
        f"{len(tasks)} task families, fully observed mode, training seeds {list(seeds)}. Completed {len(rows)}/{len(tasks) * len(seeds) * len(CONDITIONS)} runs. Scores are held-out stochastic-policy episode returns; scores are not averaged across different tasks.",
        "",
        "| Task | " + " | ".join(LABELS) + " |",
        "|---|" + "---:|" * len(CONDITIONS),
    ]
    for task in tasks:
        values = []
        for variant in CONDITIONS:
            selected = [
                r for r in rows if r["task"] == task and r["variant"] == variant
            ]
            values.append(
                f"{np.mean([r['score'] for r in selected]):.4f}"
                if len(selected) == len(seeds)
                else "pending"
            )
        lines.append("| " + task + " | " + " | ".join(values) + " |")
    lines += [
        "",
        "Memory bias (MLP score − condition score; lower is better):",
        "",
        "| Task | " + " | ".join(LABELS[1:]) + " |",
        "|---|" + "---:|" * (len(CONDITIONS) - 1),
    ]
    for task in tasks:
        values = []
        for variant in CONDITIONS[1:]:
            selected = [
                r
                for r in rows
                if r["task"] == task
                and r["variant"] == variant
                and r["memory_bias"] is not None
            ]
            values.append(
                f"{np.mean([r['memory_bias'] for r in selected]):+.4f}"
                if len(selected) == len(seeds)
                else "pending"
            )
        lines.append("| " + task + " | " + " | ".join(values) + " |")
    complete = []
    for task in tasks:
        means = {}
        for variant in CONDITIONS:
            selected = [
                r for r in rows if r["task"] == task and r["variant"] == variant
            ]
            if len(selected) == len(seeds):
                means[variant] = float(np.mean([r["score"] for r in selected]))
        if len(means) == len(CONDITIONS):
            complete.append((task, means))
    if complete:
        lines += [
            "",
            "Observed comparisons (point estimates, not significance tests):",
            "",
        ]
        for task, means in complete:
            best_ib = max(IB_DIVERGENCES, key=lambda variant: means[variant])
            delta = means[best_ib] - means["lru"]
            lines.append(
                f"- {task}: best observed IB is {IB_DIVERGENCES[best_ib]} (score {means[best_ib]:.4f}), a score change of {delta:+.4f} versus LRU."
            )
        counts = {
            IB_DIVERGENCES[variant]: sum(
                means[variant] > means["lru"] for _, means in complete
            )
            for variant in IB_DIVERGENCES
        }
        lines += [
            "",
            "Observed wins over LRU on completed tasks: "
            + "; ".join(
                f"{kind}: {count}/{len(complete)}" for kind, count in counts.items()
            )
            + ". No raw-score average across tasks is used.",
            "",
            "Negative memory bias means the condition outperformed the MLP reference. A positive memory deficit is not present in every task's baseline; reductions in the signed gap are equivalent to improvements in score versus LRU.",
        ]
    lines += [
        "",
        "All agents receive `(previous action, previous reward, current observation)`. Every bottleneck head receives `(input, prior mean, prior log variance)`, with no latent sampling. Each actor/critic Gaussian has 1,365 coordinates (floor(2/3 × 2,048 real scalar LRU-state dimensions)). The posterior uses the current full LRU state. All bottlenecks use `max(1, divergence_bits)`, summed over the full Gaussian event, averaged over samples, then summed across actor and critic with coefficient 1. Symmetric KL is the average of both KL directions. Rényi-½ is twice the Bhattacharyya distance. All have exact closed forms; no Monte Carlo divergence estimator is used.",
        "",
        "This is a 100K-requested-step pilot (98,304 actual steps per model), with upstream PPO hyperparameters, 64 held-out episodes per model, and a learning-rate schedule spanning the pilot. Conditional episode SEs are in summary.csv; they do not capture variability across training seeds. Divergence values have different geometries despite common bit units and coefficient; regularization strengths are not locally calibrated. No coefficient tuning was performed. All IB policies have no recurrent-state inference path; the memory is auxiliary during training. Model sizes differ between LRU and IB, while all four IB variants have identical architectures and initial parameters.",
        "",
        "CartPole MLP, LRU, and forward-KL results may be reused from the earlier matching pilot, with original source hashes and provenance retained. Other conditions are freshly trained. Task-specific episode horizons are used for evaluation. Per-run configs, logs, scores, and local weights are retained under each task directory.",
        "",
        "[Divergence definitions](https://arxiv.org/abs/1206.2459) · [Experiment instructions](../../DIVERGENCE_SUITE.md) · [CSV](summary.csv)",
    ]
    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            math.ceil(len(tasks) / 2),
            2,
            figsize=(13, 4 * math.ceil(len(tasks) / 2)),
            squeeze=False,
            layout="constrained",
        )
        for ax, task in zip(axes.ravel(), tasks):
            for i, variant in enumerate(CONDITIONS):
                selected = [
                    r for r in rows if r["task"] == task and r["variant"] == variant
                ]
                if len(selected) != len(seeds):
                    continue
                scores = np.array([r["score"] for r in selected])
                se = (
                    selected[0]["episode_se"]
                    if len(seeds) == 1
                    else scores.std(ddof=1) / np.sqrt(len(scores))
                )
                ax.bar(i, scores.mean(), yerr=se, capsize=3, color=f"C{i}")
            ax.set_xticks(range(len(CONDITIONS)), LABELS, rotation=20, ha="right")
            ax.set_title(task)
            ax.set_ylabel("Held-out episode return ± SE")
        for ax in axes.ravel()[len(tasks) :]:
            ax.set_visible(False)
        fig.suptitle("Gaussian divergence comparison · 100K-step pilot · preliminary")
        fig.savefig(output / "comparison.png", dpi=180)
        plt.close(fig)
        lines += ["", "![Per-task evaluation](comparison.png)"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(
        f"Completed {len(rows)}/{len(tasks) * len(seeds) * len(CONDITIONS)} task/condition/seed runs",
        flush=True,
    )


def reuse_cartpole(output, tasks, args):
    if "CartPoleHard" not in tasks or args.timesteps != 100000 or 0 not in args.seeds:
        return
    for variant in ("mlp", "lru", "information_bottleneck"):
        source = Path("results/memory_bias_pilot") / f"{variant}_seed0"
        target = output / "CartPoleHard" / source.name
        if not (source / "result.json").exists() or target.exists():
            continue
        result = json.loads((source / "result.json").read_text())
        if (
            result["actual_steps"] != 98304
            or result["hidden"] != 512
            or result["layers"] != 2
            or result["evaluation_episodes"] != 64
        ):
            raise ValueError(f"Earlier pilot settings differ: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        result["reused_from"] = str(source.resolve())
        result["original_result_sha256"] = hashlib.sha256(
            (source / "result.json").read_bytes()
        ).hexdigest()
        (target / "result.json").write_text(json.dumps(result, indent=2))
        print(f"Reusing matching {source}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--output", type=Path, default=Path("results/divergence_suite"))
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        summarize(args.output, args.tasks, args.seeds, plot=True)
        return
    if jax.default_backend() != "gpu":
        raise RuntimeError("CUDA GPU required for this experiment")
    manifest = {
        "tasks": args.tasks,
        "conditions": list(CONDITIONS),
        "seeds": args.seeds,
        "requested_steps": args.timesteps,
        "device": jax.devices()[0].device_kind,
    }
    manifest_path = args.output / "suite.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing suite settings differ; use a fresh output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    reuse_cartpole(args.output, args.tasks, args)
    defaults = upstream_config()
    for task in args.tasks:
        config = defaults | {
            "ENV_NAME": task,
            "PARTIAL": False,
            "TOTAL_TIMESTEPS": args.timesteps,
            "WANDB_MODE": "disabled",
        }
        options = SimpleNamespace(
            timesteps=args.timesteps,
            output=args.output / task,
            posterior_input="state",
            lru_residual=False,
            ib_normalization="none",
            hidden=512,
            layers=2,
            dropout=0.1,
            weight_decay=1e-4,
            ib_coefficient=1.0,
            eval_episodes=64,
        )
        for seed in args.seeds:
            for variant in CONDITIONS:
                run_one(options, variant, seed, config | {"SEED": seed})
                summarize(args.output, args.tasks, args.seeds)
    summarize(args.output, args.tasks, args.seeds, plot=True)


if __name__ == "__main__":
    main()
