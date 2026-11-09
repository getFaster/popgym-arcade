"""Compare KL-balancing runs with matching original divergence-pilot baselines."""

import argparse
import json
from pathlib import Path

import numpy as np


def report(output, reference):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = [
        "# KL-balancing pilot",
        "",
        "Gradient weights: prior-heavy 1.6/0.4; posterior-heavy 0.4/1.6; original 1/1.",
        "Episode SEs condition on each trained policy; they do not measure training-seed variability.",
        "",
        "| Task | Seed | Condition | Steps | Return ± episode SE | Original return | Change vs same-direction original |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    entries = []
    for path in sorted(output.glob("*/*/result.json")):
        result = json.loads(path.read_text())
        task, variant, seed = path.parent.parent.name, result["variant"], result["seed"]
        baseline_variant = (
            "information_bottleneck" if "forward" in variant else "ib_reverse_kl"
        )
        baseline_path = (
            reference / task / f"{baseline_variant}_seed{seed}" / "result.json"
        )
        delta = "pending"
        baseline_score = "pending"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
            for field in (
                "actual_steps",
                "hidden",
                "layers",
                "evaluation_episodes",
                "ib_coefficient",
            ):
                if result[field] != baseline[field]:
                    raise ValueError(
                        f"Mismatched {field}: {path} versus {baseline_path}"
                    )
            if result["config"] != baseline["config"]:
                raise ValueError(f"Mismatched PPO config: {path}")
            baseline_score = f"{baseline['evaluation_mean']:.4f}"
            delta = f"{result['evaluation_mean'] - baseline['evaluation_mean']:+.4f}"
        returns = np.load(path.with_name("evaluation_returns.npy"))
        if not np.isfinite(returns).all() or not np.isclose(
            returns.mean(), result["evaluation_mean"]
        ):
            raise ValueError(f"Invalid evaluation: {path}")
        score, se = result["evaluation_mean"], result["evaluation_standard_error"]
        lines.append(
            f"| {task} | {seed} | {variant} | {result['actual_steps']} | {score:.4f} ± {se:.4f} | {baseline_score} | {delta} |"
        )
        entries.append((f"{task}\n{variant}\nseed {seed}", score, se))
    if entries:
        fig, ax = plt.subplots(
            figsize=(max(8, len(entries) * 2), 5), layout="constrained"
        )
        ax.bar(
            range(len(entries)),
            [x[1] for x in entries],
            yerr=[x[2] for x in entries],
            capsize=4,
        )
        ax.set_xticks(
            range(len(entries)), [x[0] for x in entries], rotation=20, ha="right"
        )
        ax.set_ylabel("Evaluation return ± episode SE")
        fig.savefig(output / "comparison.png", dpi=160)
        plt.close(fig)
        lines += ["", "![Evaluation returns](comparison.png)"]
    else:
        lines += ["", "No completed evaluations yet."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--reference", type=Path, default=Path("results/divergence_suite")
    )
    args = parser.parse_args()
    report(args.output, args.reference)


if __name__ == "__main__":
    main()
