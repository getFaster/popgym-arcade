"""Render saved pilot scores and training curves without retraining."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from popgym_arcade.experiments.memory_models import DEFAULT_IB_VARIANT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    files = list(args.output.glob("*/result.json"))
    if not files:
        parser.error("No completed result.json files")
    results = [json.loads(path.read_text()) for path in files]
    ib_variant = (
        DEFAULT_IB_VARIANT
        if any(r["variant"] == DEFAULT_IB_VARIANT for r in results)
        else "information_bottleneck"
    )
    order = ["mlp", "lru", "weight_decay", "state_dropout", ib_variant]
    labels = [
        "MLP reference",
        "LRU",
        "LRU + weight decay",
        "LRU + state dropout",
        "LRU + IB",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for variant, label in zip(order, labels):
        selected = [r for r in results if r["variant"] == variant]
        if not selected:
            continue
        means = np.array([r["evaluation_mean"] for r in selected])
        # Display episode-level SE only for a single training seed.
        uncertainty = (
            selected[0]["evaluation_standard_error"]
            if len(selected) == 1
            else means.std(ddof=1) / np.sqrt(len(selected))
        )
        axes[0].bar(label, means.mean(), yerr=uncertainty, capsize=4)
        for path in args.output.glob(f"{variant}_seed*/training.jsonl"):
            records = [json.loads(line) for line in path.read_text().splitlines()]
            window = 5
            xs, ys = [], []
            for i in range(0, len(records), window):
                group = records[i : i + window]
                episodes = sum(r["episodes"] for r in group)
                if episodes:
                    xs.append(group[-1]["steps"])
                    ys.append(
                        sum((r["training_return"] or 0) * r["episodes"] for r in group)
                        / episodes
                    )
            axes[1].plot(xs, ys, label=label)
    axes[0].set_ylabel("Held-out mean episode return")
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[0].set_title("Evaluation (error bars: SE)")
    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylabel("Training episode return")
    axes[1].set_title("Five-update windows, weighted by completed episodes")
    axes[1].legend(fontsize=8)
    fig.suptitle("CartPoleHard MDP · 100K-step pilot · preliminary")
    fig.savefig(args.output / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
