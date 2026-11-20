"""Render per-checkpoint recall heatmaps after density_analysis_memfix."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "root", nargs="?", type=Path, default=Path("results/recall_density_16")
    )
    root = parser.parse_args().root
    rows = json.loads((root / "manifest.json").read_text())
    report = root / "REPORT.md"
    text = report.read_text().split("\n## Checkpoint heatmaps")[0]
    text += "\n## Checkpoint heatmaps\n\nLabels retain source suite, mode, variant, quantization, and training seed. Each row is one saved checkpoint; copied checkpoints are not independent training seeds.\n\n"
    for task in sorted({r["task"] for r in rows}):
        selected = [r for r in rows if r["task"] == task and r["status"] == "complete"]
        if not selected:
            continue
        labels = [
            r["checkpoint"].removeprefix("results/").removesuffix("/model.eqx")
            for r in selected
        ]
        fig, ax = plt.subplots(figsize=(18, max(3, len(selected) * 0.22)))
        image = ax.imshow(
            np.array([r["bin_mean"] for r in selected]),
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="magma",
        )
        ax.set_yticks(range(len(labels)), labels, fontsize=6)
        ax.set_xticks(range(16), [str(i + 1) for i in range(16)])
        ax.set_xlabel("Normalized-time bin (1 = oldest; 16 = latest)")
        ax.set_title(task + ": PPO recall, eight evaluation samples/checkpoint")
        fig.colorbar(image, ax=ax, label="Normalized gradient mass")
        fig.tight_layout()
        fig.savefig(root / f"{task}.png", dpi=160)
        plt.close(fig)
        text += f"[{task} — {len(selected)} checkpoints]({task}.png)\n\n"
    text += "Prediction and FSQ checkpoints are available only for CartPoleHard. Other tasks have Gaussian-divergence interventions. Historical prior-head IB checkpoints and non-KL divergence variants remain individually identified by their saved configuration in manifest.json; they are not equivalent to the current default reverse-KL preset. No completed million-step model.eqx files were available.\n"
    report.write_text(text)


if __name__ == "__main__":
    main()
