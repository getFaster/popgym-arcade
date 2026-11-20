"""Plot final checkpoints of the three main MEMFIX interventions and plain LRU."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

TASKS = ["CartPoleHard", "NavigatorEasy", "MineSweeperEasy", "BreakoutEasy"]
LABELS = {
    "lru": "Plain LRU",
    "ib_reverse_prior20": "KL",
    "lru_prediction": "Prediction",
    "activation_5": "FSQ (5 levels)",
}


def matches_variant(row, variant):
    """Quantized LRU belongs only to FSQ, never to the plain control."""
    if variant == "activation_5":
        return row.get("quantization") == "activation" and row.get("levels") == 5
    if variant == "lru":
        return row["variant"] == "lru" and row.get("quantization", "none") == "none"
    return row["variant"] == variant


def selected_checkpoints(require_weights=True):
    """Use final model.eqx from the established three-seed intervention report."""
    selected = []
    for task in TASKS:
        variants = (
            list(LABELS) if task == "CartPoleHard" else ["lru", "ib_reverse_prior20"]
        )
        for variant in variants:
            for mode in ["mdp", "pomdp"]:
                seeds = [0] if variant == "lru" and task != "CartPoleHard" else range(3)
                for seed in seeds:
                    if variant == "lru" and task != "CartPoleHard":
                        directory = Path(
                            f"results/vanilla_lru_baselines_20261109/{task}/{mode}/lru_seed0"
                        )
                    elif variant == "activation_5":
                        directory = Path(
                            f"results/fsq_activation_local_20261117_2108/{mode}/activation_5/lru_seed{seed}"
                        )
                    elif seed == 0 and task == "CartPoleHard":
                        directory = Path(
                            f"results/lru_input_prediction_20261111/{task}/{mode}/{variant}_seed0"
                        )
                    elif seed == 0:
                        directory = Path(
                            f"results/posterior_rms_suite_20261109/{task}/{mode}/output/{variant}_seed0"
                        )
                    else:
                        shard = "local" if seed == 1 else "colab"
                        directory = Path(
                            f"results/memfix_short_20261119/{shard}/{task}/{mode}/{variant}_seed{seed}"
                        )
                    metadata = json.loads((directory / "config.json").read_text())
                    assert metadata["actual_steps"] == 98304
                    if require_weights:
                        assert (directory / "model.eqx").exists(), (
                            f"Checkpoint weights were pruned: {directory}; regenerate plots from saved samples instead."
                        )
                    for checkpoint in directory.glob(
                        "checkpoints/step_*/checkpoint.json"
                    ):
                        assert (
                            json.loads(checkpoint.read_text())["actual_steps"]
                            <= metadata["actual_steps"]
                        )
                    selected.append(str(directory / "model.eqx"))
    return selected


def prepare(root, source):
    from plotting.recall_queue import make_plan

    root.mkdir(parents=True, exist_ok=True)
    selected = selected_checkpoints()
    (root / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    old = {
        r["checkpoint"]: r for r in json.loads((source / "manifest.json").read_text())
    }
    plan = make_plan()
    code_hash = hashlib.sha256(
        Path("plotting/density_analysis_memfix.py").read_bytes()
    ).hexdigest()
    reused = 0
    for path in selected:
        if path not in old:
            continue
        row = old[path]
        assert row["endpoint_plan"] == plan and row["definition"] == "paper"
        checksum = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert row["sha256"] == checksum
        needs_fallback = any(
            d is None and n >= 10 for d, n in zip(row["distributions"], row["lengths"])
        )
        if needs_fallback:
            continue
        row = dict(row, reused_from=str(source / "manifest.json"))
        if not row.get("terminal_fallback"):
            row["terminal_fallback"] = True
            row["requested_endpoints"] = plan["endpoints"]
            row["endpoints"] = [
                t if d is not None else None
                for t, d in zip(plan["endpoints"], row["distributions"])
            ]
            row["window_starts"] = [
                t - 9 if t is not None else None for t in row["endpoints"]
            ]
            row["sample_status"] = [
                "defined_reused" if d is not None else "insufficient_history"
                for d in row["distributions"]
            ]
        identity = path + checksum + json.dumps(plan) + code_hash + "True"
        token = hashlib.sha256(identity.encode()).hexdigest()[:20]
        directory = root / "jobs" / token
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(json.dumps([row], indent=2) + "\n")
        reused += 1
    print(
        f"Selected {len(selected)} final checkpoints; {reused} fully reused; {len(selected) - reused} need fallback calculations."
    )


def render(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = json.loads((root / "manifest.json").read_text())
    selected = selected_checkpoints(require_weights=False)
    assert sorted(r["checkpoint"] for r in rows) == sorted(selected), (
        "Wait for all selected checkpoints."
    )
    assert all(row["status"] in ("complete", "undefined") for row in rows)
    groups = []
    for task in TASKS:
        for mode in ["MDP", "POMDP"]:
            for variant, label in LABELS.items():
                checkpoints = [
                    r
                    for r in rows
                    if r["task"] == task
                    and r["mode"] == mode
                    and matches_variant(r, variant)
                ]
                if not checkpoints:
                    continue
                checkpoints.sort(key=lambda row: row["training_seed"])
                expected_seeds = (
                    [0] if variant == "lru" and task != "CartPoleHard" else [0, 1, 2]
                )
                assert sorted(r["training_seed"] for r in checkpoints) == expected_seeds
                valid = [r for r in checkpoints if r["valid_samples"] > 0]
                values = np.asarray([r["bin_mean"] for r in valid])
                mean = values.mean(0) if len(valid) else np.full(16, np.nan)
                sd = values.std(0, ddof=1) if len(valid) > 1 else np.full(16, np.nan)
                assert not len(valid) or np.isclose(mean.sum(), 1)
                groups.append(
                    {
                        "task": task,
                        "mode": mode,
                        "intervention": label,
                        "checkpoints": len(valid),
                        "selected_checkpoints": len(checkpoints),
                        "training_seeds": [r["training_seed"] for r in checkpoints],
                        "valid_trajectories": [r["valid_samples"] for r in checkpoints],
                        "bin_mean": mean.tolist(),
                        "seed_sd": sd.tolist() if len(valid) > 1 else None,
                    }
                )
    (root / "averages.json").write_text(json.dumps(groups, indent=2) + "\n")
    with (root / "averages_16.csv").open("w") as stream:
        fields = ["task", "mode", "intervention", "checkpoints"] + [
            f"bin_{i + 1:02d}" for i in range(16)
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group in groups:
            writer.writerow(
                {
                    **{k: group[k] for k in fields[:4]},
                    **{f"bin_{i + 1:02d}": v for i, v in enumerate(group["bin_mean"])},
                }
            )
    colors = {
        "Plain LRU": "#6a51a3",
        "KL": "#3465a4",
        "Prediction": "#d46b19",
        "FSQ (5 levels)": "#238b59",
    }

    def draw(ax, task, mode):
        matches = [g for g in groups if g["task"] == task and g["mode"] == mode]
        for index, group in enumerate(matches):
            x = np.arange(16) + (index - (len(matches) - 1) / 2) * 0.25
            ax.bar(
                x,
                group["bin_mean"],
                width=0.23,
                color=colors[group["intervention"]],
                label=f"{group['intervention']} (n={group['checkpoints']} checkpoints)",
            )
            if group["checkpoints"] > 1:
                ax.errorbar(
                    x,
                    group["bin_mean"],
                    yerr=group["seed_sd"],
                    fmt="none",
                    ecolor=colors[group["intervention"]],
                    capsize=1.5,
                    linewidth=0.6,
                )
        ax.set_title(f"{task} · {mode}")
        ax.set_xticks(range(16), range(1, 17), fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Normalized window-time bin (1 oldest; 16 current)")
        ax.set_ylabel("Recall mass")
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7, loc="upper left")
        if task != "CartPoleHard":
            ax.text(
                0.02,
                0.68,
                "Prediction / FSQ: no saved checkpoints",
                transform=ax.transAxes,
                fontsize=8,
                color="#555555",
            )

    fig, axes = plt.subplots(4, 2, figsize=(15, 13), layout="constrained")
    for index, task in enumerate(TASKS):
        for col, mode in enumerate(["MDP", "POMDP"]):
            draw(axes[index, col], task, mode)
    fig.suptitle(
        "Final checkpoints · equal-weight checkpoint means\nPaper recall, 10-observation windows, 8 trajectories; whiskers = seed SD where n > 1",
        fontsize=13,
    )
    fig.savefig(root / "interventions_by_task.png", dpi=180)
    fig.savefig(root / "interventions_by_task.pdf")
    plt.close(fig)
    for task in TASKS:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), layout="constrained")
        for ax, mode in zip(axes, ["MDP", "POMDP"]):
            draw(ax, task, mode)
        fig.savefig(root / f"{task}.png", dpi=180)
        plt.close(fig)
    fallback = sum(
        s == "defined_terminal_fallback" for r in rows for s in r["sample_status"]
    )
    short = sum(s == "insufficient_history" for r in rows for s in r["sample_status"])
    text = "# Recall density: final main-intervention checkpoints\n\n[Combined graph](interventions_by_task.png) · [PDF](interventions_by_task.pdf) · [16-bin averages](averages_16.csv) · [Per-checkpoint samples](manifest.json) · [Checkpoint selection](selection.json)\n\n"
    text += f"{len(rows)} final checkpoints (98,304 training steps), one per task/mode/condition/seed. Plain LRU uses seeds 0–2 for CartPoleHard and seed 0 for the other tasks; interventions use seeds 0–2. Uses the established three-seed intervention run lineages; alternate KL configurations and intermediate snapshots are excluded. Each checkpoint averages its valid trajectory distributions, then each intervention averages checkpoint means with equal weight. Whiskers show training-seed SD, not evaluation SE; they are omitted for single-seed LRU groups.\n\n"
    text += f"Paper action-probability gradient definition, 16 bins, eight trajectories/checkpoint, ten observations per window. Shared requested endpoints: {rows[0]['endpoint_plan']['endpoints']}. If a trajectory ends early with at least ten observations, use its last pre-terminal observation as the endpoint; actual endpoints are saved separately. **{fallback} samples used this fallback; {short} samples had fewer than ten observations and remain unavailable.** No resampling or extra trajectories. Pre-window recurrent context is preserved and held fixed during differentiation; mass is normalized within the window.\n\n"
    text += "| Task | Mode | Intervention | Checkpoints | Seeds | Valid trajectories per seed |\n|---|---|---|---:|---|---|\n"
    for g in groups:
        text += f"| {g['task']} | {g['mode']} | {g['intervention']} | {g['checkpoints']} | {g['training_seeds']} | {g['valid_trajectories']} |\n"
    text += "\nPrediction and five-level post-activation FSQ are available only for CartPoleHard; missing interventions are labeled, not plotted as zero.\n\n"
    for task in TASKS:
        text += f"[{task}]({task}.png)\n\n"
    if Path("results/CHECKPOINTS.json").exists():
        text += "\nCheckpoint storage was pruned to the 36 final policies of the three main interventions, plus their matching resumable states. Plain LRU remains in these saved diagnostics and plots, but its weights have been removed. See [checkpoint inventory](../CHECKPOINTS.json). Older recall runs and superseded job records are preserved in [the diagnostic archive](../_archive/README.md).\n"
    (root / "REPORT.md").write_text(text)
    print(
        f"Replotted {len(groups)} intervention/task/mode averages; {fallback} fallback samples; {short} short samples."
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument(
        "--source", type=Path, help="Cached source results; defaults to --root"
    )
    parser.add_argument(
        "--root", type=Path, default=Path("results/recall_density_16_main_final")
    )
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root, args.source or args.root)
    else:
        render(args.root)


if __name__ == "__main__":
    main()
