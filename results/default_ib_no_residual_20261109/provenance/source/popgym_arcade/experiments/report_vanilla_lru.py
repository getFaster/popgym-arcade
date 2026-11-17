"""Audit consolidated MDP/POMDP vanilla LRU artifacts and compare saved IB scores."""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

TASKS = ("CartPoleHard", "NavigatorEasy", "MineSweeperEasy", "BreakoutEasy")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def report(root, suite):
    rows, deltas, checks, artifacts = [], [], [], []
    cached = []
    if (root / "vs_corrected_ib.csv").exists():
        with (root / "vs_corrected_ib.csv").open() as stream:
            cached = list(csv.DictReader(stream))
    for task, mode in itertools.product(TASKS, ("mdp", "pomdp")):
        path = root / task / mode / "lru_seed0" / "result.json"
        if not path.exists():
            raise ValueError(f"Missing vanilla LRU baseline: {path}")
        baseline = json.loads(path.read_text())
        scores = np.load(path.with_name("evaluation_returns.npy"))
        logs = [
            json.loads(line)
            for line in path.with_name("training.jsonl").read_text().splitlines()
        ]
        valid = (
            baseline["variant"] == "lru"
            and baseline["config"]["ENV_NAME"] == task
            and baseline["config"]["PARTIAL"] == (mode == "pomdp")
            and baseline.get("lru_residual", False) is False
            and baseline.get("head_inputs", "input_t, projected_lru_output")
            == "input_t, projected_lru_output"
            and baseline["ib_coefficient"] == 0
            and baseline["weight_decay"] == 0
            and baseline["state_dropout"] == 0
            and baseline.get("normalization") is None
            and baseline["actual_steps"] == 98304
            and baseline["seed"] == 0
            and scores.shape == (64,)
            and np.isfinite(scores).all()
            and np.isclose(scores.mean(), baseline["evaluation_mean"])
            and np.isclose(scores.std(ddof=1), baseline["evaluation_std"])
            and np.isclose(
                scores.std(ddof=1) / 8, baseline["evaluation_standard_error"]
            )
            and [r["steps"] for r in logs] == list(range(2048, 98305, 2048))
            and path.with_name("model.eqx").exists()
        )
        if not valid:
            raise ValueError(f"Invalid vanilla LRU artifacts: {path}")
        if not all(
            np.isfinite(
                [r[k] for k in ("loss", "policy_loss", "value_loss", "entropy")]
            ).all()
            for r in logs
        ):
            raise ValueError(f"Nonfinite training: {path}")
        candidates = []
        for ib_path in sorted((suite / task / mode).glob("*/*/result.json")):
            ib = json.loads(ib_path.read_text())
            for field in (
                "config",
                "seed",
                "actual_steps",
                "requested_steps",
                "evaluation_episodes",
                "hidden",
                "layers",
                "jax_version",
            ):
                if baseline[field] != ib[field]:
                    raise ValueError(f"Unmatched {field}: {ib_path}")
            if baseline["source_sha256"]["uv.lock"] != ib["source_sha256"]["uv.lock"]:
                raise ValueError(f"Different lock file: {ib_path}")
            candidates.append(
                {
                    "task": task,
                    "mode": mode,
                    "variant": ib["variant"],
                    "posterior_input": ib["posterior_input"],
                    "ib_return": ib["evaluation_mean"],
                    "ib_episode_se": ib["evaluation_standard_error"],
                    "vanilla_lru_return": baseline["evaluation_mean"],
                    "vanilla_lru_episode_se": baseline["evaluation_standard_error"],
                    "ib_minus_lru": ib["evaluation_mean"] - baseline["evaluation_mean"],
                    "ib_device": ib["device_kind"],
                    "lru_device": baseline["device_kind"],
                    "ib_result": str(ib_path),
                    "lru_result": str(path),
                }
            )
        if not candidates:
            for record in cached:
                if record["task"] == task and record["mode"] == mode:
                    record = dict(record)
                    for field in (
                        "ib_return",
                        "ib_episode_se",
                        "vanilla_lru_return",
                        "vanilla_lru_episode_se",
                        "ib_minus_lru",
                    ):
                        record[field] = float(record[field])
                    if not np.isclose(
                        record["vanilla_lru_return"], baseline["evaluation_mean"]
                    ):
                        raise ValueError(
                            f"Cached comparison baseline mismatch: {task}/{mode}"
                        )
                    record["lru_result"] = str(path)
                    candidates.append(record)
        if len(candidates) not in (0, 16):
            raise ValueError(f"Expected 0 or 16 corrected IB cells: {task}/{mode}")
        best = max(candidates, key=lambda r: r["ib_return"], default=None)
        deltas.extend(candidates)
        rows.append(
            {
                "task": task,
                "mode": mode,
                "mean_return": baseline["evaluation_mean"],
                "episode_se": baseline["evaluation_standard_error"],
                "actual_steps": baseline["actual_steps"],
                "evaluation_episodes": 64,
                "seed": 0,
                "seconds": baseline["seconds"],
                "device": baseline["device_kind"],
                "best_ib_return": best["ib_return"] if best else None,
                "best_ib_episode_se": best["ib_episode_se"] if best else None,
                "best_ib_variant": best["variant"] if best else None,
                "best_ib_posterior_input": best["posterior_input"] if best else None,
                "best_ib_minus_lru": best["ib_minus_lru"] if best else None,
                "ib_configs_above_lru": sum(r["ib_minus_lru"] > 0 for r in candidates),
                "ib_configs_available": len(candidates),
                "run_directory": str(path.parent.relative_to(root)),
            }
        )
        checks.append(
            {
                "task": task,
                "mode": mode,
                "status": "passed",
                "source_sha256": baseline["source_sha256"],
            }
        )
        for artifact in sorted(path.parent.iterdir()):
            if artifact.is_file():
                artifacts.append(
                    {
                        "path": str(artifact.relative_to(root)),
                        "bytes": artifact.stat().st_size,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                )
    write_csv(root / "summary.csv", rows)
    write_csv(root / "vs_corrected_ib.csv", deltas)
    provenance = root / "provenance"
    provenance.mkdir(exist_ok=True)
    (provenance / "artifact_manifest.json").write_text(
        json.dumps(artifacts, indent=2) + "\n"
    )
    (root / "audit.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "completed": len(rows),
                "expected": 8,
                "evaluation_episodes_total": 512,
                "runs": checks,
            },
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# Vanilla LRU baselines: MDP and POMDP",
        "",
        "**8/8 runs complete and audited.** Four tasks × two observation modes. All runs use seed 0, 100,000 requested / 98,304 actual training steps, and 64 held-out stochastic-policy evaluation episodes.",
        "",
        "[Summary CSV](summary.csv) · [All 128 IB comparisons](vs_corrected_ib.csv) · [Audit](audit.json) · [Artifact checksums](provenance/artifact_manifest.json) · [MDP import provenance](provenance/mdp_imports.json)",
        "",
        "## Scores and corrected IB comparisons",
        "",
        "± denotes episode SE conditional on one trained seed. Best IB scores select among 16 configurations and are exploratory.",
        "",
        "| Task | Mode | Vanilla LRU ± SE | Best corrected IB ± SE | Best IB − LRU | IB configs above LRU |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        best_text = (
            f"{r['best_ib_return']:.4f} ± {r['best_ib_episode_se']:.4f}"
            if r["best_ib_return"] is not None
            else "unavailable"
        )
        delta_text = (
            f"{r['best_ib_minus_lru']:+.4f}"
            if r["best_ib_minus_lru"] is not None
            else "unavailable"
        )
        lines.append(
            f"| {r['task']} | {r['mode'].upper()} | [{r['mean_return']:.4f} ± {r['episode_se']:.4f}]({r['run_directory']}/result.json) | {best_text} | {delta_text} | {r['ib_configs_above_lru']}/{r['ib_configs_available']} |"
        )
    lines += [
        "",
        "## Model and comparison details",
        "",
        "Vanilla LRU uses two 512-coordinate complex LRU layers and current input plus a 256-dimensional projected LRU output at both RL heads. There is no added inter-layer residual connection, Gaussian bottleneck, RMS normalization, weight decay, or state dropout. PPO settings match the corrected IB sweep. The prior/posterior head bug did not affect the vanilla branch.",
        "",
        "IB models enable added residual connections and larger posterior-statistics heads, so these comparisons do not isolate regularization. Device provenance appears in the CSV. Episode uncertainty does not measure training-seed variability.",
        "",
        "## Self-contained artifact layout",
        "",
        "```text",
        "Task/",
        "  mdp/lru_seed0/    # historical MDP reference, copied byte-for-byte",
        "  pomdp/lru_seed0/  # new POMDP reference",
        "provenance/        # source snapshot, original execution logs/manifests, import records",
        "```",
        "",
        "Each run contains its checkpoint, evaluation returns, config, result metadata, and full training log. All eight baseline runs are physically stored here; older result directories are not needed to retain or audit them. Original paths inside archived logs/scripts or `reused_from` metadata are historical provenance. The copied CartPole MDP result is one training run, not an additional seed; MineSweeper MDP evaluation was recovered from its checkpoint.",
        "",
        "The comparison CSV stores all IB scores, SEs, devices, and deltas, so it remains readable after the source IB results are removed. Its `ib_result` column records historical source locations, not bundled IB checkpoints. Report regeneration uses those IB artifacts when present and falls back to the cached comparison CSV when they are unavailable.",
        "",
        "Regenerate:",
        "",
        "```bash",
        f"JAX_PLATFORMS=cpu uv run python -m popgym_arcade.experiments.report_vanilla_lru {root}",
        "```",
        "",
        "Training and sync scripts under `provenance/` are archived execution records and retain their original paths; they are not active launchers.",
    ]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "completed": len(rows),
                "comparisons": len(deltas),
                "report": str(root / "RESULTS.md"),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--suite", type=Path, default=Path("results/posterior_rms_suite_20261109")
    )
    args = parser.parse_args()
    report(args.output, args.suite)


if __name__ == "__main__":
    main()
