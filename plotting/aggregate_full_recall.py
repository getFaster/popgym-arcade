"""Aggregate complete five-episode checkpoint measurements by training seed."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def aggregate(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = (row["task"], row["mode"], row["intervention"])
        seed = row["training_seed"]
        if seed in groups[key]:
            raise ValueError(f"duplicate checkpoint for {key}, seed {seed}")
        samples = row["samples"]
        if len(samples) != 5 or len({s["trajectory_seed"] for s in samples}) != 5:
            raise ValueError(f"{key}, seed {seed}: expected five distinct trajectories")
        for sample in samples:
            dist, bins = sample["distribution"], sample["bins"]
            if (
                len(dist) != sample["length"]
                or sample["endpoint"] != len(dist) - 1
                or len(bins) != 16
                or not all(math.isfinite(v) and v >= 0 for v in dist + bins)
                or not math.isclose(sum(dist), 1, abs_tol=1e-5)
                or not math.isclose(sum(bins), 1, abs_tol=1e-5)
            ):
                raise ValueError(f"{key}, seed {seed}: invalid trajectory")
            expected = [0.0] * 16
            for position, value in zip(np.linspace(0, 1, len(dist)), dist):
                index = min(int(position * 16), 15)
                expected[index] += value
            if any(
                not math.isclose(a, b, abs_tol=1e-6) for a, b in zip(expected, bins)
            ):
                raise ValueError(f"{key}, seed {seed}: inconsistent bins")
        groups[key][seed] = [sum(s["bins"][i] for s in samples) / 5 for i in range(16)]
    output, incomplete = [], []
    for key, seeds in sorted(groups.items()):
        if set(seeds) != {0, 1, 2}:
            incomplete.append(
                {
                    "task": key[0],
                    "mode": key[1],
                    "intervention": key[2],
                    "seeds": sorted(seeds),
                }
            )
            continue
        output.append(
            {
                "task": key[0],
                "mode": key[1],
                "intervention": key[2],
                "bins": [sum(seeds[s][i] for s in range(3)) / 3 for i in range(16)],
            }
        )
    return output, incomplete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(path.read_text()) for path in sorted(args.jobs.glob("*.json"))]
    output, incomplete = aggregate(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "averages.json").write_text(json.dumps(output, indent=2) + "\n")
    (args.output / "incomplete.json").write_text(
        json.dumps(incomplete, indent=2) + "\n"
    )
    with (args.output / "averages_16.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["task", "mode", "intervention"] + [f"bin_{i:02}" for i in range(1, 17)]
        )
        writer.writerows(
            [[r["task"], r["mode"], r["intervention"], *r["bins"]] for r in output]
        )


if __name__ == "__main__":
    main()
