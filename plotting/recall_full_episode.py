"""Measure last-action recall over complete episodes for selected checkpoints.

Each checkpoint is run in a separate process to bound JAX compilation caches.
Output is kept separate from earlier, window-conditional diagnostics.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

import popgym_arcade
from plotting.density_analysis_memfix import (
    bin_distribution,
    collect_episode,
    family,
    load,
    recall,
)


def measure(checkpoint, output, safety_limit):
    path = Path(checkpoint)
    metadata = json.loads(path.with_name("config.json").read_text())
    model = load(path, metadata)
    env, params = popgym_arcade.make(
        metadata["config"]["ENV_NAME"],
        partial_obs=metadata["config"]["PARTIAL"],
        obs_size=metadata["config"]["OBS_SIZE"],
    )
    samples = []
    for seed in range(1_000_000, 1_000_005):
        history = collect_episode(model, env, params, seed, safety_limit)
        distribution = np.asarray(recall(model.actor, *history, definition="paper"))
        if not np.all(np.isfinite(distribution)) or not np.isclose(
            distribution.sum(), 1, atol=1e-5
        ):
            raise ValueError(f"Undefined recall for trajectory seed {seed}")
        samples.append(
            {
                "trajectory_seed": seed,
                "length": len(history[0]),
                "endpoint": len(history[0]) - 1,
                "distribution": distribution.tolist(),
                "bins": bin_distribution(distribution).tolist(),
            }
        )
        print(f"{path}: trajectory {seed}, length {len(history[0])}", flush=True)
    result = {
        "checkpoint": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "task": metadata["config"]["ENV_NAME"],
        "mode": "POMDP" if metadata["config"]["PARTIAL"] else "MDP",
        "intervention": family(metadata),
        "training_seed": metadata["seed"],
        "actual_steps": metadata["actual_steps"],
        "definition": "paper",
        "device": str(jax.devices()[0]),
        "samples": samples,
        "bin_mean": np.mean([s["bins"] for s in samples], axis=0).tolist(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safety-limit", type=int, default=10000)
    args = parser.parse_args()
    if args.safety_limit < 1:
        parser.error("safety limit must be positive")
    measure(args.checkpoint, args.output, args.safety_limit)


if __name__ == "__main__":
    main()
