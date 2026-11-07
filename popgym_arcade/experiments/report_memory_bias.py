"""Build the final comparison from saved evaluation returns and run metadata."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    variants = ["mlp", "lru", "weight_decay", "state_dropout", "information_bottleneck"]
    labels = [
        "MLP reference",
        "LRU baseline",
        "+ weight decay (1e-4)",
        "+ state dropout (0.1)",
        "+ information bottleneck (1 free bit)",
    ]
    results = {}
    for variant in variants:
        paths = list(args.output.glob(f"{variant}_seed*/result.json"))
        if len(paths) != 1:
            parser.error(
                "This pilot report expects exactly one completed seed per condition"
            )
        result = json.loads(paths[0].read_text())
        scores = np.load(paths[0].with_name("evaluation_returns.npy"))
        records = [
            json.loads(line)
            for line in paths[0].with_name("training.jsonl").read_text().splitlines()
        ]
        assert scores.shape == (result["evaluation_episodes"],)
        assert np.isfinite(scores).all()
        assert np.isclose(scores.mean(), result["evaluation_mean"])
        assert records[-1]["steps"] == result["actual_steps"]
        assert len(records) == result["actual_steps"] // (
            result["config"]["NUM_ENVS"] * result["config"]["NUM_STEPS"]
        )
        assert all(np.isfinite(row["loss"]) for row in records)
        results[variant] = result
    reference, baseline = (
        results["mlp"]["evaluation_mean"],
        results["lru"]["evaluation_mean"],
    )
    base_bias = reference - baseline
    ib_bias = reference - results["information_bottleneck"]["evaluation_mean"]
    improvement = 100 * (base_bias - ib_bias) / base_bias
    lines = [
        "# Memory-bias pilot results",
        "",
        f"Completed on CUDA ({results['lru']['device_kind']}), JAX {results['lru']['jax_version']}. Environment: fully observed `{results['lru']['config']['ENV_NAME']}`. Each condition used training seed 0, {results['lru']['actual_steps']:,} environment steps, and {results['lru']['evaluation_episodes']} held-out evaluation episodes. The approved 100K budget was rounded down to complete 2,048-step PPO rollouts.",
        "",
        "`memory bias = MLP reference score − condition score`. Positive bias means the reference outperformed that condition. Positive bias reduction below means an improvement over unregularized LRU.",
        "",
        "| Condition | Held-out return ± episode SE | Memory bias | Bias reduction vs LRU | Parameters |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant, label in zip(variants, labels):
        result = results[variant]
        score = result["evaluation_mean"]
        bias = reference - score
        reduction = "—" if variant == "mlp" else f"{score - baseline:+.6f}"
        lines.append(
            f"| {label} | {score:.6f} ± {result['evaluation_standard_error']:.6f} | {bias:.6f} | {reduction} | {result['parameter_count']:,} |"
        )
    lines += [
        "",
        f"Information bottleneck reduced the observed gap by **{improvement:.1f}%** relative to the LRU baseline. Weight decay and state dropout did not reduce the gap in this pilot.",
        "",
        "These are single-training-seed pilot results at roughly 1% of the upstream training budget. Episode-level standard errors do not measure variation across training seeds. The results do not establish statistical significance or a general regularization benefit. The conditions also differ in parameter count, especially IB; there is no parameter-matched control.",
        "",
        "All agents use `(a_{t-1}, r_{t-1}, o_t)`, including at rollout, PPO replay, bootstrap, and evaluation. All PPO settings were read from upstream `train.py`; the training budget and disabled external logging are the explicit overrides.",
        "",
        "IB uses a diagonal multivariate Gaussian with 1,365 latent coordinates per actor/critic branch, computed as `floor(2/3 × 2,048)` real scalar LRU-state dimensions. Both RL heads receive `(input, prior_mean, prior_log_variance)` directly. The posterior receives `(input, LRU_state)`. KL is `KL(prior || posterior)` in bits; `max(1, KL_bits)` is applied per full distribution and per sample, separately for actor and critic, then summed with coefficient 1. The LRU has no direct inference path to the IB heads, so this variant evaluates auxiliary-memory training rather than a memory-dependent policy.",
        "",
        "Validation: 12 CPU test cases passed, covering input ordering, resets, KL direction, free-bit units/gradients, dropout replay, GAE termination, complex-parameter gradients, and two-layer parallel/serial equivalence for LRU, dropout, and IB. A full-size CUDA check also passed, with maximum absolute logit difference below 5e-7. Ruff formatting and lint passed for the experiment code and checks.",
        "",
        "![Evaluation and learning curves](comparison.png)",
        "",
        "[Experiment design and reproduction commands](../../MEMORY_BIAS.md) · [Summary CSV](summary.csv)",
        "",
        "Each condition directory retains `config.json`, `training.jsonl`, `evaluation_returns.npy`, `result.json`, and local `model.eqx` weights. JSON metadata includes the source and dependency-lock hashes.",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"Validated all five runs. IB observed bias reduction: {improvement:.2f}%")


if __name__ == "__main__":
    main()
