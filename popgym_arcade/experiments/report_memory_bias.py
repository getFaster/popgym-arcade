"""Build the final comparison from saved evaluation returns and run metadata."""

import argparse
import json
from pathlib import Path

import numpy as np

from popgym_arcade.experiments.memory_models import DEFAULT_IB_VARIANT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    ib_variant = (
        DEFAULT_IB_VARIANT
        if list(args.output.glob(f"{DEFAULT_IB_VARIANT}_seed*/result.json"))
        else "information_bottleneck"
    )
    variants = ["mlp", "lru", "weight_decay", "state_dropout", ib_variant]
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
    ib_bias = reference - results[ib_variant]["evaluation_mean"]
    improvement = (
        100 * (base_bias - ib_bias) / base_bias
        if abs(base_bias) > 1e-12
        else float("nan")
    )
    ib = results[ib_variant]
    lines = [
        "# Memory-bias pilot results",
        "",
        f"Completed on CUDA ({results['lru']['device_kind']}), JAX {results['lru']['jax_version']}. Environment ({'partially observed' if results['lru']['config']['PARTIAL'] else 'fully observed'}): `{results['lru']['config']['ENV_NAME']}`. Each condition used training seed 0, {results['lru']['actual_steps']:,} environment steps, and {results['lru']['evaluation_episodes']} held-out evaluation episodes. The approved 100K budget was rounded down to complete 2,048-step PPO rollouts.",
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
        f"The selected IB condition `{ib_variant}` changed the observed gap by **{improvement:.1f}%** relative to the LRU baseline (positive means a reduction).",
        "",
        "These are single-training-seed pilot results at roughly 1% of the upstream training budget. Episode-level standard errors do not measure variation across training seeds. The results do not establish statistical significance or a general regularization benefit. The conditions also differ in parameter count, especially IB; there is no parameter-matched control.",
        "",
        "All agents use `(a_{t-1}, r_{t-1}, o_t)`, including at rollout, PPO replay, bootstrap, and evaluation. All PPO settings were read from upstream `train.py`; the training budget and disabled external logging are the explicit overrides.",
        "",
        f"IB condition: `{ib_variant}`. Head inputs: `{ib.get('head_inputs', 'historical prior statistics; superseded IB implementation')}`. Posterior input: `{ib.get('posterior_input', 'state')}`; added residuals: `{ib.get('lru_residual', False)}`. Divergence: `{ib.get('divergence', 'forward_kl')}`. Prior/posterior gradient weights: `{ib.get('kl_prior_gradient_weight', 1.0)}/{ib.get('kl_posterior_gradient_weight', 1.0)}`. Normalization: `{ib.get('normalization')}`. Calibration and raw units are recorded in each run's metadata. The one-bit free region applies to the full Gaussian event, separately for actor and critic.",
        "",
        "This report rechecked episode-return statistics, complete training steps, and finite training losses against the saved artifacts. Development test results are recorded separately from these experimental scores.",
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
