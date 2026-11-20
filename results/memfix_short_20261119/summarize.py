"""Summarize completed MEMFIX runs and verify final weights and checkpoints."""

import csv
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent
PREDICTION = RESULTS / "lru_input_prediction_20261111" / "CartPoleHard"
KL_SWEEP = RESULTS / "posterior_rms_suite_20261109"
KL = "ib_reverse_prior20"
TASKS = ("CartPoleHard", "NavigatorEasy", "MineSweeperEasy", "BreakoutEasy")


def model_result(directory, mode, seed, *, fresh=False):
    result_path = directory / "result.json"
    if not result_path.exists():
        return None
    result = json.loads(result_path.read_text())
    assert result["seed"] == seed, result_path
    assert result["config"]["PARTIAL"] == (mode == "pomdp"), result_path
    assert result["actual_steps"] == 98304, result_path
    assert result["evaluation_episodes"] == 64, result_path
    assert (directory / "model.eqx").stat().st_size > 0, directory
    if fresh:
        checkpoints = list(directory.glob("checkpoints/step_*/checkpoint.json"))
        assert checkpoints, directory
        checkpoint = json.loads(checkpoints[-1].read_text())
        assert checkpoint["actual_steps"] == 98304, directory
    return result["evaluation_mean"]


def read(task, mode, variant, seed):
    name = f"{variant}_seed{seed}"
    if seed == 0 and task == "CartPoleHard":
        return model_result(PREDICTION / mode / name, mode, seed)
    if seed == 0 and variant == KL:
        return model_result(KL_SWEEP / task / mode / "output" / name, mode, seed)
    shard = "colab" if seed == 2 else "local"
    return model_result(ROOT / shard / task / mode / name, mode, seed, fresh=True)


def groups():
    for task in TASKS:
        variants = (
            ("lru", "lru_prediction", KL, KL + "_prediction")
            if task == "CartPoleHard"
            else (KL,)
        )
        for variant in variants:
            yield (
                task,
                variant,
                lambda mode, seed, task=task, variant=variant: read(
                    task, mode, variant, seed
                ),
            )
    for suite in ("fsq_lru_local_20261117_1420", "fsq_activation_local_20261117_2108"):
        source = RESULTS / suite
        for config in sorted((source / "mdp").iterdir()):
            yield (
                "CartPoleHard",
                config.name,
                lambda mode, seed, source=source, config=config: model_result(
                    source / mode / config.name / f"lru_seed{seed}", mode, seed
                ),
            )


def main():
    rows = []
    for task, intervention, measurement in groups():
        for seed in range(3):
            mdp = measurement("mdp", seed)
            pomdp = measurement("pomdp", seed)
            reference = read(task, "mdp", "mlp", seed)
            gap = mdp - pomdp if None not in (mdp, pomdp) else None
            bias = mdp - reference if None not in (mdp, reference) else None
            if None not in (gap, bias):
                assert abs(reference + bias - gap - pomdp) < 1e-12
            rows.append(
                {
                    "task": task,
                    "intervention": intervention,
                    "seed": seed,
                    "mdp": mdp,
                    "pomdp": pomdp,
                    "mlp_mdp": reference,
                    "gap": gap,
                    "bias": bias,
                }
            )
    with (ROOT / "metrics_by_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for i in range(0, len(rows), 3):
        group = rows[i : i + 3]
        summary = {"task": group[0]["task"], "intervention": group[0]["intervention"]}
        for key in ("mdp", "pomdp", "gap", "bias"):
            values = [row[key] for row in group if row[key] is not None]
            summary[f"{key}_n"] = len(values)
            summary[key] = mean(values) if len(values) == 3 else None
            summary[f"{key}_sd"] = stdev(values) if len(values) == 3 else None
        summaries.append(summary)
    with (ROOT / "metrics_three_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
