"""Full-size CUDA PPO checkpoint, same-budget resume, and extension validation."""

import json
import tempfile
from pathlib import Path

import jax
import numpy as np

from popgym_arcade.experiments import memory_bias as experiment


class SimulatedInterruption(Exception):
    pass


def main():
    assert jax.default_backend() == "gpu"
    original_save = experiment.save_checkpoint
    with tempfile.TemporaryDirectory(prefix="gpu-checkpoint-smoke-") as temporary:
        root = Path(temporary)
        for variant in ("lru", "ib_reverse_prior20"):
            args = experiment.build_parser().parse_args(
                [
                    "--variants",
                    variant,
                    "--timesteps",
                    "4096",
                    "--checkpoint-interval",
                    "2048",
                    "--gradient-log-interval",
                    "2048",
                    "--eval-episodes",
                    "16",
                    "--output",
                    str(root / variant),
                ]
            )
            config = experiment.upstream_config() | {
                "ENV_NAME": "CartPoleHard",
                "PARTIAL": False,
                "TOTAL_TIMESTEPS": 4096,
                "WANDB_MODE": "disabled",
            }

            def interrupt_after_save(*positional, **keyword):
                path = original_save(*positional, **keyword)
                raise SimulatedInterruption(str(path))

            experiment.save_checkpoint = interrupt_after_save
            try:
                experiment.run_one(args, variant, 0, config)
            except SimulatedInterruption:
                pass
            else:
                raise AssertionError("Interruption was not exercised")
            finally:
                experiment.save_checkpoint = original_save
            directory = args.output / f"{variant}_seed0"
            assert not (directory / "result.json").exists()
            experiment.run_one(args, variant, 0, config)
            records = [
                json.loads(line)
                for line in (directory / "training.jsonl").read_text().splitlines()
            ]
            assert [row["steps"] for row in records] == [2048, 4096]
            assert all(
                np.isfinite(row["gradient_norm_pre_clip_mean"]) for row in records
            )
            assert all(
                row["gradient_norm_post_clip_mean"] <= 0.50001 for row in records
            )
            assert np.load(directory / "evaluation_returns.npy").shape == (16,)
            if variant == "ib_reverse_prior20":
                extended = experiment.build_parser().parse_args(
                    [
                        "--variants",
                        variant,
                        "--timesteps",
                        "6144",
                        "--checkpoint-interval",
                        "2048",
                        "--gradient-log-interval",
                        "2048",
                        "--eval-episodes",
                        "16",
                        "--resume-from",
                        str(directory),
                        "--extend-training",
                        "--output",
                        str(root / "extension"),
                    ]
                )
                experiment.run_one(
                    extended, variant, 0, config | {"TOTAL_TIMESTEPS": 6144}
                )
                new = extended.output / f"{variant}_seed0"
                old_config = json.loads((directory / "config.json").read_text())
                new_config = json.loads((new / "config.json").read_text())
                assert old_config["rms_calibration"] == new_config["rms_calibration"]
                assert (
                    json.loads((new / "latest_checkpoint.json").read_text())[
                        "actual_steps"
                    ]
                    == 6144
                )
            jax.clear_caches()
    Path("gpu_smoke_passed.json").write_text(
        json.dumps(
            {
                "passed": True,
                "device": jax.devices()[0].device_kind,
                "hidden": 512,
                "layers": 2,
                "environments": 16,
                "rollout_steps": 128,
                "verified": [
                    "vanilla and IB real PPO",
                    "same-budget interrupted resume",
                    "checksummed full state",
                    "gradient clipping",
                    "real evaluation",
                    "explicit IB extension and exact frozen calibration reuse",
                ],
            },
            indent=2,
        )
    )
    print("PASS: full-size GPU checkpoint/resume/extension smoke", flush=True)


if __name__ == "__main__":
    main()
