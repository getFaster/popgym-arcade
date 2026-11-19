"""Atomic, checksummed PPO training checkpoints with exact optimizer/RNG resume."""

import hashlib
import json
import os
from pathlib import Path

import equinox as eqx

SIGNATURE_FIELDS = (
    "variant",
    "seed",
    "config",
    "hidden",
    "layers",
    "posterior_input",
    "lru_residual",
    "head_inputs",
    "normalization",
    "rms_scales_bits",
    "ib_coefficient",
    "requested_steps",
    "actual_steps",
    "source_sha256",
    "weight_decay",
    "state_dropout",
    "learning_rate_schedule_steps",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_signature(metadata):
    payload = {key: metadata.get(key) for key in SIGNATURE_FIELDS}
    if metadata.get("recurrent_state_quantization", "none") != "none":
        payload.update(
            {
                key: metadata[key]
                for key in ("recurrent_state_quantization", "fsq_levels", "fsq_scale")
            }
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def checkpoint_due(previous_steps, current_steps, interval):
    return interval > 0 and current_steps // interval > previous_steps // interval


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save_checkpoint(
    directory,
    training_state,
    metadata,
    completed_updates,
    actual_steps,
    elapsed_seconds,
    interval,
    final=False,
):
    parent = Path(directory) / "checkpoints"
    parent.mkdir(exist_ok=True)
    target = parent / f"step_{actual_steps:09d}"
    if target.exists():
        stored = json.loads((target / "checkpoint.json").read_text())
        if stored["run_signature"] != run_signature(metadata):
            raise ValueError("Existing checkpoint belongs to a different run")
        return target
    temporary = parent / f".step_{actual_steps:09d}_{os.getpid()}.tmp"
    temporary.mkdir()
    weights = temporary / "training_state.eqx"
    with weights.open("wb") as stream:
        eqx.tree_serialise_leaves(stream, training_state)
        stream.flush()
        os.fsync(stream.fileno())
    record = {
        "format_version": 1,
        "completed_updates": completed_updates,
        "actual_steps": actual_steps,
        "requested_steps": metadata["requested_steps"],
        "final_training_steps": metadata["actual_steps"],
        "elapsed_seconds": elapsed_seconds,
        "checkpoint_interval": interval,
        "final": final,
        "threshold_steps": metadata["requested_steps"]
        if final
        else (actual_steps // interval) * interval
        if interval
        else None,
        "run_signature": run_signature(metadata),
        "run_metadata": metadata,
        "training_state_sha256": file_sha256(weights),
        "training_state_bytes": weights.stat().st_size,
        "contents": "model, optimizer including schedule counters, environment state, observation, previous action/reward, recurrent memory, done flags, RNG",
    }
    (temporary / "checkpoint.json").write_text(json.dumps(record, indent=2) + "\n")
    temporary.rename(target)
    atomic_json(
        Path(directory) / "latest_checkpoint.json",
        {"directory": str(target.relative_to(directory)), **record},
    )
    return target


def restore_checkpoint(
    directory, template, metadata, rollout_steps, allow_budget_extension=False
):
    directory = Path(directory)
    candidates = (
        [directory / "checkpoint.json"]
        if (directory / "checkpoint.json").exists()
        else sorted((directory / "checkpoints").glob("step_*/checkpoint.json"))
    )
    if not candidates:
        return None
    path = candidates[-1]
    record = json.loads(path.read_text())
    if record["format_version"] != 1:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    if record["run_signature"] != run_signature(metadata):
        previous = record.get("run_metadata", {})
        for key, default in (
            ("recurrent_state_quantization", "none"),
            ("fsq_levels", 5),
            ("fsq_scale", 1.0),
        ):
            if previous.get(key, default) != metadata.get(key, default):
                raise ValueError("Checkpoint FSQ configuration differs")
        old_settings = {key: previous.get(key) for key in SIGNATURE_FIELDS}
        new_settings = {key: metadata.get(key) for key in SIGNATURE_FIELDS}
        for settings in (old_settings, new_settings):
            for key in (
                "requested_steps",
                "actual_steps",
                "learning_rate_schedule_steps",
            ):
                settings.pop(key, None)
            settings["config"] = {
                key: value
                for key, value in settings.get("config", {}).items()
                if key != "TOTAL_TIMESTEPS"
            }
        extended = (
            allow_budget_extension
            and metadata["actual_steps"] > previous.get("actual_steps", float("inf"))
            and metadata["requested_steps"]
            > previous.get("requested_steps", float("inf"))
            and old_settings == new_settings
        )
        if not extended:
            raise ValueError(
                f"Checkpoint configuration or training source differs: {path}"
            )
    if (
        record["actual_steps"] != record["completed_updates"] * rollout_steps
        or record["actual_steps"] > metadata["actual_steps"]
    ):
        raise ValueError(f"Invalid checkpoint training steps: {path}")
    state = path.with_name("training_state.eqx")
    if (
        state.stat().st_size != record["training_state_bytes"]
        or file_sha256(state) != record["training_state_sha256"]
    ):
        raise ValueError(f"Checkpoint checksum mismatch: {state}")
    with state.open("rb") as stream:
        restored = eqx.tree_deserialise_leaves(stream, template)
    return restored, record


def trim_training_log(path, completed_updates):
    """Discard only work after the last durable checkpoint, including a torn tail."""
    lines = path.read_text().splitlines() if path.exists() else []
    records = []
    for line in lines[:completed_updates]:
        records.append(json.loads(line))
    if [row["update"] for row in records] != list(range(1, completed_updates + 1)):
        raise ValueError("Training log does not cover the restored checkpoint")
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
