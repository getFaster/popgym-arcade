"""Exercise incremental transfer and expiration recovery without a live runtime."""

import importlib.util
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_script(name):
    path = (
        Path(__file__).resolve().parents[1] / "docs/experiment_handoff" / f"{name}.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


remote = load_script("colab_million_remote")
local = load_script("colab_million_supervisor")


def checkpoint(root, steps):
    path = (
        root
        / f"CartPoleHard/mdp/ib_residual/ib_reverse_prior20_seed0/checkpoints/step_{steps:09d}"
    )
    path.mkdir(parents=True)
    state = path / "training_state.eqx"
    state.write_bytes(f"optimizer_rng_env_state_{steps}".encode())
    (path / "checkpoint.json").write_text(
        json.dumps(
            {
                "training_state_sha256": remote.sha256(state),
                "training_state_bytes": state.stat().st_size,
                "actual_steps": steps,
            }
        )
    )
    return path


def transfer(remote_root, local_root, payload, capsys):
    remote.pack(payload)
    response = json.loads(capsys.readouterr().out)
    archive = local_root / "transfer.tar"
    archive.write_bytes(
        b"".join(Path(c["path"]).read_bytes() for c in response["chunks"])
    )
    manifest = local.install_transfer(archive, local_root)
    remote.cleanup({})
    capsys.readouterr()
    return response, manifest


def test_incremental_checkpoints_and_mutable_logs(tmp_path, monkeypatch, capsys):
    source = tmp_path / "remote"
    results = source / "results"
    destination = tmp_path / "local"
    destination.mkdir()
    monkeypatch.setattr(remote, "SOURCE", source)
    monkeypatch.setattr(remote, "RESULTS", results)
    monkeypatch.setattr(remote, "TRANSFER_ROOT", tmp_path)
    first = checkpoint(results, 100352)
    second = checkpoint(results, 200704)
    run = first.parent.parent
    (run / "training.jsonl").write_text('{"steps": 200704}\n')
    (run / "checkpoints/.step_incomplete.tmp").mkdir()
    response, manifest = transfer(results, destination, {}, capsys)
    assert response["pending_checkpoints"] == 2
    assert len(list(destination.glob("**/checkpoint.json"))) == 1
    known = {
        name: info["sha256"] for name, info in manifest.items() if name.endswith(".eqx")
    }
    (run / "training.jsonl").write_text('{"steps": 202752}\n')
    response, manifest = transfer(results, destination, {"known": known}, capsys)
    assert response["pending_checkpoints"] == 1
    assert len(list(destination.glob("**/checkpoint.json"))) == 2
    assert (
        destination / run.relative_to(results) / "training.jsonl"
    ).read_text() == '{"steps": 202752}\n'
    known.update(
        {
            name: info["sha256"]
            for name, info in manifest.items()
            if name.endswith(".eqx")
        }
    )
    response, manifest = transfer(results, destination, {"known": known}, capsys)
    assert response["pending_checkpoints"] == 0
    assert not any(name.endswith(".eqx") for name in manifest)
    assert first.exists() and second.exists()
    assert not list(destination.glob("**/.step_incomplete.tmp"))


def test_corrupt_transfer_does_not_publish_any_files(tmp_path, monkeypatch, capsys):
    source = tmp_path / "remote"
    results = source / "results"
    destination = tmp_path / "local"
    destination.mkdir()
    monkeypatch.setattr(remote, "SOURCE", source)
    monkeypatch.setattr(remote, "RESULTS", results)
    monkeypatch.setattr(remote, "TRANSFER_ROOT", tmp_path)
    path = checkpoint(results, 100352)
    record = json.loads((path / "checkpoint.json").read_text())
    record["training_state_sha256"] = "bad"
    (path / "checkpoint.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Checkpoint checksum"):
        transfer(results, destination, {}, capsys)
    assert not list(destination.glob("**/checkpoint.json"))
    assert not list(destination.glob("**/training_state.eqx"))


def test_expired_runtime_recovery_keeps_latest_state_and_frozen_calibration(
    tmp_path, monkeypatch
):
    root = tmp_path / "experiment"
    provenance = root / "provenance"
    provenance.mkdir(parents=True)
    (provenance / "source.tar.gz").write_bytes(b"frozen-source")
    first = checkpoint(root, 100352)
    second = checkpoint(root, 200704)
    run = first.parent.parent
    (run / "config.json").write_text("{}")
    (run / "training.jsonl").write_text('{"steps": 200704}\n')
    calibration = run.parent / "_calibration/frozen.json"
    calibration.parent.mkdir()
    calibration.write_text('{"scales_bits": [2, 3]}')
    supervisor = local.Supervisor(
        SimpleNamespace(output=root, session="test", gpu="L4", repository=tmp_path)
    )
    transferred = []

    def fake_cli(*args, **kwargs):
        if args[0] == "upload" and args[-1].endswith("recovery.tar"):
            with tarfile.open(args[-2]) as archive:
                transferred.extend(archive.getnames())
        return ""

    monkeypatch.setattr(supervisor, "cli", fake_cli)
    monkeypatch.setattr(supervisor, "remote", lambda *args: {})
    supervisor.deploy()
    assert str(second.relative_to(root) / "training_state.eqx") in transferred
    assert str(first.relative_to(root) / "training_state.eqx") not in transferred
    assert str(calibration.relative_to(root)) in transferred
    assert str(run.relative_to(root) / "training.jsonl") in transferred
    assert first.exists() and second.exists()
