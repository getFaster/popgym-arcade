"""Remote operations for the frozen million-step experiment (standard library only)."""

import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import tarfile
from pathlib import Path

SOURCE = Path("/content/ib_million")
RESULTS = SOURCE / "results/ib_million_steps_20261109"
TRANSFER_ROOT = SOURCE.parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alive():
    path = SOURCE / "worker.pid"
    if not path.exists():
        return False
    try:
        pid = int(path.read_text())
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except (FileNotFoundError, ValueError):
        return False


def inspect(payload):
    print(
        json.dumps(
            {
                "alive": alive(),
                "source_exists": SOURCE.exists(),
                "free_bytes": shutil.disk_usage("/content").free,
                "gpu": subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.total",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ).strip(),
                "done": json.loads((RESULTS / "done.json").read_text())
                if (RESULTS / "done.json").exists()
                else None,
            }
        )
    )


def launch(payload):
    if alive():
        raise RuntimeError("Worker already running")
    if not SOURCE.exists():
        SOURCE.mkdir()
        with tarfile.open("/content/ib_million_source.tar.gz") as archive:
            archive.extractall(SOURCE, filter="data")
    manifest = json.loads((SOURCE / "source_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(SOURCE / name) != expected:
            raise RuntimeError(f"Source hash mismatch: {name}")
    recovery = Path("/content/ib_million_recovery.tar")
    if recovery.exists():
        RESULTS.mkdir(parents=True, exist_ok=True)
        with tarfile.open(recovery) as archive:
            archive.extractall(RESULTS, filter="data")
        recovery.unlink()
    if shutil.disk_usage("/content").free < 50 * 1024**3:
        raise RuntimeError(
            "Need at least 50 GiB free for dependencies and retained checkpoints"
        )
    script = SOURCE / "launch.sh"
    script.write_text("""#!/bin/bash
set -euo pipefail
cd /content/ib_million
export MPLBACKEND=Agg
export UV_CACHE_DIR=/content/uv-cache
uv sync --frozen --extra baselines --extra cuda --dev
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ ! -f gpu_smoke_passed.json ]; then
  .venv/bin/python docs/experiment_handoff/gpu_checkpoint_smoke.py
fi
.venv/bin/python -m popgym_arcade.experiments.million_step_suite --output results/ib_million_steps_20261109
""")
    with (SOURCE / "worker.log").open("a") as log:
        process = subprocess.Popen(
            ["bash", str(script)],
            cwd=SOURCE,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (SOURCE / "worker.pid").write_text(str(process.pid))
    print(
        json.dumps(
            {
                "pid": process.pid,
                "source_manifest_sha256": sha256(SOURCE / "source_manifest.json"),
            }
        )
    )


def pack(payload):
    """Send one new complete checkpoint per batch; mutable logs are always refreshed."""
    known = payload.get("known", {})
    selected = []
    checkpoint_selected = False
    pending = 0
    if RESULTS.exists():
        for record in sorted(RESULTS.glob("**/checkpoints/step_*/checkpoint.json")):
            data = json.loads(record.read_text())
            state = record.with_name("training_state.eqx")
            relative = str(state.relative_to(RESULTS))
            if known.get(relative) == data["training_state_sha256"]:
                continue
            if state.stat().st_size != data["training_state_bytes"]:
                raise RuntimeError(f"Incomplete published state: {state}")
            pending += 1
            if not checkpoint_selected:
                selected += [record, state]
                checkpoint_selected = True
        for path in sorted(RESULTS.rglob("*")):
            relative = path.relative_to(RESULTS)
            if (
                not path.is_file()
                or any(part.startswith(".") for part in relative.parts)
                or "checkpoints" in relative.parts
            ):
                continue
            if path.suffix == ".eqx" and str(relative) in known:
                continue
            selected.append(path)
    extras = {
        "colab_worker.log": SOURCE / "worker.log",
        "gpu_smoke_passed.json": SOURCE / "gpu_smoke_passed.json",
    }
    files = {str(p.relative_to(RESULTS)): p for p in selected}
    files.update({name: path for name, path in extras.items() if path.exists()})
    manifest = {}
    archive_path = TRANSFER_ROOT / "ib_million_transfer.tar"
    with tarfile.open(archive_path, "w") as archive:
        for name, path in files.items():
            # Snapshot mutable logs so their checksum always describes transferred bytes.
            if path.suffix != ".eqx":
                data = path.read_bytes()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
                manifest[name] = {
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
            else:
                manifest[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
                archive.add(path, arcname=name)
        data = json.dumps(manifest).encode()
        info = tarfile.TarInfo("_transfer_manifest.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    chunks = []
    with archive_path.open("rb") as stream:
        while data := stream.read(32 * 1024 * 1024):
            path = TRANSFER_ROOT / f"ib_million_chunk_{len(chunks):04d}"
            path.write_bytes(data)
            chunks.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
            )
    archive_path.unlink()
    print(
        json.dumps(
            {
                "chunks": chunks,
                "pending_checkpoints": pending,
                "alive": alive(),
                "done": json.loads((RESULTS / "done.json").read_text())
                if (RESULTS / "done.json").exists()
                else None,
            }
        )
    )


def cleanup(payload):
    # Only temporary transfer chunks are removed; all checkpoints stay on the VM.
    for path in TRANSFER_ROOT.glob("ib_million_chunk_*"):
        path.unlink()
    print(json.dumps({"cleaned": True}))


def stop_worker(payload):
    if alive():
        os.killpg(int((SOURCE / "worker.pid").read_text()), signal.SIGTERM)
    print(json.dumps({"stopped": True}))
