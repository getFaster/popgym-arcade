"""Sync every checkpoint, recover expired Colab runtimes, and release on completion.

Run with the experiment's frozen source tarball already in provenance/source.tar.gz.
Only the Colab CLI accesses credentials; its session tokens are never archived.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def install_transfer(archive_path, root):
    """Verify all bytes first, then atomically publish whole checkpoint directories."""
    with tempfile.TemporaryDirectory(prefix=".incoming-", dir=root) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive_path) as archive:
            archive.extractall(staging, filter="data")
        manifest = json.loads((staging / "_transfer_manifest.json").read_text())
        for name, expected in manifest.items():
            path = staging / name
            if (
                not path.is_relative_to(staging)
                or path.stat().st_size != expected["bytes"]
                or sha256(path) != expected["sha256"]
            ):
                raise ValueError(f"Transfer checksum mismatch: {name}")
        for sidecar in staging.glob("**/checkpoints/step_*/checkpoint.json"):
            record = json.loads(sidecar.read_text())
            state = sidecar.with_name("training_state.eqx")
            if (
                state.stat().st_size != record["training_state_bytes"]
                or sha256(state) != record["training_state_sha256"]
            ):
                raise ValueError(f"Checkpoint checksum mismatch: {sidecar}")
            target = root / sidecar.parent.relative_to(staging)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if (
                    sha256(target / "training_state.eqx")
                    != record["training_state_sha256"]
                ):
                    raise ValueError(f"Conflicting retained checkpoint: {target}")
            else:
                sidecar.parent.rename(target)
        for name in manifest:
            path = staging / name
            if not path.exists() or "checkpoints" in Path(name).parts:
                continue
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)
    return manifest


class Supervisor:
    def __init__(self, args):
        self.root = args.output.resolve()
        self.provenance = self.root / "provenance"
        self.session = args.session
        self.gpu = args.gpu
        self.repository = Path(args.repository).resolve()
        self.known = {}
        for path in self.root.glob("**/checkpoints/step_*/checkpoint.json"):
            record = json.loads(path.read_text())
            state = path.with_name("training_state.eqx")
            if (
                state.stat().st_size != record["training_state_bytes"]
                or sha256(state) != record["training_state_sha256"]
            ):
                raise ValueError(f"Invalid retained local checkpoint: {state}")
            self.known[str(state.relative_to(self.root))] = record[
                "training_state_sha256"
            ]
        for path in self.root.glob("*/*/*/*/model.eqx"):
            self.known[str(path.relative_to(self.root))] = sha256(path)

    def cli(self, *arguments, timeout=300):
        result = subprocess.run(
            ["colab", *arguments],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"colab {arguments[0]} failed: {result.stderr[-1800:]} {result.stdout[-800:]}"
            )
        return result.stdout

    def remote(self, operation, payload=None):
        script = self.provenance / "remote_command.py"
        script.write_text(
            "import runpy\n"
            "_million_ops = runpy.run_path('/content/ib_million/docs/experiment_handoff/colab_million_remote.py')\n"
            f"_million_ops[{operation!r}]({payload or {}!r})\n"
        )
        result = self.cli(
            "exec",
            "-s",
            self.session,
            "-f",
            str(script),
            "--timeout",
            "180",
            timeout=240,
        )
        for line in reversed(result.splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise RuntimeError(f"No remote JSON response: {result[-1000:]}")

    def deploy(self):
        self.cli(
            "upload",
            "-s",
            self.session,
            str(self.provenance / "source.tar.gz"),
            "/content/ib_million_source.tar.gz",
        )
        bootstrap = self.provenance / "bootstrap.py"
        bootstrap.write_text(
            "from pathlib import Path\nimport tarfile\np = Path('/content/ib_million')\np.mkdir(exist_ok=True)\nwith tarfile.open('/content/ib_million_source.tar.gz') as a:\n    a.extractall(p, filter='data')\nprint('source extracted')\n"
        )
        self.cli("exec", "-s", self.session, "-f", str(bootstrap))
        # Restore logs/config/calibrations/results plus the latest full state of each
        # unfinished run. Older checkpoints remain retained on the local filesystem.
        recovery_path = self.provenance / "recovery.tar"
        recovery_files = []
        for path in self.root.rglob("*"):
            relative = path.relative_to(self.root)
            if (
                not path.is_file()
                or relative.parts[0] == "provenance"
                or any(p.startswith(".") for p in relative.parts)
                or "checkpoints" in relative.parts
            ):
                continue
            if path.name in ("done.json", "status.json", "colab_worker.log"):
                continue
            recovery_files.append(path)
        for config in self.root.glob("*/*/*/*/config.json"):
            run = config.parent
            if (run / "result.json").exists():
                continue
            checkpoints = sorted((run / "checkpoints").glob("step_*/checkpoint.json"))
            if checkpoints:
                recovery_files += [
                    checkpoints[-1],
                    checkpoints[-1].with_name("training_state.eqx"),
                ]
        if recovery_files:
            with tarfile.open(recovery_path, "w") as archive:
                for path in recovery_files:
                    archive.add(path, arcname=str(path.relative_to(self.root)))
            self.cli(
                "upload",
                "-s",
                self.session,
                str(recovery_path),
                "/content/ib_million_recovery.tar",
                timeout=900,
            )
            recovery_path.unlink()
        info = self.remote("inspect")
        atomic_json(self.provenance / "runtime.json", info)
        self.remote("launch")

    def sync(self):
        response = self.remote("pack", {"known": self.known})
        transfer = self.provenance / "transfer.tar"
        with transfer.open("wb") as output:
            for chunk in response["chunks"]:
                local = self.provenance / "chunk.partial"
                self.cli(
                    "download",
                    "-s",
                    self.session,
                    chunk["path"],
                    str(local),
                    timeout=300,
                )
                if (
                    local.stat().st_size != chunk["bytes"]
                    or sha256(local) != chunk["sha256"]
                ):
                    raise ValueError("Transfer chunk checksum mismatch")
                with local.open("rb") as incoming:
                    shutil.copyfileobj(incoming, output)
                local.unlink()
        manifest = install_transfer(transfer, self.root)
        transfer.unlink()
        for name, record in manifest.items():
            if name.endswith(".eqx"):
                self.known[name] = record["sha256"]
        self.remote("cleanup")
        atomic_json(
            self.provenance / "sync_status.json",
            {
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "verified_immutable_files": self.known,
                **response,
            },
        )
        self.update_state(response)
        return response

    def update_state(self, response, phase=None):
        state_path = self.repository / "EXPERIMENT_1M_STATE.json"
        state = json.loads(state_path.read_text())
        started = completed = steps = 0
        for job in state["jobs"]:
            variant = (
                "lru" if job["condition"] == "vanilla_lru" else "ib_reverse_prior20"
            )
            run = (
                self.root
                / job["task"]
                / job["mode"]
                / job["condition"]
                / f"{variant}_seed0"
            )
            checkpoints = list((run / "checkpoints").glob("step_*/checkpoint.json"))
            job["checkpoints_saved"] = len(checkpoints)
            log = run / "training.jsonl"
            records = []
            if log.exists():
                for line in log.read_text().splitlines():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        break
            job["actual_steps"] = records[-1]["steps"] if records else 0
            steps += job["actual_steps"]
            job["status"] = (
                "completed"
                if (run / "result.json").exists()
                else "running"
                if (run / "config.json").exists()
                else "not_started"
            )
            started += job["status"] != "not_started"
            completed += job["status"] == "completed"
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        state["status"].update(
            {
                "phase": phase
                or ("running_on_colab" if started else "colab_gpu_validation"),
                "official_runs_started": started,
                "official_runs_completed": completed,
                "official_training_steps_collected": steps,
                "latest_user_instruction": "Continue this EXPERIMENT_1M_STATE.json on colab.",
                "do_not_start_colab_without_explicit_user_resume": False,
                "resume_policy": "Explicitly resumed by user; one sequential Colab queue with checksummed sync and runtime-expiration recovery.",
                "active_colab_session": self.session,
                "supervisor_pid": os.getpid(),
                "latest_sync": str(self.provenance / "sync_status.json"),
            }
        )
        state["experiment_plan"]["planned_output_directory_created"] = True
        state["runtime_execution"] = {
            "output": str(self.root),
            "frozen_source": str(self.provenance / "source"),
            "source_archive_sha256": sha256(self.provenance / "source.tar.gz"),
            "synced_checkpoints": sum(j["checkpoints_saved"] for j in state["jobs"]),
            "remote_alive_at_sync": response.get("alive"),
            "done": response.get("done"),
        }
        if (self.root / "gpu_smoke_passed.json").exists():
            state["validation"]["gpu_validation_for_new_checkpoint_features"] = (
                json.loads((self.root / "gpu_smoke_passed.json").read_text())
            )
        atomic_json(state_path, state)

    def run(self):
        failures = 0
        recoveries = 0
        while True:
            try:
                # A stop file prevents further allocation and exits after durable sync.
                if (self.provenance / "STOP").exists():
                    self.remote("stop_worker")
                    response = self.sync()
                    while response["pending_checkpoints"]:
                        response = self.sync()
                    self.cli("stop", "-s", self.session)
                    self.update_state(response, "stopped_and_synced")
                    return
                response = self.sync()
                failures = 0
                if response["done"] is not None or not response["alive"]:
                    while response["pending_checkpoints"]:
                        response = self.sync()
                    self.cli("stop", "-s", self.session)
                    phase = (
                        "complete"
                        if response["done"]
                        and response["done"]["completed"] == 24
                        and not response["done"]["failures"]
                        else "worker_failed_and_synced"
                    )
                    self.update_state(response, phase)
                    return
                if response["pending_checkpoints"]:
                    continue
            except (
                RuntimeError,
                OSError,
                ValueError,
                subprocess.TimeoutExpired,
            ) as error:
                failures += 1
                print(
                    f"{datetime.now(timezone.utc).isoformat()} sync failure {failures}: {error}",
                    flush=True,
                )
                atomic_json(
                    self.provenance / "sync_error.json",
                    {
                        "time": time.time(),
                        "error": str(error),
                        "consecutive_failures": failures,
                    },
                )
                if failures >= 3:
                    sessions = self.cli("sessions")
                    if f"[{self.session}]" not in sessions:
                        if (self.provenance / "STOP").exists():
                            self.update_state({}, "stopped_runtime_already_expired")
                            return
                        if "[?]" in sessions:
                            self.update_state(
                                {}, "recovery_blocked_by_unidentified_runtime"
                            )
                            raise RuntimeError(
                                "Unidentified runtime exists; refusing duplicate allocation"
                            )
                        recoveries += 1
                        if recoveries > 4:
                            self.update_state({}, "recovery_limit_reached")
                            raise RuntimeError(
                                "More than four runtime replacements; stopping recovery"
                            )
                        self.cli(
                            "new", "-s", self.session, "--gpu", self.gpu, timeout=900
                        )
                        self.deploy()
                        failures = 0
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session", default="ib-million-t4")
    parser.add_argument("--gpu", choices=["T4", "L4"], default="T4")
    parser.add_argument("--repository", default="/home/user/popgym_arcade")
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args()
    args.output.joinpath("provenance").mkdir(parents=True, exist_ok=True)
    with args.output.joinpath("provenance/supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.joinpath("provenance/supervisor.pid").write_text(str(os.getpid()))
        supervisor = Supervisor(args)
        if args.deploy:
            supervisor.deploy()
        supervisor.run()


if __name__ == "__main__":
    main()
