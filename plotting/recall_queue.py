"""Resource-bounded checkpoint queue (standard library only in the coordinator)."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import random
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

MIB = 1024**2


def make_plan(seed=20261120, horizon=32):
    return {
        "seed": seed,
        "horizon": horizon,
        "window": 10,
        "endpoints": random.Random(seed).sample(range(9, horizon), 8),
        "trajectory_seeds": list(range(1_000_000, 1_000_008)),
        "sampling": "8 distinct endpoints uniformly sampled from 9..horizon-1; shared across checkpoints",
    }


def summarize(root, jobs, plan, limits):
    rows = []
    for job in jobs:
        manifest = root / "jobs" / job["id"] / "manifest.json"
        if manifest.exists():
            rows.extend(json.loads(manifest.read_text()))
    (root / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    (root / "queue.json").write_text(
        json.dumps({"limits": limits, "jobs": jobs}, indent=2) + "\n"
    )
    fields = [
        "checkpoint",
        "task",
        "mode",
        "variant",
        "training_seed",
        "status",
        "valid_samples",
    ]
    fields += [f"bin_{i + 1:02d}" for i in range(16)] + [
        f"se_{i + 1:02d}" for i in range(16)
    ]
    with (root / "recall_16.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            entry = {key: row.get(key) for key in fields[:7]}
            entry.update(
                {
                    f"bin_{i + 1:02d}": value
                    for i, value in enumerate(row.get("bin_mean", []))
                }
            )
            entry.update(
                {
                    f"se_{i + 1:02d}": value
                    for i, value in enumerate(row.get("bin_se") or [])
                }
            )
            writer.writerow(entry)
    completed = sum(job["status"] == "complete" for job in jobs)
    (root / "REPORT.md").write_text(
        "# Recall density — shared sampled endpoints, 16 bins\n\n"
        f"{completed}/{len(jobs)} checkpoint jobs completed. "
        "See [bin means and evaluation SEs](recall_16.csv), [all samples](manifest.json), "
        "[queue status and failures](queue.json), and [shared endpoint plan](endpoint_plan.json).\n\n"
        f"Paper action-probability definition. Eight trajectories/checkpoint; 10-step window; "
        f"one endpoint/trajectory: **{plan['endpoints']}**, shared across every checkpoint. "
        f"Trajectory/reset seeds are also shared. Endpoints are sampled without replacement from 9..{plan['horizon'] - 1}. "
        "The default 32-step sampling horizon matches the shortest task horizon in this archive. "
        f"{'Early termination uses the last valid observation when at least ten observations exist; actual endpoints are recorded. ' if limits.get('terminal_fallback') else 'Episodes terminating before their assigned endpoint are recorded as unavailable; no replacement endpoint. '}"
        "Zero-gradient samples are undefined and excluded from means. Counts are explicit.\n\n"
        "Each window contains observations o[t-9]..o[t], with the actual rollout carry at t-10 held fixed. "
        "This is window-conditional recall, normalized within the ten observations; it does not measure mass before the window. "
        "Gradients propagate through pixel inputs and uninterrupted sequential recurrence within the window; "
        "previous actions/rewards are held fixed. Action-wise absolute pixel L1 gradients are summed "
        "before normalization. Action gradients and trajectories run sequentially within each worker. "
        "FSQ uses the existing straight-through gradient. Equal-width bins use (j-(t-9))/9 and include 1 in the final bin. "
        "Sparse bins are expected for histories shorter than 16 observations. SEs describe evaluation samples, "
        "not training-seed variability. Old fixed-endpoint results remain in their original directories.\n\n"
        f"Hard limits: all diagnoses share {limits['cpu_cores']} allowed logical CPUs; "
        f"aggregate address-space budget {limits['memory_mib']} MiB "
        "(coordinator plus maximum worker slots). Each checkpoint runs in a fresh worker, releasing "
        "JAX compilation caches at exit. Address-space limits also bound resident memory. "
        "Concurrency decreases under host memory pressure or worker failures; queued failures remain visible.\n"
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--memory-mib", type=int, default=8192)
    parser.add_argument("--cpu-cores", type=int, default=2)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/recall_density_16_sliding")
    )
    parser.add_argument("--match", default="")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--reuse-manifest", type=Path)
    parser.add_argument("--terminal-fallback", action="store_true")
    parser.add_argument("--endpoint-horizon", type=int, default=32)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.worker:
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_mib * MIB,) * 2)
        argv = args.worker_args
        if argv and argv[0] == "--":
            argv = argv[1:]
        os.execv(
            sys.executable,
            [sys.executable, "-m", "plotting.density_analysis_memfix", *argv],
        )
    if args.selection:
        missing = [
            path
            for path in json.loads(args.selection.read_text())
            if not Path(path).is_file()
        ]
        if missing:
            parser.error(
                f"{len(missing)} selected weight files are missing or pruned. Regenerate plots from saved samples with plotting.report_main_recall, or select only retained policies. No diagnostic results were changed."
            )
    if args.endpoint_horizon < 17:
        parser.error("endpoint horizon must allow eight distinct endpoints >=9")
    if args.cpu_cores < 1 or args.max_concurrent < 1 or args.memory_mib < 1024:
        parser.error("positive CPU/concurrency and >=1024 MiB required")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    lock = (args.root / ".recall_queue.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("another recall queue is already running for this root")

    def stop_queue(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_queue)
    cpus = sorted(os.sched_getaffinity(0))[: args.cpu_cores]
    os.sched_setaffinity(0, cpus)
    os.nice(10)
    coordinator_mib = 256
    slots = min(args.max_concurrent, len(cpus))
    worker_mib = (args.memory_mib - coordinator_mib) // slots
    resource.setrlimit(
        resource.RLIMIT_AS, (coordinator_mib * MIB, args.memory_mib * MIB)
    )
    limits = {
        "cpu_cores": len(cpus),
        "cpu_affinity": cpus,
        "memory_mib": args.memory_mib,
        "worker_memory_mib": worker_mib,
        "max_concurrent": slots,
        "terminal_fallback": args.terminal_fallback,
    }
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    plan = make_plan(horizon=args.endpoint_horizon)
    (root / "endpoint_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    jobs = []
    selection = set(json.loads(args.selection.read_text())) if args.selection else None
    for path in sorted(args.root.rglob("model.eqx")):
        if selection is not None and str(path) not in selection:
            continue
        config = path.with_name("config.json")
        if (
            "provenance" in path.parts
            or not config.exists()
            or args.match not in str(path)
        ):
            continue
        m = json.loads(config.read_text())
        variant = m["variant"]
        if not (
            variant == "lru"
            or m.get("recurrent_state_quantization", "none") != "none"
            or ("prediction" in variant and not variant.startswith("mlp"))
            or variant.startswith("ib_")
            or variant == "information_bottleneck"
        ):
            continue
        identity = (
            str(path) + hashlib.sha256(path.read_bytes()).hexdigest() + json.dumps(plan)
        )
        identity += hashlib.sha256(
            Path("plotting/density_analysis_memfix.py").read_bytes()
        ).hexdigest()
        identity += str(args.terminal_fallback)
        token = hashlib.sha256(identity.encode()).hexdigest()[:20]
        manifest = root / "jobs" / token / "manifest.json"
        status = "queued"
        if manifest.exists() and all(
            r["status"] in ("complete", "undefined")
            for r in json.loads(manifest.read_text())
        ):
            status = "complete"
        jobs.append({"checkpoint": str(path), "id": token, "status": status})
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TF_NUM_INTRAOP_THREADS="1",
        TF_NUM_INTEROP_THREADS="1",
        MPLCONFIGDIR="/tmp/memfix-mpl",
        XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    )
    active = {}
    target = slots
    summarize(root, jobs, plan, limits)
    try:
        while active or any(job["status"] == "queued" for job in jobs):
            # Never exceed the configured hard slots; reduce under host pressure.
            available = (
                next(
                    int(line.split()[1])
                    for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:")
                )
                // 1024
            )
            capacity = target if available >= args.memory_mib else 1
            for job in jobs:
                if len(active) >= capacity:
                    break
                if job["status"] != "queued":
                    continue
                directory = root / "jobs" / job["id"]
                directory.mkdir(parents=True, exist_ok=True)
                log = (directory / "execution.log").open("a")
                remaining_budget = (
                    args.memory_mib
                    - coordinator_mib
                    - sum(
                        running_job["worker_memory_mib"]
                        for _, running_job, _ in active.values()
                    )
                )
                job_memory = remaining_budget // max(1, capacity - len(active))
                job["worker_memory_mib"] = job_memory
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "plotting.recall_queue",
                        "--worker",
                        "--memory-mib",
                        str(job_memory),
                        "--",
                        "--checkpoint",
                        job["checkpoint"],
                        "--definition",
                        "paper",
                        "--endpoint-plan",
                        str(root / "endpoint_plan.json"),
                        "--output",
                        str(directory),
                        *(["--terminal-fallback"] if args.terminal_fallback else []),
                        *(
                            ["--reuse-manifest", str(args.reuse_manifest)]
                            if args.reuse_manifest
                            else []
                        ),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                job.update(status="running", pid=process.pid)
                active[process.pid] = (process, job, log)
            for pid, (process, job, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                manifest = root / "jobs" / job["id"] / "manifest.json"
                ok = (
                    code == 0
                    and manifest.exists()
                    and all(
                        r["status"] in ("complete", "undefined")
                        for r in json.loads(manifest.read_text())
                    )
                )
                job.update(status="complete" if ok else "failed", returncode=code)
                if not ok:
                    target = max(1, target - 1)
                    log_text = (root / "jobs" / job["id"] / "execution.log").read_text()
                    if job.get("retries", 0) < 1 and (
                        code != 0 or "MemoryError" in log_text
                    ):
                        job.update(status="queued", retries=job.get("retries", 0) + 1)
                del active[pid]
                print(
                    f"{sum(j['status'] == 'complete' for j in jobs)}/{len(jobs)} complete; {job['status']}: {job['checkpoint']}",
                    flush=True,
                )
                summarize(root, jobs, plan, limits)
            time.sleep(1)
    finally:
        for process, job, log in active.values():
            process.terminate()
            process.wait()
            log.close()
            job["status"] = "interrupted"
        summarize(root, jobs, plan, limits)


if __name__ == "__main__":
    main()
