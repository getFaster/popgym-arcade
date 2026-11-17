# Colab million-step experiment

The user explicitly resumed this experiment on 2026-11-09. T4, L4, and A100
allocations all returned HTTP 503. No runtime or official run was created and no
supervisor was started. The final source and launch workflow are ready for a
future allocation attempt; GPU validation remains unrun. The actual device will
be recorded in every training run. Training must remain on Colab.

`source/`, `source_manifest.json`, and `source.tar.gz` preserve the exact source
including uncommitted implementation, tests, dependency lock, and experiment docs.
`handoff_original_state.json` is the original stopped-state handoff. The live
repository-level `EXPERIMENT_1M_STATE.json` is updated after each verified sync.

The supervisor uses 32 MiB checksummed chunks, validates all downloaded files,
and publishes whole checkpoint directories only after checking the state against
its sidecar. It retains every checkpoint locally and remotely. Verified immutable
weights are acknowledged so subsequent syncs transfer only new weights and fresh
logs/metadata. Transfer files are temporary; checkpoint files are never deleted.

For an expired runtime, the supervisor checks server assignments before allocating
one replacement. It restores the frozen source, run configs, logs, calibrations,
completed results, and latest verified full checkpoint of every unfinished run.
All older checkpoints remain local. Each new runtime repeats the full-size CUDA
smoke check before resuming the queue. It permits at most four replacements.

The full-size CUDA smoke uses actual PPO updates and actual evaluation for vanilla
LRU and default IB, interrupts after a durable checkpoint, resumes at the same
budget, verifies gradients, and extends IB into a fresh directory while checking
exact frozen calibration reuse. Smoke artifacts are temporary and excluded from
official experiment results.

The supervisor releases its runtime after final artifact transfer, and also on a
worker failure after transfer. Check `sync_status.json`, `sync_error.json`,
`supervisor.log`, and the synced `../colab_worker.log` when diagnosing progress.
To request an orderly stop, create `provenance/STOP`: the supervisor terminates the
worker, drains published checkpoints, and releases its runtime. Remove that marker
only when explicitly resuming again.

Only one supervisor may hold `supervisor.lock`. To reattach to a healthy runtime,
run the frozen supervisor without `--deploy`. Use `--deploy` only for a newly
allocated empty runtime:

```bash
.venv/bin/python results/ib_million_steps_20261109/provenance/source/docs/experiment_handoff/colab_million_supervisor.py \
  --output results/ib_million_steps_20261109 --session ib-million-l4 --gpu L4
```

This file describes the workflow, not evidence that a runtime or training run has
started. Consult the live state and sync timestamp for the current status.
