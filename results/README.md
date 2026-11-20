# Results index

## Current recall-density results

[Final intervention and plain-LRU report](recall_density_16_main_final/REPORT.md) contains the current task-level graphs, PDF, 16-bin averages, and sample-level results. The report uses eight trajectories per final checkpoint, shared sampled endpoints, and the last-valid-step fallback for episodes with at least ten observations. Completed execution logs are under `recall_density_16_main_final/logs/`; `jobs/` contains only the 48 jobs contributing to the report.

Regenerate the plots from saved diagnostics without loading trained weights:

```bash
MPLCONFIGDIR=/tmp/memfix-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0,1 .venv/bin/python -m plotting.report_main_recall
```

## Retained checkpoint weights

[CHECKPOINTS.json](CHECKPOINTS.json) records every retained and removed weight file and the verification of rewritten transfer archives. Only the 36 final policies for default reverse-KL, LRU prediction, and five-level post-activation FSQ are retained, along with 20 matching resumable training states. All are final 98,304-step checkpoints. Other configurations, MLP and plain-LRU controls, and earlier snapshots have no remaining weights. Their configurations, metrics, training logs, evaluation records, and source provenance remain available. Historical checkpoint metadata may describe deliberately removed weights.

The recall report still contains all 48 selected checkpoint measurements, including 12 plain-LRU controls; its `selection.json` documents the measurement sources and is not an inventory of remaining weight files. Re-evaluating a removed policy requires restoring its weights from another copy. Plot regeneration uses the saved measurements and needs no policy weights.

## Experiment records

| Directory | Contents |
|---|---|
| `memfix_short_20261119/` | Three-seed completion runs and consolidated return metrics |
| `lru_input_prediction_20261111/` | Main seed-0 CartPoleHard KL and prediction runs |
| `fsq_activation_local_20261117_2108/` | Post-activation FSQ and state-plus-activation sweep |
| `fsq_lru_local_20261117_1420/` | State/pre-readout FSQ sweep and continuous controls |
| `posterior_rms_suite_20261109/` | Four-task Gaussian-divergence sweep |
| `default_ib_no_residual_20261109/` | KL residual ablation |
| `vanilla_lru_baselines_20261109/` | Plain-LRU control records |
| `memory_bias_pilot/` | Historical pilot records |
| `lru_prediction_20261111/` | Superseded prediction pilot records |
| `ib_million_steps_20261109/` | Incomplete million-step suite and source provenance |
| `development_checks/` | Development-validation records |

## Superseded diagnostics

[`_archive/`](_archive/README.md) contains a verified compressed archive of the earlier recall definitions, full sweeps, and duplicate job records, with original paths and SHA-256 hashes. Smoke outputs were removed. Checkpoint weights were deleted according to the inventory above, not moved into the archive.
