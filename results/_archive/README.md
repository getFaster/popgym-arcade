# Archived recall diagnostics

`recall_diagnostics_superseded.tar.gz` preserves the superseded upstream-definition, initial paper-definition, full sliding-window sweep, pre-LRU snapshot, and duplicate final-job records. Archive member paths are relative to `results/`, retaining original provenance locations. `recall_cleanup.json` records original paths and verified SHA-256 hashes. Smoke-test outputs were removed.

Current results: [final intervention and plain-LRU report](../recall_density_16_main_final/REPORT.md). To restore historical paths, extract this archive into `results/` after checking for conflicting files. No training artifacts were moved or deleted.
