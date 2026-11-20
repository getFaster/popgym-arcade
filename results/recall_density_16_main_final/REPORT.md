# Whole-trajectory recall density, 16 bins

[Metrics](averages_16.csv) · [Plot](interventions_by_task.png) · [Per-trajectory results and checkpoint provenance](manifest.json)

Each task and observation mode has plain LRU, reverse-KL, prediction, and five-level post-activation FSQ, with three independently trained checkpoints (seeds 0–2, 98,304 steps). Each checkpoint has five episodes with reset seeds 1000000–1000004. The target is the final action distribution before termination. For each action, absolute gradients with respect to each observation's pixels are summed; action-wise masses are summed, normalized across the complete episode, and placed into 16 equal normalized-time bins. Prior actions and rewards are held fixed. The 16 values are one normalized recall distribution per intervention, task and observation mode, averaging trajectories within checkpoint and then the three checkpoint means equally. Zero-gradient trajectories are invalid and are never silently omitted. FSQ uses its straight-through gradient. Checkpoint paths and hashes appear in the linked manifest; training configuration and device are recorded alongside each checkpoint.

Previous window-conditional results are preserved in repository history. Checkpoints are stored separately from the report.
