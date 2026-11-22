# Million-step LRU and default IB comparison

**0/24 runs complete.** Colab GPU; actual accelerator is recorded per run. Four tasks × MDP/POMDP × vanilla LRU / default IB with residuals / default IB without residuals. Seed 0, 1,000,000 requested / 999,424 actual training steps, 64 stochastic-policy evaluation episodes. PPO settings are unchanged; the learning-rate schedule spans this longer budget.

IB uses reverse KL, output posterior input, prior20 gradient weights 0.4/1.6, coefficient 1, one free bit per Gaussian event, and fresh frozen branch RMS scales. Vanilla LRU has no added residual or bottleneck. Every condition receives `(previous_action, previous_reward, current_observation)`; IB heads receive posterior statistics.

Resumable checkpoints retain model, optimizer/schedule state, environment and recurrent state, and RNG. Periodic checkpoints occur at the first complete 2,048-step rollout crossing 100K-step thresholds, with an additional final checkpoint at 999,424 steps. Completed checkpoints are synced locally. A resumed run continues its original 1M schedule and optimizer state.

| Task | Mode | Condition | Return ± episode SE | Checkpoints |
|---|---|---|---:|---:|

The three conditions are compared at the same training budget. One seed and episode SE do not establish training-seed variability. Calibration is repeated for each IB architecture; normalization scales therefore differ. Historical 100K results are not reused for training.
