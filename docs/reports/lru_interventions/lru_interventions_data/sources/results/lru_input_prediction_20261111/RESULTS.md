# CartPoleHard MDP/POMDP prediction results

Seed 0; 98,304 steps; 64 held-out episodes per model. Episode SE describes evaluation uncertainty, not training-seed variability.

| Mode | Condition | Return | Episode SE |
|---|---|---:|---:|
| MDP | MLP | 0.705572 | 0.022491 |
| MDP | MLP + prediction | 0.640859 | 0.022095 |
| MDP | LRU | 0.455911 | 0.016830 |
| MDP | LRU + prediction | 0.605885 | 0.022459 |
| MDP | Default IB | 0.507656 | 0.022777 |
| MDP | IB + prediction | 0.623854 | 0.024923 |
| POMDP | MLP | 0.315807 | 0.015760 |
| POMDP | MLP + prediction | 0.299088 | 0.011625 |
| POMDP | LRU | 0.262448 | 0.010552 |
| POMDP | LRU + prediction | 0.336484 | 0.018281 |
| POMDP | Default IB | 0.280338 | 0.012669 |
| POMDP | IB + prediction | 0.287344 | 0.012312 |

| Mode | MLP prediction gain | LRU prediction gain | IB prediction gain |
|---|---:|---:|---:|
| MDP | -0.064713 | +0.149974 | +0.116198 |
| POMDP | -0.016719 | +0.074036 | +0.007005 |

### MDP-to-POMDP return gaps

| Condition | MDP − POMDP return |
|---|---:|
| MLP | +0.389765 |
| MLP + prediction | +0.341771 |
| LRU | +0.193463 |
| LRU + prediction | +0.269401 |
| Default IB | +0.227318 |
| IB + prediction | +0.336510 |

Hardware: the reused POMDP default-IB baseline was trained on a Tesla T4; all new runs and the reused MLP/LRU references use RTX 2060. JAX is 0.6.2. Hardware differences may affect training numerics, so the small POMDP IB difference is not a clean hardware-matched auxiliary ablation.

Both IB conditions use the default reverse KL, posterior output input, 0.4/1.6 gradient balancing, one-bit full-distribution free region, frozen RMS calibration, and inter-layer LRU residuals. Vanilla LRU conditions both disable added inter-layer residuals. LRU/IB prediction inputs include input_t; MLP prediction uses its shared 256-dimensional hidden layer. CNN gradients from prediction are stopped on current and target paths. Prediction coefficient is 1.

These are short, single-seed comparisons. Prediction adds parameters; observation/reward losses are tested together. The common coefficient does not match auxiliary gradient strength across architectures; prediction MSE is not RMS normalized. Existing matching MDP MLP and POMDP LRU/IB references are copied byte-for-byte with provenance; MDP LRU/IB were completed before extending to both modes.

[Design and commands](EXPERIMENT.md). [Artifact audit](artifact_audit.json).
