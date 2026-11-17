# CartPoleHard MDP/POMDP prediction and memory bias

Seed 0; 98,304 steps; 64 held-out episodes per model. Memory bias = MLP return − condition return within each mode. Lower bias is better. Episode SE describes evaluation uncertainty, not training-seed variability.

| Mode | Condition | Return | Episode SE | Memory bias |
|---|---|---:|---:|---:|
| MDP | MLP | 0.705572 | 0.022491 | 0.000000 |
| MDP | MLP + prediction | 0.640859 | 0.022095 | 0.064713 |
| MDP | LRU | 0.455911 | 0.016830 | 0.249661 |
| MDP | LRU + prediction | 0.605885 | 0.022459 | 0.099687 |
| MDP | Default IB | 0.507656 | 0.022777 | 0.197916 |
| MDP | IB + prediction | 0.623854 | 0.024923 | 0.081719 |
| POMDP | MLP | 0.315807 | 0.015760 | 0.000000 |
| POMDP | MLP + prediction | 0.299088 | 0.011625 | 0.016719 |
| POMDP | LRU | 0.262448 | 0.010552 | 0.053359 |
| POMDP | LRU + prediction | 0.336484 | 0.018281 | -0.020677 |
| POMDP | Default IB | 0.280338 | 0.012669 | 0.035469 |
| POMDP | IB + prediction | 0.287344 | 0.012312 | 0.028464 |

| Mode | MLP prediction gain | LRU prediction gain | IB prediction gain |
|---|---:|---:|---:|
| MDP | -0.064713 | +0.149974 | +0.116198 |
| POMDP | -0.016719 | +0.074036 | +0.007005 |

Matched auxiliary memory bias uses MLP + prediction as the reference for prediction variants:

| Mode | LRU + prediction bias | LRU bias reduction vs plain pair | IB + prediction bias | IB bias reduction vs plain pair |
|---|---:|---:|---:|---:|
| MDP | +0.034974 | +0.214687 | +0.017005 | +0.180911 |
| POMDP | -0.037396 | +0.090755 | +0.011745 | +0.023724 |

MDP-to-POMDP return gaps are a separate comparison from MLP-relative memory bias:

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

[Design and commands](EXPERIMENT.md). [Comparison plot](comparison.png). [Artifact audit](artifact_audit.json).
