# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns. Lower memory bias is better.

| Variant | Steps/seed | Seeds | Score | MLP − score |
|---|---:|---:|---:|---:|
| mlp | 98304 | 1 | -0.024048 | 0.000000 |
| lru | 98304 | 1 | 0.404907 | -0.428955 |
| information_bottleneck | 98304 | 1 | 0.480347 | -0.504395 |
| ib_reverse_kl | 98304 | 1 | -0.372192 | 0.348145 |
| ib_symmetric_kl | 98304 | 1 | -0.024170 | 0.000122 |
| ib_renyi_half | 98304 | 1 | 0.159424 | -0.183472 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB uses prior mean/log variance at both heads, with no direct recurrent-state path. Its score is therefore an auxiliary-memory training control, not a memory-dependent policy.
