# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns. Lower memory bias is better.

| Variant | Steps/seed | Seeds | Score | MLP − score |
|---|---:|---:|---:|---:|
| mlp | 98304 | 1 | -0.731693 | 0.000000 |
| lru | 98304 | 1 | -0.703411 | -0.028281 |
| information_bottleneck | 98304 | 1 | -0.614297 | -0.117396 |
| ib_reverse_kl | 98304 | 1 | -0.532500 | -0.199193 |
| ib_symmetric_kl | 98304 | 1 | -0.515651 | -0.216042 |
| ib_renyi_half | 98304 | 1 | -0.726250 | -0.005443 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB uses prior mean/log variance at both heads, with no direct recurrent-state path. Its score is therefore an auxiliary-memory training control, not a memory-dependent policy.
