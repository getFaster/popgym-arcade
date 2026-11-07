# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns. Lower memory bias is better.

| Variant | Steps/seed | Seeds | Score | MLP − score |
|---|---:|---:|---:|---:|
| mlp | 98304 | 1 | 0.396763 | 0.000000 |
| lru | 98304 | 1 | 0.518694 | -0.121931 |
| information_bottleneck | 98304 | 1 | 0.441685 | -0.044922 |
| ib_reverse_kl | 98304 | 1 | 0.457031 | -0.060268 |
| ib_symmetric_kl | 98304 | 1 | 0.347656 | 0.049107 |
| ib_renyi_half | 98304 | 1 | 0.407087 | -0.010324 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB uses prior mean/log variance at both heads, with no direct recurrent-state path. Its score is therefore an auxiliary-memory training control, not a memory-dependent policy.
