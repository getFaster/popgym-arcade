# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns. Lower memory bias is better.

| Variant | Steps/seed | Seeds | Score | MLP − score |
|---|---:|---:|---:|---:|
| mlp | 98304 | 1 | 0.705572 | 0.000000 |
| lru | 98304 | 1 | 0.600703 | 0.104870 |
| information_bottleneck | 98304 | 1 | 0.646822 | 0.058750 |
| ib_reverse_kl | 98304 | 1 | 0.548593 | 0.156979 |
| ib_symmetric_kl | 98304 | 1 | 0.590078 | 0.115495 |
| ib_renyi_half | 98304 | 1 | 0.625859 | 0.079713 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB uses prior mean/log variance at both heads, with no direct recurrent-state path. Its score is therefore an auxiliary-memory training control, not a memory-dependent policy.
