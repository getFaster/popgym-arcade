# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns. Lower memory bias is better.

| Variant | Steps/seed | Seeds | Score | MLP − score |
|---|---:|---:|---:|---:|
| mlp | 98304 | 1 | 0.371015 | 0.000000 |
| lru | 98304 | 1 | 0.225625 | 0.145390 |
| lru_prediction | 98304 | 1 | 0.279349 | 0.091667 |
| mlp_prediction | 98304 | 1 | 0.322526 | 0.048490 |
| ib_reverse_prior20 | 98304 | 1 | 0.319557 | 0.051458 |
| ib_reverse_prior20_prediction | 98304 | 1 | 0.282604 | 0.088411 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
