# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns. Lower memory bias is better.

| Variant | Steps/seed | Seeds | Score | MLP − score |
|---|---:|---:|---:|---:|
| mlp | 98304 | 1 | 0.520546 | 0.000000 |
| lru | 98304 | 1 | 0.408750 | 0.111797 |
| lru_prediction | 98304 | 1 | 0.512578 | 0.007969 |
| mlp_prediction | 98304 | 1 | 0.599348 | -0.078802 |
| ib_reverse_prior20 | 98304 | 1 | 0.310026 | 0.210521 |
| ib_reverse_prior20_prediction | 98304 | 1 | 0.350833 | 0.169713 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
