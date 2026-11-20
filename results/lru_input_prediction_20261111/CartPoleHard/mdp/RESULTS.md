# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns.

| Variant | Steps/seed | Seeds | Score |
|---|---:|---:|---:|
| mlp | 98304 | 1 | 0.705572 |
| lru | 98304 | 1 | 0.455911 |
| lru_prediction | 98304 | 1 | 0.605885 |
| mlp_prediction | 98304 | 1 | 0.640859 |
| ib_reverse_prior20 | 98304 | 1 | 0.507656 |
| ib_reverse_prior20_prediction | 98304 | 1 | 0.623854 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
