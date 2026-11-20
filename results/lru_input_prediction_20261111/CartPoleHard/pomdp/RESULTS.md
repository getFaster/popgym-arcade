# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns.

| Variant | Steps/seed | Seeds | Score |
|---|---:|---:|---:|
| mlp | 98304 | 1 | 0.315807 |
| lru | 98304 | 1 | 0.262448 |
| lru_prediction | 98304 | 1 | 0.336484 |
| mlp_prediction | 98304 | 1 | 0.299088 |
| ib_reverse_prior20 | 98304 | 1 | 0.280338 |
| ib_reverse_prior20_prediction | 98304 | 1 | 0.287344 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
