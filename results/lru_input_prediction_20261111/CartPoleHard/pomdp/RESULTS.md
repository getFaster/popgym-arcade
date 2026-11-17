# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns. Lower memory bias is better.

| Variant | Steps/seed | Seeds | Score | MLP − score |
|---|---:|---:|---:|---:|
| mlp | 98304 | 1 | 0.315807 | 0.000000 |
| lru | 98304 | 1 | 0.262448 | 0.053359 |
| lru_prediction | 98304 | 1 | 0.336484 | -0.020677 |
| mlp_prediction | 98304 | 1 | 0.299088 | 0.016719 |
| ib_reverse_prior20 | 98304 | 1 | 0.280338 | 0.035469 |
| ib_reverse_prior20_prediction | 98304 | 1 | 0.287344 | 0.028464 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
