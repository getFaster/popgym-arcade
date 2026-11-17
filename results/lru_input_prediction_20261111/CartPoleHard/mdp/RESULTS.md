# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns. Lower memory bias is better.

| Variant | Steps/seed | Seeds | Score | MLP − score |
|---|---:|---:|---:|---:|
| mlp | 98304 | 1 | 0.705572 | 0.000000 |
| lru | 98304 | 1 | 0.455911 | 0.249661 |
| lru_prediction | 98304 | 1 | 0.605885 | 0.099687 |
| mlp_prediction | 98304 | 1 | 0.640859 | 0.064713 |
| ib_reverse_prior20 | 98304 | 1 | 0.507656 | 0.197916 |
| ib_reverse_prior20_prediction | 98304 | 1 | 0.623854 | 0.081719 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
