# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns.

| Variant | Steps/seed | Seeds | Score |
|---|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | 0.290755 |
| ib_forward_prior80 | 98304 | 1 | 0.334974 |
| ib_reverse_prior20 | 98304 | 1 | 0.280338 |
| ib_reverse_prior80 | 98304 | 1 | 0.322317 |
| ib_symmetric_prior20 | 98304 | 1 | 0.313489 |
| ib_symmetric_prior80 | 98304 | 1 | 0.326276 |
| ib_renyi_half_prior20 | 98304 | 1 | 0.294114 |
| ib_renyi_half_prior80 | 98304 | 1 | 0.297552 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
