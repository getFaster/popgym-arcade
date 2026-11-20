# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns.

| Variant | Steps/seed | Seeds | Score |
|---|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | -0.653750 |
| ib_forward_prior80 | 98304 | 1 | -0.768594 |
| ib_reverse_prior20 | 98304 | 1 | -0.684349 |
| ib_reverse_prior80 | 98304 | 1 | -0.608646 |
| ib_symmetric_prior20 | 98304 | 1 | -0.728412 |
| ib_symmetric_prior80 | 98304 | 1 | -0.801042 |
| ib_renyi_half_prior20 | 98304 | 1 | -0.789766 |
| ib_renyi_half_prior80 | 98304 | 1 | -0.691328 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
