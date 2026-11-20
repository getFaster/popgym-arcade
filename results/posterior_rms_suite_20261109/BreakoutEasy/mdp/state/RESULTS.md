# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns.

| Variant | Steps/seed | Seeds | Score |
|---|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | -0.654974 |
| ib_forward_prior80 | 98304 | 1 | -0.642813 |
| ib_reverse_prior20 | 98304 | 1 | -0.803646 |
| ib_reverse_prior80 | 98304 | 1 | -0.743255 |
| ib_symmetric_prior20 | 98304 | 1 | -0.736094 |
| ib_symmetric_prior80 | 98304 | 1 | -0.847370 |
| ib_renyi_half_prior20 | 98304 | 1 | -0.741146 |
| ib_renyi_half_prior80 | 98304 | 1 | -0.876172 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
