# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns.

| Variant | Steps/seed | Seeds | Score |
|---|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | -0.125488 |
| ib_forward_prior80 | 98304 | 1 | 0.121826 |
| ib_reverse_prior20 | 98304 | 1 | 0.048828 |
| ib_reverse_prior80 | 98304 | 1 | 0.117920 |
| ib_symmetric_prior20 | 98304 | 1 | 0.419434 |
| ib_symmetric_prior80 | 98304 | 1 | 0.200562 |
| ib_renyi_half_prior20 | 98304 | 1 | 0.131958 |
| ib_renyi_half_prior80 | 98304 | 1 | -0.413452 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
