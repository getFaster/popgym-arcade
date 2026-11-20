# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns.

| Variant | Steps/seed | Seeds | Score |
|---|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | 0.312134 |
| ib_forward_prior80 | 98304 | 1 | 0.375610 |
| ib_reverse_prior20 | 98304 | 1 | 0.404541 |
| ib_reverse_prior80 | 98304 | 1 | 0.296387 |
| ib_symmetric_prior20 | 98304 | 1 | 0.032715 |
| ib_symmetric_prior80 | 98304 | 1 | 0.125854 |
| ib_renyi_half_prior20 | 98304 | 1 | 0.220093 |
| ib_renyi_half_prior80 | 98304 | 1 | 0.345459 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
