# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns.

| Variant | Steps/seed | Seeds | Score |
|---|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | 0.451328 |
| ib_forward_prior80 | 98304 | 1 | 0.309687 |
| ib_reverse_prior20 | 98304 | 1 | 0.394817 |
| ib_reverse_prior80 | 98304 | 1 | 0.355885 |
| ib_symmetric_prior20 | 98304 | 1 | 0.418567 |
| ib_symmetric_prior80 | 98304 | 1 | 0.319661 |
| ib_renyi_half_prior20 | 98304 | 1 | 0.478463 |
| ib_renyi_half_prior80 | 98304 | 1 | 0.532395 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
