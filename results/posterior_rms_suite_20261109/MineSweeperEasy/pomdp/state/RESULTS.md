# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns.

| Variant | Steps/seed | Seeds | Score |
|---|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | 0.325056 |
| ib_forward_prior80 | 98304 | 1 | 0.318080 |
| ib_reverse_prior20 | 98304 | 1 | 0.297433 |
| ib_reverse_prior80 | 98304 | 1 | 0.285156 |
| ib_symmetric_prior20 | 98304 | 1 | 0.345145 |
| ib_symmetric_prior80 | 98304 | 1 | 0.340123 |
| ib_renyi_half_prior20 | 98304 | 1 | 0.301897 |
| ib_renyi_half_prior80 | 98304 | 1 | 0.374163 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
