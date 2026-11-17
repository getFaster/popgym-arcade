# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns. Lower memory bias is better.

| Variant | Steps/seed | Seeds | Score | MLP − score |
|---|---:|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | 0.325056 | pending |
| ib_forward_prior80 | 98304 | 1 | 0.318080 | pending |
| ib_reverse_prior20 | 98304 | 1 | 0.297433 | pending |
| ib_reverse_prior80 | 98304 | 1 | 0.285156 | pending |
| ib_symmetric_prior20 | 98304 | 1 | 0.345145 | pending |
| ib_symmetric_prior80 | 98304 | 1 | 0.340123 | pending |
| ib_renyi_half_prior20 | 98304 | 1 | 0.301897 | pending |
| ib_renyi_half_prior80 | 98304 | 1 | 0.374163 | pending |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
