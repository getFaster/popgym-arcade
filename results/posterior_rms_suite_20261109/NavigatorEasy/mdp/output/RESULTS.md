# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns.

| Variant | Steps/seed | Seeds | Score |
|---|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | 0.264404 |
| ib_forward_prior80 | 98304 | 1 | 0.103149 |
| ib_reverse_prior20 | 98304 | 1 | 0.403442 |
| ib_reverse_prior80 | 98304 | 1 | 0.217285 |
| ib_symmetric_prior20 | 98304 | 1 | 0.462524 |
| ib_symmetric_prior80 | 98304 | 1 | 0.496460 |
| ib_renyi_half_prior20 | 98304 | 1 | 0.273438 |
| ib_renyi_half_prior80 | 98304 | 1 | 0.261353 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
