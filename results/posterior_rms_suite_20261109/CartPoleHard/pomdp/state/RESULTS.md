# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns.

| Variant | Steps/seed | Seeds | Score |
|---|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | 0.256432 |
| ib_forward_prior80 | 98304 | 1 | 0.278958 |
| ib_reverse_prior20 | 98304 | 1 | 0.289609 |
| ib_reverse_prior80 | 98304 | 1 | 0.312448 |
| ib_symmetric_prior20 | 98304 | 1 | 0.330130 |
| ib_symmetric_prior80 | 98304 | 1 | 0.345963 |
| ib_renyi_half_prior20 | 98304 | 1 | 0.293906 |
| ib_renyi_half_prior80 | 98304 | 1 | 0.281328 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
