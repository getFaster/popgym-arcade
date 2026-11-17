# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns. Lower memory bias is better.

| Variant | Steps/seed | Seeds | Score | MLP − score |
|---|---:|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | 0.256432 | pending |
| ib_forward_prior80 | 98304 | 1 | 0.278958 | pending |
| ib_reverse_prior20 | 98304 | 1 | 0.289609 | pending |
| ib_reverse_prior80 | 98304 | 1 | 0.312448 | pending |
| ib_symmetric_prior20 | 98304 | 1 | 0.330130 | pending |
| ib_symmetric_prior80 | 98304 | 1 | 0.345963 | pending |
| ib_renyi_half_prior20 | 98304 | 1 | 0.293906 | pending |
| ib_renyi_half_prior80 | 98304 | 1 | 0.281328 | pending |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
