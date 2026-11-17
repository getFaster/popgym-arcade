# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns. Lower memory bias is better.

| Variant | Steps/seed | Seeds | Score | MLP − score |
|---|---:|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | -0.125488 | pending |
| ib_forward_prior80 | 98304 | 1 | 0.121826 | pending |
| ib_reverse_prior20 | 98304 | 1 | 0.048828 | pending |
| ib_reverse_prior80 | 98304 | 1 | 0.117920 | pending |
| ib_symmetric_prior20 | 98304 | 1 | 0.419434 | pending |
| ib_symmetric_prior80 | 98304 | 1 | 0.200562 | pending |
| ib_renyi_half_prior20 | 98304 | 1 | 0.131958 | pending |
| ib_renyi_half_prior80 | 98304 | 1 | -0.413452 | pending |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
