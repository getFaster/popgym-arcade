# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns. Lower memory bias is better.

| Variant | Steps/seed | Seeds | Score | MLP − score |
|---|---:|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | -0.654974 | pending |
| ib_forward_prior80 | 98304 | 1 | -0.642813 | pending |
| ib_reverse_prior20 | 98304 | 1 | -0.803646 | pending |
| ib_reverse_prior80 | 98304 | 1 | -0.743255 | pending |
| ib_symmetric_prior20 | 98304 | 1 | -0.736094 | pending |
| ib_symmetric_prior80 | 98304 | 1 | -0.847370 | pending |
| ib_renyi_half_prior20 | 98304 | 1 | -0.741146 | pending |
| ib_renyi_half_prior80 | 98304 | 1 | -0.876172 | pending |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
