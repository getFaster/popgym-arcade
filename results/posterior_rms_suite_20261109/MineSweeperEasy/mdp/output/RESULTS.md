# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns.

| Variant | Steps/seed | Seeds | Score |
|---|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | 0.505859 |
| ib_forward_prior80 | 98304 | 1 | 0.440011 |
| ib_reverse_prior20 | 98304 | 1 | 0.455357 |
| ib_reverse_prior80 | 98304 | 1 | 0.343192 |
| ib_symmetric_prior20 | 98304 | 1 | 0.339565 |
| ib_symmetric_prior80 | 98304 | 1 | 0.456194 |
| ib_renyi_half_prior20 | 98304 | 1 | 0.410435 |
| ib_renyi_half_prior80 | 98304 | 1 | 0.389230 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
