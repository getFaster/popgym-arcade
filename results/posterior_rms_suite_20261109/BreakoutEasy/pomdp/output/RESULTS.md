# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns.

| Variant | Steps/seed | Seeds | Score |
|---|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | -0.874167 |
| ib_forward_prior80 | 98304 | 1 | -0.819766 |
| ib_reverse_prior20 | 98304 | 1 | -0.742162 |
| ib_reverse_prior80 | 98304 | 1 | -0.686797 |
| ib_symmetric_prior20 | 98304 | 1 | -0.773750 |
| ib_symmetric_prior80 | 98304 | 1 | -0.814349 |
| ib_renyi_half_prior20 | 98304 | 1 | -0.660677 |
| ib_renyi_half_prior80 | 98304 | 1 | -0.717760 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
