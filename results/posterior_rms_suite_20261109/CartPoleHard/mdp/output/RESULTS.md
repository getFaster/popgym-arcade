# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns.

| Variant | Steps/seed | Seeds | Score |
|---|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | 0.473125 |
| ib_forward_prior80 | 98304 | 1 | 0.485729 |
| ib_reverse_prior20 | 98304 | 1 | 0.418593 |
| ib_reverse_prior80 | 98304 | 1 | 0.418489 |
| ib_symmetric_prior20 | 98304 | 1 | 0.485234 |
| ib_symmetric_prior80 | 98304 | 1 | 0.370494 |
| ib_renyi_half_prior20 | 98304 | 1 | 0.526666 |
| ib_renyi_half_prior80 | 98304 | 1 | 0.354661 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
