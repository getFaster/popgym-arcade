# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns.

| Variant | Steps/seed | Seeds | Score |
|---|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | 0.173462 |
| ib_forward_prior80 | 98304 | 1 | 0.207153 |
| ib_reverse_prior20 | 98304 | 1 | 0.328735 |
| ib_reverse_prior80 | 98304 | 1 | 0.280640 |
| ib_symmetric_prior20 | 98304 | 1 | -0.027954 |
| ib_symmetric_prior80 | 98304 | 1 | 0.155884 |
| ib_renyi_half_prior20 | 98304 | 1 | 0.143188 |
| ib_renyi_half_prior80 | 98304 | 1 | 0.039551 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
