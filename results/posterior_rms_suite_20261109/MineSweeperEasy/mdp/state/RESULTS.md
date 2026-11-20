# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns.

| Variant | Steps/seed | Seeds | Score |
|---|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | 0.478237 |
| ib_forward_prior80 | 98304 | 1 | 0.366909 |
| ib_reverse_prior20 | 98304 | 1 | 0.444196 |
| ib_reverse_prior80 | 98304 | 1 | 0.480190 |
| ib_symmetric_prior20 | 98304 | 1 | 0.459263 |
| ib_symmetric_prior80 | 98304 | 1 | 0.446987 |
| ib_renyi_half_prior20 | 98304 | 1 | 0.443359 |
| ib_renyi_half_prior80 | 98304 | 1 | 0.423549 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
