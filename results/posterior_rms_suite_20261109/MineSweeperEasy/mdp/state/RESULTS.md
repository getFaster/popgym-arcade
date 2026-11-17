# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns. Lower memory bias is better.

| Variant | Steps/seed | Seeds | Score | MLP − score |
|---|---:|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | 0.478237 | pending |
| ib_forward_prior80 | 98304 | 1 | 0.366909 | pending |
| ib_reverse_prior20 | 98304 | 1 | 0.444196 | pending |
| ib_reverse_prior80 | 98304 | 1 | 0.480190 | pending |
| ib_symmetric_prior20 | 98304 | 1 | 0.459263 | pending |
| ib_symmetric_prior80 | 98304 | 1 | 0.446987 | pending |
| ib_renyi_half_prior20 | 98304 | 1 | 0.443359 | pending |
| ib_renyi_half_prior80 | 98304 | 1 | 0.423549 | pending |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
