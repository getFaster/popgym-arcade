# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns. Lower memory bias is better.

| Variant | Steps/seed | Seeds | Score | MLP − score |
|---|---:|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | 0.173462 | pending |
| ib_forward_prior80 | 98304 | 1 | 0.207153 | pending |
| ib_reverse_prior20 | 98304 | 1 | 0.328735 | pending |
| ib_reverse_prior80 | 98304 | 1 | 0.280640 | pending |
| ib_symmetric_prior20 | 98304 | 1 | -0.027954 | pending |
| ib_symmetric_prior80 | 98304 | 1 | 0.155884 | pending |
| ib_renyi_half_prior20 | 98304 | 1 | 0.143188 | pending |
| ib_renyi_half_prior80 | 98304 | 1 | 0.039551 | pending |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
