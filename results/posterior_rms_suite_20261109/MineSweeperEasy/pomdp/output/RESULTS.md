# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns. Lower memory bias is better.

| Variant | Steps/seed | Seeds | Score | MLP − score |
|---|---:|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | 0.298270 | pending |
| ib_forward_prior80 | 98304 | 1 | 0.271205 | pending |
| ib_reverse_prior20 | 98304 | 1 | 0.352958 | pending |
| ib_reverse_prior80 | 98304 | 1 | 0.292969 | pending |
| ib_symmetric_prior20 | 98304 | 1 | 0.321987 | pending |
| ib_symmetric_prior80 | 98304 | 1 | 0.330078 | pending |
| ib_renyi_half_prior20 | 98304 | 1 | 0.263393 | pending |
| ib_renyi_half_prior80 | 98304 | 1 | 0.308315 | pending |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
