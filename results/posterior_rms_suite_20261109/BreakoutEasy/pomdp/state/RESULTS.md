# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns.

| Variant | Steps/seed | Seeds | Score |
|---|---:|---:|---:|
| ib_forward_prior20 | 98304 | 1 | -0.873724 |
| ib_forward_prior80 | 98304 | 1 | -0.767240 |
| ib_reverse_prior20 | 98304 | 1 | -0.852865 |
| ib_reverse_prior80 | 98304 | 1 | -0.793984 |
| ib_symmetric_prior20 | 98304 | 1 | -0.779896 |
| ib_symmetric_prior80 | 98304 | 1 | -0.826563 |
| ib_renyi_half_prior20 | 98304 | 1 | -0.797083 |
| ib_renyi_half_prior80 | 98304 | 1 | -0.872500 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
