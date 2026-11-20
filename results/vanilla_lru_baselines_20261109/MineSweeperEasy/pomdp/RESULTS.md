# Memory-bias pilot results

Scores are held-out stochastic-policy episode returns.

| Variant | Steps/seed | Seeds | Score |
|---|---:|---:|---:|
| lru | 98304 | 1 | 0.352958 |

A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.

IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.
