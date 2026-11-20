# LRU intervention results

Snapshot of saved results as of 2026-11-19. CartPoleHard is the only task shared by all three interventions. Each reported run used 98,304 actual training steps (100,000 requested) and 64 held-out evaluation episodes. Inputs were previous action, previous reward, and current observation.

## Metrics

- **MDP return** is $J(f,\pi,M)$ and **POMDP return** is $J(f,\pi,P)$. The artifacts contain separately trained and evaluated policies for each observation mode, so the gap compares those mode-specific runs. A fixed-policy gap would require evaluating the same checkpoint in both modes.
- **Observability gap** is $J(f,\pi,M)-J(f,\pi,P)$. A positive gap means higher return in MDP.
- **Memory bias** is $J(f,\pi,M)-J(\pi,M)$, using the plain feedforward MLP as the memoryless reference. Negative bias means the recurrent model scored below the MLP.
- The decomposition is $J(f,\pi,P)=J(\pi,M)+\mathrm{Bias}-\mathrm{Gap}$. Differences were computed before rounding.

## CartPoleHard KL and next-observation prediction

All rows use seed 0 and the plain MLP MDP reference of **0.705572**. Vanilla LRU is the control from the prediction suite.

| Condition | MDP return | POMDP return | Gap | Bias |
|---|---:|---:|---:|---:|
| Vanilla LRU | 0.455911 | 0.262448 | +0.193463 | -0.249661 |
| KL auxiliary loss (default IB) | 0.507656 | 0.280338 | +0.227318 | -0.197916 |
| LRU + next-observation and reward prediction | 0.605885 | 0.336484 | +0.269401 | -0.099687 |
| Default IB + prediction | 0.623854 | 0.287344 | +0.336510 | -0.081719 |

Prediction improved both returns over this vanilla LRU control. The MDP gain was larger (+0.149974 versus +0.074036), so the gap increased. Default IB also improved both returns within this comparison, although it changes the policy and value head inputs and adds inter-layer residuals. Its POMDP reference ran on a Tesla T4, while the prediction suite’s new runs used an RTX 2060.

Default IB uses reverse KL (posterior relative to prior), LRU output as posterior memory input, posterior statistics plus current input in the actor and critic, gradient weights of 0.4/1.6 for the prior and posterior, a one-bit free region per Gaussian distribution, and frozen RMS normalization with coefficient 1. Prediction uses coefficient 1 and jointly predicts the next 512-dimensional CNN embedding and reward for the action taken. Terminal embeddings are masked, and prediction gradients are stopped on both CNN paths. The prediction gain therefore covers the joint objective.

[Prediction results](../../results/lru_input_prediction_20261111/RESULTS.md) · [Run design](../../results/lru_input_prediction_20261111/EXPERIMENT.md) · [Plain MLP MDP result](../../results/lru_input_prediction_20261111/CartPoleHard/mdp/mlp_seed0/result.json)

## CartPoleHard FSQ

FSQ uses scalar levels of 3, 5, or 7 at scale 1. State FSQ quantizes the carried state, pre-readout FSQ quantizes the state used by the readout while carrying continuous state, and post-activation FSQ quantizes the activated output. The continuous control comes from the original FSQ sweep. All rows use seeds 0, 1, and 2 with the same per-run budget and evaluation count as above.

The main columns average the three seed-level returns and paired gaps. Gap SD is the sample standard deviation across training seeds. Bias averages the three seed-matched MDP differences against plain MLP. The MLP MDP returns were 0.705572, 0.520546, and 0.676953 for seeds 0–2. The seed-2 MLP ran on a Colab T4, while all FSQ runs used an RTX 2060.

| Condition | MDP return | POMDP return | Gap | Gap SD | Bias |
|---|---:|---:|---:|---:|---:|
| Continuous LRU | 0.605182 | 0.307291 | +0.297890 | 0.159962  -0.029175 |
| State FSQ, 3 levels | 0.674236 | 0.302760 | +0.371475 | 0.165109  +0.039878 |
| State FSQ, 5 levels | 0.645460 | 0.294739 | +0.350720 | 0.150268  +0.011102 |
| State FSQ, 7 levels | 0.657725 | 0.305373 | +0.352352 | 0.052849  +0.023368 |
| Pre-readout FSQ, 3 levels | 0.562656 | 0.364826 | +0.197830 | 0.041135  -0.071701 |
| Pre-readout FSQ, 5 levels | 0.650772 | 0.291015 | +0.359757 | 0.234504  +0.016415 |
| Pre-readout FSQ, 7 levels | 0.648619 | 0.317014 | +0.331606 | 0.112800  +0.014262 |
| Post-activation FSQ, 3 levels | 0.629999 | 0.347769 | +0.282231 | 0.163530  -0.004358 |
| Post-activation FSQ, 5 levels | 0.607309 | 0.364904 | +0.242404 | 0.135930  -0.027049 |
| Post-activation FSQ, 7 levels | 0.468576 | 0.331041 | +0.137535 | 0.071393  -0.165781 |
| State + post-activation FSQ, 3 levels | 0.673819 | 0.324383 | +0.349435 | 0.017304  +0.039462 |
| State + post-activation FSQ, 5 levels | 0.661527 | 0.335555 | +0.325972 | 0.075091  +0.027170 |
| State + post-activation FSQ, 7 levels | 0.642604 | 0.308663 | +0.333941 | 0.085446  +0.008247 |

Five-level post-activation FSQ had the highest mean POMDP return (0.364904 versus 0.307291 for continuous LRU) and a smaller gap (0.242404 versus 0.297890). Seven-level post-activation FSQ had a smaller gap still (0.137535), alongside a lower MDP return than the continuous control. These comparisons are descriptive across three seeds and multiple configurations.

[Follow-up seed metrics](../../results/memfix_short_20261119/metrics_by_seed.csv) · [Original FSQ results](../../results/fsq_lru_local_20261117_1420/) · [Post-activation FSQ results](../../results/fsq_activation_local_20261117_2108/) · [FSQ design](../fsq/README.md)

## Additional KL results

The separate residual ablation covers two reverse-KL configurations at seed 0. Its residual-enabled CartPoleHard row comes from the divergence suite. The prediction suite above uses a different saved MDP run for the same preset, so its MDP return differs. The earlier 16-configuration divergence sweep also has paired returns for four tasks in its CSV. The table below uses the default KL preset and the tested residual ablation. The seed-0 MDP MLP references are 0.408447 for NavigatorEasy, 0.391741 for MineSweeperEasy, and -0.838359 for BreakoutEasy.

| Task | KL residuals | MDP return | POMDP return | Gap | Bias |
|---|---|---:|---:|---:|---:|
| CartPoleHard | Enabled | 0.418593 | 0.280338 | +0.138255 | -0.286979 |
| CartPoleHard | Disabled | 0.324349 | 0.284245 | +0.040104 | -0.381224 |
| NavigatorEasy | Enabled | 0.403442 | 0.328735 | +0.074707 | -0.005005 |
| NavigatorEasy | Disabled | 0.087524 | -0.108765 | +0.196289 | -0.320923 |
| MineSweeperEasy | Enabled | 0.455357 | 0.352958 | +0.102400 | +0.063616 |
| MineSweeperEasy | Disabled | 0.344587 | 0.315011 | +0.029576 | -0.047154 |
| BreakoutEasy | Enabled | -0.684349 | -0.742162 | +0.057813 | +0.154010 |
| BreakoutEasy | Disabled | -0.704271 | -0.702865 | -0.001406 | +0.134089 |

[Residual ablation results](../../results/default_ib_no_residual_20261109/summary.csv) · [Divergence sweep](../../results/posterior_rms_suite_20261109/summary.csv)

## Missing results

- **Prediction and KL training-seed variability** is unavailable for the reported single-seed runs. The source reports’ episode standard errors quantify evaluation uncertainty conditional on each trained policy.
- **Longer-budget KL results** are unavailable in the million-step directory, which has no completed evaluation results. The final-output-only prediction pilot was superseded before its prediction run finished. The completed prediction variant above uses current input in its prediction heads.

[Million-step KL status](../../results/ib_million_steps_20261109/RESULTS.md) · [Superseded prediction pilot](../../results/lru_prediction_20261111/EXPERIMENT.md)
