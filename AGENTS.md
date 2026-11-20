# MEMFIX

MEMFIX (Mitigating Memory-Induced Performance Degradation) builds on POPGym Arcade. It tests interventions that improve linear recurrent unit (LRU) policies under full and partial observability.

## Interventions and latest results

CartPoleHard results below use 98,304 training steps per run and 64 held-out evaluation episodes. See [the intervention report](docs/reports/lru_interventions.md) for controls, all FSQ settings, and source artifacts.

| Intervention | MDP | POMDP | Gap | Bias |
|---|---:|---:|---:|---:|
| Reverse-KL auxiliary loss (default information bottleneck) | 0.507656 | 0.280338 | +0.227318 | −0.197916 |
| Next-observation embedding and reward prediction | 0.605885 | 0.336484 | +0.269401 | −0.099687 |
| Five-level post-activation finite scalar quantization (FSQ) | 0.607309 | 0.364904 | +0.242404 | −0.027049 |

KL and prediction use seed 0. FSQ metrics average seeds 0–2. Its seed-2 MLP reference used a Colab T4, while FSQ used an RTX 2060. The million-step KL suite has no completed evaluations.

## Experiment integrity

Gap = recurrent MDP return − recurrent POMDP return. Bias = recurrent MDP return − memoryless MLP MDP return. Recurrent POMDP return = memoryless MDP return + Bias − Gap. Saved gaps compare separately trained policies for each observation mode.

Preserve `input_t = (previous_action, previous_reward, current_observation)` across training and evaluation. Record divergence direction, head inputs, free-bit units, seeds, actual steps, and hyperparameter changes. Use fresh output directories for changed configurations and retain provenance. Distinguish evaluation episode uncertainty from training-seed variability.

## Repository workflow

JAX environments and drawing utilities live in `popgym_arcade/environments/`. Task registration and wrappers are in `registration.py` and `wrappers.py` within the package. Upstream defaults are in `popgym_arcade/train.py`. Models, PPO experiments, and reporting live in `popgym_arcade/experiments/`. Store artifacts in `results/`. Read [MEMORY_BIAS.md](MEMORY_BIAS.md) and [DIVERGENCE_SUITE.md](DIVERGENCE_SUITE.md) before changing experiment behavior.

- Install dependencies with `uv sync --extra baselines --extra cuda --dev`. Build with `uv build`.
- Run focused tests with `JAX_PLATFORMS=cpu uv run pytest -q tests/test_memory_bias.py tests/test_fsq.py`. Validate CUDA with `uv run python tests/check_lru_cuda.py`.
- Use Ruff, four-space indentation, snake_case for modules and functions, and PascalCase for classes. Keep JAX shapes and PRNG keys explicit. Avoid unrelated upstream changes.
- Add focused coverage for changed formulas, recurrent resets, rollout/replay consistency, and evaluation boundaries. Run affected environment or training tests.
- Use short, descriptive commit subjects. PRs should explain motivation, behavior changes, and validation. Include plots for experimental claims and screenshots for rendering changes. Keep large checkpoints and credentials out of commits.
