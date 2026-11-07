# Repository Guidelines

## Project Structure & Module Organization

`popgym_arcade/environments/` contains JAX environments and drawing utilities. `registration.py` registers tasks; `wrappers.py` provides environment wrappers. `popgym_arcade/train.py` defines upstream training defaults. Memory-bias models, PPO experiments, and reporting live in `popgym_arcade/experiments/`. Tests are in `tests/`; documentation and visual assets are in `docs/` and `imgs/`. Experiment artifacts belong in `results/`. Read `MEMORY_BIAS.md` and `DIVERGENCE_SUITE.md` before changing experiment behavior.

## Build, Test, and Development Commands

- `uv sync --extra baselines --extra cuda --dev`: install training, CUDA, and development dependencies.
- `uv build`: build distribution artifacts.
- `JAX_PLATFORMS=cpu uv run pytest -q tests/test_memory_bias.py`: run focused experiment tests without GPU allocation.
- `uv run python tests/check_lru_cuda.py`: validate LRU behavior on CUDA.
- `uv run ruff check popgym_arcade/experiments tests/test_memory_bias.py tests/check_lru_cuda.py`: lint experiment code.
- `uv run ruff format popgym_arcade/experiments tests/test_memory_bias.py tests/check_lru_cuda.py`: format experiment code.
- `JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python -m popgym_arcade.experiments.divergence_suite --output results/new_suite`: run the GPU pilot in a fresh directory.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` for modules/functions/variables, and `PascalCase` for classes. Use Ruff for formatting and linting. Keep JAX transformations compatible with explicit array shapes and PRNG keys. Avoid unrelated formatting changes to upstream files.

## Testing Guidelines

Use pytest; name files `test_*.py` and functions `test_*`. Cover changed numerical formulas, recurrent resets, rollout/replay consistency, and evaluation boundaries. Run affected environment or training tests when modifying those components. No numerical coverage threshold is configured. GPU validation complements CPU unit tests.

## Experiment Integrity

Preserve `input_t = (previous_action, previous_reward, current_observation)` across training and evaluation. Document divergence direction, head inputs, free-bit units, seeds, actual steps, and hyperparameter changes. Use fresh output directories for changed configurations; retain existing results and provenance. Distinguish evaluation-episode uncertainty from training-seed variability.

## Commit & Pull Request Guidelines

History uses short descriptive subjects, sometimes with PR numbers; no strict Conventional Commits pattern is evident. Describe the concrete change in commit subjects. PRs should explain motivation, behavior changes, validation commands, and relevant issues. Include plots for experimental claims and screenshots for rendering changes. Keep large checkpoints and credentials out of commits.
