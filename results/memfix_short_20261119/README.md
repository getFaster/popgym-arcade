# MEMFIX short-budget completion runs

This batch extends the saved seed-0 KL and prediction experiments to seeds 0–2 and supplies the plain MDP MLP references for FSQ and the four KL tasks. Every fresh run requests 100,000 steps, reaches 98,304 actual steps, and evaluates 64 episodes.

The local RTX 2060 trains seed 1 and the previously missing seed-0 MLP references on NavigatorEasy, MineSweeperEasy, and BreakoutEasy. The Colab T4 trains seed 2. CartPoleHard includes the vanilla, prediction, default reverse-KL, and combined KL-plus-prediction conditions in both modes, along with plain and prediction MLP controls. The other three tasks extend the default reverse-KL condition in both modes. Their seed-0 recurrent runs come from the residual-enabled sweep. All saved FSQ conditions and their continuous control already have three seeds and final weights.

Training source commit: `c4012b1f1a3afdb48938f52e670d67f16fe7c7e6`. Each fresh run saves `model.eqx` and a full final training checkpoint under `checkpoints/step_000098304/`. Completed T4 runs are copied into `colab/` by `sync_colab.py`. The local runs are saved under `local/`.

`summarize.py` verifies the required model files and writes seed-level metrics and three-seed means with sample standard deviations. Blank aggregates indicate fewer than three measurements. The CSVs use recurrent MDP minus MLP MDP for bias. Device is confounded with training seed in the new runs, so the spread across seeds also includes any device effects.
