"""Regenerate the pilot report and figure from retained raw artifacts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
runs = root / 'CartPoleHard/mdp'
variants = ['lru', 'lru_prediction']
results = [json.loads((runs / f'{v}_seed0/result.json').read_text()) for v in variants]
returns = [np.load(runs / f'{v}_seed0/evaluation_returns.npy') for v in variants]
means = [float(r.mean()) for r in returns]
ses = [float(r.std(ddof=1) / np.sqrt(r.size)) for r in returns]
delta = means[1] - means[0]
lines = ['# CartPoleHard MDP prediction pilot', '',
         '| Condition | Evaluation mean | Episode SE | Actual steps |',
         '|---|---:|---:|---:|']
for v, m, se, r in zip(variants, means, ses, results):
    lines.append(f'| {v} | {m:.6f} | {se:.6f} | {r["actual_steps"]} |')
lines += ['', f'Prediction minus baseline: **{delta:+.6f}**.', '',
          'Seed 0, 64 evaluation episodes per model. Episode SE is conditional on each trained model; it does not measure training-seed variability. This short single-seed pilot cannot establish a reproducible benefit.', '',
          '[Design and reproduction command](EXPERIMENT.md). [Figure](comparison.png).', '']
(root / 'RESULTS.md').write_text('\n'.join(lines))
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].bar(['Vanilla LRU', 'LRU + prediction'], means, yerr=ses, capsize=4)
axes[0].set_ylabel('Evaluation return (mean ± episode SE)')
for v in variants:
    logs = [json.loads(x) for x in (runs / f'{v}_seed0/training.jsonl').read_text().splitlines()]
    axes[1].plot([r['steps'] for r in logs], [r['training_return'] for r in logs], label=v, alpha=.8)
    if v == 'lru_prediction':
        for name in ['observation', 'reward']:
            axes[2].plot([r['steps'] for r in logs], [r[f'prediction_{name}_mse'] for r in logs], label=name)
axes[1].set_ylabel('Rollout episode return')
axes[2].set_ylabel('Auxiliary MSE (sum of branches)')
for ax in axes[1:]:
    ax.set_xlabel('Environment steps')
    ax.legend()
fig.suptitle('CartPoleHard MDP · seed 0 · 98,304 steps · coefficient 1')
fig.tight_layout()
fig.savefig(root / 'comparison.png', dpi=160)
print('\n'.join(lines))
