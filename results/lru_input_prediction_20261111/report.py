"""Audit and report paired MDP/POMDP memory-bias experiments (allows pending runs)."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
variants = ['mlp', 'mlp_prediction', 'lru', 'lru_prediction', 'ib_reverse_prior20', 'ib_reverse_prior20_prediction']
labels = ['MLP', 'MLP + prediction', 'LRU', 'LRU + prediction', 'Default IB', 'IB + prediction']
reference_records = json.loads((root/'provenance/reused_references.json').read_text())
reused = {(r['mode'], r['variant']): r for r in reference_records}
records = {}
rows = []
config = None
for mode in ['mdp', 'pomdp']:
    for v in variants:
        directory = root/f'CartPoleHard/{mode}/{v}_seed0'
        if not (directory/'result.json').exists():
            continue
        r = json.loads((directory/'result.json').read_text())
        scores = np.load(directory/'evaluation_returns.npy')
        cfg = dict(r['config'])
        assert cfg.pop('PARTIAL') == (mode == 'pomdp')
        config = cfg if config is None else config
        assert cfg == config
        assert r['seed'] == 0 and r['actual_steps'] == 98304 and scores.size == r['evaluation_episodes'] == 64
        assert np.isfinite(scores).all()
        np.testing.assert_allclose(scores.mean(), r['evaluation_mean'], atol=1e-7)
        np.testing.assert_allclose(scores.std(ddof=1)/8, r['evaluation_standard_error'], atol=1e-7)
        if (mode, v) in reused:
            for name, digest in reused[mode, v]['sha256'].items():
                assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
        else:
            for name, digest in r['source_sha256'].items():
                assert any(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p in (root/'provenance').rglob(name)), (v, name)
        if v.endswith('_prediction'):
            assert r['prediction_coefficient'] == 1
            expected = {'lru_prediction':'input_t, final_lru_output', 'ib_reverse_prior20_prediction':'input_t, posterior_mean, posterior_log_variance', 'mlp_prediction':'shared_mlp_hidden'}.get(v)
            if expected is None:
                expected = r['prediction_head_inputs']
            assert r['prediction_head_inputs'] == expected
        records[mode, v] = r
    baseline = records.get((mode, 'mlp'))
    for v, label in zip(variants, labels):
        r = records.get((mode,v))
        if r is None:
            continue
        bias = baseline['evaluation_mean'] - r['evaluation_mean'] if baseline else None
        rows.append(dict(mode=mode, variant=v, mean=r['evaluation_mean'], episode_se=r['evaluation_standard_error'], memory_bias=bias, actual_steps=r['actual_steps'], device=r.get('device_kind'), parameters=r['parameter_count']))
    if (mode,'ib_reverse_prior20_prediction') in records:
        a,b = [records[mode,v] for v in ['ib_reverse_prior20','ib_reverse_prior20_prediction']]
        for field in ['rms_calibration','rms_scales_bits','lru_residual','posterior_input','ib_coefficient','ib_free_bits','divergence','kl_prior_gradient_weight','kl_posterior_gradient_weight']:
            assert a[field] == b[field], field
        assert a['lru_residual'] and a['posterior_input'] == 'output'
        assert a['divergence'] == 'reverse_kl' and a['kl_prior_gradient_weight'] == .4 and a['kl_posterior_gradient_weight'] == 1.6
    if (mode,'lru_prediction') in records:
        assert records[mode,'lru']['lru_residual'] == records[mode,'lru_prediction']['lru_residual'] == False
with (root/'summary.csv').open('w') as f:
    writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
lines = ['# CartPoleHard MDP/POMDP prediction and memory bias', '',
         'Seed 0; 98,304 steps; 64 held-out episodes per model. Memory bias = MLP return − condition return within each mode. Lower bias is better. Episode SE describes evaluation uncertainty, not training-seed variability.', '',
         '| Mode | Condition | Return | Episode SE | Memory bias |', '|---|---|---:|---:|---:|']
for mode in ['mdp','pomdp']:
    for v,label in zip(variants,labels):
        row=next((r for r in rows if r['mode']==mode and r['variant']==v),None)
        if row:
            bias=f"{row['memory_bias']:.6f}" if row['memory_bias'] is not None else 'pending MLP'
            lines.append(f"| {mode.upper()} | {label} | {row['mean']:.6f} | {row['episode_se']:.6f} | {bias} |")
        else:
            lines.append(f'| {mode.upper()} | {label} | pending | | |')
lines += ['', '| Mode | MLP prediction gain | LRU prediction gain | IB prediction gain |','|---|---:|---:|---:|']
for mode in ['mdp','pomdp']:
    gains=[]
    for base,pred in [('mlp','mlp_prediction'),('lru','lru_prediction'),('ib_reverse_prior20','ib_reverse_prior20_prediction')]:
        gain = records[mode,pred]['evaluation_mean'] - records[mode,base]['evaluation_mean'] if (mode,pred) in records and (mode,base) in records else None
        gains.append(f'{gain:+.6f}' if gain is not None else 'pending')
    lines.append(f'| {mode.upper()} | {gains[0]} | {gains[1]} | {gains[2]} |')
lines += ['', 'Matched auxiliary memory bias uses MLP + prediction as the reference for prediction variants:', '', '| Mode | LRU + prediction bias | LRU bias reduction vs plain pair | IB + prediction bias | IB bias reduction vs plain pair |', '|---|---:|---:|---:|---:|']
for mode in ['mdp','pomdp']:
    ref=records.get((mode,'mlp_prediction'))
    biases=[]
    for v in ['lru_prediction','ib_reverse_prior20_prediction']:
        value=ref['evaluation_mean']-records[mode,v]['evaluation_mean'] if ref and (mode,v) in records else None
        biases.append(f'{value:+.6f}' if value is not None else 'pending')
        base_v=v.removesuffix('_prediction')
        reduction=(records[mode,'mlp']['evaluation_mean']-records[mode,base_v]['evaluation_mean'])-value if value is not None else None
        biases.append(f'{reduction:+.6f}' if reduction is not None else 'pending')
    lines.append(f'| {mode.upper()} | {biases[0]} | {biases[1]} | {biases[2]} | {biases[3]} |')
lines += ['', 'MDP-to-POMDP return gaps are a separate comparison from MLP-relative memory bias:', '', '| Condition | MDP − POMDP return |', '|---|---:|']
for v,label in zip(variants,labels):
    if all((m,v) in records for m in ['mdp','pomdp']):
        gap=records['mdp',v]['evaluation_mean'] - records['pomdp',v]['evaluation_mean']
        lines.append(f'| {label} | {gap:+.6f} |')
lines += ['', 'Hardware: the reused POMDP default-IB baseline was trained on a Tesla T4; all new runs and the reused MLP/LRU references use RTX 2060. JAX is 0.6.2. Hardware differences may affect training numerics, so the small POMDP IB difference is not a clean hardware-matched auxiliary ablation.', '', 'Both IB conditions use the default reverse KL, posterior output input, 0.4/1.6 gradient balancing, one-bit full-distribution free region, frozen RMS calibration, and inter-layer LRU residuals. Vanilla LRU conditions both disable added inter-layer residuals. LRU/IB prediction inputs include input_t; MLP prediction uses its shared 256-dimensional hidden layer. CNN gradients from prediction are stopped on current and target paths. Prediction coefficient is 1.', '',
          'These are short, single-seed comparisons. Prediction adds parameters; observation/reward losses are tested together. The common coefficient does not match auxiliary gradient strength across architectures; prediction MSE is not RMS normalized. Existing matching MDP MLP and POMDP LRU/IB references are copied byte-for-byte with provenance; MDP LRU/IB were completed before extending to both modes.', '',
          '[Design and commands](EXPERIMENT.md). [Comparison plot](comparison.png). [Artifact audit](artifact_audit.json).', '']
(root/'RESULTS.md').write_text('\n'.join(lines))
fig,axes=plt.subplots(3,2,figsize=(13,11))
x=np.arange(len(variants))
for col,mode in enumerate(['mdp','pomdp']):
    means=[records[mode,v]['evaluation_mean'] if (mode,v) in records else np.nan for v in variants]
    ses=[records[mode,v]['evaluation_standard_error'] if (mode,v) in records else 0 for v in variants]
    axes[0,col].bar(x,means,yerr=ses,capsize=4,color=['gray','silver','steelblue','skyblue','darkorange','gold'])
    axes[0,col].set_title(mode.upper()); axes[0,col].set_ylabel('Return ± episode SE')
    mlp=records.get((mode,'mlp'))
    if mlp:
        axes[1,col].bar(x,[mlp['evaluation_mean']-m for m in means],color=['gray','silver','steelblue','skyblue','darkorange','gold'])
    axes[1,col].axhline(0,color='black',linewidth=.6)
    axes[1,col].set_ylabel('Memory bias: MLP − condition')
    for ax in axes[:2,col]:
        ax.set_xticks(x,['MLP','MLP + aux','LRU','LRU + aux','IB','IB + aux'],rotation=20,ha='right')
    aux_ref=records.get((mode,'mlp_prediction'))
    if mlp and aux_ref and all((mode,v) in records for v in variants[2:]):
        matched=[mlp['evaluation_mean']-records[mode,'lru']['evaluation_mean'], aux_ref['evaluation_mean']-records[mode,'lru_prediction']['evaluation_mean'], mlp['evaluation_mean']-records[mode,'ib_reverse_prior20']['evaluation_mean'], aux_ref['evaluation_mean']-records[mode,'ib_reverse_prior20_prediction']['evaluation_mean']]
        axes[2,col].bar(range(4),matched,color=['steelblue','skyblue','darkorange','gold'])
    axes[2,col].axhline(0,color='black',linewidth=.6)
    axes[2,col].set_xticks(range(4),['LRU','LRU + prediction','Default IB','IB + prediction'],rotation=20,ha='right')
    axes[2,col].set_ylabel('Bias vs MLP with matching auxiliary setting')
fig.suptitle('CartPoleHard · seed 0 · 98,304 steps · 64 evaluation episodes')
fig.tight_layout();fig.savefig(root/'comparison.png',dpi=160)
(root/'artifact_audit.json').write_text(json.dumps({'passed':True,'completed_runs':len(records),'expected_runs':12,'verified':['source hashes or copied artifact digests','matching PPO settings and steps','raw return statistics','evaluation counts','shared IB calibration and defaults','prediction head inputs']},indent=2))
print('\n'.join(lines))
