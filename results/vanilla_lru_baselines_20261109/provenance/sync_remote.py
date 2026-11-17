from pathlib import Path
import json, tarfile
root = Path('/content/vanilla_pomdp/results/vanilla_lru_pomdp_20261109')
known = set(['MineSweeperEasy/lru_seed0', 'NavigatorEasy/lru_seed0'])
with tarfile.open('/content/vanilla_results.tar.gz', 'w:gz') as tar:
    if root.exists():
        for p in root.rglob('*'):
            if not p.is_file():
                continue
            # Retrieve checkpoints only once the corresponding result is complete.
            if p.suffix == '.eqx' and (not (p.parent/'result.json').exists() or str(p.parent.relative_to(root)) in known):
                continue
            tar.add(p, arcname=str(p.relative_to(root)))
    p = Path('/content/vanilla_pomdp/worker.log')
    if p.exists(): tar.add(p, arcname='colab_worker.log')
print('DONE', (root/'colab_done.json').exists())
