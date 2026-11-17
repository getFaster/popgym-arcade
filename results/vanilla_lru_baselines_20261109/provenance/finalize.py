from pathlib import Path
import json
import subprocess
import time

root = Path('/home/user/popgym_arcade/results/vanilla_lru_pomdp_20261109')
for _ in range(1440):
    if all((root/f'{worker}_done.json').exists() for worker in ('local','colab')) and all((root/task/'lru_seed0'/name).exists() for task in ('CartPoleHard','NavigatorEasy','MineSweeperEasy','BreakoutEasy') for name in ('result.json','model.eqx','evaluation_returns.npy')):
        for worker in ('local','colab'):
            done=json.loads((root/f'{worker}_done.json').read_text())
            if done['failures']:
                raise RuntimeError(done)
        result=subprocess.run(['/home/user/popgym_arcade/.venv/bin/python','-m','popgym_arcade.experiments.report_vanilla_pomdp',str(root)],cwd='/home/user/popgym_arcade',check=True)
        break
    time.sleep(60)
