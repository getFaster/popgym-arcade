from pathlib import Path
import subprocess
import tarfile

root = Path('/content/vanilla_pomdp')
root.mkdir(exist_ok=True)
with tarfile.open('/content/vanilla_source.tar.gz') as archive:
    archive.extractall(root, filter='data')
script = root/'launch.sh'
script.write_text('''#!/bin/bash
set -e
export MPLBACKEND=Agg
cd /content/vanilla_pomdp
uv sync --frozen --extra baselines --extra cuda --dev
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
.venv/bin/python -m popgym_arcade.experiments.vanilla_pomdp --worker colab --output results/vanilla_lru_pomdp_20261109
''')
log = (root/'worker.log').open('a')
p = subprocess.Popen(['bash', str(script)], cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(root/'worker.pid').write_text(str(p.pid))
print('Started T4 setup + two vanilla LRU POMDP runs; pid', p.pid)
