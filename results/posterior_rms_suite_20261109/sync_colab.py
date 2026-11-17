"""Periodically retrieve completed T4 runs, then release its runtime."""
import json
from pathlib import Path
import subprocess
import tarfile
import time

ROOT = Path('/home/user/popgym_arcade/results/posterior_rms_suite_20261109')
SESSION = 'posterior-rms-t4'
for attempt in range(1000):
    try:
        known = [str(p.parent.relative_to(ROOT)) for p in ROOT.rglob('result.json')]
        code = '''from pathlib import Path
import json, tarfile
root = Path('/content/posterior_suite/results/posterior_rms_suite_20261109')
known = set(KNOWN)
with tarfile.open('/content/posterior_results.tar.gz', 'w:gz') as tar:
    if root.exists():
        for p in root.rglob('*'):
            if not p.is_file():
                continue
            # Retrieve checkpoints only once the corresponding result is complete.
            if p.suffix == '.eqx' and (not (p.parent/'result.json').exists() or str(p.parent.relative_to(root)) in known):
                continue
            tar.add(p, arcname=str(p.relative_to(root)))
    p = Path('/content/posterior_suite/worker.log')
    if p.exists(): tar.add(p, arcname='colab_worker.log')
print('DONE', (root/'colab_done.json').exists())
'''.replace('KNOWN', repr(known))
        script = ROOT / 'sync_remote.py'
        script.write_text(code)
        result = subprocess.run(['colab','exec','-s',SESSION,'-f',str(script)], capture_output=True, text=True, timeout=180, check=True)
        subprocess.run(['colab','download','-s',SESSION,'/content/posterior_results.tar.gz',str(ROOT/'colab_transfer.tar.gz')], timeout=300, check=True)
        with tarfile.open(ROOT/'colab_transfer.tar.gz') as archive:
            archive.extractall(ROOT, filter='data')
        (ROOT/'sync_status.json').write_text(json.dumps({'time':time.time(),'attempt':attempt,'status':result.stdout}))
        if 'DONE True' in result.stdout:
            subprocess.run(['colab','stop','-s',SESSION], timeout=180, check=True)
            break
    except Exception as error:
        (ROOT/'sync_error.txt').write_text(str(error))
    time.sleep(300)
