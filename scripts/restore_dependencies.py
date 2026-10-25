"""Fetch recorded public upstream sources without copying environments or data."""
import json
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[1]
for dep in json.loads((ROOT / 'UPSTREAM.json').read_text())['external_sources']:
    target = ROOT / dep['path']
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}; inspect existing checkout first')
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '--no-checkout', dep['url'], str(target)], check=True)
    subprocess.run(['git', '-C', str(target), 'checkout', '--detach', dep['commit']], check=True)
    for patch in dep.get('patches', []):
        subprocess.run(['git', '-C', str(target), 'apply', '--check', str(ROOT / patch)], check=True)
        subprocess.run(['git', '-C', str(target), 'apply', str(ROOT / patch)], check=True)
    if dep.get('recursive', True):
        subprocess.run(['git', '-C', str(target), 'submodule', 'update', '--init', '--recursive'], check=True)
