"""Copy matching, Python-independent CUDA distributions into the project environment."""
import importlib.metadata as metadata
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/linxiaoyuan/miniconda3/envs/dronesplat/lib/python3.11/site-packages')
TARGET = ROOT / '.venv/lib/python3.10/site-packages'
EXPECTED = {
    'nvidia-cublas-cu12': '12.1.3.1', 'nvidia-cuda-cupti-cu12': '12.1.105',
    'nvidia-cuda-nvrtc-cu12': '12.1.105', 'nvidia-cuda-runtime-cu12': '12.1.105',
    'nvidia-cufft-cu12': '11.0.2.54', 'nvidia-curand-cu12': '10.3.2.106',
    'nvidia-cusolver-cu12': '11.4.5.107', 'nvidia-cusparse-cu12': '12.1.0.106',
    'nvidia-nccl-cu12': '2.20.5', 'nvidia-nvtx-cu12': '12.1.105',
    'nvidia-nvjitlink-cu12': '12.9.86',
}


def main():
    report = []
    for distribution in metadata.distributions(path=[str(SOURCE)]):
        name = distribution.metadata.get('Name')
        if name not in EXPECTED:
            continue
        if distribution.version != EXPECTED[name]:
            raise ValueError(f'Version mismatch for {name}')
        count, size = 0, 0
        for relative in distribution.files or []:
            if '__pycache__' in relative.parts:
                continue
            # Copy CUDA namespace and wheel metadata only; no unrelated environment files.
            if relative.parts[0] != 'nvidia' and not relative.parts[0].endswith('.dist-info'):
                continue
            source = Path(distribution.locate_file(relative))
            destination = TARGET / relative
            if not destination.resolve().is_relative_to(TARGET.resolve()):
                raise ValueError('Path outside target site-packages')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            count += 1
            size += source.stat().st_size
        report.append({'name': name, 'version': distribution.version, 'files': count, 'bytes': size})
        print(name, distribution.version, count, 'files', flush=True)
    if {item['name'] for item in report} != set(EXPECTED):
        raise ValueError('Some required CUDA distributions were not found')
    (ROOT / 'results/cuda_reuse_manifest.json').write_text(json.dumps({
        'source': str(SOURCE), 'destination': str(TARGET),
        'method': 'copies of matching CUDA wheels; source environment unchanged',
        'distributions': report}, indent=2) + '\n')


if __name__ == '__main__':
    main()
