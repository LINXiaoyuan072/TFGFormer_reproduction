"""Download an official PyPI wheel in resumable ranges and verify its published SHA256."""
import argparse
import hashlib
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('package')
    parser.add_argument('version')
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    with urllib.request.urlopen(f'https://pypi.org/pypi/{args.package}/{args.version}/json', timeout=60) as response:
        metadata = json.load(response)
    wheels = [f for f in metadata['urls'] if f['filename'].endswith('.whl') and 'x86_64' in f['filename']
              and 'manylinux' in f['filename'] and ('-py3-none-' in f['filename'] or '-cp310-cp310-' in f['filename'])]
    if len(wheels) != 1:
        raise ValueError(f'Expected one compatible Linux wheel, found {len(wheels)}')
    wheel = wheels[0]
    folder = ROOT / 'packages'
    folder.mkdir(exist_ok=True)
    target = folder / wheel['filename']
    size = wheel['size']
    expected_hash = wheel['digests']['sha256']
    if target.exists():
        digest = sha256_file(target)
        if digest == expected_hash:
            print('Already verified', target.name, flush=True)
            return
        raise ValueError('Existing wheel has unexpected hash; preserve and inspect it')
    chunk_size = (size + args.workers - 1) // args.workers
    print('Downloading', wheel['filename'], size, 'bytes in', args.workers, 'ranges', flush=True)

    def get_chunk(index):
        start = index * chunk_size
        end = min(start + chunk_size - 1, size - 1)
        chunk = target.with_suffix(target.suffix + f'.chunk{index}')
        if chunk.exists() and chunk.stat().st_size == end - start + 1:
            return chunk
        for attempt in range(4):
            try:
                request = urllib.request.Request(wheel['url'], headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                        raise ValueError('Server did not honor requested range')
                    with chunk.open('wb') as out:
                        shutil.copyfileobj(response, out)
                if chunk.stat().st_size != end - start + 1:
                    raise ValueError('Incomplete range')
                print('Range', index, 'complete', flush=True)
                return chunk
            except Exception:
                if attempt == 3:
                    raise
    with ThreadPoolExecutor(args.workers) as pool:
        chunks = list(pool.map(get_chunk, range(args.workers)))
    partial = target.with_suffix(target.suffix + '.part')
    with partial.open('wb') as out:
        for chunk in chunks:
            with chunk.open('rb') as source:
                shutil.copyfileobj(source, out)
    digest = sha256_file(partial)
    if partial.stat().st_size != size or digest != expected_hash:
        raise ValueError('Full wheel integrity verification failed')
    partial.rename(target)
    (folder / (wheel['filename'] + '.provenance.json')).write_text(json.dumps(wheel, indent=2) + '\n')
    print('Verified SHA256', digest, flush=True)


if __name__ == '__main__':
    main()
