"""Download public NVA artifacts, verifying size and recording SHA256 (no credentials)."""
import argparse
import hashlib
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = '01977e2c97f8-09ea4ff5-fd74-48c0-bd99-790e3d8c19ac'
API = 'https://api.nva.unit.no/publication/' + PUBLICATION


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        'Accept': 'application/json', 'User-Agent': 'TFGFormer-reproduction/1.0'}), timeout=60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    metadata_path = ROOT / 'paper/nva_publication_api.json'
    if not metadata_path.exists():
        with request(API) as response:
            metadata_path.write_bytes(response.read())
    files = {f['name']: f for f in json.loads(metadata_path.read_text())['associatedArtifacts']}
    manifests = []
    for name in args.names:
        artifact = files[name]
        target = ROOT / 'data/raw' / name
        if not target.exists() or target.stat().st_size != artifact['size']:
            with request(API + '/filelink/' + artifact['identifier']) as response:
                link = json.load(response)['id']
            partial = target.with_suffix(target.suffix + '.part')
            print('Downloading', name, artifact['size'], 'bytes', flush=True)
            if args.workers > 1 and artifact['size'] > 2**20:
                size = artifact['size']
                chunk_size = (size + args.workers - 1) // args.workers
                def download_chunk(index):
                    start = index * chunk_size
                    end = min(size - 1, start + chunk_size - 1)
                    chunk = target.with_suffix(target.suffix + f'.chunk{index}')
                    req = urllib.request.Request(link, headers={'Range': f'bytes={start}-{end}'})
                    with urllib.request.urlopen(req, timeout=90) as response, chunk.open('wb') as out:
                        if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                            raise ValueError('Server did not honor byte range')
                        shutil.copyfileobj(response, out)
                    if chunk.stat().st_size != end - start + 1:
                        raise ValueError('Chunk length mismatch')
                    return chunk
                with ThreadPoolExecutor(args.workers) as pool:
                    chunks = list(pool.map(download_chunk, range(args.workers)))
                with partial.open('wb') as out:
                    for chunk in chunks:
                        with chunk.open('rb') as source:
                            shutil.copyfileobj(source, out)
            else:
                with urllib.request.urlopen(link, timeout=90) as response, partial.open('wb') as out:
                    shutil.copyfileobj(response, out)
            if partial.stat().st_size != artifact['size']:
                raise ValueError('Download size mismatch: ' + name)
            partial.rename(target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        record = {k: artifact[k] for k in ['name', 'identifier', 'size', 'license']}
        record.update(sha256=digest, source=API)
        target.with_suffix(target.suffix + '.provenance.json').write_text(json.dumps(record, indent=2) + '\n')
        manifests.append(record)
        print('Verified', name, digest, flush=True)


if __name__ == '__main__':
    main()
