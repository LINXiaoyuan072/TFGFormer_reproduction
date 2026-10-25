"""Verify file pairing, masks, finite values, and nonoverlapping split windows."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'data/processed/hardanger_2014_10_fig4')
    args = parser.parse_args()
    manifest = json.loads((args.data / 'manifest.json').read_text())
    intervals = defaultdict(list)
    ids = set()
    for split, samples in manifest['samples'].items():
        for sample in samples:
            assert sample['id'] not in ids
            ids.add(sample['id'])
            intervals[sample['recording']].append((sample['start_sample'], sample['end_sample_exclusive'], split))
    for spans in intervals.values():
        spans.sort()
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    pairs = 0
    for scenario, channels in manifest['masked_channel_ids_1based'].items():
        expected_mask = np.ones((16, 1024), dtype=np.uint8)
        expected_mask[np.array(channels) - 1] = 0
        for split, samples in manifest['samples'].items():
            folder = args.data / ('S' + scenario) / split
            expected_names = {sample['id'] + '.npy' for sample in samples}
            for kind in ['feature', 'label', 'mask']:
                assert {p.name for p in (folder / kind).glob('*.npy')} == expected_names
            for name in sorted(expected_names):
                x, y, m = [np.load(folder / kind / name, allow_pickle=False) for kind in ['feature', 'label', 'mask']]
                assert x.dtype == np.float32 and y.dtype == np.float32
                assert x.shape == y.shape == m.shape == (16, 1024)
                assert all(np.isfinite(a).all() for a in [x, y, m])
                assert np.array_equal(m, expected_mask)
                assert np.array_equal(x, y * m)
                pairs += 1
    report = {'status': 'passed', 'dataset': str(args.data), 'sample_count': len(ids),
              'verified_scenario_sample_pairs': pairs, 'split_counts': manifest['split_counts'],
              'split_ids_disjoint': True, 'raw_sample_intervals_nonoverlapping': True,
              'observed_values_exact': True, 'fixed_masks_match_supplementary': True,
              'finite_float32_labels_and_features': True,
              'limitations': 'Does not remove statistical dependence between adjacent nonoverlapping windows.'}
    (args.data / 'checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
