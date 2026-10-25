"""Build a transparent subset experiment from the official monthly MAT ZIP.

Sensor order and selected month are reproduction choices, not verified paper IDs.
"""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
# Fig. 4b: Bu -> Vallavik, upper row is Vest (negative y), lower row East.
# Two missing physical sensors (H2 East/H8 Vest) explain the irregular pairs.
FIGURE_SENSOR_ORDER = ['H1 Vest', 'H1 East', 'H2 Vest', 'H3 East', 'H3 Vest',
                      'H4 East', 'H4 Vest', 'H5 East', 'H5 Vest', 'H6 East',
                      'H6 Vest', 'H7 East', 'H7 Vest', 'H8 East', 'H9 Vest', 'H9 East']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=ROOT / 'data/raw/MidFreq_2014_10.zip')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/processed/hardanger_2014_10_fig4')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output} already exists; choose a new output to preserve splits')
    windows, mapping, records, excluded = [], None, [], []
    with zipfile.ZipFile(args.archive) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC validation failed')
        for member in sorted(archive.namelist()):
            if not member.endswith('.mat'):
                continue
            recording = loadmat(io.BytesIO(archive.read(member)), simplify_cells=True)['recording']
            if float(recording['samplerate']) != 10:
                raise ValueError(f'Unexpected sampling rate in {member}')
            # The first 16 logger channels are inside the main deck; 17-20 are pylon sensors.
            deck_sensors = {str(s['sensor_name']): s for s in recording['sensor'] if int(s['channel']) <= 16}
            if set(deck_sensors) != set(FIGURE_SENSOR_ORDER):
                raise ValueError('Missing required deck sensors')
            sensors = [deck_sensors[name] for name in FIGURE_SENSOR_ORDER]
            current = []
            signals = []
            for index, sensor in enumerate(sensors, 1):
                names = [str(s).lower() for s in np.atleast_1d(sensor['component_names'])]
                z_index = names.index('z')
                unit = str(np.atleast_1d(sensor['component_units'])[z_index])
                if unit != 'g' or 'bridge deck' not in str(sensor['location']).lower():
                    raise ValueError('Expected deck acceleration in g')
                current.append({'model_channel_1based': index, 'logger_channel': int(sensor['channel']),
                                'sensor_name': str(sensor['sensor_name']), 'component': 'z', 'unit': unit})
                signals.append(np.asarray(sensor['data'])[:, z_index])
            if mapping is not None and current != mapping:
                raise ValueError('Sensor mapping changed between recordings')
            mapping = current
            signal = np.stack(signals).astype(np.float32)
            before = len(windows)
            # Do not concatenate discontinuous recordings or bridge invalid time gaps.
            for start in range(867, signal.shape[1] - 1024 + 1, 1024):
                window = signal[:, start:start + 1024]
                info = {'recording': member, 'start_sample': start, 'end_sample_exclusive': start + 1024}
                if not np.isfinite(window).all():
                    excluded.append({**info, 'reason': 'nonfinite window'})
                    continue
                windows.append((window.copy(), info))
            records.append({'recording': member, 'samples': signal.shape[1],
                            'accepted_windows': len(windows) - before,
                            'nonfinite_values': int((~np.isfinite(signal)).sum())})
    count = len(windows)
    if count < 10:
        raise ValueError(f'Only {count} valid windows; insufficient for 7:2:1 split')
    permutation = np.random.default_rng(args.seed).permutation(count)
    n_train, n_val = int(0.7 * count), int(0.2 * count)
    splits = {'train': permutation[:n_train], 'val': permutation[n_train:n_train + n_val],
              'test': permutation[n_train + n_val:]}
    scenarios = json.loads((ROOT / 'configs/hardanger_protocol.json').read_text())['verified_supplementary']['masked_channel_ids_1based']
    samples = {}
    for split, indices in splits.items():
        samples[split] = []
        for scenario, channel_ids in scenarios.items():
            for folder in ['feature', 'label', 'mask']:
                (args.output / ('S' + scenario) / split / folder).mkdir(parents=True, exist_ok=True)
            for i in indices:
                label, info = windows[int(i)]
                mask = np.ones((16, 1024), dtype=np.uint8)
                mask[np.array(channel_ids) - 1] = 0
                feature = label * mask
                for folder, data in [('feature', feature), ('label', label), ('mask', mask)]:
                    np.save(args.output / ('S' + scenario) / split / folder / f'{i:06d}.npy', data)
        for i in indices:
            samples[split].append({'id': f'{i:06d}', **windows[int(i)][1]})
    report = {'status': 'subset_method_reproduction', 'archive': args.archive.name,
              'archive_sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest(),
              'sampling_rate_hz': 10, 'unit': 'g', 'axis': 'z', 'dtype': 'float32',
              'window': 1024, 'stride': 1024, 'seed': args.seed, 'additional_normalization': False,
              'additional_filtering': False, 'split_counts': {k: len(v) for k, v in splits.items()},
              'sensor_mapping': mapping, 'records': records, 'excluded_windows': excluded,
              'mapping_source': 'Inferred from paper Fig.4b (Bu/Vallavik, X/Y arrows) and raw sensor names/coordinates; not author-confirmed.',
              'samples': samples, 'masked_channel_ids_1based': scenarios,
              'limitations': ['Paper month is unspecified; 2014-10 is a selected subset.',
                             'Model IDs inferred from Fig.4b and raw coordinates; first/last logger pairs swapped versus raw order.',
                             'First 867 points trimmed from EACH independent recording; paper trimming scope unspecified.',
                             'Invalid windows dropped without joining across gaps.',
                             'Random nonoverlapping-window split follows supplementary; adjacent windows may remain correlated.',
                             'Official MAT values in g retained without additional normalization.']}
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'split_counts', 'records', 'limitations']}, indent=2))


if __name__ == '__main__':
    main()
