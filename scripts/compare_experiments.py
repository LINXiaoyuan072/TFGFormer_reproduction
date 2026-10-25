"""Compare recorded experiments on the identical held-out inputs, labels and masks."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def spectral_diagnostics(prediction, target, observed):
    keep = ~observed[:, 0, :, 0]
    p, y = prediction[:, 0][keep].astype(np.float64), target[:, 0][keep].astype(np.float64)
    p -= p.mean(axis=-1, keepdims=True)
    y -= y.mean(axis=-1, keepdims=True)
    pf, yf = np.fft.rfft(p, axis=-1), np.fft.rfft(y, axis=-1)
    error_power = np.abs(pf - yf) ** 2
    error_power[:, 1:-1] *= 2
    denominator = np.linalg.norm(np.abs(yf))
    return {'demeaned_missing_magnitude_spectrum_relative_error': float(np.linalg.norm(np.abs(pf) - np.abs(yf)) / denominator) if denominator else None,
            'missing_error_nonDC_power_at_32sample_harmonics_fraction': float(error_power[:, 32::32].sum() / error_power[:, 1:].sum()) if error_power[:, 1:].sum() else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/framework_comparison')
    args = parser.parse_args()
    if args.output.with_suffix('.json').exists():
        raise FileExistsError('Choose a new comparison output')
    report, reference = [], None
    fig, axes = plt.subplots(len(args.runs), 2, figsize=(12, 3.5 * len(args.runs)), squeeze=False)
    for row, path in enumerate(args.runs):
        run = json.loads((path / 'run.json').read_text())
        m = json.loads((path / 'metrics.json').read_text())
        if run['status'] != 'completed':
            raise ValueError(f'Run did not complete: {path}')
        with np.load(path / 'test_predictions.npz') as archive:
            x, y, p, observed = [archive[k] for k in ['input', 'target', 'prediction', 'observed']]
        if reference is None:
            reference = (x, y, observed)
        elif not all(np.array_equal(a, b) for a, b in zip(reference, (x, y, observed))):
            raise ValueError('Cannot compare different test windows, labels or masks')
        assert np.isfinite(p).all() and np.array_equal(p[observed], x[observed])
        item = {'run': str(path), 'torch': run['torch'], 'gate_layout': run.get('gate_layout', 'released'),
                'released_epoch_reseed': run.get('released_epoch_reseed', False),
                'seed': run['seed'], 'learning_rate': run['lr'], 'epochs': run['epochs'],
                'best_epoch': m['best_epoch'], 'global': m['model_global'],
                'missing_only': m['model_missing_only'], **spectral_diagnostics(p, y, observed)}
        report.append(item)
        label = f"{run['torch']}, gate={item['gate_layout']}"
        t = np.arange(1024) / 10
        frequency = np.fft.rfftfreq(1024, d=.1)
        for name, signal in [('Measured', y[0, 0, 1]), ('Reconstructed', p[0, 0, 1])]:
            axes[row, 0].plot(t, signal, label=name, lw=.8)
            a = signal.astype(np.float64)
            spectrum = np.abs(np.fft.rfft(a - a.mean())) / 1024
            spectrum[1:-1] *= 2
            axes[row, 1].plot(frequency, spectrum, label=name, lw=.8)
        axes[row, 0].set(xlabel='Time (s)', ylabel='Acceleration (g)', title=label + ', same held-out channel 2')
        axes[row, 1].set(xlabel='Frequency (Hz)', ylabel='Amplitude (g)', title='Mean-removed spectrum')
        for axis in axes[row]:
            axis.legend()
    fig.tight_layout()
    fig.savefig(args.output.with_suffix('.png'), dpi=180)
    plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps({'identical_test_arrays_verified': True, 'runs': report,
        'scope': 'One seed on the 2014-10 subset; environment changes involve multiple package versions, not only torch.'}, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
