"""Create an auditable, fixed-sample comparison between two completed runs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from train_hardanger import metrics


def load_run(path):
    run = json.loads((path / 'run.json').read_text())
    metrics = json.loads((path / 'metrics.json').read_text())
    if run['status'] != 'completed':
        raise ValueError(f'Incomplete run: {path}')
    with np.load(path / 'test_predictions.npz') as archive:
        arrays = {name: archive[name] for name in ['input', 'target', 'prediction', 'observed']}
    return run, metrics, arrays


def spectrum(signal):
    signal = signal.astype(np.float64)
    values = np.abs(np.fft.rfft(signal - signal.mean())) / signal.size
    values[1:-1] *= 2
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--released', required=True, type=Path)
    parser.add_argument('--variant', required=True, type=Path)
    parser.add_argument('--variant-label', default='Channel-layout gate')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--sample', type=int, default=0)
    args = parser.parse_args()
    if args.output.with_suffix('.png').exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('Choose a new output prefix')

    released_run, released_metrics, a = load_run(args.released)
    variant_run, variant_metrics, b = load_run(args.variant)
    for name in ['input', 'target', 'observed']:
        if not np.array_equal(a[name], b[name]):
            raise ValueError(f'Runs differ in {name}')
    target, observed = a['target'], a['observed']
    missing_channels = np.flatnonzero(~observed[args.sample, 0, :, 0])
    if not len(missing_channels):
        raise ValueError('Selected sample has no fully missing channels')
    channel = int(missing_channels[0])
    t = np.arange(target.shape[-1]) / 10
    f = np.fft.rfftfreq(target.shape[-1], d=.1)
    truth = target[args.sample, 0, channel]
    released = a['prediction'][args.sample, 0, channel]
    variant = b['prediction'][args.sample, 0, channel]
    ensemble_prediction = (a['prediction'].astype(np.float64) + b['prediction'].astype(np.float64)) / 2
    ensemble_prediction[observed] = a['input'][observed]
    ensemble = ensemble_prediction[args.sample, 0, channel]

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(t, truth, label='Measured', color='#202020', lw=1.25)
    ax.plot(t, released, label='Released gate', color='#d95f02', lw=.85, alpha=.85)
    ax.plot(t, variant, label=args.variant_label, color='#1b9e77', lw=.9, alpha=.9)
    ax.plot(t, ensemble, label='Fixed 50/50 ensemble', color='#377eb8', lw=1, ls='--', alpha=.9)
    ax.set(title=f'Fixed test window {args.sample}, missing channel {channel + 1}', xlabel='Time (s)', ylabel='Acceleration (g)')
    ax.legend(ncol=3, fontsize=9)

    ax = axes[0, 1]
    floor = 1e-8
    ax.semilogy(f, np.maximum(spectrum(truth), floor), label='Measured', color='#202020', lw=1.25)
    ax.semilogy(f, np.maximum(spectrum(released), floor), label='Released gate', color='#d95f02', lw=.85)
    ax.semilogy(f, np.maximum(spectrum(variant), floor), label=args.variant_label, color='#1b9e77', lw=.9)
    ax.semilogy(f, np.maximum(spectrum(ensemble), floor), label='Fixed 50/50 ensemble', color='#377eb8', lw=1, ls='--')
    ax.set(title='Mean-removed amplitude spectrum', xlabel='Frequency (Hz)', ylabel='Amplitude (g)', xlim=(0, 5))
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.plot(t, released - truth, label='Released residual', color='#d95f02', lw=.8)
    ax.plot(t, variant - truth, label=f'{args.variant_label} residual', color='#1b9e77', lw=.8)
    ax.plot(t, ensemble - truth, label='Ensemble residual', color='#377eb8', lw=.9, ls='--')
    ax.axhline(0, color='.25', lw=.6)
    ax.set(title='Pointwise reconstruction residual', xlabel='Time (s)', ylabel='Prediction - measured (g)')
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    labels = ['Zero fill', 'Released', args.variant_label, '50/50 ensemble']
    ensemble_missing = metrics(ensemble_prediction, target, ~observed)
    mae = [released_metrics['zero_fill_missing_only']['MAE_g'], released_metrics['model_missing_only']['MAE_g'],
           variant_metrics['model_missing_only']['MAE_g'], ensemble_missing['MAE_g']]
    rmse = [released_metrics['zero_fill_missing_only']['RMSE_g'], released_metrics['model_missing_only']['RMSE_g'],
            variant_metrics['model_missing_only']['RMSE_g'], ensemble_missing['RMSE_g']]
    x = np.arange(2)
    width = .19
    colors = ['#7570b3', '#d95f02', '#1b9e77', '#377eb8']
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.bar(x + (i - 1.5) * width, np.array([mae[i], rmse[i]]) * 1e3, width, label=label, color=color)
    ax.set(title='All test windows, missing positions only', ylabel='Error (10$^{-3}$ g)', xticks=x, xticklabels=['MAE', 'RMSE'])
    ax.legend(fontsize=9)

    rel_error = np.abs(a['prediction'][args.sample, 0, missing_channels] - target[args.sample, 0, missing_channels])
    var_error = np.abs(b['prediction'][args.sample, 0, missing_channels] - target[args.sample, 0, missing_channels])
    vmax = float(np.percentile(np.concatenate([rel_error.ravel(), var_error.ravel()]), 99))
    images = []
    for ax, error, title in [(axes[2, 0], rel_error, 'Released gate: absolute error'),
                             (axes[2, 1], var_error, f'{args.variant_label}: absolute error')]:
        image = ax.imshow(error, aspect='auto', origin='lower', extent=[0, t[-1], -.5, len(missing_channels) - .5],
                          cmap='magma', vmin=0, vmax=vmax)
        images.append(image)
        ax.set(title=title, xlabel='Time (s)', ylabel='Missing model channel',
               yticks=np.arange(len(missing_channels)), yticklabels=(missing_channels + 1))
    fig.colorbar(images[-1], ax=axes[2, :], label='Absolute error (g)', shrink=.85)
    fig.suptitle('TFGFormer reconstruction comparison — identical held-out arrays', fontsize=15)
    fig.savefig(args.output.with_suffix('.png'), dpi=220)
    plt.close(fig)

    report = {
        'identical_input_target_mask_verified': True,
        'fixed_sample_index': args.sample,
        'fixed_model_channel_1based': channel + 1,
        'missing_channels_1based': (missing_channels + 1).tolist(),
        'released': {'run': str(args.released), 'missing_only': released_metrics['model_missing_only']},
        'variant': {'run': str(args.variant), 'missing_only': variant_metrics['model_missing_only']},
        'variant_label': args.variant_label,
        'fixed_equal_weight_ensemble': {'missing_only': ensemble_missing},
        'variant_improvement_percent': {
            'MAE': float(100 * (1 - mae[2] / mae[1])),
            'RMSE': float(100 * (1 - rmse[2] / rmse[1]))
        },
        'ensemble_improvement_over_released_percent': {
            'MAE': float(100 * (1 - mae[3] / mae[1])),
            'RMSE': float(100 * (1 - rmse[3] / rmse[1]))
        },
        'heatmap_shared_vmax_g_p99': vmax,
        'selection_note': 'sample 0 and the first masked channel are fixed by rule; no best-case sample selection'
    }
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
