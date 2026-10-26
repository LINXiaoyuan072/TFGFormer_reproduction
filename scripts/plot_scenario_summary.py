"""Plot selected subset results against official five-run paper means."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s25', required=True, type=Path)
    parser.add_argument('--s50', required=True, type=Path)
    parser.add_argument('--s75', required=True, type=Path)
    parser.add_argument('--paper', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.with_suffix('.png').exists():
        raise FileExistsError(args.output.with_suffix('.png'))
    scenarios = ['S25', 'S50', 'S75']
    paths = [args.s25, args.s50, args.s75]
    reports = [json.loads((path / 'metrics.json').read_text()) for path in paths]
    paper = json.loads(args.paper.read_text())
    current = {
        'MAE': np.array([r['model_global']['MAE_g'] for r in reports]),
        'RMSE': np.array([r['model_global']['RMSE_g'] for r in reports]),
        'RE': np.array([r['model_global']['RE_ratio'] for r in reports])
    }
    target = {
        'MAE': np.array([paper[s]['MAE_g_mean'] for s in scenarios]),
        'RMSE': np.array([paper[s]['RMSE_g_mean'] for s in scenarios]),
        'RE': np.array([paper[s]['RE_mean'] for s in scenarios])
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    x = np.arange(3)
    width = .36
    for ax, key, ylabel in zip(axes.flat[:3], ['MAE', 'RMSE', 'RE'],
                               ['Global MAE (g)', 'Global RMSE (g)', 'Global RE ratio']):
        ax.bar(x - width / 2, target[key], width, label='Paper mean (5 runs)', color='#7570b3')
        ax.bar(x + width / 2, current[key], width, label='Current 2014-10 subset', color='#1b9e77')
        ax.set(xticks=x, xticklabels=scenarios, ylabel=ylabel, title=f'{key}: current vs official source data', yscale='log')
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=9)
    ax = axes[1, 1]
    mae_improvement = [100 * (1 - r['model_missing_only']['MAE_g'] / r['zero_fill_missing_only']['MAE_g']) for r in reports]
    rmse_improvement = [100 * (1 - r['model_missing_only']['RMSE_g'] / r['zero_fill_missing_only']['RMSE_g']) for r in reports]
    ax.bar(x - width / 2, mae_improvement, width, label='MAE reduction', color='#66c2a5')
    ax.bar(x + width / 2, rmse_improvement, width, label='RMSE reduction', color='#3288bd')
    ax.axhline(0, color='.25', lw=.7)
    ax.set(xticks=x, xticklabels=scenarios, ylabel='Reduction vs zero fill (%)',
           title='Missing positions: improvement over zero fill')
    ax.legend(fontsize=9)
    fig.suptitle('Hardanger Bridge subset reproduction — selected stable results', fontsize=15)
    fig.savefig(args.output.with_suffix('.png'), dpi=220)
    plt.close(fig)
    result = {
        'selected_runs': dict(zip(scenarios, map(str, paths))),
        'current_global': {key: dict(zip(scenarios, map(float, values))) for key, values in current.items()},
        'paper_means': {key: dict(zip(scenarios, map(float, values))) for key, values in target.items()},
        'missing_improvement_vs_zero_percent': {
            'MAE': dict(zip(scenarios, mae_improvement)), 'RMSE': dict(zip(scenarios, rmse_improvement))},
        'selection_note': 'Uses the best stable method tried per scenario; this is a diagnostic summary, not a like-for-like paper table reproduction.'
    }
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
