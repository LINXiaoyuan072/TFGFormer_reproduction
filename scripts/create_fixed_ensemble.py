"""Create a fixed equal-weight ensemble from two completed, identical-test runs."""
import argparse
import json
from pathlib import Path

import numpy as np
from train_hardanger import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_a', type=Path)
    parser.add_argument('run_b', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    archives = []
    for path in [args.run_a, args.run_b]:
        run = json.loads((path / 'run.json').read_text())
        if run['status'] != 'completed':
            raise ValueError(f'Incomplete source run: {path}')
        archives.append(np.load(path / 'test_predictions.npz'))
    a, b = archives
    for name in ['input', 'target', 'observed']:
        if not np.array_equal(a[name], b[name]):
            raise ValueError(f'Source runs differ in {name}')
    x, y, observed = a['input'], a['target'], a['observed']
    prediction = (a['prediction'].astype(np.float64) + b['prediction'].astype(np.float64)) / 2
    prediction[observed] = x[observed]
    report = {
        'best_epoch': None,
        'best_val_loss': None,
        'test_windows': len(y),
        'model_global': metrics(prediction, y),
        'model_missing_only': metrics(prediction, y, ~observed),
        'zero_fill_global': metrics(x, y),
        'zero_fill_missing_only': metrics(x, y, ~observed),
        'observed_values_preserved': bool(np.array_equal(prediction[observed], x[observed])),
        'ensemble': 'fixed arithmetic mean; no test-set weight fitting',
        'source_runs': [str(args.run_a), str(args.run_b)]
    }
    run = {
        'status': 'completed', 'method': 'fixed_50_50_prediction_ensemble',
        'source_runs': [str(args.run_a), str(args.run_b)], 'test_weight_tuning': False
    }
    args.output.mkdir(parents=True)
    (args.output / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(args.output / 'test_predictions.npz', input=x, target=y,
                        prediction=prediction, observed=observed)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
