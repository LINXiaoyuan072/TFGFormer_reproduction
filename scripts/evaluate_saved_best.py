"""Evaluate a saved best checkpoint, including runs that later became nonfinite."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_hardanger import PairedDataset, TFGFormer, configure_gate_layout, metrics, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_run = json.loads((args.run / 'run.json').read_text())
    checkpoint_path = args.run / 'best.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = TFGFormer().to(device)
    configure_gate_layout(model, source_run.get('gate_layout', 'released'))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    dataset = PairedDataset(Path(source_run['data']) / 'test')
    loader = DataLoader(dataset, batch_size=source_run['batch_size'], shuffle=False, num_workers=0)
    collections = [[], [], [], []]
    with torch.no_grad():
        for x, y, observed in loader:
            x, y, observed = x.to(device), y.to(device), observed.to(device)
            prediction = predict(model, x, observed)
            if not torch.isfinite(prediction).all() or not torch.equal(prediction[observed], x[observed]):
                raise FloatingPointError('Best checkpoint produced invalid predictions')
            for items, tensor in zip(collections, [x, y, prediction, observed]):
                items.append(tensor.cpu().numpy())
    x, y, prediction, observed = [np.concatenate(items) for items in collections]
    report = {
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': checkpoint['val_loss'],
        'test_windows': len(y),
        'metrics_aggregation': 'all test elements pooled; g units; RE given both as ratio and percent',
        'model_global': metrics(prediction, y),
        'model_missing_only': metrics(prediction, y, ~observed),
        'zero_fill_global': metrics(x, y),
        'zero_fill_missing_only': metrics(x, y, ~observed),
        'observed_values_preserved': bool(np.array_equal(prediction[observed], x[observed])),
        'source_run_status': source_run['status'],
        'source_checkpoint': str(checkpoint_path),
        'limitation': 'Best-checkpoint evaluation recovered from a run that later became nonfinite.'
    }
    run = dict(source_run)
    run.update(status='completed', evaluation_only=True, source_run=str(args.run),
               source_run_status=source_run['status'])
    args.output.mkdir(parents=True)
    (args.output / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(args.output / 'test_predictions.npz', input=x, target=y,
                        prediction=prediction, observed=observed)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
