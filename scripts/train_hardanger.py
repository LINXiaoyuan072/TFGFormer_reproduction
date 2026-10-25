"""Reproducible training/evaluation adapter around the untouched released network."""
import argparse
import hashlib
import importlib.metadata
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'upstream/TFGFormer-for-SHM/TFGFormer/HardangerBridge'))
from Network.TFGFormer import TFGFormer, weights_init_normal
from Utils.TimeFrequencyLoss import JointTimeFreqLoss
from model_variants import configure_gate_layout


class PairedDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.paths = sorted((folder / 'label').glob('*.npy'))
        if not self.paths:
            raise ValueError(f'No samples: {folder}')
        names = {p.name for p in self.paths}
        for kind in ['feature', 'mask']:
            if {p.name for p in (folder / kind).glob('*.npy')} != names:
                raise ValueError(f'Mismatched {kind}/label filenames')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        name = self.paths[index].name
        arrays = [np.load(self.folder / kind / name, allow_pickle=False) for kind in ['feature', 'label', 'mask']]
        x, y, m = arrays
        if any(a.shape != (16, 1024) or not np.isfinite(a).all() for a in arrays):
            raise ValueError(f'Invalid sample: {name}')
        if not np.isin(m, [0, 1]).all() or not np.array_equal(x, y * m):
            raise ValueError(f'Invalid observed values or mask: {name}')
        return torch.from_numpy(x[None]), torch.from_numpy(y[None]), torch.from_numpy(m[None]).bool()


def predict(model, x, observed):
    # The upstream model infers missingness from x==0. Explicit masks also preserve real measured zeros.
    return torch.where(observed, x, model(x))


def metrics(pred, true, selected=None):
    if selected is not None:
        pred, true = pred[selected], true[selected]
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    residual = pred - true
    mse = float(np.mean(residual ** 2))
    energy = float(np.sum(true ** 2))
    error = float(np.sum(residual ** 2))
    variance = float(np.sum((true - true.mean()) ** 2))
    return {'MAE_g': float(np.mean(np.abs(residual))), 'RMSE_g': float(np.sqrt(mse)),
            'RE_ratio': float(np.sqrt(error / energy)) if energy else None,
            'RE_percent': float(100 * np.sqrt(error / energy)) if energy else None,
            'R2_flattened': float(1 - error / variance) if variance else None,
            'SNR_dB': float(10 * np.log10(energy / error)) if energy and error else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'data/processed/hardanger_2014_10_fig4/S25')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/hardanger_S25_pilot')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--clip-grad', type=float, default=None,
                        help='Optional gradient-norm limit; diagnostic deviation from released training')
    parser.add_argument('--gate-layout', choices=['released', 'channel'], default='released',
                        help='channel is a separate diagnostic variant, not the author implementation')
    parser.add_argument('--released-epoch-reseed', action='store_true',
                        help='Mirror released train.py: reset seed 42 and enable cuDNN benchmark every epoch')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output directory to preserve previous experiment')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    datasets = {s: PairedDataset(args.data / s) for s in ['train', 'val', 'test']}
    sets = [{p.stem for p in d.paths} for d in datasets.values()]
    assert all(not sets[i].intersection(sets[j]) for i in range(3) for j in range(i))
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=(s == 'train'),
                            drop_last=(s == 'train'), num_workers=0, generator=generator if s == 'train' else None)
               for s, d in datasets.items()}
    model = TFGFormer().to(device)
    changed_gates = configure_gate_layout(model, args.gate_layout)
    model.apply(weights_init_normal)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = JointTimeFreqLoss()
    args.output.mkdir(parents=True)
    run = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run.update(torch=torch.__version__, python=sys.version, device=str(device),
               gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
               parameter_count=sum(p.numel() for p in model.parameters()),
               loss='released JointTimeFreqLoss(spatial_weight=1, freq_weight=1), internal /2 and ortho rfft2',
               learning_rate_source='paper, user confirmed',
               dataset_manifest=str(args.data.parent / 'manifest.json'),
               adapter_changes=['filename-matched pairs', 'explicit observed-value preservation',
                                'seed before model initialization', 'all validation/test samples included'],
               train_drop_last=True,
               released_epoch_reseed=args.released_epoch_reseed,
               changed_gates=changed_gates,
               environment={name: importlib.metadata.version(name) for name in
                            ['torch', 'torchvision', 'numpy', 'scipy', 'matplotlib', 'timm', 'einops']},
               adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               variant_adapter_sha256=hashlib.sha256((ROOT / 'scripts/model_variants.py').read_bytes()).hexdigest(),
               dataset_manifest_sha256=hashlib.sha256((args.data.parent / 'manifest.json').read_bytes()).hexdigest(),
               tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
               tf32_cudnn=torch.backends.cudnn.allow_tf32,
               status='running')
    (args.output / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    history, best, best_epoch = [], float('inf'), None
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        if args.released_epoch_reseed:
            # The released train.py calls seed_all(42) at every epoch.  Reset the
            # dedicated loader generator too, so shuffle order actually repeats.
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            generator.manual_seed(42)
            torch.backends.cudnn.benchmark = True
        row = {'epoch': epoch}
        for split in ['train', 'val']:
            model.train(split == 'train')
            total, count = 0., 0
            for x, y, observed in loaders[split]:
                x, y, observed = x.to(device), y.to(device), observed.to(device)
                with torch.set_grad_enabled(split == 'train'):
                    pred = predict(model, x, observed)
                    loss = loss_fn(pred, y)
                    if not torch.isfinite(loss):
                        run.update(status='failed_nonfinite_loss', failed_epoch=epoch, failed_split=split)
                        (args.output / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
                        raise FloatingPointError(f'Nonfinite {split} loss at epoch {epoch}')
                    if split == 'train':
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        if args.clip_grad is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, error_if_nonfinite=True)
                        optimizer.step()
                total += loss.item() * len(x)
                count += len(x)
            row[split + '_loss'] = total / count
        if row['val_loss'] < best:
            best, best_epoch = row['val_loss'], epoch
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': best}, args.output / 'best.pt')
        history.append(row)
        (args.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    checkpoint = torch.load(args.output / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    xs, ys, ps, ms = [], [], [], []
    with torch.no_grad():
        for x, y, observed in loaders['test']:
            x, y, observed = x.to(device), y.to(device), observed.to(device)
            pred = predict(model, x, observed)
            assert torch.isfinite(pred).all() and torch.equal(pred[observed], x[observed])
            for collection, tensor in zip([xs, ys, ps, ms], [x, y, pred, observed]):
                collection.append(tensor.cpu().numpy())
    x, y, p, m = [np.concatenate(items) for items in [xs, ys, ps, ms]]
    report = {'best_epoch': best_epoch, 'best_val_loss': best, 'test_windows': len(y),
              'metrics_aggregation': 'all test elements pooled; g units; RE given both as ratio and percent',
              'model_global': metrics(p, y), 'model_missing_only': metrics(p, y, ~m),
              'zero_fill_global': metrics(x, y), 'zero_fill_missing_only': metrics(x, y, ~m),
              'observed_values_preserved': bool(np.array_equal(p[m], x[m])),
              'elapsed_seconds': time.monotonic() - start,
              'limitation': 'Subset pilot with unverified paper month/channel mapping; not a paper-table reproduction.'}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(args.output / 'test_predictions.npz', input=x, target=y, prediction=p, observed=m)
    sample, channel = 0, 1  # Paper's channel 2, missing in S25.
    t = np.arange(1024) / 10
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].plot(t, y[sample, 0, channel], label='Measured', lw=1)
    axes[0].plot(t, p[sample, 0, channel], label='Reconstructed', lw=.9, alpha=.8)
    axes[0].set(xlabel='Time (s)', ylabel='Vertical acceleration (g)', title='Held-out window, model channel 2')
    axes[0].legend()
    frequencies = np.fft.rfftfreq(1024, d=.1)
    for name, signals in [('Measured', y), ('Reconstructed', p)]:
        a = signals[sample, 0, channel].astype(np.float64)
        spectrum = np.abs(np.fft.rfft(a - a.mean())) / 1024
        spectrum[1:-1] *= 2
        axes[1].plot(frequencies, spectrum, label=name, lw=1)
    axes[1].set(xlabel='Frequency (Hz)', ylabel='Single-sided amplitude (g)', title='Spectrum (mean removed)')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.output / 'reconstruction.png', dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 3.5))
    for split in ['train', 'val']:
        axis.plot([r['epoch'] for r in history], [r[split + '_loss'] for r in history], label=split)
    axis.set(xlabel='Epoch', ylabel='Released joint loss', yscale='log')
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output / 'loss.png', dpi=180)
    plt.close(fig)
    run.update(status='completed', best_epoch=best_epoch)
    (args.output / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
