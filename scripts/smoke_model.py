"""Check the unmodified released network; synthetic data are NOT research results."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'upstream/TFGFormer-for-SHM/TFGFormer/HardangerBridge'))
from Network.TFGFormer import TFGFormer
from Utils.TimeFrequencyLoss import JointTimeFreqLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'results/model_smoke.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output path to preserve previous checks')
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = TFGFormer().to(device)
    target = torch.randn(1, 1, 16, 1024, device=device) * 0.01
    observed = torch.ones_like(target, dtype=torch.bool)
    observed[:, :, [1, 7, 10, 12], :] = False
    inputs = target * observed
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Retain the released loss's internal normalization; record this choice explicitly.
    loss_fn = JointTimeFreqLoss()
    start = time.monotonic()
    model.train()
    output = model(inputs)
    assert output.shape == target.shape
    assert torch.isfinite(output).all()
    assert torch.equal(output[observed], inputs[observed])
    loss = loss_fn(output, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().max() > 0 for g in grads)
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        after = model(inputs)
    assert torch.isfinite(after).all()
    report = {
        'status': 'passed', 'data': 'synthetic; no reconstruction accuracy claim',
        'torch': torch.__version__, 'device': str(device),
        'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'parameters': sum(p.numel() for p in model.parameters()),
        'input_shape': list(inputs.shape), 'learning_rate': 0.01,
        'loss_definition': 'released JointTimeFreqLoss, spatial_weight=1, freq_weight=1; internal /2',
        'loss_before_step': loss.item(), 'finite_gradients': True,
        'observed_values_preserved': True, 'finite_output_after_step': True,
        'elapsed_seconds': time.monotonic() - start,
        'peak_cuda_memory_mib': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None,
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
