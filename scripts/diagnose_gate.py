"""Evidence for a tensor-layout concern; does not change or retrain the released model."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'upstream/TFGFormer-for-SHM/TFGFormer/HardangerBridge'))
from Network.TFGFormer import HybridGate


def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    gate = HybridGate().eval()
    x = (torch.arange(32, dtype=torch.float32) / 32)[None, :, None, None].expand(1, 32, 16, 1024).contiguous()
    with torch.no_grad():
        original = gate(x)
        left, right = gate.expand(x).chunk(2, dim=1)
        # Hypothetical pointwise channel MLP, same weights, no training.
        corrected = gate.ca(left) * gate.mlp(right.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    a = original[0, :, 8, 64:-64].numpy()
    b = corrected[0, :, 8, 64:-64].numpy()
    report = {
        'torch': torch.__version__,
        'probe': 'Temporally constant, channel-varying input; same random weights; compare interior only.',
        'released_gate_temporal_variance_mean': float(a.var(axis=-1).mean()),
        'channel_MLP_probe_temporal_variance_mean': float(b.var(axis=-1).mean()),
        'released_gate_32step_difference_max': float(np.max(np.abs(a[:, 32:] - a[:, :-32]))),
        'interpretation': 'Released NCHW.view(B,-1,C) groups 32 consecutive time samples, not 32 feature channels. This probe demonstrates a 32-sample periodic output for constant input. Author intent is unconfirmed.',
        'scope': 'Diagnostic hypothesis, not an architectural change or corrected reproduction result.',
    }
    path = ROOT / 'results/hardanger_S25_fig4_200ep_torch24/test_predictions.npz'
    if path.exists():
        with np.load(path) as prediction:
            residual = prediction['prediction'] - prediction['target']
            masked = ~prediction['observed']
            keep = masked[:, 0, :, 0]
            spectra = np.fft.rfft(residual[:, 0][keep].astype(np.float64), axis=-1)
            power = np.abs(spectra) ** 2
            power[:, 1:-1] *= 2
            report['previous_test_error_nonDC_power_at_32sample_harmonics_fraction'] = float(
                power[:, 32::32].sum() / power[:, 1:].sum())
            report['hypothesized_harmonic_spacing_hz'] = 10 / 32
    (ROOT / 'results/gate_layout_diagnostic.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
