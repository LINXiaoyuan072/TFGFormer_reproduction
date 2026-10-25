"""Explicit diagnostic variants; default preserves the archived author implementation."""
from types import MethodType


def channel_gate_forward(self, x):
    left, right = self.expand(x).chunk(2, dim=1)
    left = self.ca(left)
    right = self.mlp(right.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
    return left * right


def configure_gate_layout(model, layout):
    if layout == 'released':
        return 0
    if layout != 'channel':
        raise ValueError(layout)
    from Network.TFGFormer import HybridGate
    changed = 0
    for module in model.modules():
        if isinstance(module, HybridGate):
            module.forward = MethodType(channel_gate_forward, module)
            changed += 1
    if changed != 6:
        raise ValueError(f'Expected 6 gates, found {changed}')
    return changed
