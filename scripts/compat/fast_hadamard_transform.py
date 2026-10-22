"""Exact PyTorch FWHT fallback; accuracy experiments only, no kernel speed claims."""
import torch

def hadamard_transform(x, scale=1.0):
    n = x.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError(f'FWHT requires a power of two, got {n}')
    y = x.contiguous()
    h = 1
    while h < n:
        z = y.reshape(*y.shape[:-1], -1, 2, h)
        a, b = z.unbind(-2)
        y = torch.stack((a + b, a - b), dim=-2).reshape_as(y)
        h *= 2
    return y * scale
