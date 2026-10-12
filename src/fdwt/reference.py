"""Pure PyTorch implementations of every operator.

They are used as the CPU fallback and as the oracle in the test-suite.  They are
written for clarity, not speed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .kernels.dwt import mode_period, normalize_mode


def _ext_index(p: torch.Tensor, n: int, mode: int):
    mode = normalize_mode(mode, n)
    if mode == 0:
        valid = (p >= 0) & (p < n)
        return p.clamp(0, max(n - 1, 0)), valid
    P = mode_period(mode, n)
    q = torch.remainder(p, P)
    if mode == 1:
        idx = q
    elif mode == 2:
        idx = torch.where(q < n, q, P - 1 - q)
    elif mode == 3:
        idx = torch.where(q < n, q, P - q)
    else:
        idx = q.clamp(max=n - 1)
    return idx, torch.ones_like(p, dtype=torch.bool)


def _taps(n: int, n_out: int, L: int, mode: int, off: int, device):
    o = torch.arange(n_out, device=device)[:, None]
    j = torch.arange(L, device=device)[None, :]
    return _ext_index(2 * o + off - j, n, mode)


def analysis(x, h0, h1, mode: int, off: int, n_out: int):
    idx, valid = _taps(x.shape[-1], n_out, h0.numel(), mode, off, x.device)
    xs = x[..., idx] * valid
    return (xs * h0.to(x.dtype)).sum(-1), (xs * h1.to(x.dtype)).sum(-1)


def synthesis(a, d, h0, h1, mode: int, off: int, n: int):
    idx, valid = _taps(n, a.shape[-1], h0.numel(), mode, off, a.device)
    contrib = (a[..., :, None] * h0.to(a.dtype) + d[..., :, None] * h1.to(a.dtype)) * valid
    y = a.new_zeros(*a.shape[:-1], n)
    return y.index_add(-1, idx.flatten(), contrib.flatten(-2))


def filterbank_conv(x, h, n_out: int, off: int, dil: int = 1):
    """y[..., f, n] = sum_k h[f, k] * x[..., n + off - dil * k]  (x is zero outside its support).

    x: (..., N), h: (F, K) -> (..., F, n_out)
    """
    lead, N = x.shape[:-1], x.shape[-1]
    Fn, K = h.shape
    left = dil * (K - 1) - off  # conv1d is a cross-correlation: flip the filters
    right = n_out + off - N
    xp = x.reshape(-1, 1, N)
    xp = F.pad(xp, (max(left, 0), max(right, 0)))
    if left < 0:
        xp = xp[..., -left:]
    y = F.conv1d(xp, h.flip(-1).to(x.dtype)[:, None, :], dilation=dil)[..., :n_out]
    return y.reshape(*lead, Fn, n_out)
