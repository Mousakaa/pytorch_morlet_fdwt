"""Fast (Mallat) discrete wavelet transform with fused Triton kernels.

The functional API mirrors PyWavelets (``dwt``, ``idwt``, ``wavedec``,
``waverec``) and produces identical coefficients; the transform is applied along
the last dimension and every leading dimension is treated as a batch dimension.
Each level costs O(N) and the full decomposition is O(N) overall.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn

from . import reference
from .kernels import dwt as K
from .wavelets import WaveletLike, get_filter_bank

__all__ = ["dwt", "idwt", "wavedec", "waverec", "dwt_max_level", "DWT1D", "IDWT1D"]


def _use_triton(t: torch.Tensor) -> bool:
    return t.is_cuda


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def _check_dtype(x: torch.Tensor):
    if not x.is_floating_point():
        raise TypeError(f"expected a floating point tensor, got {x.dtype}")


class _Analysis(torch.autograd.Function):
    """(y0, y1) = A x ; backward = A^T, implemented by :class:`_Synthesis`."""

    @staticmethod
    def forward(ctx, x, h0, h1, mode: int, off: int, n_out: int):
        ctx.save_for_backward(h0, h1)
        ctx.cfg = (mode, off, x.shape[-1])
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        if _use_triton(x2):
            y0, y1 = K.analysis(x2, h0, h1, mode, off, n_out)
        else:
            y0, y1 = reference.analysis(x2, h0, h1, mode, off, n_out)
        return y0.reshape(*lead, n_out), y1.reshape(*lead, n_out)

    @staticmethod
    def backward(ctx, g0, g1):
        h0, h1 = ctx.saved_tensors
        mode, off, n = ctx.cfg
        gx = _Synthesis.apply(g0, g1, h0, h1, mode, off, n) if ctx.needs_input_grad[0] else None
        return gx, None, None, None, None, None


class _Synthesis(torch.autograd.Function):
    """y = A^T (a, d) ; backward = A, implemented by :class:`_Analysis`."""

    @staticmethod
    def forward(ctx, a, d, h0, h1, mode: int, off: int, n: int):
        ctx.save_for_backward(h0, h1)
        ctx.cfg = (mode, off, a.shape[-1])
        lead = a.shape[:-1]
        a2 = a.reshape(-1, a.shape[-1]).contiguous()
        d2 = d.reshape(-1, d.shape[-1]).contiguous()
        if _use_triton(a2):
            y = K.synthesis(a2, d2, h0, h1, mode, off, n)
        else:
            y = reference.synthesis(a2, d2, h0, h1, mode, off, n)
        return y.reshape(*lead, n)

    @staticmethod
    def backward(ctx, gy):
        h0, h1 = ctx.saved_tensors
        mode, off, nc = ctx.cfg
        ga, gd = None, None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            ga, gd = _Analysis.apply(gy, h0, h1, mode, off, nc)
        return ga, gd, None, None, None, None, None


def _mode_id(mode: str) -> int:
    try:
        return K.MODES[mode]
    except KeyError:
        raise ValueError(f"unsupported mode '{mode}', expected one of {sorted(K.MODES)}") from None


def _bank(wavelet, x: torch.Tensor):
    bank = get_filter_bank(wavelet)
    return bank, bank.tensors(x.device, _compute_dtype(x.dtype))


def dwt_coeff_len(n: int, filter_len: int, mode: str) -> int:
    if mode == "periodization":
        return (n + 1) // 2
    return (n + filter_len - 1) // 2


def dwt(x: torch.Tensor, wavelet: WaveletLike = "db4", mode: str = "symmetric"
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single level DWT along the last dimension. Returns ``(cA, cD)`` like ``pywt.dwt``."""
    _check_dtype(x)
    m = _mode_id(mode)
    bank, f = _bank(wavelet, x)
    L = bank.length
    off = L // 2 if m == K.MODES["periodization"] else 1
    return _Analysis.apply(x, f["dec_lo"], f["dec_hi"], m, off, dwt_coeff_len(x.shape[-1], L, mode))


def idwt(cA: Optional[torch.Tensor], cD: Optional[torch.Tensor], wavelet: WaveletLike = "db4",
         mode: str = "symmetric") -> torch.Tensor:
    """Single level inverse DWT, identical to ``pywt.idwt``."""
    if cA is None and cD is None:
        raise ValueError("at least one of cA, cD must be given")
    cA = torch.zeros_like(cD) if cA is None else cA
    cD = torch.zeros_like(cA) if cD is None else cD
    if cA.shape != cD.shape:
        raise ValueError(f"cA and cD shapes differ: {tuple(cA.shape)} vs {tuple(cD.shape)}")
    _check_dtype(cA)
    m = _mode_id(mode)
    bank, f = _bank(wavelet, cA)
    L, nc = bank.length, cA.shape[-1]
    if m == K.MODES["periodization"]:
        # pywt.idwt is the transpose of a periodized analysis with reversed rec filters
        n, off = 2 * nc, L // 2
    else:
        # ... and for every other mode the transpose of a zero-padded one, cropped to 2*nc-L+2
        n, off, m = 2 * nc - L + 2, 1, K.MODES["zero"]
        if n <= 0:
            raise ValueError(f"coefficients of length {nc} are too short for a filter of length {L}")
    return _Synthesis.apply(cA, cD, f["rec_lo_rev"], f["rec_hi_rev"], m, off, n)


def dwt_max_level(n: int, wavelet: WaveletLike = "db4") -> int:
    """Same as ``pywt.dwt_max_level``."""
    L = get_filter_bank(wavelet).length
    if L < 2 or n < L - 1:
        return 0
    return (n // (L - 1)).bit_length() - 1


def wavedec(x: torch.Tensor, wavelet: WaveletLike = "db4", mode: str = "symmetric",
            level: Optional[int] = None) -> List[torch.Tensor]:
    """Multilevel DWT: ``[cA_n, cD_n, ..., cD_1]`` exactly like ``pywt.wavedec``."""
    if level is None:
        level = dwt_max_level(x.shape[-1], wavelet)
    if level < 0:
        raise ValueError("level must be >= 0")
    details = []
    a = x
    for _ in range(level):
        a, d = dwt(a, wavelet, mode)
        details.append(d)
    return [a] + details[::-1]


def waverec(coeffs: Sequence[torch.Tensor], wavelet: WaveletLike = "db4",
            mode: str = "symmetric") -> torch.Tensor:
    """Inverse of :func:`wavedec`, like ``pywt.waverec``."""
    if len(coeffs) == 0:
        raise ValueError("coeffs must not be empty")
    a = coeffs[0]
    for d in coeffs[1:]:
        if a.shape[-1] == d.shape[-1] + 1:  # same trimming rule as pywt.waverec
            a = a[..., :-1]
        a = idwt(a, d, wavelet, mode)
    return a


class DWT1D(nn.Module):
    """Multilevel 1D DWT along the last dimension.

    ``forward(x)`` returns ``[cA_J, cD_J, ..., cD_1]`` (PyWavelets ordering).
    """

    def __init__(self, wavelet: WaveletLike = "db4", level: Optional[int] = None,
                 mode: str = "symmetric"):
        super().__init__()
        _mode_id(mode)
        self.bank = get_filter_bank(wavelet)
        self.level = level
        self.mode = mode

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return wavedec(x, self.bank, self.mode, self.level)

    def extra_repr(self) -> str:
        return f"wavelet={self.bank.name}, level={self.level}, mode={self.mode}"


class IDWT1D(nn.Module):
    """Inverse of :class:`DWT1D`."""

    def __init__(self, wavelet: WaveletLike = "db4", mode: str = "symmetric"):
        super().__init__()
        _mode_id(mode)
        self.bank = get_filter_bank(wavelet)
        self.mode = mode

    def forward(self, coeffs: Sequence[torch.Tensor]) -> torch.Tensor:
        return waverec(coeffs, self.bank, self.mode)

    def extra_repr(self) -> str:
        return f"wavelet={self.bank.name}, mode={self.mode}"
