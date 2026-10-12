"""Filter banks for the discrete wavelet transform.

Coefficients follow the PyWavelets conventions so that results are directly
comparable with ``pywt.dwt`` / ``pywt.wavedec``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence, Union

import torch

__all__ = ["FilterBank", "get_filter_bank", "WaveletLike"]


@dataclass(frozen=True)
class FilterBank:
    """Analysis (``dec_*``) and synthesis (``rec_*``) filters of a two-channel filter bank."""

    name: str
    dec_lo: tuple
    dec_hi: tuple
    rec_lo: tuple
    rec_hi: tuple

    @property
    def length(self) -> int:
        return len(self.dec_lo)

    def tensors(self, device, dtype) -> dict:
        return _as_tensors(self, torch.device(device), dtype)


WaveletLike = Union[str, FilterBank, Sequence[Sequence[float]], "pywt.Wavelet"]  # noqa: F821


def _even(f: Sequence[float], length: int) -> tuple:
    f = tuple(float(v) for v in f)
    return f + (0.0,) * (length - len(f))


def get_filter_bank(wavelet: WaveletLike) -> FilterBank:
    """Resolve ``wavelet`` into a :class:`FilterBank`.

    ``wavelet`` may be a PyWavelets name (``"haar"``, ``"db4"``, ``"sym8"``,
    ``"coif3"``, ``"bior4.4"``...), a ``pywt.Wavelet``, a :class:`FilterBank`, or
    a sequence ``(dec_lo, dec_hi, rec_lo, rec_hi)``.  All filters are padded to a
    common even length (padding at the end does not change the transform).
    """
    if isinstance(wavelet, FilterBank):
        return wavelet
    if isinstance(wavelet, str):
        return _named_bank(wavelet)
    if hasattr(wavelet, "filter_bank"):
        name, bank = getattr(wavelet, "name", "custom"), wavelet.filter_bank
    else:
        name, bank = "custom", wavelet
    if len(bank) != 4:
        raise ValueError("a custom filter bank must be (dec_lo, dec_hi, rec_lo, rec_hi)")
    length = max(len(f) for f in bank)
    length += length % 2
    return FilterBank(name, *(_even(f, length) for f in bank))


@lru_cache(maxsize=None)
def _named_bank(name: str) -> FilterBank:
    if name.lower() in ("morl", "morlet", "cmor", "mexh", "gaus1", "shan", "fbsp"):
        raise ValueError(
            f"'{name}' is a continuous wavelet without a finite two-channel filter bank; "
            "use fdwt.cwt / fdwt.CWT1D for the Morlet transform."
        )
    try:
        import pywt
    except ImportError as e:  # pragma: no cover - pywt is a declared dependency
        raise ImportError("PyWavelets is required to look up wavelets by name") from e
    return get_filter_bank(pywt.Wavelet(name))


@lru_cache(maxsize=None)
def _as_tensors(bank: FilterBank, device: torch.device, dtype: torch.dtype) -> dict:
    mk = lambda f: torch.tensor(f, dtype=dtype, device=device)  # noqa: E731
    return {
        "dec_lo": mk(bank.dec_lo),
        "dec_hi": mk(bank.dec_hi),
        # synthesis is the transpose of an analysis operator built on reversed rec filters
        "rec_lo_rev": mk(bank.rec_lo[::-1]),
        "rec_hi_rev": mk(bank.rec_hi[::-1]),
    }
