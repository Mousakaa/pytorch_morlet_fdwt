"""Fast 1D wavelet transforms for PyTorch with fused Triton kernels.

* :func:`dwt`, :func:`idwt`, :func:`wavedec`, :func:`waverec`, :class:`DWT1D`,
  :class:`IDWT1D` — Mallat fast wavelet transform (O(N)), PyWavelets-compatible.
* :func:`cwt`, :class:`CWT1D` — Morlet continuous wavelet transform, with an
  O(N)-per-scale octave pyramid (``method="fast"``).
* :class:`MorletTokenMixer`, :class:`WaveletTokenMixer`, :class:`MorletAttention`,
  :class:`MorletChannelMixAttention` — attention replacements.
"""

from . import config
from .cwt import CWT1D, MorletPlan, cwt, freqs_to_scales, scales_to_freqs
from .dwt import DWT1D, IDWT1D, dwt, dwt_max_level, idwt, wavedec, waverec
from .layers import (MorletAttention, MorletChannelMixAttention, MorletTokenMixer,
                     WaveletTokenMixer)
from .wavelets import FilterBank, get_filter_bank

__version__ = "0.1.0"

__all__ = [
    "config", "CWT1D", "MorletPlan", "cwt", "freqs_to_scales", "scales_to_freqs",
    "DWT1D", "IDWT1D", "dwt", "dwt_max_level", "idwt", "wavedec", "waverec",
    "MorletAttention", "MorletChannelMixAttention", "MorletTokenMixer", "WaveletTokenMixer", "FilterBank", "get_filter_bank",
]
