"""Token-mixing layers built on the wavelet transforms.

``MorletTokenMixer`` and ``WaveletTokenMixer`` are drop-in replacements for a
self-attention block: they map ``(batch, length, dim)`` to ``(batch, length, dim)``.
``MorletAttention`` and ``MorletChannelMixAttention`` work on multi-channel patch
tokens ``(B, H, C, T, D)``.  The
cost of all three is linear in the sequence length (attention is quadratic).
"""

from __future__ import annotations

from typing import Optional, Sequence

import math

import torch
from torch import nn

from .cwt import CWT1D, morlet_mix, spectral_softmax
from .dwt import wavedec, waverec
from .wavelets import WaveletLike

__all__ = ["MorletTokenMixer", "WaveletTokenMixer", "MorletAttention", "MorletChannelMixAttention"]


class MorletTokenMixer(nn.Module):
    """Mix tokens through a Morlet scalogram.

    Each of the ``dim`` channels is transformed along the sequence with a
    (learnable) Morlet CWT; the per-scale magnitudes — a shift-robust, local
    time-frequency energy — and optionally the real parts are combined with a
    per-channel learned weighting over scales, then projected back::

        y[b, l, d] = sum_s mag_weight[d, s] |W_s x[b, :, d]|(l) + phase_weight[d, s] Re W_s x(l)

    Cost: O(L * dim * n_scales) time with the ``fast`` pyramid, and — thanks to
    the fused kernel — O(L * dim) activation memory: no O(L^2) term, and the
    scalogram itself is never stored.

    Args:
        dim: model width.
        freqs / scales: Morlet centre frequencies (with ``fs``) or scales in tokens.
        fs: token rate, e.g. the sampling rate when every token is one sample.
        learnable_scales: learn the (log-)scales.
        use_phase: also feed the real parts (carries phase information).
    """

    def __init__(self, dim: int, freqs: Optional[Sequence[float]] = None,
                 scales: Optional[Sequence[float]] = None, fs: float = 1.0, w0: float = 6.0,
                 method: str = "fast", learnable_scales: bool = True, use_phase: bool = True,
                 out_proj: bool = True):
        super().__init__()
        if freqs is None and scales is None:
            scales = torch.logspace(0, 6, 16, base=2).tolist()  # 1 .. 64 tokens
        self.cwt = CWT1D(freqs=freqs, scales=scales, fs=fs, w0=w0, method=method,
                         learnable=learnable_scales)
        S = self.cwt.plan.num_scales
        self.use_phase = use_phase
        self.mag_weight = nn.Parameter(torch.randn(dim, S) / S ** 0.5)
        self.phase_weight = nn.Parameter(torch.randn(dim, S) / S ** 0.5, requires_grad=use_phase)
        if not use_phase:
            self.phase_weight.data.zero_()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.out_proj = nn.Linear(dim, dim) if out_proj else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fused on CUDA: the (B, dim, S, L) scalogram is never materialised
        dt = torch.float64 if x.dtype == torch.float64 else torch.float32
        H = self.cwt._filters(x.device, dt)
        y = morlet_mix(x.transpose(1, 2), self.cwt.plan, H, self.mag_weight, self.phase_weight)
        y = y.transpose(1, 2) + self.bias
        return self.out_proj(y.to(x.dtype))


class WaveletTokenMixer(nn.Module):
    """Mix tokens in the DWT domain (O(L) transform, O(L * dim^2) mixing).

    The sequence is decomposed with the fast wavelet transform, every sub-band
    is mixed across channels by its own small MLP (coarse bands see long-range
    context, fine bands local detail), and the signal is reconstructed.

    Args:
        dim: model width.
        wavelet, mode: DWT filter bank and boundary mode.
        level: number of decomposition levels; the coarsest band summarises
            ``2**level`` tokens.
        hidden: hidden width of the per-band MLPs (``0``: a single linear map).
    """

    def __init__(self, dim: int, wavelet: WaveletLike = "db4", mode: str = "periodization",
                 level: int = 4, hidden: int = 0):
        super().__init__()
        if level < 1:
            raise ValueError("level must be >= 1")
        self.wavelet, self.mode, self.level = wavelet, mode, level

        def band():
            if hidden:
                return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
            return nn.Linear(dim, dim)

        self.bands = nn.ModuleList(band() for _ in range(level + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        coeffs = wavedec(x.transpose(1, 2), self.wavelet, self.mode, self.level)  # (B, D, n_j)
        mixed = [band(c.transpose(1, 2)).transpose(1, 2) for band, c in zip(self.bands, coeffs)]
        y = waverec(mixed, self.wavelet, self.mode)[..., :L]
        return y.transpose(1, 2)


class _MorletPatchAttention(nn.Module):
    """Shared plumbing: (B, H, C, T, D) patch tokens, flattened to (B, H, C, T*D) signals
    sampled at ``hidden_dim / patch_duration`` Hz, and a learnable Morlet CWT."""

    def __init__(self, num_channels: int, hidden_dim: int, freqs: Optional[Sequence[float]],
                 patch_duration: float, w0: float, learnable: bool, temperature: float,
                 method: str):
        super().__init__()
        self.num_channels, self.hidden_dim = num_channels, hidden_dim
        fs = hidden_dim / patch_duration
        if freqs is None:
            f_max = min(40.0, fs / 4)
            freqs = torch.logspace(0, math.log10(max(f_max, 1.5)), 16).tolist()
        if max(freqs) >= fs / 2:
            raise ValueError(f"frequencies must be below the Nyquist frequency {fs / 2:g} Hz")
        self.cwt = CWT1D(freqs=freqs, fs=fs, w0=w0, method=method,
                         learnable=learnable, learnable_w0=learnable)
        self.temperature = temperature

    @property
    def num_freqs(self) -> int:
        return self.cwt.plan.num_scales

    @property
    def freqs(self) -> torch.Tensor:
        """Current (learned) centre frequencies in Hz."""
        return self.cwt.freqs

    def _check(self, x: torch.Tensor):
        if x.dim() != 5:
            raise ValueError(f"expected (B, H, C, T, D) input, got shape {tuple(x.shape)}")
        B, H, C, T, D = x.shape
        if C != self.num_channels or D != self.hidden_dim:
            raise ValueError(f"expected C={self.num_channels} and D={self.hidden_dim}, "
                             f"got C={C} and D={D}")
        return B, H, C, T, D

    def _probs(self, x: torch.Tensor) -> torch.Tensor:
        """softmax_f |W_f x| / tau, shape (B, H, C, f, T*D) — materialised (inspection, CPU)."""
        B, H, C, T, D = self._check(x)
        w = self.cwt(x.reshape(B, H, C, T * D))
        return torch.softmax(w.abs() / self.temperature, dim=-2)

    def _spectral_softmax(self, x: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """sum_f softmax_f(|W_f x| / tau) V_f -> (B, H, C, T, D), fused on CUDA."""
        B, H, C, T, D = x.shape
        out = self.cwt.spectral_softmax(x.reshape(B * H * C, T * D), V, C, self.temperature)
        return out.reshape(B, H, C, T, D).to(x.dtype)


class MorletAttention(_MorletPatchAttention):
    """Attention, but with QKV replaced by a Morlet Wavelet Transform to
    capture structured token interactions.

    Input is of shape (B, H, C, T, D) for batch, head, channel, time and hidden
    dimension. This module supposes that tokens were formed by chunking the input
    signal into 1s patches, then projected to D. Therefore, it will flatten D into
    T to approximate the original temporal structure. Then, the CWT is computed on
    the sequence for every (B, H, C). We get a tensor of shape (B, H, C, T*D, f)
    with f the number of frequencies of the WT, which is softmaxed over f,
    transposed to (B, H, T*D, C, f) and multiplied, channel by channel, with a
    learned (C, f, 1) matrix, which gives (B, H, T*D, C); this is rearranged to
    get the input tensor shape back.

    The frequencies and scales of the CWT are learned as well.

    Concretely, with ``x_c[n]`` the flattened sequence of channel ``c`` (n < T*D)::

        out[c, n] = sum_f softmax_f(|W_f x_c|(n) / temperature) * w[c, f]

    i.e. at every position, a softmax-weighted average of channel ``c``'s learned
    per-frequency values, weighted by which frequencies dominate locally.

    On CUDA the whole layer is one fused operator: the CWT, the softmax (kept online
    across the octaves of the CWT pyramid) and the weighting run in the Triton
    kernels, so only (B, H, C, T*D) — plus two softmax statistics per position — is
    stored, never the (B, H, C, f, T*D) scalogram.  The backward recomputes the
    scalogram one octave at a time.

    Args:
        num_channels: C, number of signal channels (e.g. EEG electrodes).
        hidden_dim: D, samples per token after flattening.
        freqs: CWT centre frequencies in Hz (default: 16 log-spaced frequencies
            from 1 Hz to ``min(40, fs / 4)`` Hz).
        patch_duration: duration of one token in seconds; the flattened sequence is
            treated as sampled at ``fs = hidden_dim / patch_duration`` Hz.
        w0: initial Morlet wave number.
        learnable: learn the CWT scales (centre frequencies) and the wave number.
        temperature: softmax temperature applied to the scalogram magnitudes.
        method: CWT algorithm, ``"fast"`` (O(N) per frequency) or ``"direct"``.
    """

    def __init__(self, num_channels: int, hidden_dim: int,
                 freqs: Optional[Sequence[float]] = None, patch_duration: float = 1.0,
                 w0: float = 6.0, learnable: bool = True, temperature: float = 1.0,
                 method: str = "fast"):
        super().__init__(num_channels, hidden_dim, freqs, patch_duration, w0, learnable,
                         temperature, method)
        # the learned (C, f, 1) matrix, stored as (C, f)
        self.freq_weight = nn.Parameter(torch.randn(num_channels, self.num_freqs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        return self._spectral_softmax(x, self.freq_weight)


class MorletChannelMixAttention(_MorletPatchAttention):
    """Channel attention on (B, H, C, T, D) patch tokens driven by Morlet spectra.

    Same pre-processing as :class:`MorletAttention` (flatten T*D, learnable CWT per
    (B, H, C), softmax over the f frequencies).  The softmaxed spectrum of every
    channel is then turned into a C x C channel-attention map through a learned
    (f, C) matrix ``M``, and that map is applied to the input (values = the input)::

        P[c, n, f]  = softmax_f |W_f x_c|(n) / temperature   (spectral "query" of channel c)
        A[n, c, c'] = sum_f P[c, n, f] * M[f, c']            (C x C attention map)
        out[c, n]   = sum_c' A[n, c, c'] * x_c'[n]

    ``A`` is never formed: ``out[c, n] = sum_f P[c, n, f] * U[f, n]`` with
    ``U[f, n] = sum_c' M[f, c'] x_c'[n]`` (cost O(T*D * C * f) rather than
    O(T*D * C^2)); nothing is quadratic in the sequence length.  On CUDA the CWT,
    the softmax and the weighting by ``U`` are one fused operator (see
    :class:`MorletAttention`); ``U`` itself is a small (B, H, f, T*D) einsum.

    Args: as :class:`MorletAttention`.
    """

    def __init__(self, num_channels: int, hidden_dim: int,
                 freqs: Optional[Sequence[float]] = None, patch_duration: float = 1.0,
                 w0: float = 6.0, learnable: bool = True, temperature: float = 1.0,
                 method: str = "fast"):
        super().__init__(num_channels, hidden_dim, freqs, patch_duration, w0, learnable,
                         temperature, method)
        # M[f, c']: how much frequency band f of a query channel attends to key channel c'
        self.freq_to_channel = nn.Parameter(torch.randn(self.num_freqs, num_channels)
                                            / self.num_freqs ** 0.5)

    def attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """Explicit (B, H, T*D, C, C) attention map, for inspection (memory heavy)."""
        p = self._probs(x)  # (B, H, C, f, N)
        return torch.einsum("bhcfn,fk->bhnck", p, self.freq_to_channel.to(p.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, C, T, D = self._check(x)
        dt = torch.float64 if x.dtype == torch.float64 else torch.float32
        v = x.reshape(B, H, C, T * D).to(dt)                 # values: the input itself
        U = torch.einsum("fk,bhkn->bhfn", self.freq_to_channel.to(dt), v)
        return self._spectral_softmax(x, U.reshape(B * H, self.num_freqs, T * D))
