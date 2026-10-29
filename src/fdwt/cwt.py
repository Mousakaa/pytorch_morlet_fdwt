"""Continuous wavelet transform with the (complex) Morlet wavelet.

    W[s, n] = sum_m x[m] * conj(psi((m - n) / s)) / norm(s)
    psi(t)  = pi^(-1/4) * (exp(i*w0*t) - exp(-w0^2/2)) * exp(-t^2/2)

``norm(s)`` is ``sqrt(s)`` (``norm="l2"``, energy preserving) or ``s``
(``norm="l1"``, scale-independent amplitude for sinusoids).  The signal is
zero-padded and the Gaussian envelope is truncated at ``truncate`` standard
deviations.

Three algorithms compute the same quantity:

``method="direct"``
    Exact convolution with the sampled wavelets: O(N * sum_s s) — the cost
    grows with the scale.
``method="fast"`` (default)
    Octave pyramid ("algorithme a trous").  Scales are split into octaves;
    octave j convolves a half-band low-passed copy of the signal with Morlet
    filters dilated by 2^j.  Every filter keeps O(1) taps, so the cost is
    O(N) per scale whatever its size.  Scales below the first octave boundary
    are computed exactly; coarser ones match ``direct`` to ~1e-4 relative
    error (~5e-6 with ``truncate=5``).
``method="fft"``
    Same filters as ``direct`` applied through FFTs (pure PyTorch).

``direct`` and ``fast`` run fused Triton kernels on CUDA tensors (forward,
gradient w.r.t. the input, and w.r.t. the scales when they are learnable) and
fall back to PyTorch on CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import triton
from torch import nn

from . import config, reference
from .kernels import conv as KC
from .kernels import filters as KF
from .kernels import mix as KM
from .kernels import softmax as KS

__all__ = ["cwt", "CWT1D", "MorletPlan", "freqs_to_scales", "scales_to_freqs", "spectral_softmax",
           "spectral_softmax_params"]

_PI_M14 = math.pi ** -0.25
_BLOCK_F = 16  # filters per program in the forward kernel (8 complex scales)


def freqs_to_scales(freqs, fs: float = 1.0, w0: float = 6.0) -> torch.Tensor:
    """Scale (in samples) whose Morlet centre frequency is ``freqs`` (same unit as ``fs``)."""
    f = torch.as_tensor(freqs, dtype=torch.float64)
    return w0 * fs / (2 * math.pi * f)


def scales_to_freqs(scales, fs: float = 1.0, w0: float = 6.0) -> torch.Tensor:
    s = torch.as_tensor(scales, dtype=torch.float64)
    return w0 * fs / (2 * math.pi * s)


def _halfband(c: int, beta: float) -> torch.Tensor:
    n = torch.arange(-c, c + 1, dtype=torch.float64)
    h = 0.5 * torch.sinc(n / 2) * torch.from_numpy(np.kaiser(2 * c + 1, beta))
    return h / h.sum()


@dataclass
class _Stage:
    octave: int
    dil: int                # filter dilation 2**octave
    scale_idx: List[int]    # global scale indices, sorted by support
    half: int               # filters have 2*half+1 taps (at spacing `dil`)
    support: List[int]      # per scale half-support (taps)


class MorletPlan:
    """Static description of a Morlet CWT: scale grouping, filter sizes and margins.

    Everything here depends only on the *initial* scales, so the plan can be built
    once and reused while learnable scales evolve (filters are regenerated from the
    current scales at every call, with the supports fixed by the plan).

    ``banks > 1``: the scales are ``banks`` filter banks of ``S / banks`` scales each (bank-major),
    each with its own softmax in :func:`spectral_softmax`.  Scale ``k`` of every bank goes in the
    same octave (the finest of the banks' own octaves, so none aliases), and every stage lists
    its scales bank by bank: the kernels run all the banks in one launch per octave.
    """

    def __init__(self, scales: Sequence[float], w0: float = 6.0, truncate: float = 4.0,
                 norm: str = "l2", complex: bool = True, method: str = "fast",
                 support_margin: float = 1.0, octaves: Optional[Sequence[int]] = None,
                 banks: int = 1):
        if method not in ("fast", "direct", "fft"):
            raise ValueError(f"unknown method '{method}' (expected 'fast', 'direct' or 'fft')")
        if norm not in ("l1", "l2"):
            raise ValueError("norm must be 'l1' or 'l2'")
        scales = [float(s) for s in scales]
        if not scales or min(scales) <= 0:
            raise ValueError("scales must be a non-empty sequence of positive numbers")
        self.scales, self.w0, self.truncate, self.norm = scales, float(w0), float(truncate), norm
        self.complex, self.method = bool(complex), method
        self.support_margin = float(support_margin)
        self.channels = 2 if complex else 1
        reach = truncate * support_margin
        if octaves is not None and (len(octaves) != len(scales) or min(octaves) < 0):
            raise ValueError("octaves must hold one non-negative octave index per scale")
        if banks < 1 or len(scales) % banks:
            raise ValueError(f"{len(scales)} scales cannot form {banks} equal filter banks")
        self.banks = int(banks)
        nb = len(scales) // self.banks

        # octave j holds the scales in [2^j s_th, 2^(j+1) s_th): after j half-band
        # low-pass stages the Morlet band (w0 +- truncate)/s' must stay below 0.8 pi.
        self.s_th = (w0 + truncate) / (0.8 * math.pi)
        js = []
        for i, s in enumerate(scales):
            if octaves is not None:
                js.append(int(octaves[i]) if method == "fast" else 0)
            else:
                js.append(0 if method != "fast" else max(0, math.floor(math.log2(s / self.s_th))))
        if self.banks > 1:
            if octaves is None:  # one octave per bank position: the finest one never aliases
                js = [min(js[b * nb + k] for b in range(self.banks)) for k in range(nb)] * self.banks
            elif any(js[i] != js[i % nb] for i in range(len(js))):
                raise ValueError("with filter banks, scale k of every bank must be in the same octave")
        groups: Dict[int, List[int]] = {}
        for i, j in enumerate(js):
            groups.setdefault(j, []).append(i)
        self.stages: List[_Stage] = []
        for j in sorted(groups):
            dil = 2 ** j
            sup = {i: max(1, math.ceil(reach * scales[i] / dil)) for i in groups[j]}
            idx = sorted(groups[j], key=lambda i: (i // nb, sup[i]))  # bank by bank
            self.stages.append(_Stage(j, dil, idx, max(sup.values()), [sup[i] for i in idx]))
        self.n_octaves = self.stages[-1].octave + 1
        # all filters live in one (S*C, 2*center+1) tensor, row i*C + c in user scale order
        self.center = max(st.half for st in self.stages)
        self._dil = [0.0] * len(scales)
        self._sup = [0.0] * len(scales)
        for st in self.stages:
            for i, c in zip(st.scale_idx, st.support):
                self._dil[i], self._sup[i] = float(st.dil), float(c)

        # half-band anti-aliasing filter of the pyramid
        c_lp, beta = (16, 8.0) if truncate <= 4.5 else (24, 10.0)
        self.lowpass = _halfband(c_lp, beta) if self.n_octaves > 1 else torch.zeros(1, dtype=torch.float64)
        self.c_lp = c_lp if self.n_octaves > 1 else 0

        # x_j (j >= 1) is materialised on [-M_j, N + M_j) so that coarse octaves see the
        # zero-padded signal exactly; x_0 is the input itself (implicitly zero outside).
        J = self.n_octaves - 1
        by_oct = {st.octave: st for st in self.stages}
        self.margins = [0] * (J + 1)
        for j in range(J, 0, -1):
            own = by_oct[j].half * 2 ** j if j in by_oct else 0
            nxt = self.margins[j + 1] + self.c_lp * 2 ** j if j < J else 0
            self.margins[j] = max(own, nxt)
        self._static_cache: Dict[torch.device, list] = {}

    @property
    def num_scales(self) -> int:
        return len(self.scales)

    @property
    def octaves(self) -> List[int]:
        """Octave (pyramid level) of every scale, in user order."""
        out = [0] * len(self.scales)
        for st in self.stages:
            for i in st.scale_idx:
                out[i] = st.octave
        return out

    def static_tensors(self, device: torch.device) -> list:
        """Per stage ``(kb, cmap, block_s, kf, rows, cols)``: tap range per scale block,
        output channels, scales per block, tap range per scale, rows of the stage in the
        filter tensor and its column slice.  Plus ``(lowpass, kb, cmap)``."""
        if device not in self._static_cache:
            out = []
            C = self.channels
            for st in self.stages:
                bs = min(_BLOCK_F // C, triton.next_power_of_2(len(st.scale_idx)))
                kb = [[st.half - max(st.support[b:b + bs]), st.half + max(st.support[b:b + bs]) + 1]
                      for b in range(0, len(st.support), bs)]
                kf = [[st.half - c, st.half + c + 1] for c in st.support]
                rows = [i * C + c for i in st.scale_idx for c in range(C)]
                out.append((
                    torch.tensor(kb, dtype=torch.int32, device=device),
                    torch.tensor(st.scale_idx, dtype=torch.int32, device=device),
                    bs,
                    torch.tensor(kf, dtype=torch.int32, device=device),
                    torch.tensor(rows, dtype=torch.int64, device=device),
                    slice(self.center - st.half, self.center + st.half + 1),
                ))
            lp = (
                self.lowpass.to(device=device),
                torch.tensor([[0, self.lowpass.numel()]], dtype=torch.int32, device=device),
                torch.zeros(1, dtype=torch.int32, device=device),
            )
            self._static_cache[device] = [out, lp]
        return self._static_cache[device]

    def halfband_taps(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """``taps[t] = lowpass[c_lp + t]`` (t = 0..c_lp), as used by the half-band kernel."""
        key = ("hb", device, dtype)
        if key not in self._static_cache:
            self._static_cache[key] = self.lowpass[self.c_lp:].to(device=device, dtype=dtype).contiguous()
        return self._static_cache[key]

    def _grid(self, dev: torch.device):
        """``(dil, sup, k)``: per-scale dilation and half-support (``(S, 1)``) and the tap
        offsets ``(1, 2*center+1)``, fp64, cached per device."""
        key = ("grid", dev)
        if key not in self._static_cache:  # host -> device copies would sync the stream
            f64 = torch.float64
            self._static_cache[key] = (
                torch.tensor(self._dil, dtype=f64, device=dev)[:, None],
                torch.tensor(self._sup, dtype=f64, device=dev)[:, None],
                torch.arange(-self.center, self.center + 1, device=dev, dtype=f64)[None, :],
            )
        return self._static_cache[key]

    def _w0(self, w0, dev, detach: bool = False) -> torch.Tensor:
        """``w0`` as fp64: a scalar, or ``(S, 1)`` when it holds one value per bank."""
        w = torch.as_tensor(w0, dtype=torch.float64, device=dev)
        if detach:
            w = w.detach()
        if w.numel() == 1:
            return w.reshape(())
        return w.reshape(-1).repeat_interleave(self.num_scales // w.numel())[:, None]

    def _pack(self, re: torch.Tensor, im: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if not self.complex:
            return re.to(dtype)
        return torch.stack([re, im], 1).reshape(-1, re.shape[1]).to(dtype)

    def filters(self, scales: torch.Tensor, w0, dtype: torch.dtype) -> torch.Tensor:
        """All sampled Morlet filters as one ``(S*C, 2*center+1)`` tensor (row ``i*C + c``).

        Scale ``i`` is sampled with the spacing (dilation) of its octave and truncated to
        its support.  Differentiable w.r.t. ``scales`` and ``w0`` (when it is a tensor).
        """
        dev = scales.device
        f64 = torch.float64
        s = scales.to(f64)[:, None]
        dil, sup, k = self._grid(dev)
        t = k * dil / s
        env = torch.exp(-0.5 * t * t) * (k.abs() <= sup)
        # a dilated sum approximates the Riemann sum with spacing `dil`
        amp = _PI_M14 * dil / (s.sqrt() if self.norm == "l2" else s)
        w0_ = self._w0(w0, dev)
        w0t = w0_ * t
        re = (torch.cos(w0t) - torch.exp(-0.5 * w0_ * w0_)) * env * amp
        im = torch.sin(w0t) * env * amp if self.complex else None
        return self._pack(re, im, dtype)

    def filter_tangents(self, log_scales: torch.Tensor, w0, dtype: torch.dtype,
                        wrt_scales: bool = True, wrt_w0: bool = True) -> List[torch.Tensor]:
        """Derivatives of :meth:`filters` (``exp(log_scales)``, ``w0``), each shaped like it:
        dH/dlog s (row ``i*C + c`` w.r.t. its own scale ``i``) and dH/dw0.

        In closed form, with ``t = k dil / s``, ``E = amp * envelope`` and ``c0 = exp(-w0^2/2)``
        (``amp ~ s^-p``, p = 1/2 for the l2 norm, 1 for l1)::

            dRe/dlog s = (w0 t sin + (cos - c0) (t^2 - p)) E    dIm/dlog s = (sin (t^2 - p) - w0 t cos) E
            dRe/dw0    = (w0 c0 - t sin) E                       dIm/dw0    = t cos E
        """
        f64 = torch.float64
        ls = log_scales.detach().to(f64)
        dev = ls.device
        dil, sup, k = self._grid(dev)
        s = ls.exp()[:, None]
        l2 = self.norm == "l2"
        t = k * dil / s
        e = torch.exp(-0.5 * t * t) * (k.abs() <= sup) * (_PI_M14 * dil / (s.sqrt() if l2 else s))
        w = self._w0(w0, dev, detach=True)
        wt = w * t
        cs, sn = torch.cos(wt), torch.sin(wt)
        c0 = torch.exp(-0.5 * w * w)
        out = []
        if wrt_scales:
            q = t * t - (0.5 if l2 else 1.0)
            out.append(self._pack((wt * sn + (cs - c0) * q) * e, (sn * q - wt * cs) * e, dtype))
        if wrt_w0:
            out.append(self._pack((w * c0 - t * sn) * e, t * cs * e, dtype))  # summed per bank by the caller
        return out


# --------------------------------------------------------------------------- GPU path


def _out_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


class _MorletCWTFunction(torch.autograd.Function):
    """All octaves of the transform in one autograd node (fused forward and backward)."""

    @staticmethod
    def forward(ctx, x, plan: MorletPlan, H):
        R, N = x.shape
        stat, _ = plan.static_tensors(x.device)
        Cc = plan.channels
        y = torch.empty((R, plan.num_scales, N, Cc), device=x.device, dtype=_out_dtype(x.dtype))
        need_h = ctx.needs_input_grad[2]

        hTs = []
        for kb, _, bs, _, rows, cols in stat:
            hTs.append(KC.pad_filters(H.index_select(0, rows)[:, cols].t(), kb.shape[0], bs, Cc))
        xs = [x]
        by_oct = {st.octave: k for k, st in enumerate(plan.stages)}
        for j in range(plan.n_octaves):
            dil, mj = 2 ** j, plan.margins[j]
            if j in by_oct:
                k = by_oct[j]
                kb, cmap, bs = stat[k][:3]
                KC.forward(xs[j], hTs[k], kb, cmap, y, mj + plan.stages[k].half * dil, dil, bs)
            if j + 1 < plan.n_octaves:
                m1 = plan.margins[j + 1]
                xs.append(_lowpass_step(plan, xs[j], j))

        ctx.plan = plan
        ctx.x_dtype = x.dtype
        ctx.need_h = need_h
        ctx.n_x = len(xs) if need_h else 0
        ctx.save_for_backward(*(xs if need_h else []), H)
        return y

    @staticmethod
    def backward(ctx, g):
        plan: MorletPlan = ctx.plan
        saved = ctx.saved_tensors
        xs, H = saved[:ctx.n_x], saved[-1]
        g = g.contiguous()
        R, _, N, _ = g.shape
        stat, _ = plan.static_tensors(g.device)
        by_oct = {st.octave: k for k, st in enumerate(plan.stages)}

        gH = torch.zeros_like(H) if ctx.need_h else None
        g_next = None
        for j in range(plan.n_octaves - 1, -1, -1):
            dil, mj = 2 ** j, plan.margins[j]
            gx = torch.empty((R, N + 2 * mj), device=g.device, dtype=g.dtype)
            filled = False
            if g_next is not None:
                m1 = plan.margins[j + 1]
                KC.halfband(g_next, plan.halfband_taps(gx.device, gx.dtype), gx,
                            mj - m1 + plan.c_lp * dil, dil, transpose=True)
                filled = True
            if j in by_oct:
                k = by_oct[j]
                kb, cmap, bs, kf, rows, cols = stat[k]
                off = mj + plan.stages[k].half * dil
                h = H.index_select(0, rows)[:, cols].contiguous()
                KC.backward_x(g, h, kf, cmap, gx, off, dil, accumulate=filled)
                filled = True
                if ctx.need_h:
                    gh = KC.backward_h(xs[j], g, kb, cmap, off, dil, h.shape[1], bs)
                    gH[rows, cols] = gh.to(H.dtype)
            if not filled:
                gx.zero_()
            g_next = gx
        gx = g_next.to(ctx.x_dtype) if ctx.needs_input_grad[0] else None
        return gx, None, gH


class _MorletMixFunction(torch.autograd.Function):
    """y[r, n] = sum_s a[d, s] |W_s[r, n]| + b[d, s] Re W_s[r, n]  without materialising W.

    Forward reduces each scalogram tile in registers.  Backward recomputes one octave at a
    time (in row chunks bounded by ``MIX_CHUNK_BYTES``) to form dL/dW, then reuses the
    grad-x / grad-h kernels of the plain transform.
    """

    MIX_CHUNK_BYTES = 256 * 2**20

    @staticmethod
    def forward(ctx, x, plan: MorletPlan, H, a, b):
        R, N = x.shape
        D = a.shape[0]
        stat, _ = plan.static_tensors(x.device)
        dt = _out_dtype(x.dtype)
        a_, b_ = a.to(dt).contiguous(), b.to(dt).contiguous()
        y = torch.zeros((R, N), device=x.device, dtype=dt)
        hTs = [KC.pad_filters(H.index_select(0, rows)[:, cols].t(), kb.shape[0], bs, plan.channels)
               for kb, _, bs, _, rows, cols in stat]
        xs = [x]
        by_oct = {st.octave: k for k, st in enumerate(plan.stages)}
        for j in range(plan.n_octaves):
            dil, mj = 2 ** j, plan.margins[j]
            if j in by_oct:
                k = by_oct[j]
                kb, gidx, bs = stat[k][:3]
                KM.mix_forward(xs[j], hTs[k], kb, gidx, a_, b_, y,
                               mj + plan.stages[k].half * dil, dil, bs, D, 0)
            if j + 1 < plan.n_octaves:
                m1 = plan.margins[j + 1]
                xs.append(_lowpass_step(plan, xs[j], j))
        ctx.plan = plan
        ctx.x_dtype = x.dtype
        ctx.save_for_backward(*xs, H, a_, b_, *hTs)
        ctx.n_x = len(xs)
        return y

    @staticmethod
    def backward(ctx, gy):
        plan: MorletPlan = ctx.plan
        saved = ctx.saved_tensors
        nx, n_st = ctx.n_x, len(plan.stages)
        xs = saved[:nx]
        H, a, b = saved[nx:nx + 3]
        hTs = saved[nx + 3:]
        gy = gy.contiguous().to(a.dtype)
        R, N = gy.shape
        D = a.shape[0]
        Cc = plan.channels
        stat, _ = plan.static_tensors(gy.device)
        need_h = ctx.needs_input_grad[2]
        gH = torch.zeros_like(H) if need_h else None
        ga = torch.zeros_like(a)
        gb = torch.zeros_like(b)
        by_oct = {st.octave: k for k, st in enumerate(plan.stages)}

        g_next = None
        for j in range(plan.n_octaves - 1, -1, -1):
            dil, mj = 2 ** j, plan.margins[j]
            gx = torch.empty((R, N + 2 * mj), device=gy.device, dtype=gy.dtype)
            filled = False
            if g_next is not None:
                m1 = plan.margins[j + 1]
                KC.halfband(g_next, plan.halfband_taps(gx.device, gx.dtype), gx,
                            mj - m1 + plan.c_lp * dil, dil, transpose=True)
                filled = True
            if j in by_oct:
                k = by_oct[j]
                kb, gidx, bs, kf, rows, cols = stat[k]
                S_j = gidx.numel()
                off = mj + plan.stages[k].half * dil
                h = H.index_select(0, rows)[:, cols].contiguous()
                local = torch.arange(S_j, device=gy.device, dtype=torch.int32)
                chunk = max(1, _MorletMixFunction.MIX_CHUNK_BYTES // (S_j * N * Cc * gy.element_size()))
                gh_acc = None
                for r0 in range(0, R, chunk):
                    r1 = min(R, r0 + chunk)
                    gw = torch.empty((r1 - r0, S_j, N, Cc), device=gy.device, dtype=gy.dtype)
                    KM.mix_backward(xs[j][r0:r1], hTs[k], kb, gidx, a, b, gy[r0:r1], gw, ga, gb,
                                    off, dil, bs, D, r0)
                    KC.backward_x(gw, h, kf, local, gx[r0:r1], off, dil, accumulate=filled)
                    if need_h:
                        gh = KC.backward_h(xs[j][r0:r1], gw, kb, local, off, dil, h.shape[1], bs)
                        gh_acc = gh if gh_acc is None else gh_acc + gh
                if need_h:
                    gH[rows, cols] = gh_acc.to(H.dtype)
                filled = True
            if not filled:
                gx.zero_()
            g_next = gx
        gx = g_next.to(ctx.x_dtype) if ctx.needs_input_grad[0] else None
        return gx, None, gH, ga, gb


def morlet_mix(x: torch.Tensor, plan: MorletPlan, H: torch.Tensor, a: torch.Tensor,
               b: torch.Tensor) -> torch.Tensor:
    """``x (B, D, N)`` -> ``sum_s a[d, s] |W_s| + b[d, s] Re W_s``, shape ``(B, D, N)``.

    Fused on CUDA (memory O(B*D*N) instead of O(B*D*S*N)); plain PyTorch elsewhere.
    """
    B, D, N = x.shape
    if x.is_cuda:
        x2 = x.reshape(B * D, N).contiguous()
        if x2.dtype not in (torch.float32, torch.float64):
            x2 = x2.float()
        return _MorletMixFunction.apply(x2, plan, H, a, b).view(B, D, N)
    w = _apply_plan(x, plan, H)
    mag = w.abs()
    re = w.real if plan.complex else w
    return torch.einsum("bdsn,ds->bdn", mag, a.to(mag.dtype)) + torch.einsum("bdsn,ds->bdn", re, b.to(mag.dtype))


def _ss_v_layout(V):
    """Strides ``(SV_G, SV_C, SV_F, SV_N, SV_H)`` of the canonical V layouts, and whether V is
    per position: ``(HV, NCH, S)`` (row r uses ``V[:, r % NCH]``) or ``(G, HV, S, N)``
    (row r uses ``V[r // NCH]``), with HV value heads."""
    if V.dim() == 3:
        return (0, V.shape[2], 1, 0, V.shape[1] * V.shape[2]), False
    return (V.shape[1] * V.shape[2] * V.shape[3], 0, V.shape[3], 1, V.shape[2] * V.shape[3]), True


def _ss_heads(V) -> int:
    return V.shape[0] if V.dim() == 3 else V.shape[1]


def _ss_stage_filters(plan, H, stat):
    return [H.index_select(0, rows)[:, cols].contiguous() for *_, rows, cols in stat]


def _ss_fwd_tiles(plan: MorletPlan, H, N: int, precision: str):
    """The forward's band tiles of every stage (see :func:`kernels.softmax.band_tiles`)."""
    stat, _ = plan.static_tensors(H.device)
    hs = _ss_stage_filters(plan, H, stat)
    return [KS.band_tiles(h.detach(), st_[3], N, 2 ** plan.stages[k].octave, precision)
            for k, (h, st_) in enumerate(zip(hs, stat))]


def _ss_forward(x, plan: MorletPlan, H, V, nch: int, tau: float, precision: str, tiles=None):
    """Forward of the spectral softmax: ``(out, m, l, V_, tiles)`` (``tiles``: prebuilt band
    tiles, otherwise built from ``H``)."""
    if not plan.complex:
        raise ValueError("the spectral softmax needs the complex Morlet transform")
    R, N = x.shape
    dt = _out_dtype(x.dtype)
    stat, _ = plan.static_tensors(x.device)
    V_ = V.to(dt).contiguous()
    v_strides, v_pos = _ss_v_layout(V_)
    hv, nb = _ss_heads(V_), plan.banks
    if tiles is None:
        tiles = _ss_fwd_tiles(plan, H, N, precision)
    m = torch.full((nb * R, N), float("-inf"), device=x.device, dtype=dt)
    l = torch.zeros((nb * R, N), device=x.device, dtype=dt)
    num = torch.zeros((hv, nb * R, N), device=x.device, dtype=dt)
    xj = x
    by_oct = {st.octave: k for k, st in enumerate(plan.stages)}
    for j in range(plan.n_octaves):
        dil, mj = 2 ** j, plan.margins[j]
        if j in by_oct:
            k = by_oct[j]
            _, gidx, _, kf = stat[k][:4]
            KS.forward_stage(xj, tiles[k], kf, gidx, V_, v_strides, v_pos, nch, 1.0 / tau,
                             mj + plan.stages[k].half * dil, dil, precision, m, l, num, banks=nb, hv=hv)
        if j + 1 < plan.n_octaves:
            xj = _lowpass_step(plan, xj, j)
    out = num / l
    return out, m, l, V_, tiles


def _ss_bwd_modes(st: _Stage, need_x: bool, need_h: bool, nt: int, multi: bool = False):
    """``(sweep, fuse_gx, fuse_th)`` of one stage's backward, from :mod:`fdwt.config`
    (``multi``: filter banks or value heads, which only have the fused paths)."""
    K = 2 * st.half + 1
    mode = True if multi else config.fused_input_grad
    fuse_gx = need_x and (mode if isinstance(mode, bool) else K <= config.fused_input_grad_max_taps)
    mode = True if multi else config.fused_param_grad
    fuse_th = nt > 0 and (mode if isinstance(mode, bool) else K > config.fused_param_grad_min_taps)
    mode = config.sweep_backward  # atomic-free one-pass backward (needs dL/dx, no explicit dL/dH)
    sweep = need_x and not need_h and (mode if isinstance(mode, bool) else K <= config.sweep_backward_max_taps)
    return sweep, fuse_gx, fuse_th


def _ss_bwd_inputs(plan: MorletPlan, H, tangents, tiles, N: int, prec: str, vdt, need_x: bool, need_h: bool,
                   multi: bool = False):
    """Per stage, the filter operands of the backward: a dict with ``sweep`` (one-pass kernel),
    ``tl`` (band tiles replacing the forward's, or None), ``kf``, ``tt`` (tangent tiles),
    ``ta`` / ``K`` (sweep adjoint blocks, taps), ``h`` (stage filters, for the dL/dW path)
    and ``jt`` (stage tangents, for the grad-h contraction).  Only layout operations on ``H``
    and ``tangents``, so :func:`_param_operands` can run it once on index codes."""
    stat, _ = plan.static_tensors(H.device)
    nt = len(tangents)
    hs = _ss_stage_filters(plan, H, stat)
    out = []
    for k, st in enumerate(plan.stages):
        kb, gidx, bs, kf, rows, cols = stat[k]
        dil = 2 ** st.octave
        sweep, fuse_gx, fuse_th = _ss_bwd_modes(st, need_x, need_h, nt, multi)
        need_gh = need_h or (nt > 0 and not fuse_th)
        need_gw = (need_x and not fuse_gx) or need_gh
        d = {"sweep": False, "tl": None, "kf": kf, "tt": None, "ta": None, "K": 0, "h": None, "jt": []}
        if sweep and KS.sweep_supported(2 * st.half + 1, N, dil):
            h_k = hs[k].to(vdt)
            jt = [t.index_select(0, rows)[:, cols].to(vdt) for t in tangents]
            tl_k, kf_k, tt = KS.sweep_tiles(h_k, kf, N, dil, prec, jt)
            K, bu = h_k.shape[1], tl_k[1]
            d.update(sweep=True, tl=tl_k, kf=kf_k, tt=tt, K=K,
                     ta=KS.adjoint_tiles(h_k, bu, tl_k[0].dtype, 1 if K <= bu + 1 else 2))
        else:
            if nt:
                jt = [t.index_select(0, rows)[:, cols] for t in tangents]
                if fuse_th:
                    d["tt"] = torch.stack([KS.band_tiles(t.to(vdt), kf, N, dil, prec, window=False)[0]
                                           for t in jt])
                    if tiles[k][3]:  # the tangent tiles need the chunked layout for the filters too
                        d["tl"] = KS.band_tiles(H.index_select(0, rows)[:, cols].to(vdt), kf, N, dil,
                                                prec, window=False)
                else:
                    d["jt"] = jt
            if need_gw:
                d["h"] = hs[k]
        out.append(d)
    return out


def _ss_backward(plan: MorletPlan, x, H, V, m, l, out, tiles, gy, nch: int, tau: float, prec: str,
                 need_x: bool, need_h: bool, tangents=(), pre=None, nt=None):
    """Backward of the spectral softmax -> ``(gx, gH, gV, gtheta)``.

    ``tangents`` are filter tangents dH/dtheta_t (each shaped like ``H``); ``gtheta[t, f]`` is
    then dL/dtheta_t restricted to the rows of frequency ``f`` (the parameter gradient
    without dL/dH).  ``pre``: the operands of :func:`_ss_bwd_inputs`, prebuilt (with ``nt``
    tangents; ``H`` and ``tangents`` are then not used).  Which kernels run follows
    :mod:`fdwt.config`:

    * ``fused_input_grad``: dL/dx is scattered by the softmax backward kernel itself (adjoint
      of its Toeplitz tiles), instead of writing dL/dW to memory for the grad-x kernel;
    * ``fused_param_grad``: dL/dtheta comes from the same kernel (inner products with the
      tangent tiles), instead of the grad-h kernel (dL/dH, then contracted with the tangents).

    Both are decided per octave ("auto": by the number of taps).  Octaves with short filters
    (``config.sweep_backward``) instead run the one-pass sweep kernel
    (:func:`kernels.softmax.backward_stage_sweep`): dL/dx, dL/dV and dL/dtheta without dL/dW
    in memory and without atomics on dL/dx.

    dL/dW (``(rows, S_j, N, 2)``, in row chunks) is only materialised for what is not fused,
    and always for ``need_h`` (an explicit dL/dH).

    ``out`` and ``gy`` are ``(HV, banks * R, N)`` (see :func:`_ss_forward`).  Filter banks and
    value heads run the fused / sweep kernels only (no dL/dW, no explicit dL/dH).
    """
    R = x.shape[0]
    N = gy.shape[-1]
    hv, nb = gy.shape[0], plan.banks
    multi = hv > 1 or nb > 1
    if multi and need_h:
        raise NotImplementedError("dL/dH (explicit filters) with filter banks or value heads")
    # the kernels read one dL/dout and out; with value heads, out is c = sum_h gy_h * out_h
    out_k = (gy * out).sum(0) if hv > 1 else out[0]
    gy_k = gy if hv > 1 else gy[0]
    stat, _ = plan.static_tensors(gy.device)
    v_strides, v_pos = _ss_v_layout(V)
    if pre is None:
        nt = len(tangents)
        pre = _ss_bwd_inputs(plan, H, tangents, tiles, N, prec, V.dtype, need_x, need_h, multi)
    modes = [_ss_bwd_modes(st, need_x, need_h, nt, multi) for st in plan.stages]

    fuse_any_gx = any(sw or fx for sw, fx, _ in modes)
    ml = (m + torch.log(l)) if any(d["sweep"] for d in pre) else None  # shared by the sweep launches
    gV = torch.zeros_like(V)
    gH = torch.zeros_like(H) if need_h else None
    gth = torch.zeros((nt, plan.num_scales), device=gy.device, dtype=V.dtype) if nt else None
    by_oct = {st.octave: k for k, st in enumerate(plan.stages)}
    xs = _lowpass_chain(plan, x.to(V.dtype))
    Cc = plan.channels

    g_next = None
    for j in range(plan.n_octaves - 1, -1, -1):
        dil, mj = 2 ** j, plan.margins[j]
        gx, filled = None, False
        if need_x:
            if g_next is not None:
                gx = torch.empty((R, N + 2 * mj), device=gy.device, dtype=gy.dtype)
                m1 = plan.margins[j + 1]
                KC.halfband(g_next, plan.halfband_taps(gx.device, gx.dtype), gx,
                            mj - m1 + plan.c_lp * dil, dil, transpose=True)
                filled = True
            elif fuse_any_gx:
                gx = torch.zeros((R, N + 2 * mj), device=gy.device, dtype=gy.dtype)
                filled = True
            else:
                gx = torch.empty((R, N + 2 * mj), device=gy.device, dtype=gy.dtype)
        if j in by_oct:
            k = by_oct[j]
            kb, gidx, bs, kf, rows, cols = stat[k]
            d = pre[k]
            S_j = gidx.numel()
            off = mj + plan.stages[k].half * dil
            _, fuse_gx, fuse_th = modes[k]
            need_gh = need_h or (nt > 0 and not fuse_th)
            need_gw = (need_x and not fuse_gx) or need_gh
            if d["sweep"]:
                gxs = gx if nb == 1 else torch.zeros((nb, *gx.shape), device=gx.device, dtype=gx.dtype)
                KS.backward_stage_sweep(xs[j], d["tl"], None, d["kf"], gidx, V, v_strides, v_pos, nch,
                                        1.0 / tau, off, dil, prec, m, l, out_k, gy_k, gV, gxs,
                                        filled and nb == 1, ttiles=d["tt"], gth=gth, ml=ml,
                                        uniform=KS.sweep_uniform(d["K"], N, dil), ta=d["ta"], taps=d["K"],
                                        banks=nb, hv=hv)
                if nb > 1:
                    gx += gxs.sum(0)  # gx is zero-filled or holds the coarser octaves' part
                g_next = gx
                xs[j] = None
                continue
            tt = d["tt"]
            tl_k = tiles[k] if d["tl"] is None else d["tl"]
            if not need_gw:
                KS.backward_stage(xs[j], tl_k, kf, gidx, V, v_strides, v_pos, nch, 0, 1.0 / tau,
                                  off, dil, prec, m, l, out_k, gy_k, None, gV, gx=gx if fuse_gx else None,
                                  ttiles=tt, gth=gth, banks=nb, hv=hv)
            else:
                h = d["h"]
                local = torch.arange(S_j, device=gy.device, dtype=torch.int32)
                chunk = max(1, _SpectralSoftmaxFunction.CHUNK_BYTES // (S_j * N * Cc * gy.element_size()))
                gh_acc = None
                for r0 in range(0, R, chunk):
                    r1 = min(R, r0 + chunk)
                    gw = torch.empty((r1 - r0, S_j, N, Cc), device=gy.device, dtype=gy.dtype)
                    KS.backward_stage(xs[j][r0:r1], tl_k, kf, gidx, V, v_strides, v_pos, nch,
                                      r0, 1.0 / tau, off, dil, prec, m[r0:r1], l[r0:r1],
                                      out_k[r0:r1], gy_k[r0:r1], gw, gV,
                                      gx=gx[r0:r1] if fuse_gx else None, ttiles=tt, gth=gth)
                    if need_x and not fuse_gx:
                        KC.backward_x(gw, h, kf, local, gx[r0:r1], off, dil, accumulate=filled,
                                      precision=prec)
                    if need_gh:
                        gh = KC.backward_h(xs[j][r0:r1], gw, kb, local, off, dil, h.shape[1], bs,
                                           precision=prec)
                        gh_acc = gh if gh_acc is None else gh_acc + gh
                    del gw
                if need_h:
                    gH[rows, cols] = gh_acc.to(H.dtype)
                if nt and not fuse_th:
                    for t in range(nt):
                        contrib = (gh_acc * d["jt"][t].to(gh_acc.dtype)).view(S_j, -1).sum(1)
                        gth[t].index_add_(0, gidx.long(), contrib.to(gth.dtype))
            filled = True
        if need_x and not filled:
            gx.zero_()
        g_next = gx
        xs[j] = None  # free each octave signal once its octave is done
    return g_next, gH, gV, gth


class _Slot:
    __slots__ = ("i",)

    def __init__(self, i: int):
        self.i = i


def _encode(obj, slots: list):
    """Replace the floating tensors of a nested dict / list / tuple by slots (kept in ``slots``)."""
    if isinstance(obj, torch.Tensor) and obj.is_floating_point():
        slots.append(obj)
        return _Slot(len(slots) - 1)
    if isinstance(obj, dict):
        return {k: _encode(v, slots) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_encode(v, slots) for v in obj)
    return obj


def _decode(obj, views: list):
    if isinstance(obj, _Slot):
        return views[obj.i]
    if isinstance(obj, dict):
        return {k: _decode(v, views) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_decode(v, views) for v in obj)
    return obj


def _layout_knobs() -> tuple:
    """The settings that shape the tile layouts (part of the code-map cache key)."""
    return (config.fused_input_grad, config.fused_input_grad_max_taps, config.fused_param_grad,
            config.fused_param_grad_min_taps, config.sweep_backward, config.sweep_backward_max_taps,
            KS.BU, KS.BV, KS.USE_WINDOW, KS.WIN_MAX, KS.SWEEP_BU, KS.SWEEP_SMALL_BU,
            KS.SWEEP_SMALL_MAX_CARRY, KS.SWEEP_UNI, KS.SWEEP_UNI_MAXCH)


def _param_operands(plan: MorletPlan, log_scales, w0, dtype, key: tuple, build):
    """``build(codes)`` with every filter / tangent sample evaluated at the current
    ``log_scales`` and ``w0``, in one :func:`kernels.filters.morlet_eval` launch.

    ``build`` lays out the plan's filters (``codes[0]``), dH/dlog s (``codes[1]``) and
    dH/dw0 (``codes[2]``) with index selects, gathers and masks only; it runs once per
    ``key`` on fp64 index codes, and its floating outputs become one int32 code map.
    """
    dev = log_scales.device
    ck = ("operands", dev) + key
    if ck not in plan._static_cache:
        W = 2 * plan.center + 1
        n = 3 * plan.num_scales * plan.channels * W
        codes = torch.arange(1, n + 1, device=dev, dtype=torch.float64).view(3, -1, W)
        slots: list = []
        template = _encode(build(codes), slots)
        maps = (torch.cat([t.reshape(-1) for t in slots]).to(torch.int32) if slots
                else torch.zeros(0, device=dev, dtype=torch.int32))
        sizes = [t.numel() for t in slots]
        offs = [sum(sizes[:i]) for i in range(len(sizes))]
        plan._static_cache[ck] = (template, maps, [(o, s_, t.shape) for o, s_, t in zip(offs, sizes, slots)])
    template, maps, spans = plan._static_cache[ck]
    buf = torch.empty(maps.numel(), device=dev, dtype=dtype)
    dil, sup, _ = plan._grid(dev)
    KF.morlet_eval(maps, buf, log_scales, w0, dil, sup, plan.channels, 2 * plan.center + 1, plan.center,
                   plan.norm == "l2")
    return _decode(template, [buf[o:o + s_].view(sh) for o, s_, sh in spans])


def _bf16_tiles(pre):
    """Cast the tile operands of :func:`_ss_bwd_inputs` to the bf16 dot operands."""
    for d in pre:
        if d["tl"] is not None:
            d["tl"] = (d["tl"][0].to(torch.bfloat16), *d["tl"][1:])
        for key in ("tt", "ta"):
            if d[key] is not None:
                d[key] = d[key].to(torch.bfloat16)
    return pre


class _SpectralSoftmaxFunction(torch.autograd.Function):
    """out[r, n] = sum_f softmax_f(|W_f x_r|(n) / tau) * V_f[r, n], without materialising W.

    ``V`` is in a canonical layout of :func:`_ss_v_layout`; the output is
    ``(HV, banks * R, N)``.  The forward keeps a flash-attention style
    online softmax across octave launches; the backward recomputes one octave at a time.
    This variant takes the filters ``H`` and returns dL/dH (see
    :class:`_SpectralSoftmaxParamFunction` for the faster scales / w0 parametrisation).
    """

    CHUNK_BYTES = 256 * 2**20  # bound on the dL/dW buffer, when it is materialised

    @staticmethod
    def forward(ctx, x, plan: MorletPlan, H, V, nch: int, tau: float, precision: str):
        out, m, l, V_, tiles = _ss_forward(x, plan, H, V, nch, tau, precision)
        ctx.plan, ctx.nch, ctx.tau, ctx.precision = plan, nch, tau, precision
        ctx.x_dtype, ctx.v_dtype = x.dtype, V.dtype
        # the low-passed octave signals are recomputed in the backward, not stored
        ctx.tiles = [(None, *rest) for _, *rest in tiles]
        ctx.save_for_backward(x, H, V_, m, l, out, *[t[0] for t in tiles])
        return out

    @staticmethod
    def backward(ctx, gy):
        x, H, V, m, l, out, *tbs = ctx.saved_tensors
        tiles = [(tb, *rest) for tb, (_, *rest) in zip(tbs, ctx.tiles)]
        gy = gy.contiguous().to(V.dtype)
        gx, gH, gV, _ = _ss_backward(ctx.plan, x, H, V, m, l, out, tiles, gy, ctx.nch, ctx.tau,
                                     ctx.precision, ctx.needs_input_grad[0], ctx.needs_input_grad[2])
        gx = gx.to(ctx.x_dtype) if gx is not None else None
        return gx, None, gH, gV.to(ctx.v_dtype), None, None, None


class _SpectralSoftmaxParamFunction(torch.autograd.Function):
    """Same operator, parametrised by ``log_scales`` and ``w0`` (the filters of
    :meth:`MorletPlan.filters`) instead of the filter tensor: their gradients are inner
    products of dL/dW with the convolutions by the filter tangents dH/dlog s and dH/dw0,
    accumulated inside the softmax backward kernel (no dL/dH, no grad-h pass).  Filters,
    tangents and all their tiles come from one :func:`kernels.filters.morlet_eval` launch
    per pass (see :func:`_param_operands`)."""

    @staticmethod
    def forward(ctx, x, plan: MorletPlan, log_scales, w0, V, nch: int, tau: float, precision: str):
        dt = _out_dtype(x.dtype)
        N = x.shape[1]
        tiles = _param_operands(plan, log_scales, _detach(w0), dt, ("fwd", N) + _layout_knobs(),
                                lambda codes: _ss_fwd_tiles(plan, codes[0], N, "ieee"))
        if precision == "bf16":
            tiles = [(t.to(torch.bfloat16), *rest) for t, *rest in tiles]
        out, m, l, V_, tiles = _ss_forward(x, plan, None, V, nch, tau, precision, tiles=tiles)
        ctx.plan, ctx.nch, ctx.tau, ctx.precision = plan, nch, tau, precision
        ctx.x_dtype, ctx.v_dtype = x.dtype, V.dtype
        ctx.w0_float = None if isinstance(w0, torch.Tensor) else float(w0)
        ctx.tiles = [(None, *rest) for _, *rest in tiles]
        w0_t = w0 if isinstance(w0, torch.Tensor) else log_scales.new_tensor(float(w0))
        ctx.save_for_backward(x, log_scales, w0_t, V_, m, l, out, *[t[0] for t in tiles])
        return out

    @staticmethod
    def backward(ctx, gy):
        x, log_scales, w0, V, m, l, out, *tbs = ctx.saved_tensors
        tiles = [(tb, *rest) for tb, (_, *rest) in zip(tbs, ctx.tiles)]
        plan: MorletPlan = ctx.plan
        need_x = ctx.needs_input_grad[0]
        need_s = ctx.needs_input_grad[2]
        need_w = ctx.needs_input_grad[3] and ctx.w0_float is None
        N = gy.shape[-1]
        meta = ctx.tiles

        multi = plan.banks > 1 or gy.shape[0] > 1

        def build(codes):
            tangents = [codes[1]] * need_s + [codes[2]] * need_w
            return _ss_bwd_inputs(plan, codes[0], tangents, meta, N, "ieee", torch.float64, need_x, False,
                                  multi)

        pre = _param_operands(plan, log_scales, w0.detach(), V.dtype,
                              ("bwd", N, need_x, need_s, need_w, multi) + _layout_knobs(), build)
        if ctx.precision == "bf16":
            pre = _bf16_tiles(pre)
        gy = gy.contiguous().to(V.dtype)
        gx, _, gV, gth = _ss_backward(plan, x, None, V, m, l, out, tiles, gy, ctx.nch, ctx.tau,
                                      ctx.precision, need_x, False, pre=pre, nt=int(need_s) + int(need_w))
        gx = gx.to(ctx.x_dtype) if gx is not None else None
        g_s = gth[0].to(log_scales.dtype) if need_s else None
        g_w = gth[int(need_s)].view(w0.numel(), -1).sum(1).view(w0.shape).to(w0.dtype) if need_w else None
        return gx, None, g_s, g_w, gV.to(ctx.v_dtype), None, None, None


def _detach(w0):
    return w0.detach() if isinstance(w0, torch.Tensor) else w0


def _lowpass_step(plan: MorletPlan, xj, j):
    """x_{j+1} = dilated half-band low-pass of x_j (with the plan's margins)."""
    R = xj.shape[0]
    N = xj.shape[1] - 2 * plan.margins[j]
    m1 = plan.margins[j + 1]
    nxt = torch.empty((R, N + 2 * m1), device=xj.device, dtype=xj.dtype)
    return KC.halfband(xj, plan.halfband_taps(xj.device, xj.dtype), nxt,
                       plan.margins[j] - m1 + plan.c_lp * 2 ** j, 2 ** j)


def _lowpass_chain(plan: MorletPlan, x):
    xs = [x]
    for j in range(plan.n_octaves - 1):
        xs.append(_lowpass_step(plan, xs[j], j))
    return xs


def _canonical_v(V: torch.Tensor, heads: Optional[int]) -> torch.Tensor:
    if heads is None:  # (nch, S) / (G, S, N): one value head
        return V[None] if V.dim() == 2 else V[:, None]
    if V.dim() not in (3, 4) or _ss_heads(V) != heads:
        raise ValueError(f"V must be (heads, nch, S) or (G, heads, S, N) with heads={heads}, got {tuple(V.shape)}")
    return V


def _spectral_softmax_out(out: torch.Tensor, plan: MorletPlan, heads: Optional[int], R: int, N: int):
    out = out.view(out.shape[0], plan.banks, R, N)
    return out[0, 0] if heads is None and plan.banks == 1 else out


def _spectral_softmax_gpu(fn, x, plan, V, heads, *args):
    """Run the autograd ``fn(x, plan, *args[:-3], V, *args[-3:])`` on the canonical V, padding
    the value heads to a power of two (the kernels' head axis)."""
    R, N = x.shape
    V = _canonical_v(V, heads)
    hv = _ss_heads(V)
    hp = triton.next_power_of_2(hv)
    if hp != hv:
        pad = [0, 0] * (V.dim() - (1 if V.dim() == 3 else 2)) + [0, hp - hv]
        V = torch.nn.functional.pad(V, pad)
    x2 = x.contiguous()
    if x2.dtype not in (torch.float32, torch.float64):
        x2 = x2.float()
    out = fn(x2, plan, *args[:-3], V, *args[-3:])[:hv]
    return _spectral_softmax_out(out, plan, heads, R, N)


def _spectral_softmax_torch(x, plan: MorletPlan, H, V, nch: int, tau: float, heads: Optional[int]):
    R, N = x.shape
    nb = plan.banks
    W = _apply_plan(x, plan, H)                                        # (R, S, N)
    p = torch.softmax(W.abs().view(R, nb, -1, N) / tau, dim=2)        # (R, banks, S_b, N)
    V = _canonical_v(V, heads).to(p.dtype)
    rows = torch.arange(R, device=x.device)
    if V.dim() == 3:                                                   # (HV, nch, S)
        Vr = V[:, rows % nch].view(V.shape[0], R, nb, -1)
        out = torch.einsum("rbsn,hrbs->hbrn", p, Vr)
    else:                                                              # (G, HV, S, N)
        Vr = V[rows // nch].view(R, V.shape[1], nb, -1, N)
        out = torch.einsum("rbsn,rhbsn->hbrn", p, Vr)
    return out[0, 0] if heads is None and nb == 1 else out


def spectral_softmax(x: torch.Tensor, plan: MorletPlan, H: torch.Tensor, V: torch.Tensor,
                     nch: int, tau: float = 1.0, heads: Optional[int] = None) -> torch.Tensor:
    """``x (R, N)`` -> ``sum_f softmax_f(|W_f x|/tau) V_f``, shape ``(R, N)``.

    ``V`` is ``(nch, S)`` (row r uses ``V[r % nch]``) or ``(R // nch, S, N)`` (row r uses
    ``V[r // nch]``).  Fused on CUDA (memory O(R*N)); plain PyTorch elsewhere.  The
    tensor-core precision follows :mod:`fdwt.config` and the dtype of ``x`` (bf16 / fp16
    inputs use bf16 operands with fp32 accumulation).

    Several outputs from one transform:

    * ``heads=HV`` value heads share the softmax of each row: ``V`` is ``(HV, nch, S)`` or
      ``(R // nch, HV, S, N)``, and ``out[h] = sum_f p_f V[h]_f``;
    * a plan with ``banks=B`` filter banks (:class:`MorletPlan`) gives every row B softmaxes,
      each over its bank's ``S / B`` scales (``V`` then indexes all ``S`` scales, bank-major).

    The output is then ``(HV, B, R, N)`` (``HV = 1`` without ``heads``).
    """
    if x.is_cuda:
        return _spectral_softmax_gpu(_SpectralSoftmaxFunction.apply, x, plan, V, heads, H, nch, float(tau),
                                     KC.resolve_precision(x.dtype))
    return _spectral_softmax_torch(x, plan, H, V, nch, tau, heads)


def spectral_softmax_params(x: torch.Tensor, plan: MorletPlan, log_scales: torch.Tensor, w0,
                            V: torch.Tensor, nch: int, tau: float = 1.0,
                            heads: Optional[int] = None) -> torch.Tensor:
    """:func:`spectral_softmax` with the filters ``plan.filters(exp(log_scales), w0)``,
    differentiable w.r.t. ``log_scales`` and ``w0`` (a tensor or a float; one value per filter
    bank, or one for all).  On CUDA their gradients are computed inside the fused backward
    (see :mod:`fdwt.config`)."""
    if x.is_cuda:
        return _spectral_softmax_gpu(_SpectralSoftmaxParamFunction.apply, x, plan, V, heads, log_scales, w0,
                                     nch, float(tau), KC.resolve_precision(x.dtype))
    dt = torch.float64 if x.dtype == torch.float64 else torch.float32
    H = plan.filters(log_scales.to(torch.float64).exp(), w0, dt)
    return spectral_softmax(x, plan, H, V, nch, tau, heads)


# --------------------------------------------------------------------------- torch paths


def _scatter_stage(y, out, stage_idx, Cc):
    """y: (R, F, n) for one stage -> out[:, idx, :, :] (R, S, n, Cc)."""
    R, Fn, n = y.shape
    y = y.reshape(R, Fn // Cc, Cc, n).transpose(-1, -2)
    return out.index_copy(1, stage_idx, y.to(out.dtype))


def _torch_cwt(x, plan: MorletPlan, H: torch.Tensor) -> torch.Tensor:
    R, N = x.shape
    Cc = plan.channels
    out = x.new_zeros((R, plan.num_scales, N, Cc), dtype=_out_dtype(x.dtype))
    lp = plan.lowpass.to(device=x.device, dtype=x.dtype)
    xj = x
    stat = plan.static_tensors(x.device)[0]
    by_oct = {st.octave: (st, H[sv[4]][:, sv[5]]) for st, sv in zip(plan.stages, stat)}
    for j in range(plan.n_octaves):
        dil, mj = 2 ** j, plan.margins[j]
        if j in by_oct:
            st, h = by_oct[j]
            idx = torch.tensor(st.scale_idx, device=x.device)
            if plan.method == "fft":
                y = _fft_conv(xj, h, N, mj + st.half * dil)
            else:
                y = reference.filterbank_conv(xj, h, N, mj + st.half * dil, dil)
            out = _scatter_stage(y, out, idx, Cc)
        if j + 1 < plan.n_octaves:
            m1 = plan.margins[j + 1]
            xj = reference.filterbank_conv(xj, lp[None], N + 2 * m1, mj - m1 + plan.c_lp * dil, dil)[:, 0]
    return out


def _fft_conv(x, h, n_out, off):
    # y[f, n] = sum_k h[f, k] x[n + off - k] = (h * x)[n + off]
    R, N = x.shape
    K = h.shape[1]
    nfft = 1 << (N + K - 1 - 1).bit_length()
    X = torch.fft.rfft(x.to(h.dtype), nfft)
    H = torch.fft.rfft(h, nfft)
    y = torch.fft.irfft(X[:, None, :] * H[None], nfft)
    return y[..., off:off + n_out]


# --------------------------------------------------------------------------- public API


def _apply_plan(x: torch.Tensor, plan: MorletPlan, H: torch.Tensor) -> torch.Tensor:
    if not x.is_floating_point() or x.is_complex():
        raise TypeError("cwt expects a real floating point tensor")
    lead, N = x.shape[:-1], x.shape[-1]
    x2 = x.reshape(-1, N).contiguous()
    if x2.is_cuda and plan.method != "fft":
        if x2.dtype not in (torch.float32, torch.float64):
            x2 = x2.float()
        y = _MorletCWTFunction.apply(x2, plan, H)
    else:
        y = _torch_cwt(x2.to(_out_dtype(x2.dtype)), plan, H)
    y = y.reshape(*lead, plan.num_scales, N, plan.channels)
    return torch.view_as_complex(y) if plan.complex else y[..., 0]


@lru_cache(maxsize=64)
def _cached_plan(scales: Tuple[float, ...], w0, truncate, norm, complex, method) -> MorletPlan:
    return MorletPlan(scales, w0, truncate, norm, complex, method)


@lru_cache(maxsize=64)
def _cached_filters(plan: MorletPlan, device: torch.device, dtype: torch.dtype):
    s = torch.tensor(plan.scales, dtype=torch.float64, device=device)
    return plan.filters(s, plan.w0, dtype)


def _resolve_scales(scales, freqs, fs, w0) -> List[float]:
    if (scales is None) == (freqs is None):
        raise ValueError("pass exactly one of `scales` or `freqs`")
    s = freqs_to_scales(freqs, fs, w0) if freqs is not None else torch.as_tensor(scales, dtype=torch.float64)
    return [float(v) for v in s.flatten()]


def cwt(x: torch.Tensor, scales: Optional[Sequence[float]] = None, *, freqs=None, fs: float = 1.0,
        w0: float = 6.0, method: str = "fast", norm: str = "l2", complex: bool = True,
        truncate: float = 4.0) -> torch.Tensor:
    """Morlet CWT along the last dimension.

    Args:
        x: real tensor ``(..., N)``.
        scales: wavelet scales in samples, or
        freqs: centre frequencies (same unit as ``fs``); ``scale = w0 * fs / (2 pi f)``.
        method: ``"fast"`` (O(N) pyramid), ``"direct"`` (exact) or ``"fft"``.
        complex: complex Morlet (default) or its real part only.

    Returns:
        ``(..., S, N)`` complex tensor (real if ``complex=False``).
    """
    s = _resolve_scales(scales, freqs, fs, w0)
    plan = _cached_plan(tuple(s), float(w0), float(truncate), norm, bool(complex), method)
    dt = torch.float64 if x.dtype == torch.float64 else torch.float32
    return _apply_plan(x, plan, _cached_filters(plan, x.device, dt))


class CWT1D(nn.Module):
    """Morlet continuous wavelet transform as a module.

    Input ``(..., N)`` real, output ``(..., S, N)`` complex (or real with
    ``complex=False``).  With ``learnable=True`` the (log-)scales are parameters
    and receive gradients through the fused backward kernel; ``learnable_w0``
    additionally learns the Morlet wave number.

    ``banks=B``: B independent filter banks, each starting from ``freqs`` / ``scales`` and with
    its own learnable scales and wave number (``log_scales`` holds the ``B * S`` scales
    bank-major, ``w0`` one value per bank); see :func:`spectral_softmax`.
    """

    def __init__(self, freqs=None, scales=None, fs: float = 1.0, w0: float = 6.0,
                 method: str = "fast", norm: str = "l2", complex: bool = True,
                 truncate: float = 4.0, learnable: bool = False, learnable_w0: bool = False,
                 support_margin: Optional[float] = None, banks: int = 1):
        super().__init__()
        s = _resolve_scales(scales, freqs, fs, w0) * banks
        if support_margin is None:
            support_margin = 1.25 if learnable else 1.0
        self.fs = fs
        self.banks = banks
        self.plan = MorletPlan(s, w0, truncate, norm, complex, method, support_margin, banks=banks)
        log_s = torch.tensor(s, dtype=torch.float64).log().float()
        if learnable:
            self.log_scales = nn.Parameter(log_s)
        else:
            self.register_buffer("log_scales", log_s)
        if learnable_w0:
            self.w0 = nn.Parameter(torch.full((banks,), float(w0)) if banks > 1 else torch.tensor(float(w0)))
        else:
            self.w0 = float(w0)
        self._learn = learnable or learnable_w0
        self._filter_cache: Dict = {}

    @property
    def scales(self) -> torch.Tensor:
        return self.log_scales.exp()

    @property
    def w0_value(self) -> float:
        """The wave number (with filter banks: the largest bank's, which sets the octaves)."""
        return float(self.w0.detach().max()) if isinstance(self.w0, torch.Tensor) else float(self.w0)

    @property
    def freqs(self) -> torch.Tensor:
        if isinstance(self.w0, torch.Tensor) and self.w0.numel() > 1:
            w = self.w0.detach().cpu().double().repeat_interleave(self.plan.num_scales // self.w0.numel())
            return scales_to_freqs(self.scales.detach().cpu(), self.fs, 1.0) * w
        return scales_to_freqs(self.scales.detach(), self.fs, self.w0_value)

    def _filters(self, device, dtype):
        if self._learn:
            return self.plan.filters(self.log_scales.to(torch.float64).exp(), self.w0, dtype)
        key = (device, dtype)
        if key not in self._filter_cache:
            self._filter_cache = {key: self.plan.filters(self.scales.to(torch.float64), self.w0, dtype)}
        return self._filter_cache[key]

    def _apply(self, fn, *args, **kwargs):
        self._filter_cache = {}
        return super()._apply(fn, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = torch.float64 if x.dtype == torch.float64 else torch.float32
        return _apply_plan(x, self.plan, self._filters(x.device, dt))

    def spectral_softmax(self, x: torch.Tensor, V: torch.Tensor, nch: int, tau: float = 1.0,
                         heads: Optional[int] = None) -> torch.Tensor:
        """``x (R, N)`` -> ``sum_f softmax_f(|W_f x| / tau) V_f`` with this module's filters
        (see :func:`spectral_softmax`, also for ``heads`` and the banks' output layout);
        learnable scales / w0 get their gradients from the fused backward kernel."""
        if self._learn:
            return spectral_softmax_params(x, self.plan, self.log_scales, self.w0, V, nch, tau, heads)
        dt = torch.float64 if x.dtype == torch.float64 else torch.float32
        return spectral_softmax(x, self.plan, self._filters(x.device, dt), V, nch, tau, heads)

    @torch.no_grad()
    def replan(self, support_margin: Optional[float] = None, regroup: bool = False) -> "CWT1D":
        """Rebuild the plan (filter supports, octaves) around the *current* scales.

        The plan is built from the initial scales and fixes every filter's support (and its
        octave), so learned scales that have shrunk keep being convolved with mostly zero
        taps, and grown ones get truncated.  ``replan`` resizes the supports to
        ``truncate * support_margin`` standard deviations of the current scales:

        * ``regroup=False``: every scale stays in its octave (same dilation, same low-pass
          signal), so the transform is unchanged up to the truncated Gaussian tails (the
          envelope beyond 4 standard deviations is below 3.4e-4 of its peak);
        * ``regroup=True``: scales are also re-assigned to octaves from the current scales
          and wave number, like a fresh plan.  Faster when scales have shrunk below their
          octave (fewer pyramid levels), but it *changes* the transform where the old
          octave under-sampled the wavelet.

        Call it after loading trained weights (or periodically while training: with
        ``support_margin > 1`` scales may grow by that factor before being truncated).
        """
        p = self.plan
        margin = p.support_margin if support_margin is None else float(support_margin)
        scales = [float(v) for v in self.scales.detach().to(torch.float64).cpu()]
        w0 = self.w0_value
        octaves = None if regroup else p.octaves
        self.plan = MorletPlan(scales, w0 if regroup else p.w0, p.truncate, p.norm, p.complex, p.method,
                               margin, octaves, banks=p.banks)
        self._filter_cache = {}
        return self

    def extra_repr(self) -> str:
        p = self.plan
        return (f"scales={p.num_scales}, w0={self.w0_value:g}, method={p.method}, norm={p.norm}, "
                f"complex={p.complex}, octaves={p.n_octaves}, learnable={self._learn}")
