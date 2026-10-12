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

from . import reference
from .kernels import conv as KC
from .kernels import mix as KM
from .kernels import softmax as KS

__all__ = ["cwt", "CWT1D", "MorletPlan", "freqs_to_scales", "scales_to_freqs"]

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
    """

    def __init__(self, scales: Sequence[float], w0: float = 6.0, truncate: float = 4.0,
                 norm: str = "l2", complex: bool = True, method: str = "fast",
                 support_margin: float = 1.0):
        if method not in ("fast", "direct", "fft"):
            raise ValueError(f"unknown method '{method}' (expected 'fast', 'direct' or 'fft')")
        if norm not in ("l1", "l2"):
            raise ValueError("norm must be 'l1' or 'l2'")
        scales = [float(s) for s in scales]
        if not scales or min(scales) <= 0:
            raise ValueError("scales must be a non-empty sequence of positive numbers")
        self.scales, self.w0, self.truncate, self.norm = scales, float(w0), float(truncate), norm
        self.complex, self.method = bool(complex), method
        self.channels = 2 if complex else 1
        reach = truncate * support_margin

        # octave j holds the scales in [2^j s_th, 2^(j+1) s_th): after j half-band
        # low-pass stages the Morlet band (w0 +- truncate)/s' must stay below 0.8 pi.
        self.s_th = (w0 + truncate) / (0.8 * math.pi)
        groups: Dict[int, List[int]] = {}
        for i, s in enumerate(scales):
            j = 0 if method != "fast" else max(0, math.floor(math.log2(s / self.s_th)))
            groups.setdefault(j, []).append(i)
        self.stages: List[_Stage] = []
        for j in sorted(groups):
            dil = 2 ** j
            sup = {i: max(1, math.ceil(reach * scales[i] / dil)) for i in groups[j]}
            idx = sorted(groups[j], key=lambda i: sup[i])
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

    def filters(self, scales: torch.Tensor, w0, dtype: torch.dtype) -> torch.Tensor:
        """All sampled Morlet filters as one ``(S*C, 2*center+1)`` tensor (row ``i*C + c``).

        Scale ``i`` is sampled with the spacing (dilation) of its octave and truncated to
        its support.  Differentiable w.r.t. ``scales`` and ``w0`` (when it is a tensor).
        """
        dev = scales.device
        f64 = torch.float64
        s = scales.to(f64)[:, None]
        dil = torch.tensor(self._dil, dtype=f64, device=dev)[:, None]
        sup = torch.tensor(self._sup, dtype=f64, device=dev)[:, None]
        k = torch.arange(-self.center, self.center + 1, device=dev, dtype=f64)[None, :]
        t = k * dil / s
        env = torch.exp(-0.5 * t * t) * (k.abs() <= sup)
        # a dilated sum approximates the Riemann sum with spacing `dil`
        amp = _PI_M14 * dil / (s.sqrt() if self.norm == "l2" else s)
        w0_ = torch.as_tensor(w0, dtype=f64, device=dev)
        w0t = w0_ * t
        re = (torch.cos(w0t) - torch.exp(-0.5 * w0_ * w0_)) * env * amp
        if not self.complex:
            return re.to(dtype)
        im = torch.sin(w0t) * env * amp
        return torch.stack([re, im], 1).reshape(-1, k.shape[1]).to(dtype)


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


class _SpectralSoftmaxFunction(torch.autograd.Function):
    """out[r, n] = sum_f softmax_f(|W_f x_r|(n) / tau) * V_f[r, n], without materialising W.

    ``V`` is ``(NCH, S)`` (per channel, row r uses ``V[r % NCH]``) or ``(G, S, N)`` (per
    position, row r uses ``V[r // NCH]``).  The forward keeps a flash-attention style
    online softmax across octave launches; the backward recomputes one octave at a time.
    """

    CHUNK_BYTES = 256 * 2**20

    @staticmethod
    def _v_layout(V):
        if V.dim() == 2:  # (NCH, S)
            return (0, V.shape[1], 1, 0), False
        return (V.shape[1] * V.shape[2], 0, V.shape[2], 1), True  # (G, S, N)

    @staticmethod
    def _stage_filters(plan, H, stat):
        return [H.index_select(0, rows)[:, cols].contiguous() for *_, rows, cols in stat]

    @staticmethod
    def forward(ctx, x, plan: MorletPlan, H, V, nch: int, tau: float, precision: str):
        if not plan.complex:
            raise ValueError("the spectral softmax needs the complex Morlet transform")
        R, N = x.shape
        dt = _out_dtype(x.dtype)
        stat, _ = plan.static_tensors(x.device)
        V_ = V.to(dt).contiguous()
        v_strides, v_pos = _SpectralSoftmaxFunction._v_layout(V_)
        hs = _SpectralSoftmaxFunction._stage_filters(plan, H, stat)
        tiles = [KS.band_tiles(h.detach(), st_[3], N, 2 ** plan.stages[k].octave, precision)
                 for k, (h, st_) in enumerate(zip(hs, stat))]
        m = torch.full((R, N), float("-inf"), device=x.device, dtype=dt)
        l = torch.zeros((R, N), device=x.device, dtype=dt)
        num = torch.zeros((R, N), device=x.device, dtype=dt)
        xj = x
        by_oct = {st.octave: k for k, st in enumerate(plan.stages)}
        for j in range(plan.n_octaves):
            dil, mj = 2 ** j, plan.margins[j]
            if j in by_oct:
                k = by_oct[j]
                _, gidx, _, kf = stat[k][:4]
                KS.forward_stage(xj, tiles[k], kf, gidx, V_, v_strides, v_pos, nch, 1.0 / tau,
                                 mj + plan.stages[k].half * dil, dil, precision, m, l, num)
            if j + 1 < plan.n_octaves:
                xj = _lowpass_step(plan, xj, j)
        out = num / l
        del num
        ctx.plan, ctx.nch, ctx.tau, ctx.precision = plan, nch, tau, precision
        ctx.x_dtype, ctx.v_dtype = x.dtype, V.dtype
        # the low-passed octave signals are recomputed in the backward, not stored
        ctx.tiles = [(None, bu, nc) for _, bu, nc in tiles]
        ctx.save_for_backward(x, H, V_, m, l, out, *[t for t, _, _ in tiles])
        return out

    @staticmethod
    def backward(ctx, gy):
        plan: MorletPlan = ctx.plan
        x, H, V, m, l, out, *tbs = ctx.saved_tensors
        tiles = [(tb, bu, nc) for tb, (_, bu, nc) in zip(tbs, ctx.tiles)]
        prec = ctx.precision
        gy = gy.contiguous().to(V.dtype)
        R, N = gy.shape
        stat, _ = plan.static_tensors(gy.device)
        v_strides, v_pos = _SpectralSoftmaxFunction._v_layout(V)
        hs = _SpectralSoftmaxFunction._stage_filters(plan, H, stat)
        need_h = ctx.needs_input_grad[2]
        gH = torch.zeros_like(H) if need_h else None
        gV = torch.zeros_like(V)
        by_oct = {st.octave: k for k, st in enumerate(plan.stages)}
        xs = _lowpass_chain(plan, x.to(V.dtype))

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
                h = hs[k]
                local = torch.arange(S_j, device=gy.device, dtype=torch.int32)
                chunk = max(1, _SpectralSoftmaxFunction.CHUNK_BYTES // (S_j * N * 2 * gy.element_size()))
                gh_acc = None
                for r0 in range(0, R, chunk):
                    r1 = min(R, r0 + chunk)
                    gw = torch.empty((r1 - r0, S_j, N, 2), device=gy.device, dtype=gy.dtype)
                    KS.backward_stage(xs[j][r0:r1], tiles[k], kf, gidx, V, v_strides, v_pos, ctx.nch,
                                      r0, 1.0 / ctx.tau, off, dil, prec, m[r0:r1], l[r0:r1],
                                      out[r0:r1], gy[r0:r1], gw, gV)
                    KC.backward_x(gw, h, kf, local, gx[r0:r1], off, dil, accumulate=filled,
                                  precision=prec)
                    if need_h:
                        gh = KC.backward_h(xs[j][r0:r1], gw, kb, local, off, dil, h.shape[1], bs,
                                           precision=prec)
                        gh_acc = gh if gh_acc is None else gh_acc + gh
                    del gw
                if need_h:
                    gH[rows, cols] = gh_acc.to(H.dtype)
                filled = True
            if not filled:
                gx.zero_()
            g_next = gx
            xs[j] = None  # free each octave signal once its octave is done
        gx = g_next.to(ctx.x_dtype) if ctx.needs_input_grad[0] else None
        return gx, None, gH, gV.to(ctx.v_dtype), None, None, None


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


def spectral_softmax(x: torch.Tensor, plan: MorletPlan, H: torch.Tensor, V: torch.Tensor,
                     nch: int, tau: float = 1.0) -> torch.Tensor:
    """``x (R, N)`` -> ``sum_f softmax_f(|W_f x|/tau) V_f``, shape ``(R, N)``.

    ``V`` is ``(nch, S)`` (row r uses ``V[r % nch]``) or ``(R // nch, S, N)`` (row r uses
    ``V[r // nch]``).  Fused on CUDA (memory O(R*N)); plain PyTorch elsewhere.  The
    tensor-core precision follows :mod:`fdwt.config` and the dtype of ``x`` (bf16 / fp16
    inputs use bf16 operands with fp32 accumulation).
    """
    R, N = x.shape
    if x.is_cuda:
        prec = KC.resolve_precision(x.dtype)
        x2 = x.contiguous()
        if x2.dtype not in (torch.float32, torch.float64):
            x2 = x2.float()
        return _SpectralSoftmaxFunction.apply(x2, plan, H, V, nch, float(tau), prec)
    p = torch.softmax(_apply_plan(x, plan, H).abs() / tau, dim=-2)  # (R, S, N)
    p = p.view(-1, nch, p.shape[1], N)                              # (G, nch, S, N)
    V = V.to(p.dtype)
    Vb = V[None, :, :, None] if V.dim() == 2 else V[:, None]          # broadcast to p
    return (p * Vb).sum(2).reshape(R, N)


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
    """

    def __init__(self, freqs=None, scales=None, fs: float = 1.0, w0: float = 6.0,
                 method: str = "fast", norm: str = "l2", complex: bool = True,
                 truncate: float = 4.0, learnable: bool = False, learnable_w0: bool = False,
                 support_margin: Optional[float] = None):
        super().__init__()
        s = _resolve_scales(scales, freqs, fs, w0)
        if support_margin is None:
            support_margin = 1.25 if learnable else 1.0
        self.fs = fs
        self.plan = MorletPlan(s, w0, truncate, norm, complex, method, support_margin)
        log_s = torch.tensor(s, dtype=torch.float64).log().float()
        if learnable:
            self.log_scales = nn.Parameter(log_s)
        else:
            self.register_buffer("log_scales", log_s)
        if learnable_w0:
            self.w0 = nn.Parameter(torch.tensor(float(w0)))
        else:
            self.w0 = float(w0)
        self._learn = learnable or learnable_w0
        self._filter_cache: Dict = {}

    @property
    def scales(self) -> torch.Tensor:
        return self.log_scales.exp()

    @property
    def w0_value(self) -> float:
        return float(self.w0.detach()) if isinstance(self.w0, torch.Tensor) else float(self.w0)

    @property
    def freqs(self) -> torch.Tensor:
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

    def extra_repr(self) -> str:
        p = self.plan
        return (f"scales={p.num_scales}, w0={self.w0_value:g}, method={p.method}, norm={p.norm}, "
                f"complex={p.complex}, octaves={p.n_octaves}, learnable={self._learn}")
