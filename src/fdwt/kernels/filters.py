"""Morlet filters, their tangents and every tile layout of the spectral softmax, in one launch.

The tiles the softmax kernels read (banded Toeplitz tiles, adjoint blocks, per-octave filter
slices) are gathers of the sampled filters ``H`` and tangents ``dH/dlog s``, ``dH/dw0``.  Their
index pattern depends only on the plan, so :class:`fdwt.cwt.MorletPlan` builds it once as an
int32 *code map* (``0``: zero, else ``1 + (which * S*C + row) * W + col``) and
:func:`morlet_eval` evaluates the Morlet expressions directly at the coded positions: one
kernel instead of a few hundred small PyTorch ops per call.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

BLOCK = 1024
_PI_M14 = math.pi ** -0.25


@triton.jit
def _morlet_eval_kernel(map_ptr, out_ptr, ls_ptr, w0_ptr, dil_ptr, sup_ptr, n, W0_VAL,
                        RW, W, C, CENTER, SB,
                        W0_PTR: tl.constexpr, L2: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = e < n
    code = tl.load(map_ptr + e, mask=ok, other=0)
    nz = code > 0
    c0 = tl.where(nz, code - 1, 0)
    which = c0 // RW                      # 0: H, 1: dH/dlog s, 2: dH/dw0
    rem = c0 - which * RW
    row = rem // W
    col = rem - row * W
    i = row // C
    part = row - i * C                    # 0: real, 1: imaginary
    s = tl.exp(tl.load(ls_ptr + i, mask=ok, other=0.0).to(tl.float64))
    dil = tl.load(dil_ptr + i, mask=ok, other=1.0)
    sup = tl.load(sup_ptr + i, mask=ok, other=0.0)
    if W0_PTR:
        w = tl.load(w0_ptr + i // SB, mask=ok, other=0.0).to(tl.float64)  # one w0 per bank of SB scales
    else:
        w = W0_VAL.to(tl.float64)
    k = (col - CENTER).to(tl.float64)
    t = k * dil / s
    if L2:
        amp = 0.7511255444649425 * dil / tl.sqrt(s)
        p = 0.5
    else:
        amp = 0.7511255444649425 * dil / s
        p = 1.0
    env = tl.where(tl.abs(k) <= sup, tl.exp(-0.5 * t * t), 0.0) * amp
    wt = w * t
    cs = tl.cos(wt)
    sn = tl.sin(wt)
    cw = tl.exp(-0.5 * w * w)
    q = t * t - p
    re = tl.where(which == 0, cs - cw, tl.where(which == 1, wt * sn + (cs - cw) * q, w * cw - t * sn))
    im = tl.where(which == 0, sn, tl.where(which == 1, sn * q - wt * cs, t * cs))
    v = tl.where(nz, tl.where(part == 0, re, im) * env, 0.0)
    tl.store(out_ptr + e, v.to(out_ptr.dtype.element_ty), mask=ok)


def morlet_eval(codes: torch.Tensor, out: torch.Tensor, log_scales: torch.Tensor, w0, dil: torch.Tensor,
                sup: torch.Tensor, channels: int, width: int, center: int, l2: bool) -> torch.Tensor:
    """``out[e]`` = the coded filter / tangent sample ``codes[e]`` (int32, see module doc), from
    the current ``log_scales`` ``(S,)`` and ``w0`` (float, or a tensor of 1 or B values: one per
    bank of S / B consecutive scales); ``dil``, ``sup`` are the per-scale fp64 dilations and
    half-supports.  Computed in fp64, stored in ``out.dtype``."""
    n = codes.numel()
    if n == 0:
        return out
    S = log_scales.numel()
    w0_t = isinstance(w0, torch.Tensor)
    _morlet_eval_kernel[(triton.cdiv(n, BLOCK),)](
        codes, out, log_scales.detach(), w0.detach() if w0_t else codes, dil, sup, n,
        0.0 if w0_t else float(w0), S * channels * width, width, channels, center,
        S // w0.numel() if w0_t else S,
        W0_PTR=w0_t, L2=l2, BLOCK=BLOCK,
    )
    return out
