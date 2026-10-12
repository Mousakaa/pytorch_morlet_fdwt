"""Fused "scalogram mixing" kernels.

    y[r, n] += sum_s a[d, s] * |W_s[r, n]| + b[d, s] * Re W_s[r, n]        (d = (r + ROW0) % D)

where ``W_s`` is the filter-bank convolution of :mod:`.conv`.  The scalogram
tile is produced in registers and reduced over scales before anything is
written, so the forward pass only writes ``(R, N)``.  With ``GRAD=True`` the
same kernel recomputes the tile and, from the incoming gradient ``gy``,

* accumulates ``ga[d, s] = sum gy * |W|`` and ``gb[d, s] = sum gy * Re W``, and
* writes ``gW = dL/dW`` for one octave, consumed by the regular grad-x /
  grad-h kernels (so the backward only ever holds one octave of the
  scalogram, optionally in row chunks).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .conv import _acc, _conv_tile


@triton.jit
def _fb_mix_kernel(
    x_ptr, hT_ptr, kb_ptr, gidx_ptr, a_ptr, b_ptr, y_ptr,
    gy_ptr, gw_ptr, ga_ptr, gb_ptr,
    N_IN, N_OUT, OFF, DIL, S, S_W, D, ROW0,
    C: tl.constexpr, ACC: tl.constexpr, GRAD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_S: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    n0 = tl.program_id(1) * BLOCK_N
    n = n0 + tl.arange(0, BLOCK_N)
    sb = tl.program_id(2)
    s = sb * BLOCK_S + tl.arange(0, BLOCK_S)
    smask = s < S
    nmask = n < N_OUT
    klo = tl.load(kb_ptr + 2 * sb)
    khi = tl.load(kb_ptr + 2 * sb + 1)
    acc = _conv_tile(x_ptr + r * N_IN, hT_ptr, tl.num_programs(2) * BLOCK_S * C, s, n, n0,
                     klo, khi, N_IN, OFF, DIL, C, ACC, BLOCK_N, BLOCK_S)

    if C == 2:
        re, im = tl.split(acc)
        mag = tl.sqrt(re * re + im * im)
    else:
        re = tl.sum(acc, axis=2)
        im = re * 0.0
        mag = tl.abs(re)
    d = (r + ROW0) % D
    w_off = d * S_W + tl.load(gidx_ptr + s, mask=smask, other=0)
    av = tl.load(a_ptr + w_off, mask=smask, other=0.0).to(ACC)
    bv = tl.load(b_ptr + w_off, mask=smask, other=0.0).to(ACC)

    if GRAD:
        gyv = tl.load(gy_ptr + r * N_OUT + n, mask=nmask, other=0.0).to(ACC)
        tl.atomic_add(ga_ptr + w_off, tl.sum(gyv[None, :] * mag, axis=1), mask=smask)
        tl.atomic_add(gb_ptr + w_off, tl.sum(gyv[None, :] * re, axis=1), mask=smask)
        inv = tl.where(mag > 0, 1.0 / tl.where(mag > 0, mag, 1.0), 0.0)
        g_re = gyv[None, :] * (av[:, None] * re * inv + bv[:, None])
        g_off = r * S * N_OUT * C + (s * (N_OUT * C))[:, None] + (n * C)[None, :]
        m = smask[:, None] & nmask[None, :]
        tl.store(gw_ptr + g_off, g_re.to(gw_ptr.dtype.element_ty), mask=m)
        if C == 2:
            g_im = gyv[None, :] * (av[:, None] * im * inv)
            tl.store(gw_ptr + g_off + 1, g_im.to(gw_ptr.dtype.element_ty), mask=m)
    else:
        contrib = tl.sum(av[:, None] * mag + bv[:, None] * re, axis=0)
        tl.atomic_add(y_ptr + r * N_OUT + n, contrib, mask=nmask)


def mix_forward(x, hT, kb, gidx, a, b, y, off: int, dil: int, block_s: int, d: int, row0: int):
    """y (R, N_OUT) fp32 += mixed scalogram of one stage; a, b: (D, S_total) fp32."""
    R, n_in = x.shape
    n_out = y.shape[1]
    S = gidx.numel()
    C = hT.shape[1] // (kb.shape[0] * block_s)
    if R == 0 or n_out == 0 or S == 0:
        return y
    block_n = min(256, triton.next_power_of_2(n_out))
    grid = (R, triton.cdiv(n_out, block_n), kb.shape[0])
    _fb_mix_kernel[grid](
        x, hT, kb, gidx, a, b, y, y, y, a, a,
        n_in, n_out, off, dil, S, a.shape[1], d, row0,
        C=C, ACC=_acc(x.dtype), GRAD=False, BLOCK_N=block_n, BLOCK_S=block_s, num_warps=4,
    )
    return y


def mix_backward(x, hT, kb, gidx, a, b, gy, gw, ga, gb, off: int, dil: int, block_s: int,
                 d: int, row0: int):
    """Writes gw (R, S, N_OUT, C) = dL/dW for one stage; accumulates ga, gb (D, S_total)."""
    R, n_in = x.shape
    n_out = gy.shape[1]
    S = gidx.numel()
    C = gw.shape[-1]
    if R == 0 or n_out == 0 or S == 0:
        return gw
    block_n = min(256, triton.next_power_of_2(n_out))
    grid = (R, triton.cdiv(n_out, block_n), kb.shape[0])
    _fb_mix_kernel[grid](
        x, hT, kb, gidx, a, b, gy, gy, gw, ga, gb,
        n_in, n_out, off, dil, S, a.shape[1], d, row0,
        C=C, ACC=_acc(x.dtype), GRAD=True, BLOCK_N=block_n, BLOCK_S=block_s, num_warps=4,
    )
    return gw
