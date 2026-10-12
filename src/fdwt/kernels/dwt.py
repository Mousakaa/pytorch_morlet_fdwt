"""Triton kernels for one level of the fast (Mallat) wavelet transform.

A DWT level is the linear *analysis* operator

    y0[o] = sum_j h0[j] * x_ext[2*o + OFF - j]
    y1[o] = sum_j h1[j] * x_ext[2*o + OFF - j]

where ``x_ext`` is ``x`` extended past its boundaries according to the
signal-extension mode.  Both sub-bands are produced by a single fused pass
(boundary handling + filtering + down-sampling), so ``x`` is read once.

The *synthesis* kernel applies the exact transpose of that operator: it
up-samples, filters, sums both sub-bands and folds the boundary extension back
onto the signal.  It is simultaneously

* the backward pass of the analysis operator (any mode), and
* the inverse DWT (``pywt.idwt``) when fed with time-reversed synthesis filters.

Conversely the analysis kernel is the backward pass of the synthesis operator,
so every forward/backward pair in this library is one fused kernel launch per
level.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

MODES = {"zero": 0, "periodic": 1, "symmetric": 2, "reflect": 3, "periodization": 4}
_ZERO, _PERIODIC, _SYMMETRIC, _REFLECT, _PERIODIZATION = range(5)


def mode_period(mode: int, n: int) -> int:
    """Period of the extended signal (unused for zero padding)."""
    if mode == _PERIODIC:
        return n
    if mode == _SYMMETRIC:
        return 2 * n
    if mode == _REFLECT:
        return 2 * n - 2
    if mode == _PERIODIZATION:
        return n + n % 2
    return 1


def normalize_mode(mode: int, n: int) -> int:
    # reflect on a single sample degenerates to a constant (period-1) extension
    return _PERIODIC if (mode == _REFLECT and n == 1) else mode


@triton.jit
def _ext_index(p, N, P, MODE: tl.constexpr):
    """Map a position of the extended signal onto ``[0, N)``."""
    if MODE == 0:
        idx = p
        valid = (p >= 0) & (p < N)
    else:
        q = p % P
        q = tl.where(q < 0, q + P, q)  # robust to C- or Python-style modulo
        if MODE == 1:
            idx = q
        elif MODE == 2:
            idx = tl.where(q < N, q, P - 1 - q)
        elif MODE == 3:
            idx = tl.where(q < N, q, P - q)
        else:
            idx = tl.minimum(q, N - 1)
        valid = p == p
    return idx, valid


@triton.jit
def _analysis_kernel(
    x_ptr, h0_ptr, h1_ptr, y0_ptr, y1_ptr,
    R, N, NO, P, OFF,
    L: tl.constexpr, MODE: tl.constexpr, ACC: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_O: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    o = tl.program_id(1) * BLOCK_O + tl.arange(0, BLOCK_O)
    rmask = rows < R
    omask = o < NO
    x_row = x_ptr + rows.to(tl.int64)[:, None] * N

    acc0 = tl.zeros([BLOCK_R, BLOCK_O], ACC)
    acc1 = tl.zeros([BLOCK_R, BLOCK_O], ACC)
    for j in tl.static_range(L):
        idx, valid = _ext_index(2 * o + OFF - j, N, P, MODE)
        m = rmask[:, None] & (omask & valid)[None, :]
        xv = tl.load(x_row + idx[None, :], mask=m, other=0.0).to(ACC)
        acc0 += xv * tl.load(h0_ptr + j).to(ACC)
        acc1 += xv * tl.load(h1_ptr + j).to(ACC)

    out = rows.to(tl.int64)[:, None] * NO + o[None, :]
    m = rmask[:, None] & omask[None, :]
    tl.store(y0_ptr + out, acc0.to(y0_ptr.dtype.element_ty), mask=m)
    tl.store(y1_ptr + out, acc1.to(y1_ptr.dtype.element_ty), mask=m)


@triton.jit
def _transpose_tap_sum(a_row, d_row, h0_ptr, h1_ptr, p, NC, OFF, lane_mask,
                       L: tl.constexpr, ACC: tl.constexpr):
    """sum over (o, j) with 2*o + OFF - j == p of h0[j]*a[o] + h1[j]*d[o].

    Only taps with the right parity contribute, so each lane visits L/2 taps.
    """
    q = p - OFF
    par = q & 1
    o_base = (q + par) >> 1
    acc = tl.zeros(p.shape, ACC)
    for t in tl.static_range(L // 2):
        o = o_base + t
        m = lane_mask & (o >= 0) & (o < NC)
        av = tl.load(a_row + o, mask=m, other=0.0).to(ACC)
        dv = tl.load(d_row + o, mask=m, other=0.0).to(ACC)
        h0e = tl.load(h0_ptr + 2 * t).to(ACC)
        h0o = tl.load(h0_ptr + 2 * t + 1).to(ACC)
        h1e = tl.load(h1_ptr + 2 * t).to(ACC)
        h1o = tl.load(h1_ptr + 2 * t + 1).to(ACC)
        acc += av * tl.where(par == 1, h0o, h0e) + dv * tl.where(par == 1, h1o, h1e)
    return acc


@triton.jit
def _synthesis_kernel(
    a_ptr, d_ptr, h0_ptr, h1_ptr, y_ptr,
    R, N, NC, P, OFF, PMIN, PMAX, W_LO, W_HI,
    L: tl.constexpr, MODE: tl.constexpr, ACC: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_I: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    i0 = tl.program_id(1) * BLOCK_I
    i = i0 + tl.arange(0, BLOCK_I)
    i1 = tl.minimum(i0 + BLOCK_I, N) - 1
    rmask = rows < R
    imask = i < N
    row64 = rows.to(tl.int64)[:, None]
    a_row = a_ptr + row64 * NC
    d_row = d_ptr + row64 * NC
    lane = rmask[:, None] & imask[None, :]
    zero = tl.zeros([BLOCK_R, BLOCK_I], tl.int32)

    if MODE == 0:
        acc = _transpose_tap_sum(a_row, d_row, h0_ptr, h1_ptr, zero + i[None, :], NC, OFF,
                                 lane, L, ACC)
    else:
        # Every extended position p with ext_index(p) == i contributes to y[i].  Those are
        # p = base + w*P for the direct class (base = i) and, for mirrored modes, the
        # mirror class.  Wraps that cannot reach [PMIN, PMAX] are skipped per tile, so
        # interior tiles only do the w == 0 direct pass.
        acc = tl.zeros([BLOCK_R, BLOCK_I], ACC)
        for w in range(W_LO, W_HI + 1):
            if (i0 + w * P <= PMAX) & (i1 + w * P >= PMIN):
                acc += _transpose_tap_sum(a_row, d_row, h0_ptr, h1_ptr,
                                          zero + (i + w * P)[None, :], NC, OFF, lane, L, ACC)
            if MODE == 2:  # symmetric: mirror of i is P - 1 - i
                if (P - 1 - i1 + w * P <= PMAX) & (P - 1 - i0 + w * P >= PMIN):
                    acc += _transpose_tap_sum(a_row, d_row, h0_ptr, h1_ptr,
                                              zero + (P - 1 - i + w * P)[None, :], NC, OFF,
                                              lane, L, ACC)
            if MODE == 3:  # reflect: mirror of i is P - i, except at both end points
                if (P - i1 + w * P <= PMAX) & (P - i0 + w * P >= PMIN):
                    mm = lane & ((i > 0) & (i < N - 1))[None, :]
                    acc += _transpose_tap_sum(a_row, d_row, h0_ptr, h1_ptr,
                                              zero + (P - i + w * P)[None, :], NC, OFF,
                                              mm, L, ACC)
            if MODE == 4:  # periodization of an odd-length signal duplicates x[N-1]
                if (P != N) & (i1 == N - 1):
                    if (N + w * P <= PMAX) & (N + w * P >= PMIN):
                        mm = lane & (i == N - 1)[None, :]
                        acc += _transpose_tap_sum(a_row, d_row, h0_ptr, h1_ptr,
                                                  zero + (N + w * P), NC, OFF, mm, L, ACC)

    tl.store(y_ptr + row64 * N + i[None, :], acc.to(y_ptr.dtype.element_ty), mask=lane)


# --------------------------------------------------------------------------- launchers


def _acc_dtype(dtype: torch.dtype):
    return tl.float64 if dtype == torch.float64 else tl.float32


def _tiles(rows: int, n: int):
    block_n = min(512, triton.next_power_of_2(max(n, 1)))
    block_r = max(1, min(triton.next_power_of_2(rows), 1024 // block_n))
    return block_r, block_n


def analysis(x: torch.Tensor, h0: torch.Tensor, h1: torch.Tensor, mode: int, off: int, n_out: int):
    """x: (R, N) contiguous -> (y0, y1) each (R, n_out)."""
    R, N = x.shape
    mode = normalize_mode(mode, N)
    y0 = torch.empty((R, n_out), device=x.device, dtype=x.dtype)
    y1 = torch.empty_like(y0)
    if R == 0 or n_out == 0:
        return y0, y1
    br, bo = _tiles(R, n_out)
    grid = (triton.cdiv(R, br), triton.cdiv(n_out, bo))
    _analysis_kernel[grid](
        x, h0, h1, y0, y1, R, N, n_out, mode_period(mode, N), off,
        L=h0.numel(), MODE=mode, ACC=_acc_dtype(x.dtype), BLOCK_R=br, BLOCK_O=bo,
        num_warps=4,
    )
    return y0, y1


def synthesis(a: torch.Tensor, d: torch.Tensor, h0: torch.Tensor, h1: torch.Tensor,
              mode: int, off: int, n: int):
    """Transpose of :func:`analysis`: (a, d) each (R, NC) -> y (R, n)."""
    R, NC = a.shape
    L = h0.numel()
    mode = normalize_mode(mode, n)
    y = torch.empty((R, n), device=a.device, dtype=a.dtype)
    if R == 0 or n == 0:
        return y
    P = mode_period(mode, n)
    pmin, pmax = off - (L - 1), 2 * (NC - 1) + off
    w_lo, w_hi = (pmin - P + 1) // P, pmax // P
    br, bi = _tiles(R, n)
    grid = (triton.cdiv(R, br), triton.cdiv(n, bi))
    _synthesis_kernel[grid](
        a, d, h0, h1, y, R, n, NC, P, off, pmin, pmax, w_lo, w_hi,
        L=L, MODE=mode, ACC=_acc_dtype(a.dtype), BLOCK_R=br, BLOCK_I=bi,
        num_warps=4,
    )
    return y
