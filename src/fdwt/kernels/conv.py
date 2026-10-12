"""Triton kernels for a real signal convolved with a bank of (possibly dilated) filters.

Forward::

    y[r, ch(s), n, c] = sum_k h[k, s*C + c] * x[r, n + OFF - DIL*k]      (x = 0 outside [0, N_IN))

``C`` is 2 for complex filters (interleaved re/im columns of ``h``); the output
is laid out as ``(R, S, N, C)`` so that it can be viewed as a complex tensor
without a copy, and every thread stores (re, im) pairs: stores are fully
coalesced.  ``ch`` maps the scales of a launch to output channels so that
several launches (one per octave of the Morlet pyramid) fill one user-ordered
output.  Filters are grouped in blocks of ``BLOCK_S`` scales of similar
support, and each block only visits its own tap range ``kb[block]``.

Kernels
-------
forward   tile (scales x time): each x sample feeds every scale of the block.
grad x    gx[r, m] = sum_{s,c,k} h[k, s*C+c] * g[r, ch(s), m - OFF + DIL*k, c].
          Lanes cover a polyphase-blocked tile (PB consecutive phases x U
          positions spaced DIL apart) so that, whatever the dilation, the
          gradient window of a tile stays L1-resident and loads are coalesced.
grad h    gh[k, s*C+c] = sum_{r,n} g[r, ch(s), n, c] * x[r, n + OFF - DIL*k],
          a (filters x taps) tile reduced over rows and time with tl.dot.

All three take an "interior" fast path without bounds checks for tiles whose
whole window is inside the signal.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .. import config


@triton.jit
def _conv_tile(x_row, hT_ptr, h_stride, s, n, n0, klo, khi, N_IN, OFF, DIL,
               C: tl.constexpr, ACC: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_S: tl.constexpr):
    """acc[s, n, c] = sum_k hT[k, s*C + c] * x_row[n + OFF - DIL*k] for k in [klo, khi)."""
    c = tl.arange(0, C)
    # h is padded host-side to whole blocks: no mask needed on filter loads
    h_ptrs = hT_ptr + klo * h_stride + (s * C)[:, None, None] + c[None, None, :]
    x_ptrs = x_row + n + OFF - DIL * klo
    acc = tl.zeros([BLOCK_S, BLOCK_N, C], ACC)
    lo = n0 + OFF - DIL * (khi - 1)
    hi = n0 + BLOCK_N - 1 + OFF - DIL * klo
    if (lo >= 0) & (hi < N_IN):
        for _k in range(klo, khi):
            xv = tl.load(x_ptrs).to(ACC)
            hv = tl.load(h_ptrs).to(ACC)
            acc += hv * xv[None, :, None]
            x_ptrs -= DIL
            h_ptrs += h_stride
    else:
        pos = n + OFF - DIL * klo
        for _k in range(klo, khi):
            xv = tl.load(x_ptrs, mask=(pos >= 0) & (pos < N_IN), other=0.0).to(ACC)
            hv = tl.load(h_ptrs).to(ACC)
            acc += hv * xv[None, :, None]
            x_ptrs -= DIL
            h_ptrs += h_stride
            pos -= DIL
    return acc


@triton.jit
def _fb_fwd_kernel(
    x_ptr, hT_ptr, kb_ptr, cmap_ptr, y_ptr,
    N_IN, N_OUT, OFF, DIL, S, S_TOT,
    C: tl.constexpr, ACC: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_S: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    n0 = tl.program_id(1) * BLOCK_N
    n = n0 + tl.arange(0, BLOCK_N)
    sb = tl.program_id(2)
    s = sb * BLOCK_S + tl.arange(0, BLOCK_S)
    c = tl.arange(0, C)
    smask = s < S
    klo = tl.load(kb_ptr + 2 * sb)
    khi = tl.load(kb_ptr + 2 * sb + 1)
    acc = _conv_tile(x_ptr + r * N_IN, hT_ptr, tl.num_programs(2) * BLOCK_S * C, s, n, n0,
                     klo, khi, N_IN, OFF, DIL, C, ACC, BLOCK_N, BLOCK_S)

    ch = tl.load(cmap_ptr + s, mask=smask, other=0).to(tl.int64)
    y_off = r * S_TOT * N_OUT * C + (ch * (N_OUT * C))[:, None, None] \
        + (n * C)[None, :, None] + c[None, None, :]
    m = smask[:, None, None] & (n < N_OUT)[None, :, None] & (c < C)[None, None, :]
    tl.store(y_ptr + y_off, acc.to(y_ptr.dtype.element_ty), mask=m)


@triton.jit
def _fb_fwd_dot_kernel(
    x_ptr, hT_ptr, kb_ptr, cmap_ptr, y_ptr,
    N_IN, N_OUT, OFF, DIL, S, S_TOT,
    C: tl.constexpr, ACC: tl.constexpr, PRECISION: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # acc[(s, c), n] += H[(s, c), k] @ X[k, n]   with the Hankel tile X[k, n] = x[n + OFF - DIL*k]
    r = tl.program_id(0).to(tl.int64)
    n0 = tl.program_id(1) * BLOCK_N
    n = n0 + tl.arange(0, BLOCK_N)
    sb = tl.program_id(2)
    BF: tl.constexpr = BLOCK_S * C
    f = sb * BF + tl.arange(0, BF)
    klo = tl.load(kb_ptr + 2 * sb)
    khi = tl.load(kb_ptr + 2 * sb + 1)
    h_stride = tl.num_programs(2) * BF
    kk = tl.arange(0, BLOCK_K)

    x_row = x_ptr + r * N_IN
    acc = tl.zeros([BF, BLOCK_N], ACC)
    for k0 in range(klo, khi, BLOCK_K):
        k = k0 + kk
        H = tl.load(hT_ptr + k[None, :] * h_stride + f[:, None], mask=(k < khi)[None, :], other=0.0)
        pos = n[None, :] + OFF - DIL * k[:, None]
        lo = n0 + OFF - DIL * (k0 + BLOCK_K - 1)
        hi = n0 + BLOCK_N - 1 + OFF - DIL * k0
        if (lo >= 0) & (hi < N_IN):
            X = tl.load(x_row + pos)
        else:
            X = tl.load(x_row + pos, mask=(pos >= 0) & (pos < N_IN), other=0.0)
        acc += _dot(H.to(ACC), X.to(ACC), ACC, PRECISION)

    s = sb * BLOCK_S + tl.arange(0, BLOCK_S)
    smask = s < S
    out = tl.permute(tl.reshape(acc, [BLOCK_S, C, BLOCK_N]), (0, 2, 1))
    ch = tl.load(cmap_ptr + s, mask=smask, other=0).to(tl.int64)
    c = tl.arange(0, C)
    y_off = r * S_TOT * N_OUT * C + (ch * (N_OUT * C))[:, None, None] \
        + (n * C)[None, :, None] + c[None, None, :]
    m = smask[:, None, None] & (n < N_OUT)[None, :, None] & (c < C)[None, None, :]
    tl.store(y_ptr + y_off, out.to(y_ptr.dtype.element_ty), mask=m)


@triton.jit
def _fb_bwd_x_kernel(
    g_ptr, h_ptr, kf_ptr, cmap_ptr, gx_ptr,
    R, N_IN, N_OUT, OFF, DIL, S, K, S_TOT, N_PB,
    C: tl.constexpr, ACC: tl.constexpr, ACCUMULATE: tl.constexpr, PRECISION: tl.constexpr,
    BU: tl.constexpr, BV: tl.constexpr, PB: tl.constexpr, BR: tl.constexpr,
):
    # Toeplitz form: for output phase p and u, m = p0 + p + DIL*u and n = p0 + p - OFF + DIL*v,
    #   gx[r, m] = sum_{s,c} sum_v h[s*C+c, v - u] * g[r, ch(s), n, c]
    # i.e. acc[u, (p, r)] += T[u, v] @ G[v, (p, r)] with T[u, v] = h[v - u] (banded).
    tid = tl.program_id(1)
    p0 = (tid % N_PB) * PB
    u0 = (tid // N_PB) * BU
    u = tl.arange(0, BU)
    pr = tl.arange(0, PB * BR)
    p = pr % PB
    r = tl.program_id(0) * BR + pr // PB
    col_ok = (r < R) & (p0 + p < DIL)
    g_base = g_ptr + r.to(tl.int64) * (S_TOT * N_OUT * C)
    nb = p0 - OFF

    acc = tl.zeros([BU, PB * BR], ACC)
    for s in range(0, S):
        klo = tl.load(kf_ptr + 2 * s)
        khi = tl.load(kf_ptr + 2 * s + 1)
        g_ch = g_base + tl.load(cmap_ptr + s).to(tl.int64) * (N_OUT * C)
        for cc in tl.static_range(C):
            h_row = h_ptr + (s * C + cc) * K
            for v0 in range(u0 + klo, u0 + BU - 1 + khi, BV):
                n_lo = nb + DIL * v0
                n_hi = nb + PB - 1 + DIL * (v0 + BV - 1)
                if (n_hi >= 0) & (n_lo < N_OUT):
                    v = v0 + tl.arange(0, BV)
                    kk = v[None, :] - (u0 + u)[:, None]
                    T = tl.load(h_row + kk, mask=(kk >= klo) & (kk < khi), other=0.0).to(ACC)
                    n = nb + p[None, :] + DIL * v[:, None]
                    G = tl.load(g_ch[None, :] + n * C + cc,
                                mask=(n >= 0) & (n < N_OUT) & col_ok[None, :], other=0.0).to(ACC)
                    acc += _dot(T, G, ACC, PRECISION)

    m = p0 + p[None, :] + DIL * (u0 + u)[:, None]
    o_ptrs = gx_ptr + r.to(tl.int64)[None, :] * N_IN + m
    omask = (m < N_IN) & col_ok[None, :]
    if ACCUMULATE:
        acc += tl.load(o_ptrs, mask=omask, other=0.0).to(ACC)
    tl.store(o_ptrs, acc.to(gx_ptr.dtype.element_ty), mask=omask)


@triton.jit
def _split_tf32(v):
    hi = (v.to(tl.uint32, bitcast=True) & 0xFFFFE000).to(tl.float32, bitcast=True)
    return hi, v - hi


@triton.jit
def _dot(a, b, ACC: tl.constexpr, PRECISION: tl.constexpr):
    """tl.dot with an extra "3xtf32" mode (hi*hi + hi*lo + lo*hi), written out by hand: as
    accurate as fp32 FMAs, on tensor cores, independent of the compiler's tf32x3 lowering."""
    if PRECISION == "bf16":
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), out_dtype=tl.float32).to(ACC)
    elif PRECISION == "3xtf32":
        a_hi, a_lo = _split_tf32(a)
        b_hi, b_lo = _split_tf32(b)
        acc = tl.dot(a_lo, b_hi, input_precision="tf32", out_dtype=ACC)
        acc = tl.dot(a_hi, b_lo, acc, input_precision="tf32", out_dtype=ACC)
        return tl.dot(a_hi, b_hi, acc, input_precision="tf32", out_dtype=ACC)
    else:
        return tl.dot(a, b, input_precision=PRECISION, out_dtype=ACC)


@triton.jit
def _fb_bwd_h_kernel(
    x_ptr, g_ptr, kb_ptr, cmap_ptr, gh_ptr,
    R, N_IN, N_OUT, OFF, DIL, F, K, S_TOT, N_CHUNKS, ITEMS_PER_PROG,
    C: tl.constexpr, ACC: tl.constexpr, PRECISION: tl.constexpr,
    BLOCK_F: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # gh[f, k] = sum_r sum_n g[r, ch(f // C), n, f % C] * x[r, n + OFF - DIL*k]
    # work items = (row, time chunk) pairs, split evenly across programs
    fb = tl.program_id(0)
    k0 = tl.program_id(1) * BLOCK_K
    klo = tl.load(kb_ptr + 2 * fb)
    khi = tl.load(kb_ptr + 2 * fb + 1)
    if (k0 < khi) & (k0 + BLOCK_K > klo):
        f = fb * BLOCK_F + tl.arange(0, BLOCK_F)
        k = k0 + tl.arange(0, BLOCK_K)
        fmask = f < F
        ch = tl.load(cmap_ptr + f // C, mask=fmask, other=0).to(tl.int64)
        g_col = ch * (N_OUT * C) + f % C
        t = tl.arange(0, BLOCK_T)
        acc = tl.zeros([BLOCK_F, BLOCK_K], ACC)
        w_start = tl.program_id(2) * ITEMS_PER_PROG
        w_end = tl.minimum(w_start + ITEMS_PER_PROG, R * N_CHUNKS)
        for w in range(w_start, w_end):
            r = (w // N_CHUNKS).to(tl.int64)
            n0 = (w % N_CHUNKS) * BLOCK_T
            n = n0 + t
            g_ptrs = g_ptr + r * S_TOT * N_OUT * C + g_col[:, None] + (n * C)[None, :]
            pos = n[:, None] + OFF - DIL * k[None, :]
            x_ptrs = x_ptr + r * N_IN + pos
            lo = n0 + OFF - DIL * (k0 + BLOCK_K - 1)
            hi = n0 + BLOCK_T - 1 + OFF - DIL * k0
            if (n0 + BLOCK_T <= N_OUT) & (lo >= 0) & (hi < N_IN):
                gv = tl.load(g_ptrs, mask=fmask[:, None], other=0.0)
                xv = tl.load(x_ptrs)
            else:
                nm = n < N_OUT
                gv = tl.load(g_ptrs, mask=fmask[:, None] & nm[None, :], other=0.0)
                xv = tl.load(x_ptrs, mask=(pos >= 0) & (pos < N_IN) & nm[:, None], other=0.0)
            acc += _dot(gv.to(ACC), xv.to(ACC), ACC, PRECISION)
        tl.atomic_add(gh_ptr + f[:, None] * K + k[None, :], acc,
                      mask=fmask[:, None] & ((k >= klo) & (k < khi))[None, :])


@triton.jit
def _halfband_kernel(
    x_ptr, tap_ptr, y_ptr, N_IN, N_OUT, B, DIL,
    C_LP: tl.constexpr, ACC: tl.constexpr, ACCUMULATE: tl.constexpr, BLOCK: tl.constexpr,
):
    # y[n] = t[0] x[n+B] + sum_{t odd <= C_LP} t[t] (x[n+B-DIL*t] + x[n+B+DIL*t])
    # i.e. a symmetric half-band filter, whose even taps (except the centre) are zero.
    r = tl.program_id(0).to(tl.int64)
    n0 = tl.program_id(1) * BLOCK
    n = n0 + tl.arange(0, BLOCK)
    x_row = x_ptr + r * N_IN
    c = n + B
    acc = tl.load(tap_ptr).to(ACC) * tl.load(x_row + c, mask=(c >= 0) & (c < N_IN), other=0.0).to(ACC)
    if (n0 + B - DIL * C_LP >= 0) & (n0 + BLOCK - 1 + B + DIL * C_LP < N_IN):
        for t in tl.static_range(1, C_LP + 1, 2):
            pair = tl.load(x_row + c - DIL * t).to(ACC) + tl.load(x_row + c + DIL * t).to(ACC)
            acc += tl.load(tap_ptr + t).to(ACC) * pair
    else:
        for t in tl.static_range(1, C_LP + 1, 2):
            a = c - DIL * t
            b = c + DIL * t
            pair = tl.load(x_row + a, mask=(a >= 0) & (a < N_IN), other=0.0).to(ACC) \
                + tl.load(x_row + b, mask=(b >= 0) & (b < N_IN), other=0.0).to(ACC)
            acc += tl.load(tap_ptr + t).to(ACC) * pair
    nm = n < N_OUT
    y_row = y_ptr + r * N_OUT + n
    if ACCUMULATE:
        acc += tl.load(y_row, mask=nm, other=0.0).to(ACC)
    tl.store(y_row, acc.to(y_ptr.dtype.element_ty), mask=nm)


# --------------------------------------------------------------------------- launchers

FWD_BLOCK_N = 256
FWD_USE_DOT = True
FWD_DOT_BLOCK_N, FWD_DOT_BLOCK_K = 128, 32
BWD_BU, BWD_BV, BWD_COLS = 64, 32, 64
BWD_H_BLOCK_K, BWD_H_BLOCK_T = 32, 128


def _acc(dtype):
    return tl.float64 if dtype == torch.float64 else tl.float32


def pad_filters(hT: torch.Tensor, n_blocks: int, block_s: int, C: int) -> torch.Tensor:
    """(K, F) -> (K, n_blocks*block_s*C) zero padded, so that kernels never mask filter loads."""
    K, F = hT.shape
    width = n_blocks * block_s * C
    if width == F:
        return hT.contiguous()
    out = hT.new_zeros((K, width))
    out[:, :F] = hT
    return out


def forward(x, hT, kb, cmap, y, off: int, dil: int, block_s: int, precision=None):
    """x: (R, N_IN); hT: (K, n_blocks*block_s*C) padded filters; kb: (n_blocks, 2) int32;
    cmap: (S,) int32; y: (R, S_TOT, N_OUT, C) written in place."""
    R, n_in = x.shape
    _, S_TOT, n_out, C = y.shape
    S = cmap.numel()
    if R == 0 or n_out == 0 or S == 0:
        return y
    if FWD_USE_DOT and block_s * C >= 16:
        block_n = min(FWD_DOT_BLOCK_N, max(16, triton.next_power_of_2(n_out)))
        grid = (R, triton.cdiv(n_out, block_n), kb.shape[0])
        _fb_fwd_dot_kernel[grid](
            x, hT, kb, cmap, y, n_in, n_out, off, dil, S, S_TOT,
            C=C, ACC=_acc(x.dtype), PRECISION=precision or _dot_precision(x.dtype),
            BLOCK_N=block_n, BLOCK_S=block_s, BLOCK_K=FWD_DOT_BLOCK_K, num_warps=4,
        )
        return y
    block_n = min(FWD_BLOCK_N, triton.next_power_of_2(n_out))
    grid = (R, triton.cdiv(n_out, block_n), kb.shape[0])
    _fb_fwd_kernel[grid](
        x, hT, kb, cmap, y, n_in, n_out, off, dil, S, S_TOT,
        C=C, ACC=_acc(x.dtype), BLOCK_N=block_n, BLOCK_S=block_s, num_warps=4,
    )
    return y


def resolve_precision(dtype, compute_dtype=None) -> str:
    """Dot precision for inputs of ``dtype`` (the user's dtype, before any upcast)."""
    if (compute_dtype or dtype) == torch.float64:
        return "ieee"
    if dtype in (torch.bfloat16, torch.float16) or config.allow_bf16:
        return "bf16"
    return "tf32" if config.allow_tf32 else "ieee"


def _dot_precision(dtype) -> str:
    return resolve_precision(dtype)


def backward_x(g, h, kf, cmap, gx, off: int, dil: int, accumulate: bool, precision=None):
    """g: (R, S_TOT, N_OUT, C); h: (S*C, K) filters; kf: (S, 2) int32 tap range per scale;
    cmap: (S,) int32  ->  gx: (R, N_IN) written (or accumulated) in place."""
    R, S_TOT, n_out, C = g.shape
    n_in = gx.shape[1]
    S = cmap.numel()
    if R == 0 or n_in == 0:
        return gx
    if S == 0:
        if not accumulate:
            gx.zero_()
        return gx
    pb = min(triton.next_power_of_2(dil), 8)
    br = max(16 // pb, min(BWD_COLS // pb, triton.next_power_of_2(R)))
    bu = max(16, min(BWD_BU, triton.next_power_of_2(triton.cdiv(n_in, dil))))
    n_pb = triton.cdiv(dil, pb)
    n_ub = triton.cdiv(triton.cdiv(n_in, dil), bu)
    grid = (triton.cdiv(R, br), n_pb * n_ub)
    _fb_bwd_x_kernel[grid](
        g, h, kf, cmap, gx, R, n_in, n_out, off, dil, S, h.shape[1], S_TOT, n_pb,
        C=C, ACC=_acc(g.dtype), ACCUMULATE=accumulate,
        PRECISION=precision or _dot_precision(g.dtype),
        BU=bu, BV=BWD_BV, PB=pb, BR=br, num_warps=4,
    )
    return gx


def backward_h(x, g, kb, cmap, off: int, dil: int, K: int, block_s: int, precision=None):
    """Gradient w.r.t. the filters, returned as (S*C, K) (row f = s*C + c)."""
    R, n_in = x.shape
    _, S_TOT, n_out, C = g.shape
    S = cmap.numel()
    F = S * C
    gdtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    gh = torch.zeros((F, K), device=x.device, dtype=gdtype)
    if R == 0 or n_out == 0 or F == 0:
        return gh
    block_k, block_t = BWD_H_BLOCK_K, BWD_H_BLOCK_T
    block_f = block_s * C
    bf = max(16, block_f)  # tl.dot needs >= 16 rows
    if bf != block_f:  # merge the tap ranges of consecutive scale blocks
        ratio = bf // block_f
        kb = kb.view(-1, 2)
        pad = (-kb.shape[0]) % ratio
        if pad:
            kb = torch.cat([kb, kb[-1:].expand(pad, 2)])
        kb = kb.view(-1, ratio, 2)
        kb = torch.stack([kb[..., 0].amin(1), kb[..., 1].amax(1)], 1).contiguous()
    n_fb, n_kb = triton.cdiv(F, bf), triton.cdiv(K, block_k)
    n_chunks = triton.cdiv(n_out, block_t)
    items = R * n_chunks
    sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    progs = max(1, min(items, (8 * sm) // max(1, n_fb * n_kb)))
    per_prog = triton.cdiv(items, progs)
    grid = (n_fb, n_kb, triton.cdiv(items, per_prog))
    # parameter gradients: 3xTF32 is as accurate as "ieee" here and markedly faster
    prec = precision or _dot_precision(x.dtype)
    prec = "3xtf32" if prec == "ieee" and x.dtype != torch.float64 else prec
    _fb_bwd_h_kernel[grid](
        x, g, kb, cmap, gh, R, n_in, n_out, off, dil, F, K, S_TOT, n_chunks, per_prog,
        C=C, ACC=_acc(x.dtype), PRECISION=prec,
        BLOCK_F=bf, BLOCK_K=block_k, BLOCK_T=block_t, num_warps=4,
    )
    return gh


def halfband(x, taps, y, off: int, dil: int, transpose: bool = False, accumulate: bool = False):
    """Symmetric half-band filter ``h`` (2c+1 taps, ``taps[t] = h[c + t]``, even t > 0 zero).

    Forward:   y[n] = sum_k h[k] x[n + off - dil*k]
    Transpose: y[m] = sum_k h[k] x[m - off + dil*k]   (its adjoint; same kernel by symmetry)
    x: (R, N_IN), y: (R, N_OUT) written (or accumulated) in place.
    """
    R, n_in = x.shape
    n_out = y.shape[1]
    c = taps.numel() - 1
    if R == 0 or n_out == 0:
        return y
    b = (dil * c - off) if transpose else (off - dil * c)
    block = min(1024, triton.next_power_of_2(n_out))
    grid = (R, triton.cdiv(n_out, block))
    _halfband_kernel[grid](
        x, taps, y, n_in, n_out, b, dil,
        C_LP=c, ACC=_acc(x.dtype), ACCUMULATE=accumulate, BLOCK=block, num_warps=4,
    )
    return y
