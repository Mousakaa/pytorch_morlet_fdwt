"""Fused "spectral softmax" kernels.

For every row r (one signal) and position n::

    a_f[r, n]  = |W_f x_r|(n) / tau                       (Morlet magnitude, frequency f)
    out[r, n]  = sum_f softmax_f(a)[r, n] * V_f[r, n]

``V`` is given by strides, so it can be a per-row constant (``V[c, f]`` with
``c = row % NCH``, used by ``MorletAttention``) or a per-position tensor
(``V[g, f, n]`` with ``g = row // NCH``, used by ``MorletChannelMixAttention``).

Tensor-core formulation
-----------------------
With ``n = p0 + p + DIL*u`` (polyphase-blocked output tile) and ``OFF = DIL*q + rho``,
the Morlet coefficients of one frequency are a banded-Toeplitz product::

    W_f[u, (p, r)] = sum_v T_f[u, v] * X[v, (p, r)],   T_f[u, v] = h_f[u + q - v],
                                                       X[v, (p, r)] = x[r, p0 + p + rho + DIL*v]

``T_f`` does not depend on the output tile, so its band is materialised once per octave as
small dense tiles (:func:`band_tiles`), already in the dot's operand dtype; the rows (and
phases) of a tile form the N dimension of ``tl.dot``: no padding waste
however few frequencies an octave holds, and dilated filters cost the same as undilated
ones.  A program loops over the frequencies of the octave one at a time, folds each
into a flash-attention style online softmax ``(max, denominator, numerator)`` held in
registers, and carries that state from one octave launch to the next through three
(R, N) buffers: the scalogram is never written.

Backward (``GRAD=True``) recomputes one octave's coefficients, rebuilds the softmax
probabilities from the saved max and denominator, writes ``dL/dW`` for that octave
(consumed by the grad-x / grad-h kernels) and accumulates ``dL/dV`` with atomics::

    out = sum_f p_f V_f,  p = softmax(a)  =>  dL/da_f = gy * p_f * (V_f - out),
                                              dL/dW_f = dL/da_f / tau * W_f / |W_f|
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .conv import _acc, _dot

BU, BV, COLS = 32, 16, 64  # tuned on RTX 4080 SUPER (fp32 and bf16 alike)
NUM_WARPS, NUM_STAGES = 4, 3


@triton.jit
def _spectral_softmax_kernel(
    x_ptr, tb_ptr, kf_ptr, gidx_ptr, v_ptr,
    m_ptr, l_ptr, num_ptr,
    out_ptr, gy_ptr, gw_ptr, gv_ptr,
    R, N_IN, N_OUT, OFF, DIL, S, NCHUNK, NCH, ROW0, INV_TAU, N_PB,
    SV_G, SV_C, SV_F, SV_N,
    ACC: tl.constexpr, GRAD: tl.constexpr, V_PER_POS: tl.constexpr, PRECISION: tl.constexpr,
    BU: tl.constexpr, BV: tl.constexpr, PB: tl.constexpr, BR: tl.constexpr,
):
    tid = tl.program_id(1)
    p0 = (tid % N_PB) * PB
    u0 = (tid // N_PB) * BU
    u = tl.arange(0, BU)
    pr = tl.arange(0, PB * BR)
    p = pr % PB
    r = tl.program_id(0) * BR + pr // PB
    col_ok = (r < R) & (p0 + p < DIL)
    r64 = r.to(tl.int64)
    n = p0 + p[None, :] + DIL * (u0 + u)[:, None]           # [BU, COLS] output positions
    nm = (n < N_OUT) & col_ok[None, :]
    q = OFF // DIL
    rho = OFF - q * DIL
    x_col = x_ptr + r64 * N_IN + p0 + p + rho                # [COLS]
    xpos0 = p0 + p + rho
    st = r64[None, :] * N_OUT + n
    rg = r + ROW0
    v_col = (rg // NCH) * SV_G + (rg % NCH) * SV_C            # [COLS]

    m = tl.load(m_ptr + st, mask=nm, other=0.0).to(ACC)
    l = tl.load(l_ptr + st, mask=nm, other=1.0).to(ACC)
    if GRAD:
        o = tl.load(out_ptr + st, mask=nm, other=0.0).to(ACC)
        gyv = tl.load(gy_ptr + st, mask=nm, other=0.0).to(ACC)
    else:
        num = tl.load(num_ptr + st, mask=nm, other=0.0).to(ACC)

    # columns are all real rows/phases: interior chunks need no bounds checks at all
    full_cols = (tl.program_id(0) * BR + BR <= R) & (p0 + PB <= DIL)
    t_off = u[:, None] * BV + tl.arange(0, BV)[None, :]
    for s in range(0, S):
        klo = tl.load(kf_ptr + 2 * s)
        khi = tl.load(kf_ptr + 2 * s + 1)
        gi = tl.load(gidx_ptr + s)
        t_re = tb_ptr + (s * 2 * NCHUNK) * (BU * BV) + t_off
        t_im = t_re + NCHUNK * (BU * BV)
        re = tl.zeros([BU, PB * BR], ACC)
        im = tl.zeros([BU, PB * BR], ACC)
        v_first = u0 + q - khi + 1
        for ci in range(0, (BU + khi - klo - 1 + BV - 1) // BV):
            v0 = v_first + ci * BV
            v = v0 + tl.arange(0, BV)
            x_ptrs = x_col[None, :] + DIL * v[:, None]
            lo = p0 + rho + DIL * v0
            hi = p0 + PB - 1 + rho + DIL * (v0 + BV - 1)
            if full_cols & (lo >= 0) & (hi < N_IN):
                X = tl.load(x_ptrs).to(ACC)
            else:
                pos = xpos0[None, :] + DIL * v[:, None]
                X = tl.load(x_ptrs, mask=(pos >= 0) & (pos < N_IN) & col_ok[None, :],
                            other=0.0).to(ACC)
            re += _dot(tl.load(t_re + ci * (BU * BV)), X, ACC, PRECISION)
            im += _dot(tl.load(t_im + ci * (BU * BV)), X, ACC, PRECISION)

        mag = tl.sqrt(re * re + im * im)
        a = mag * INV_TAU
        if V_PER_POS:
            v_off = v_col[None, :] + gi * SV_F + n * SV_N
            val = tl.load(v_ptr + v_off, mask=nm, other=0.0).to(ACC)
        else:
            val = tl.load(v_ptr + v_col + gi * SV_F, mask=col_ok, other=0.0).to(ACC)[None, :]

        if GRAD:
            pf = tl.where(nm, tl.exp(a - m) / l, 0.0)
            gp = gyv * pf
            if V_PER_POS:
                tl.atomic_add(gv_ptr + v_off, gp, mask=nm)
            else:
                tl.atomic_add(gv_ptr + v_col + gi * SV_F, tl.sum(gp, axis=0), mask=col_ok)
            inv = tl.where(mag > 0, 1.0 / tl.where(mag > 0, mag, 1.0), 0.0)
            coef = gp * (val - o) * INV_TAU * inv
            g_off = r64[None, :] * (S * N_OUT * 2) + s * (N_OUT * 2) + n * 2
            tl.store(gw_ptr + g_off, (coef * re).to(gw_ptr.dtype.element_ty), mask=nm)
            tl.store(gw_ptr + g_off + 1, (coef * im).to(gw_ptr.dtype.element_ty), mask=nm)
        else:
            m_new = tl.maximum(m, a)
            alpha = tl.exp(m - m_new)
            e = tl.exp(a - m_new)
            l = l * alpha + e
            num = num * alpha + e * val
            m = m_new

    if not GRAD:
        tl.store(m_ptr + st, m.to(m_ptr.dtype.element_ty), mask=nm)
        tl.store(l_ptr + st, l.to(l_ptr.dtype.element_ty), mask=nm)
        tl.store(num_ptr + st, num.to(num_ptr.dtype.element_ty), mask=nm)


def band_tiles(h, kf, n_out: int, dil: int, precision: str):
    """Dense banded-Toeplitz tiles of one octave's filters, laid out for the kernel.

    ``T[s, part, ci, u, vv] = h[2*s + part, u - ci*BV - vv + khi_s - 1]`` (zero outside the
    tap range ``[klo_s, khi_s)``).  They do not depend on the output tile, so they are built
    once per octave instead of being gathered element by element inside the kernel.
    Returns ``(tiles, bu, n_chunks)``.
    """
    S, K = kf.shape[0], h.shape[1]
    bu = max(16, min(BU, triton.next_power_of_2(triton.cdiv(n_out, dil))))
    n_chunks = triton.cdiv(bu + K - 1, BV)
    dev = h.device
    ci = torch.arange(n_chunks, device=dev)[:, None, None]
    u = torch.arange(bu, device=dev)[None, :, None]
    vv = torch.arange(BV, device=dev)[None, None, :]
    klo = kf[:, 0].long()[:, None, None, None]
    khi = kf[:, 1].long()[:, None, None, None]
    kk = (u - ci * BV - vv)[None] + khi - 1                         # (S, n_chunks, bu, BV)
    valid = (kk >= klo) & (kk < khi)
    hh = h.view(S, 2, K)
    tiles = hh[torch.arange(S, device=dev)[:, None, None, None, None],
               torch.arange(2, device=dev)[None, :, None, None, None],
               kk.clamp(0, K - 1)[:, None]] * valid[:, None]
    dt = torch.bfloat16 if precision == "bf16" else h.dtype
    return tiles.to(dt).contiguous(), bu, n_chunks


def _launch(x, tiles, kf, gidx, v, v_strides, v_per_pos, nch, row0, inv_tau, off, dil, precision,
            m, l, num, out, gy, gw, gv, grad):
    R, n_in = x.shape
    n_out = m.shape[1]
    S = gidx.numel()
    if R == 0 or n_out == 0 or S == 0:
        return
    tb, bu, n_chunks = tiles
    pb = min(triton.next_power_of_2(dil), 8)
    br = max(16 // pb, min(COLS // pb, triton.next_power_of_2(R)))
    n_pb = triton.cdiv(dil, pb)
    n_ub = triton.cdiv(triton.cdiv(n_out, dil), bu)
    grid = (triton.cdiv(R, br), n_pb * n_ub)
    _spectral_softmax_kernel[grid](
        x, tb, kf, gidx, v, m, l, num, out, gy, gw, gv,
        R, n_in, n_out, off, dil, S, n_chunks, nch, row0, inv_tau, n_pb, *v_strides,
        ACC=_acc(x.dtype), GRAD=grad, V_PER_POS=v_per_pos, PRECISION=precision,
        BU=bu, BV=BV, PB=pb, BR=br, num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )


def forward_stage(x, tiles, kf, gidx, v, v_strides, v_per_pos, nch, inv_tau, off, dil, precision,
                  m, l, num):
    """Fold one octave into the online-softmax state (m, l, num), each (R, N_OUT).

    ``tiles`` comes from :func:`band_tiles`, ``kf`` holds the (S, 2) tap ranges and
    ``gidx`` the global frequency indices of the octave.
    """
    _launch(x, tiles, kf, gidx, v, v_strides, v_per_pos, nch, 0, inv_tau, off, dil, precision,
            m, l, num, m, m, m, m, grad=False)


def backward_stage(x, tiles, kf, gidx, v, v_strides, v_per_pos, nch, row0, inv_tau, off, dil,
                   precision, m, l, out, gy, gw, gv):
    """Write gw (R, S, N_OUT, 2) = dL/dW for one octave and accumulate gv (= dL/dV)."""
    _launch(x, tiles, kf, gidx, v, v_strides, v_per_pos, nch, row0, inv_tau, off, dil, precision,
            m, l, m, out, gy, gw, gv, grad=True)
