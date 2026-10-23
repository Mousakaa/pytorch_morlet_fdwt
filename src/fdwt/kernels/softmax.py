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

Backward, short filters (:func:`backward_stage_sweep`, the default): one program sweeps
its rows' output tiles in order and applies the adjoint as two banded blocks, carrying the
part that lands in the next input tile in registers; dL/dx, dL/dV and the filter-tangent
gradients come out of one pass, without atomics on dL/dx and without dL/dW in memory.

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
BWD_NUM_WARPS = 4  # fused backward (adjoint + tangents)
USE_WINDOW, WIN_MAX, WIN_STAGES = True, 64, 2  # one-window tiles for short filters (see band_tiles)
WIN_BWD_COLS = 32  # narrower tiles for the fused window backward (register pressure)
# sweep backward (backward_stage_sweep), measured on an RTX 4080 SUPER (REVE EEG shape):
# 16-row output tiles for filters of at most 2 * 16 + 1 taps (the adjoint then reaches one or
# two following input tiles: NCARRY), each with its (columns, warps); 32-row tiles otherwise.
SWEEP_BU, SWEEP_COLS, SWEEP_NUM_WARPS, SWEEP_STAGES = 32, 64, 4, 2
SWEEP_SMALL_BU = 16
SWEEP_SMALL_CFG = {1: (32, 2), 2: (64, 4)}  # NCARRY -> (columns, warps)
SWEEP_SMALL_MAX_CARRY = 2
SWEEP_UNI, SWEEP_UNI_MAXCH = True, 3  # shared-alignment path, at most 3 input chunks per tile
SWEEP_MAXNREG = 0  # register cap for the sweep kernel (0: compiler's choice)
SWEEP_PREFETCH = False  # load the next tile's state / input chunks early (RTX 4080: +-1% morlet, +10% channel mix)
SWEEP_RELAX = True  # relaxed dL/dV atomics (no memory fence)
SWEEP_ML_PRE = True  # use m + log l precomputed once per backward (``ml`` argument)
SWEEP_FAST = True  # rsqrt for |W| and 1/|W|, exp2 (fp32 only)
SWEEP_MIN_PROGRAMS_PER_SM, SWEEP_MIN_SEG_TILES = 32, 4  # split rows into segments (0: one segment per row)


def sweep_tile_rows(K: int, n_out: int, dil: int) -> int:
    """Output-tile rows of the sweep backward for an octave with ``K``-tap filters."""
    if SWEEP_SMALL_BU and K <= SWEEP_SMALL_MAX_CARRY * SWEEP_SMALL_BU + 1:
        return SWEEP_SMALL_BU
    return max(16, min(SWEEP_BU, triton.next_power_of_2(triton.cdiv(n_out, dil))))


def sweep_supported(K: int, n_out: int, dil: int) -> bool:
    return K <= 2 * sweep_tile_rows(K, n_out, dil) + 1


def uniform_kf(kf, K: int):
    """Every frequency on the octave's full tap range [0, K) (filters are zero outside their own
    support): tiles with a common alignment, so all frequencies read the same input chunks."""
    out = kf.clone()
    out[:, 0] = 0
    out[:, 1] = K
    return out


def sweep_tiles(h, kf, n_out: int, dil: int, precision: str, tangents=()):
    """Tiles for :func:`backward_stage_sweep`: ``(tiles, kf, tangent tiles or None)``, with the
    common-alignment tap ranges when an output tile's window fits ``SWEEP_UNI_MAXCH`` chunks."""
    K = h.shape[1]
    bu = sweep_tile_rows(K, n_out, dil)
    if SWEEP_UNI and bu + K - 1 <= SWEEP_UNI_MAXCH * BV:
        kf = uniform_kf(kf, K)
    tiles = band_tiles(h, kf, n_out, dil, precision, window=False, bu=bu)
    tt = (torch.stack([band_tiles(t, kf, n_out, dil, precision, window=False, bu=bu)[0] for t in tangents])
          if len(tangents) else None)
    return tiles, kf, tt


@triton.jit
def _spectral_softmax_kernel(
    x_ptr, tb_ptr, kf_ptr, gidx_ptr, v_ptr,
    m_ptr, l_ptr, num_ptr,
    out_ptr, gy_ptr, gw_ptr, gv_ptr, gx_ptr, tt_ptr, gth_ptr,
    R, N_IN, N_OUT, OFF, DIL, S, NCHUNK, NCH, ROW0, INV_TAU, N_PB,
    SV_G, SV_C, SV_F, SV_N, S_TOT, KT,
    ACC: tl.constexpr, GRAD: tl.constexpr, V_PER_POS: tl.constexpr, PRECISION: tl.constexpr,
    BU: tl.constexpr, BV: tl.constexpr, PB: tl.constexpr, BR: tl.constexpr,
    WRITE_GW: tl.constexpr, FUSE_GX: tl.constexpr, NT: tl.constexpr, V_GROUPED: tl.constexpr,
    WIN: tl.constexpr,
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
    x_off = r64 * N_IN + p0 + p + rho                        # [COLS]
    x_col = x_ptr + x_off
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
    tile = BU * BV
    prog = tl.program_id(0) * tl.num_programs(1) + tid      # theta partials: one slot per program
    if WIN > 0:
        # short filters: the whole input window of the tile (BU + KT - 1 <= WIN samples per
        # column) is loaded once and shared by every frequency; tiles are (S, 2, BU, WIN)
        # with a common alignment, so each frequency is one dot and the adjoint of all the
        # frequencies accumulates in one register window.
        w_first = u0 + q - KT + 1
        vw = w_first + tl.arange(0, WIN)
        posw = xpos0[None, :] + DIL * vw[:, None]
        pmw = (posw >= 0) & (posw < N_IN) & col_ok[None, :]
        Xw = tl.load(x_col[None, :] + DIL * vw[:, None], mask=pmw, other=0.0).to(ACC)
        tw_off = u[:, None] * WIN + tl.arange(0, WIN)[None, :]
        if GRAD and FUSE_GX:
            gXw = tl.zeros([WIN, PB * BR], ACC)
    for s in range(0, S):
        klo = tl.load(kf_ptr + 2 * s)
        khi = tl.load(kf_ptr + 2 * s + 1)
        gi = tl.load(gidx_ptr + s)
        if WIN > 0:
            tw_re = tb_ptr + (s * 2) * (BU * WIN) + tw_off
            re = _dot(tl.load(tw_re), Xw, ACC, PRECISION)
            im = _dot(tl.load(tw_re + BU * WIN), Xw, ACC, PRECISION)
            n_ci = 0
        else:
            re = tl.zeros([BU, PB * BR], ACC)
            im = tl.zeros([BU, PB * BR], ACC)
            n_ci = (BU + khi - klo - 1 + BV - 1) // BV
        t_re = tb_ptr + (s * 2 * NCHUNK) * tile + t_off
        t_im = t_re + NCHUNK * tile
        v_first = u0 + q - khi + 1
        for ci in range(0, n_ci):
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
            re += _dot(tl.load(t_re + ci * tile), X, ACC, PRECISION)
            im += _dot(tl.load(t_im + ci * tile), X, ACC, PRECISION)

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
                if V_GROUPED:
                    # every row of this tile shares one V row: reduce over the rows first
                    # (NCH-fold fewer atomics on the same addresses)
                    gsum = tl.sum(tl.reshape(gp, [BU, BR, PB]), axis=1)          # [BU, PB]
                    ph = tl.arange(0, PB)
                    n2 = p0 + ph[None, :] + DIL * (u0 + u)[:, None]
                    g0 = (tl.program_id(0) * BR + ROW0) // NCH
                    m2 = (n2 < N_OUT) & (p0 + ph < DIL)[None, :]
                    tl.atomic_add(gv_ptr + g0 * SV_G + gi * SV_F + n2 * SV_N, gsum, mask=m2)
                else:
                    tl.atomic_add(gv_ptr + v_off, gp, mask=nm)
            else:
                tl.atomic_add(gv_ptr + v_col + gi * SV_F, tl.sum(gp, axis=0), mask=col_ok)
            inv = tl.where(mag > 0, 1.0 / tl.where(mag > 0, mag, 1.0), 0.0)
            coef = gp * (val - o) * INV_TAU * inv
            g_re = coef * re                                      # dL/dRe W, dL/dIm W
            g_im = coef * im
            if WRITE_GW:
                g_off = r64[None, :] * (S * N_OUT * 2) + s * (N_OUT * 2) + n * 2
                tl.store(gw_ptr + g_off, g_re.to(gw_ptr.dtype.element_ty), mask=nm)
                tl.store(gw_ptr + g_off + 1, g_im.to(gw_ptr.dtype.element_ty), mask=nm)
            if WIN > 0:
                if FUSE_GX:
                    gXw += _dot(tl.trans(tl.load(tw_re)), g_re, ACC, PRECISION)
                    gXw += _dot(tl.trans(tl.load(tw_re + BU * WIN)), g_im, ACC, PRECISION)
                # dL/dtheta = sum dL/dW * (dT X): the tangent tiles in their natural layout
                if NT > 0:
                    d_re = tt_ptr + (s * 2) * (BU * WIN) + tw_off
                    A = g_re * _dot(tl.load(d_re), Xw, ACC, PRECISION)
                    A += g_im * _dot(tl.load(d_re + BU * WIN), Xw, ACC, PRECISION)
                    tl.store(gth_ptr + s * S_TOT + prog, tl.sum(tl.sum(A, axis=1), axis=0))
                if NT > 1:
                    d_re2 = d_re + S * 2 * (BU * WIN)
                    A = g_re * _dot(tl.load(d_re2), Xw, ACC, PRECISION)
                    A += g_im * _dot(tl.load(d_re2 + BU * WIN), Xw, ACC, PRECISION)
                    tl.store(gth_ptr + (S + s) * S_TOT + prog, tl.sum(tl.sum(A, axis=1), axis=0))
            elif FUSE_GX or NT > 0:
                # adjoint of the banded-Toeplitz product, chunk by chunk:
                #   dL/dX[v] = T_re^T dL/dRe W + T_im^T dL/dIm W     (scattered into gx)
                #   dL/dtheta = sum_v X[v] * (dT_re^T dL/dRe W + dT_im^T dL/dIm W)
                # with dT the tiles of the filter tangents dH/dtheta: the filter gradient is
                # never formed, and neither is dL/dW.
                th0 = tl.zeros([BV, PB * BR], ACC)
                th1 = tl.zeros([BV, PB * BR], ACC)
                for ci in range(0, n_ci):
                    v0 = v_first + ci * BV
                    v = v0 + tl.arange(0, BV)
                    pos = xpos0[None, :] + DIL * v[:, None]
                    pm = (pos >= 0) & (pos < N_IN) & col_ok[None, :]
                    if FUSE_GX:
                        gX = _dot(tl.trans(tl.load(t_re + ci * tile)), g_re, ACC, PRECISION)
                        gX += _dot(tl.trans(tl.load(t_im + ci * tile)), g_im, ACC, PRECISION)
                        tl.atomic_add(gx_ptr + x_off[None, :] + DIL * v[:, None], gX, mask=pm)
                    if NT > 0:
                        X = tl.load(x_col[None, :] + DIL * v[:, None], mask=pm, other=0.0).to(ACC)
                        d_re = tt_ptr + (s * 2 * NCHUNK + ci) * tile + t_off
                        A = _dot(tl.trans(tl.load(d_re)), g_re, ACC, PRECISION)
                        A += _dot(tl.trans(tl.load(d_re + NCHUNK * tile)), g_im, ACC, PRECISION)
                        th0 += A * X
                        if NT > 1:
                            d_re2 = d_re + S * 2 * NCHUNK * tile
                            A = _dot(tl.trans(tl.load(d_re2)), g_re, ACC, PRECISION)
                            A += _dot(tl.trans(tl.load(d_re2 + NCHUNK * tile)), g_im, ACC, PRECISION)
                            th1 += A * X
                if NT > 0:
                    tl.store(gth_ptr + s * S_TOT + prog, tl.sum(tl.sum(th0, axis=1), axis=0))
                if NT > 1:
                    tl.store(gth_ptr + (S + s) * S_TOT + prog, tl.sum(tl.sum(th1, axis=1), axis=0))
        else:
            m_new = tl.maximum(m, a)
            alpha = tl.exp(m - m_new)
            e = tl.exp(a - m_new)
            l = l * alpha + e
            num = num * alpha + e * val
            m = m_new

    if GRAD and FUSE_GX and WIN > 0:
        tl.atomic_add(gx_ptr + x_off[None, :] + DIL * vw[:, None], gXw, mask=pmw)
    if not GRAD:
        tl.store(m_ptr + st, m.to(m_ptr.dtype.element_ty), mask=nm)
        tl.store(l_ptr + st, l.to(l_ptr.dtype.element_ty), mask=nm)
        tl.store(num_ptr + st, num.to(num_ptr.dtype.element_ty), mask=nm)


def band_tiles(h, kf, n_out: int, dil: int, precision: str, window: bool = True, bu=None):
    """Dense banded-Toeplitz tiles of one octave's filters, laid out for the kernel.

    Short filters (``BU + K - 1 <= WIN_MAX``): ``T[s, part, u, w] = h[2*s + part, u - w + K - 1]``
    (zero outside ``[0, K)``; the filters are zero outside their own support), one
    ``(BU, WIN)`` tile per frequency with a common window.  Otherwise chunks of ``BV``
    columns, ``T[s, part, ci, u, vv] = h[2*s + part, u - ci*BV - vv + khi_s - 1]`` (zero
    outside the tap range ``[klo_s, khi_s)``).  They do not depend on the output tile, so
    they are built once per octave instead of being gathered element by element inside the
    kernel.  Returns ``(tiles, bu, n_chunks, 0)`` (chunked) or ``(tiles, bu, K, win)``.
    ``window=False`` forces the chunked layout (the fused backward with filter tangents is
    faster chunked: fewer live registers).
    """
    S, K = kf.shape[0], h.shape[1]
    if bu is None:
        bu = max(16, min(BU, triton.next_power_of_2(triton.cdiv(n_out, dil))))
    dev = h.device
    dt = torch.bfloat16 if precision == "bf16" else h.dtype
    hh = h.view(S, 2, K)
    win = max(16, triton.next_power_of_2(bu + K - 1))
    if window and USE_WINDOW and win <= WIN_MAX:
        u = torch.arange(bu, device=dev)[:, None]
        w = torch.arange(win, device=dev)[None, :]
        kk = u - w + K - 1                                               # (bu, win)
        valid = (kk >= 0) & (kk < K)
        tiles = hh[:, :, kk.clamp(0, K - 1)] * valid
        return tiles.to(dt).contiguous(), bu, K, win
    n_chunks = triton.cdiv(bu + K - 1, BV)
    ci = torch.arange(n_chunks, device=dev)[:, None, None]
    u = torch.arange(bu, device=dev)[None, :, None]
    vv = torch.arange(BV, device=dev)[None, None, :]
    klo = kf[:, 0].long()[:, None, None, None]
    khi = kf[:, 1].long()[:, None, None, None]
    kk = (u - ci * BV - vv)[None] + khi - 1                         # (S, n_chunks, bu, BV)
    valid = (kk >= klo) & (kk < khi)
    tiles = hh[torch.arange(S, device=dev)[:, None, None, None, None],
               torch.arange(2, device=dev)[None, :, None, None, None],
               kk.clamp(0, K - 1)[:, None]] * valid[:, None]
    return tiles.to(dt).contiguous(), bu, n_chunks, 0


def _launch(x, tiles, kf, gidx, v, v_strides, v_per_pos, nch, row0, inv_tau, off, dil, precision,
            m, l, num, out, gy, gw, gv, grad, gx=None, ttiles=None, gth=None, nt=0, write_gw=True,
            num_warps=None):
    tb, bu, n_chunks, win = tiles
    K, n_chunks = (n_chunks, 0) if win else (0, n_chunks)
    fused = grad and (gx is not None or nt > 0)
    cols = WIN_BWD_COLS if (win and fused) else COLS
    R, n_in = x.shape
    n_out = m.shape[1]
    S = gidx.numel()
    if R == 0 or n_out == 0 or S == 0:
        return
    pb = min(triton.next_power_of_2(dil), 8)
    br = max(16 // pb, min(cols // pb, triton.next_power_of_2(R)))
    n_pb = triton.cdiv(dil, pb)
    n_ub = triton.cdiv(triton.cdiv(n_out, dil), bu)
    grid = (triton.cdiv(R, br), n_pb * n_ub)
    # V rows shared by NCH consecutive signal rows: a tile's rows all fall in one V row
    v_grouped = bool(v_per_pos and grad and nch % br == 0 and row0 % br == 0)
    fuse_gx = gx is not None
    n_prog = grid[0] * grid[1]
    # dL/dtheta: per-program partial sums (same-address atomics from every program serialise)
    part = torch.empty((nt, S, n_prog), device=x.device, dtype=gth.dtype) if nt else None
    _spectral_softmax_kernel[grid](
        x, tb, kf, gidx, v, m, l, num, out, gy, gw, gv,
        gx if fuse_gx else x, ttiles if nt else tb, part if nt else m,
        R, n_in, n_out, off, dil, S, n_chunks, nch, row0, inv_tau, n_pb, *v_strides,
        n_prog, K,
        ACC=_acc(x.dtype), GRAD=grad, V_PER_POS=v_per_pos, PRECISION=precision,
        BU=bu, BV=BV, PB=pb, BR=br, WRITE_GW=bool(grad and write_gw), FUSE_GX=fuse_gx, NT=nt,
        V_GROUPED=v_grouped, WIN=win, num_warps=num_warps or NUM_WARPS,
        num_stages=(1 if x.dtype == torch.float64 else WIN_STAGES) if win else NUM_STAGES,
    )
    if nt:
        gth.index_add_(1, gidx.long(), part.sum(-1))


def forward_stage(x, tiles, kf, gidx, v, v_strides, v_per_pos, nch, inv_tau, off, dil, precision,
                  m, l, num):
    """Fold one octave into the online-softmax state (m, l, num), each (R, N_OUT).

    ``tiles`` comes from :func:`band_tiles`, ``kf`` holds the (S, 2) tap ranges and
    ``gidx`` the global frequency indices of the octave.
    """
    _launch(x, tiles, kf, gidx, v, v_strides, v_per_pos, nch, 0, inv_tau, off, dil, precision,
            m, l, num, m, m, m, m, grad=False)


def backward_stage(x, tiles, kf, gidx, v, v_strides, v_per_pos, nch, row0, inv_tau, off, dil,
                   precision, m, l, out, gy, gw, gv, gx=None, ttiles=None, gth=None):
    """Backward of one octave: accumulates gv (= dL/dV) and, as requested,

    * writes ``gw`` (R, S, N_OUT, 2) = dL/dW (for the grad-x / grad-h kernels), if not None;
    * accumulates dL/dx into ``gx`` (R, N_IN) directly (fused adjoint, no ``gw``);
    * accumulates ``gth[t, f] += dL/dtheta_t`` for the filter tangents ``ttiles``
      (``(T, S, 2, ...)`` tiles of dH/dtheta_t, ``gth`` of shape ``(T, S_TOT)``), from
      per-program partial sums.
    """
    nt = 0 if ttiles is None else ttiles.shape[0]
    _launch(x, tiles, kf, gidx, v, v_strides, v_per_pos, nch, row0, inv_tau, off, dil, precision,
            m, l, m, out, gy, gw if gw is not None else m, gv, grad=True, gx=gx, ttiles=ttiles,
            gth=gth, nt=nt, write_gw=gw is not None,
            num_warps=BWD_NUM_WARPS if (gx is not None or nt) else None)


# --------------------------------------------------------------------------- sweep backward


@triton.jit
def _sweep_tile_loads(m_ptr, l_ptr, out_ptr, gy_ptr, x_col, xpos0, col_ok, lr, p, p0, u, u0, q, KT, DIL, N_IN,
                      N_OUT, ACC: tl.constexpr, BV: tl.constexpr, NCU: tl.constexpr, ML_PRE: tl.constexpr):
    """Softmax state (log-denominator m + log l, out, dL/dout) and the NCU shared input chunks of
    the output tile starting at u0."""
    n = p0 + p[None, :] + DIL * (u0 + u)[:, None]
    nm = (n < N_OUT) & col_ok[None, :]
    st = lr[None, :] * N_OUT + n
    if ML_PRE:
        ml = tl.load(m_ptr + st, mask=nm, other=0.0).to(ACC)
    else:
        ml = tl.load(m_ptr + st, mask=nm, other=0.0).to(ACC) + tl.log(tl.load(l_ptr + st, mask=nm, other=1.0).to(ACC))
    o = tl.load(out_ptr + st, mask=nm, other=0.0).to(ACC)
    gyv = tl.load(gy_ptr + st, mask=nm, other=0.0).to(ACC)
    vA = u0 + q - KT + 1 + tl.arange(0, BV)
    posA = xpos0[None, :] + DIL * vA[:, None]
    xA = x_col[None, :] + DIL * vA[:, None]
    X0 = tl.load(xA, mask=(posA >= 0) & (posA < N_IN) & col_ok[None, :], other=0.0).to(ACC)
    posA += DIL * BV
    X1 = tl.load(xA + DIL * BV, mask=(posA >= 0) & (posA < N_IN) & col_ok[None, :], other=0.0).to(ACC)
    if NCU > 2:
        posA += DIL * BV
        X2 = tl.load(xA + 2 * DIL * BV, mask=(posA >= 0) & (posA < N_IN) & col_ok[None, :], other=0.0).to(ACC)
    else:
        X2 = X0
    return ml, o, gyv, X0, X1, X2


@triton.jit
def _conv_u(t_ptr, tile, X0, X1, X2, NCU: tl.constexpr, ACC: tl.constexpr, PRECISION: tl.constexpr):
    """sum_ci T[ci] @ X_ci over the NCU (2 or 3) shared input chunks (tiles at t_ptr + ci * tile)."""
    acc = _dot(tl.load(t_ptr), X0, ACC, PRECISION)
    acc += _dot(tl.load(t_ptr + tile), X1, ACC, PRECISION)
    if NCU > 2:
        acc += _dot(tl.load(t_ptr + 2 * tile), X2, ACC, PRECISION)
    return acc


@triton.jit
def _spectral_softmax_bwd_sweep_kernel(
    x_ptr, tb_ptr, ta_ptr, tt_ptr, kf_ptr, gidx_ptr, v_ptr,
    m_ptr, l_ptr, out_ptr, gy_ptr, gv_ptr, gx_ptr, part_ptr,
    R, N_IN, N_OUT, OFF, DIL, NCHUNK, NCH, INV_TAU, N_PB, N_UT, KT, P_TOT, T_SEG,
    SV_G, SV_C, SV_F, SV_N,
    ACC: tl.constexpr, V_PER_POS: tl.constexpr, PRECISION: tl.constexpr,
    BU: tl.constexpr, BV: tl.constexpr, PB: tl.constexpr, BR: tl.constexpr,
    NT: tl.constexpr, V_GROUPED: tl.constexpr, ACCUM: tl.constexpr, NCU: tl.constexpr,
    NCARRY: tl.constexpr, PREFETCH: tl.constexpr, RELAX: tl.constexpr, ML_PRE: tl.constexpr,
    FAST: tl.constexpr, S: tl.constexpr,
):
    # NCU > 0: every frequency's tiles share the octave's alignment (tap range [0, KT)) and span
    # NCU (2 or 3) BV chunks: the input chunks of an output tile are loaded once for all the
    # frequencies, and the filter tangents are convolved in the same pass
    # (dL/dtheta = sum dL/dW * (dT X), no transposed tiles, no reload).
    pid0 = tl.program_id(0)
    tid = tl.program_id(1)
    # the output tiles of a row are split into segments of T_SEG tiles (grid axis 2), so that short
    # rows still fill the GPU; the first NCARRY input tiles of a segment also receive the previous
    # segment's carries: both add them atomically, every other dL/dx tile is written once
    seg = tl.program_id(2)
    t_start = seg * T_SEG
    t_end = tl.minimum(N_UT, t_start + T_SEG)
    prog = pid0 * N_PB + tid
    p0 = tid * PB
    u = tl.arange(0, BU)
    pr = tl.arange(0, PB * BR)
    p = pr % PB
    r = pid0 * BR + pr // PB
    col_ok = (r < R) & (p0 + p < DIL)
    q = OFF // DIL
    rho = OFF - q * DIL
    # one 64-bit base per program (its first row), 32-bit offsets inside it
    row0 = (pid0 * BR).to(tl.int64)
    x_ptr += row0 * N_IN
    gx_ptr += row0 * N_IN
    m_ptr += row0 * N_OUT
    l_ptr += row0 * N_OUT
    out_ptr += row0 * N_OUT
    gy_ptr += row0 * N_OUT
    lr = pr // PB
    x_off = lr * N_IN + p0 + p + rho
    x_col = x_ptr + x_off
    xpos0 = p0 + p + rho
    v_col = (r // NCH) * SV_G + (r % NCH) * SV_C
    t_off = u[:, None] * BV + tl.arange(0, BV)[None, :]
    a_off = u[:, None] * BU + u[None, :]
    tile = BU * BV
    full_cols = (pid0 * BR + BR <= R) & (p0 + PB <= DIL)

    # NCARRY = 1: an output tile's adjoint reaches its input tile (A) and the next one (B1);
    # NCARRY = 2 (KT - 1 > BU): also the one after (B2).  Carries hold what later tiles receive.
    carry = tl.zeros([BU, PB * BR], ACC)
    carry2 = tl.zeros([BU, PB * BR], ACC)
    if PREFETCH:  # software pipelining: the next tile's loads are in flight during this tile's math
        ml_n, o_n, gy_n, X0n, X1n, X2n = _sweep_tile_loads(
            m_ptr, l_ptr, out_ptr, gy_ptr, x_col, xpos0, col_ok, lr, p, p0, u, t_start * BU, q, KT, DIL, N_IN,
            N_OUT, ACC, BV, NCU, ML_PRE)
    for t in range(t_start, t_end + NCARRY):
        u0 = t * BU
        acc = carry
        carry = carry2
        carry2 = tl.zeros([BU, PB * BR], ACC)
        if t < t_end:
            n = p0 + p[None, :] + DIL * (u0 + u)[:, None]
            nm = (n < N_OUT) & col_ok[None, :]
            if PREFETCH:
                ml, o, gyv, X0, X1, X2 = ml_n, o_n, gy_n, X0n, X1n, X2n
                ml_n, o_n, gy_n, X0n, X1n, X2n = _sweep_tile_loads(
                    m_ptr, l_ptr, out_ptr, gy_ptr, x_col, xpos0, col_ok, lr, p, p0, u, u0 + BU, q, KT, DIL, N_IN,
                    N_OUT, ACC, BV, NCU, ML_PRE)
            elif NCU > 0:
                ml, o, gyv, X0, X1, X2 = _sweep_tile_loads(
                    m_ptr, l_ptr, out_ptr, gy_ptr, x_col, xpos0, col_ok, lr, p, p0, u, u0, q, KT, DIL, N_IN, N_OUT,
                    ACC, BV, NCU, ML_PRE)
            else:
                st = lr[None, :] * N_OUT + n
                # log of the softmax denominator, m + log l: one live tile instead of two
                if ML_PRE:
                    ml = tl.load(m_ptr + st, mask=nm, other=0.0).to(ACC)
                else:
                    ml = tl.load(m_ptr + st, mask=nm, other=0.0).to(ACC) + tl.log(
                        tl.load(l_ptr + st, mask=nm, other=1.0).to(ACC))
                o = tl.load(out_ptr + st, mask=nm, other=0.0).to(ACC)
                gyv = tl.load(gy_ptr + st, mask=nm, other=0.0).to(ACC)
            for s in range(0, S):
                klo = tl.load(kf_ptr + 2 * s)
                khi = tl.load(kf_ptr + 2 * s + 1)
                gi = tl.load(gidx_ptr + s)
                t_re = tb_ptr + (s * 2 * NCHUNK) * tile + t_off
                t_im = t_re + NCHUNK * tile
                if NCU > 0:
                    re = _conv_u(t_re, tile, X0, X1, X2 if NCU > 2 else X0, NCU, ACC, PRECISION)
                    im = _conv_u(t_im, tile, X0, X1, X2 if NCU > 2 else X0, NCU, ACC, PRECISION)
                    if NT > 0:
                        d_re = tt_ptr + (s * 2 * NCHUNK) * tile + t_off
                        r0t = _conv_u(d_re, tile, X0, X1, X2 if NCU > 2 else X0, NCU, ACC, PRECISION)
                        i0t = _conv_u(d_re + NCHUNK * tile, tile, X0, X1, X2 if NCU > 2 else X0,
                                      NCU, ACC, PRECISION)
                    if NT > 1:
                        d2 = d_re + S * 2 * NCHUNK * tile
                        r1t = _conv_u(d2, tile, X0, X1, X2 if NCU > 2 else X0, NCU, ACC, PRECISION)
                        i1t = _conv_u(d2 + NCHUNK * tile, tile, X0, X1, X2 if NCU > 2 else X0,
                                      NCU, ACC, PRECISION)
                    n_ci = 0
                else:
                    re = tl.zeros([BU, PB * BR], ACC)
                    im = tl.zeros([BU, PB * BR], ACC)
                    n_ci = (BU + khi - klo - 1 + BV - 1) // BV
                v_first = u0 + q - khi + 1
                for ci in range(0, n_ci):
                    v0 = v_first + ci * BV
                    v = v0 + tl.arange(0, BV)
                    x_ptrs = x_col[None, :] + DIL * v[:, None]
                    lo = p0 + rho + DIL * v0
                    hi = p0 + PB - 1 + rho + DIL * (v0 + BV - 1)
                    if full_cols & (lo >= 0) & (hi < N_IN):
                        X = tl.load(x_ptrs).to(ACC)
                    else:
                        pos = xpos0[None, :] + DIL * v[:, None]
                        X = tl.load(x_ptrs, mask=(pos >= 0) & (pos < N_IN) & col_ok[None, :], other=0.0).to(ACC)
                    re += _dot(tl.load(t_re + ci * tile), X, ACC, PRECISION)
                    im += _dot(tl.load(t_im + ci * tile), X, ACC, PRECISION)
                sq = re * re + im * im
                if FAST:  # one MUFU.RSQ gives 1/|W| and |W|; it flushes subnormal sq to 0 (-> inf),
                    # so |W|^2 below the smallest normal float counts as 0 (|W| < 1.1e-19)
                    nz = sq >= 1.1754943508222875e-38
                    inv = tl.where(nz, tl.rsqrt(tl.where(nz, sq, 1.0)), 0.0)
                    mag = sq * inv
                else:
                    mag = tl.sqrt(sq)
                a = mag * INV_TAU
                if V_PER_POS:
                    v_off = v_col[None, :] + gi * SV_F + n * SV_N
                    val = tl.load(v_ptr + v_off, mask=nm, other=0.0).to(ACC)
                else:
                    val = tl.load(v_ptr + v_col + gi * SV_F, mask=col_ok, other=0.0).to(ACC)[None, :]
                if FAST:
                    pf = tl.where(nm, tl.exp2((a - ml) * 1.4426950408889634), 0.0)
                else:
                    pf = tl.where(nm, tl.exp(a - ml), 0.0)
                gp = gyv * pf
                if V_PER_POS:
                    if V_GROUPED:
                        gsum = tl.sum(tl.reshape(gp, [BU, BR, PB]), axis=1)
                        ph = tl.arange(0, PB)
                        n2 = p0 + ph[None, :] + DIL * (u0 + u)[:, None]
                        g0 = (pid0 * BR) // NCH
                        m2 = (n2 < N_OUT) & (p0 + ph < DIL)[None, :]
                        if RELAX:  # dL/dV is only summed: no ordering, no fence
                            tl.atomic_add(gv_ptr + g0 * SV_G + gi * SV_F + n2 * SV_N, gsum, mask=m2, sem="relaxed")
                        else:
                            tl.atomic_add(gv_ptr + g0 * SV_G + gi * SV_F + n2 * SV_N, gsum, mask=m2)
                    elif RELAX:
                        tl.atomic_add(gv_ptr + v_off, gp, mask=nm, sem="relaxed")
                    else:
                        tl.atomic_add(gv_ptr + v_off, gp, mask=nm)
                elif RELAX:
                    tl.atomic_add(gv_ptr + v_col + gi * SV_F, tl.sum(gp, axis=0), mask=col_ok, sem="relaxed")
                else:
                    tl.atomic_add(gv_ptr + v_col + gi * SV_F, tl.sum(gp, axis=0), mask=col_ok)
                if not FAST:
                    inv = tl.where(mag > 0, 1.0 / tl.where(mag > 0, mag, 1.0), 0.0)
                coef = gp * (val - o) * INV_TAU * inv
                g_re = coef * re
                g_im = coef * im
                # adjoint: A -> input tile I_t, B -> head of I_{t+1}
                a_re = ta_ptr + (s * 2 * (NCARRY + 1)) * (BU * BU) + a_off
                acc += _dot(tl.load(a_re), g_re, ACC, PRECISION)
                acc += _dot(tl.load(a_re + BU * BU), g_im, ACC, PRECISION)
                carry += _dot(tl.load(a_re + 2 * BU * BU), g_re, ACC, PRECISION)
                carry += _dot(tl.load(a_re + 3 * BU * BU), g_im, ACC, PRECISION)
                if NCARRY > 1:
                    carry2 += _dot(tl.load(a_re + 4 * BU * BU), g_re, ACC, PRECISION)
                    carry2 += _dot(tl.load(a_re + 5 * BU * BU), g_im, ACC, PRECISION)
                if NCU > 0 and NT > 0:
                    slot = (prog * N_UT + t) * S + s
                    tl.store(part_ptr + slot, tl.sum(tl.sum(g_re * r0t + g_im * i0t, axis=1), axis=0))
                    if NT > 1:
                        tl.store(part_ptr + P_TOT * N_UT * S + slot,
                                 tl.sum(tl.sum(g_re * r1t + g_im * i1t, axis=1), axis=0))
                elif NT > 0:
                    th0 = tl.zeros([BV, PB * BR], ACC)
                    th1 = tl.zeros([BV, PB * BR], ACC)
                    for ci in range(0, n_ci):
                        v0 = v_first + ci * BV
                        v = v0 + tl.arange(0, BV)
                        pos = xpos0[None, :] + DIL * v[:, None]
                        pm = (pos >= 0) & (pos < N_IN) & col_ok[None, :]
                        X = tl.load(x_col[None, :] + DIL * v[:, None], mask=pm, other=0.0).to(ACC)
                        d_re = tt_ptr + (s * 2 * NCHUNK + ci) * tile + t_off
                        A = _dot(tl.trans(tl.load(d_re)), g_re, ACC, PRECISION)
                        A += _dot(tl.trans(tl.load(d_re + NCHUNK * tile)), g_im, ACC, PRECISION)
                        th0 += A * X
                        if NT > 1:
                            d2 = d_re + S * 2 * NCHUNK * tile
                            A = _dot(tl.trans(tl.load(d2)), g_re, ACC, PRECISION)
                            A += _dot(tl.trans(tl.load(d2 + NCHUNK * tile)), g_im, ACC, PRECISION)
                            th1 += A * X
                    slot = (prog * N_UT + t) * S + s
                    tl.store(part_ptr + slot, tl.sum(tl.sum(th0, axis=1), axis=0))
                    if NT > 1:
                        tl.store(part_ptr + P_TOT * N_UT * S + slot, tl.sum(tl.sum(th1, axis=1), axis=0))
        # input tile I_t starts at v = u0 + q - KT + 1 (input lattice of the phase)
        vI = u0 + q - KT + 1 + u
        posI = xpos0[None, :] + DIL * vI[:, None]
        pmI = (posI >= 0) & (posI < N_IN) & col_ok[None, :]
        g_ptrs = gx_ptr + x_off[None, :] + DIL * vI[:, None]
        if ((t < t_start + NCARRY) & (seg > 0)) | (t >= t_end):  # shared with a neighbouring segment
            tl.atomic_add(g_ptrs, acc.to(gx_ptr.dtype.element_ty), mask=pmI, sem="relaxed")
        else:
            if ACCUM:
                acc += tl.load(g_ptrs, mask=pmI, other=0.0).to(ACC)
            tl.store(g_ptrs, acc.to(gx_ptr.dtype.element_ty), mask=pmI)


def adjoint_tiles(h, bu, dtype, ncarry: int = 1):
    """(S*2, K) stage filters -> (S, 2 * (ncarry + 1), bu, bu): [A_re, A_im, B1_re, B1_im, (B2_re, B2_im)],
    ``A[v', u'] = h[u' - v' + K - 1]`` (into the aligned input tile) and
    ``Bj[v', u'] = h[u' - v' - j * bu + K - 1]`` (into the j-th next one), zero outside [0, K)."""
    S2, K = h.shape
    S = S2 // 2
    hh = h.view(S, 2, K)
    dev = h.device
    vp = torch.arange(bu, device=dev)[:, None]
    up = torch.arange(bu, device=dev)[None, :]
    out = []
    for j in range(ncarry + 1):
        kk = up - vp + K - 1 - j * bu
        valid = (kk >= 0) & (kk < K)
        out.append(hh[:, :, kk.clamp(0, K - 1)] * valid)                  # (S, 2, bu, bu)
    return torch.cat(out, 1).to(dtype).contiguous()


def backward_stage_sweep(x, tiles, h, kf, gidx, v, v_strides, v_per_pos, nch, inv_tau, off, dil, precision,
                         m, l, out, gy, gv, gx, accumulate, ttiles=None, gth=None, ml=None):
    """Atomic-free fused backward of one octave (short filters, ``K <= BU + 1``).

    A program owns ``BR`` rows x ``PB`` phases and walks their output tiles in order: it
    recomputes W and dL/dW in registers, accumulates dL/dV and the filter-tangent partials
    (``gth[t, f] += dL/dtheta_t``, as :func:`backward_stage`), and applies the adjoint of the
    filters as two banded ``BU x BU`` blocks, ``A`` into the input tile aligned with the output
    tile and ``B`` into the head of the next one; ``B @ dL/dW`` is carried to the next step, so
    every dL/dx sample is written once with a plain store (``accumulate``: added to ``gx``).
    No dL/dW buffer, no grad-x / grad-h pass.  ``ml``: ``m + log(l)``, precomputed once for all
    the octaves (otherwise formed per tile).  ``tiles`` must be chunked (``band_tiles(...,
    window=False)``), ``ttiles`` likewise.
    """
    tb, bu, n_chunks, win = tiles
    if win:
        raise ValueError("the sweep backward takes chunked tiles (band_tiles(..., window=False))")
    if v.numel() >= 2**31:
        raise ValueError("the sweep backward indexes V with 32-bit offsets")
    R, n_in = x.shape
    n_out = m.shape[1]
    K = h.shape[1]
    if K > 2 * bu + 1:
        raise ValueError(f"the sweep backward needs K <= {2 * bu + 1} taps, got {K}")
    ncarry = 1 if K <= bu + 1 else 2
    S = gidx.numel()
    if R == 0 or n_out == 0 or S == 0:
        return
    cols, warps = SWEEP_SMALL_CFG[ncarry] if bu == SWEEP_SMALL_BU else (SWEEP_COLS, SWEEP_NUM_WARPS)
    pb = min(triton.next_power_of_2(dil), 8)
    br = max(16 // pb, min(cols // pb, triton.next_power_of_2(R)))
    n_pb = triton.cdiv(dil, pb)
    n_ut = triton.cdiv(triton.cdiv(n_out, dil), bu)
    P = triton.cdiv(R, br) * n_pb
    # segments per row: enough programs for SWEEP_MIN_PROGRAMS_PER_SM per SM, segments of at least
    # SWEEP_MIN_SEG_TILES tiles (a boundary costs NCARRY atomic tiles)
    want = SWEEP_MIN_PROGRAMS_PER_SM * torch.cuda.get_device_properties(x.device).multi_processor_count
    n_seg = max(1, min(triton.cdiv(want, P), n_ut // max(1, SWEEP_MIN_SEG_TILES)))
    t_seg = triton.cdiv(n_ut, n_seg)
    n_seg = triton.cdiv(n_ut, t_seg)
    grid = (triton.cdiv(R, br), n_pb, n_seg)
    ta = adjoint_tiles(h, bu, tb.dtype, ncarry)
    nt = 0 if ttiles is None else ttiles.shape[0]
    ncu = n_chunks if (SWEEP_UNI and 2 <= n_chunks <= 3 and bool((kf[:, 0] == 0).all())
                       and bool((kf[:, 1] == K).all())) else 0
    part = torch.empty((max(nt, 1), P * n_ut * S), device=x.device, dtype=gv.dtype)
    ml_pre = bool(SWEEP_ML_PRE and ml is not None)
    _spectral_softmax_bwd_sweep_kernel[grid](
        x, tb, ta, ttiles if nt else tb, kf, gidx, v, ml if ml_pre else m, l, out, gy, gv, gx, part,
        R, n_in, n_out, off, dil, n_chunks, nch, inv_tau, n_pb, n_ut, K, P, t_seg, *v_strides,
        ACC=_acc(x.dtype), V_PER_POS=v_per_pos, PRECISION=precision, BU=bu, BV=BV, PB=pb, BR=br,
        NT=nt, V_GROUPED=bool(v_per_pos and nch % br == 0), ACCUM=bool(accumulate), NCU=ncu, NCARRY=ncarry,
        PREFETCH=bool(SWEEP_PREFETCH and ncu), RELAX=bool(SWEEP_RELAX), ML_PRE=ml_pre,
        FAST=bool(SWEEP_FAST and x.dtype != torch.float64), S=S,
        num_warps=warps, num_stages=SWEEP_STAGES, **({"maxnreg": SWEEP_MAXNREG} if SWEEP_MAXNREG else {}),
    )
    if nt:
        gth.index_add_(1, gidx.long(), part.view(nt, P * n_ut, S).sum(1))
