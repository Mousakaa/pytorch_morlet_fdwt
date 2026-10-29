import itertools

import pytest
import torch

import fdwt
from fdwt.cwt import MorletPlan, spectral_softmax, spectral_softmax_params
from fdwt.kernels import softmax as KS

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def config_restore():
    saved = (fdwt.config.fused_input_grad, fdwt.config.fused_param_grad, fdwt.config.sweep_backward, KS.USE_WINDOW)
    yield
    (fdwt.config.fused_input_grad, fdwt.config.fused_param_grad, fdwt.config.sweep_backward,
     KS.USE_WINDOW) = saved


def _module(shrink=0.0, w0=6.0):
    m = fdwt.CWT1D(freqs=torch.logspace(0, 1.2, 16).tolist(), fs=64, learnable=True, learnable_w0=True)
    with torch.no_grad():
        m.log_scales.sub_(shrink)
        m.w0.fill_(w0)
    return m.cuda()


def _grads(fn, x, V, ls, w0, gy):
    out = fn(x, V, ls, w0)
    return [out] + list(torch.autograd.grad(out, (x, V, ls, w0), gy))


@cuda
@pytest.mark.parametrize("per_position,nch", [(False, 8), (True, 8), (True, 64)])
@pytest.mark.parametrize("fused_x,fused_p", list(itertools.product([False, True, "auto"], [False, True])))
@pytest.mark.parametrize("replan,sweep", [(False, "auto"), (True, False), (True, "auto")])
def test_param_path_matches_filter_autograd(per_position, nch, fused_x, fused_p, replan, sweep, config_restore):
    # the scales / w0 parametrisation (in-kernel filter gradients, fused dL/dx) against autograd
    # through the filters with the original kernels; nch=64 takes the grouped dL/dV atomics
    torch.manual_seed(0)
    m = _module(shrink=2.0 if replan else 0.0, w0=0.7 if replan else 6.0)
    if replan:
        m.replan()
    R, N = 2 * nch if per_position else 48, 300
    dt = torch.float64
    x = torch.randn(R, N, device="cuda", dtype=dt, requires_grad=True)
    V = torch.randn((R // nch, 16, N) if per_position else (nch, 16), device="cuda", dtype=dt, requires_grad=True)
    ls = m.log_scales.detach().to(dt).clone().requires_grad_()
    w0 = torch.tensor(m.w0_value, device="cuda", dtype=dt, requires_grad=True)
    gy = torch.randn(R, N, device="cuda", dtype=dt)

    fdwt.config.fused_input_grad, fdwt.config.fused_param_grad = fused_x, fused_p
    fdwt.config.sweep_backward = sweep
    got = _grads(lambda x, V, ls, w0: spectral_softmax_params(x, m.plan, ls, w0, V, nch, 0.1), x, V, ls, w0, gy)
    fdwt.config.fused_input_grad, fdwt.config.fused_param_grad = False, False
    fdwt.config.sweep_backward = False
    KS.USE_WINDOW = False
    ref = _grads(lambda x, V, ls, w0: spectral_softmax(x, m.plan, m.plan.filters(ls.exp(), w0, dt), V, nch, 0.1),
                 x, V, ls, w0, gy)
    for a, b in zip(got, ref):
        torch.testing.assert_close(a, b, atol=1e-10 * b.abs().max().item(), rtol=0)


@cuda
def test_window_tiles_match_cpu(config_restore):
    # short (replanned) filters take the one-window kernel layout; compare with the CPU path
    torch.manual_seed(0)
    m = _module(shrink=2.0, w0=0.7).double()
    m.replan()
    assert max(2 * st.half + 1 for st in m.plan.stages) <= 33
    nch, R, N = 4, 8, 257
    x = torch.randn(R, N, dtype=torch.float64)
    V = torch.randn(nch, 16, dtype=torch.float64)
    ref = m.cpu().spectral_softmax(x, V, nch, 0.2)
    m.cuda()
    for use_window in (True, False):
        KS.USE_WINDOW = use_window
        out = m.spectral_softmax(x.cuda(), V.cuda(), nch, 0.2).cpu()
        torch.testing.assert_close(out, ref, atol=1e-12, rtol=0)


@cuda
def test_param_path_float_w0_and_frozen_scales():
    torch.manual_seed(0)
    m = fdwt.CWT1D(freqs=[2.0, 5.0, 11.0], fs=64).cuda().double()  # nothing learnable
    x = torch.randn(6, 200, dtype=torch.float64, device="cuda", requires_grad=True)
    V = torch.randn(3, 3, dtype=torch.float64, device="cuda", requires_grad=True)
    out = spectral_softmax_params(x, m.plan, m.log_scales, 6.0, V, 3, 0.5)
    ref = spectral_softmax(x, m.plan, m.plan.filters(m.scales.double(), 6.0, torch.float64), V, 3, 0.5)
    g = torch.autograd.grad(out.square().sum(), (x, V))
    g_ref = torch.autograd.grad(ref.square().sum(), (x, V))
    torch.testing.assert_close(out, ref)
    for a, b in zip(g, g_ref):
        torch.testing.assert_close(a, b)


def test_replan_keeps_transform():
    # scales that shrank: same octaves, supports cut to the learned envelope -> same transform
    torch.manual_seed(0)
    m = fdwt.CWT1D(freqs=torch.logspace(0, 1.2, 16).tolist(), fs=64, learnable=True, learnable_w0=True,
                   method="fast").double()
    with torch.no_grad():
        m.log_scales.sub_(1.5)
        m.w0.fill_(2.0)
    x = torch.randn(3, 400, dtype=torch.float64)
    octaves, taps = m.plan.octaves, [2 * st.half + 1 for st in m.plan.stages]
    y0 = m(x)
    m.replan()
    assert m.plan.octaves == octaves
    assert all(a < b for a, b in zip([2 * st.half + 1 for st in m.plan.stages], taps))
    y1 = m(x)
    assert ((y1 - y0).abs().max() / y0.abs().max()).item() < 2e-3  # Gaussian tails beyond 5 sigma
    m.replan(regroup=True)
    assert m.plan.octaves != octaves  # small scales move to finer octaves


def test_plan_octaves_argument():
    p = fdwt.MorletPlan([2.0, 40.0], octaves=[0, 0])
    assert p.octaves == [0, 0] and p.n_octaves == 1
    with pytest.raises(ValueError):
        fdwt.MorletPlan([2.0, 40.0], octaves=[0])


@cuda
@pytest.mark.parametrize("shrink", [2.0, 1.5, 1.0])  # 11-13 taps (one carried tile), 17-19 (both), 27-31 (two)
@pytest.mark.parametrize("R,N,nch,per_position", [(3, 40, 3, False), (5, 17, 5, True), (130, 1000, 65, True),
                                                  (64, 300, 64, True), (96, 300, 32, True)])
def test_sweep_backward_shapes(R, N, nch, per_position, shrink, config_restore):
    # short signals, 16-row tiles with one or two carried input tiles and 2-3 shared input chunks,
    # row counts that do not fill a tile, V rows shared by nch rows; fp64, against the grad-x /
    # grad-h path
    torch.manual_seed(1)
    m = _module(shrink=shrink, w0=0.7).double()
    m.replan()
    dt = torch.float64
    x = torch.randn(R, N, device="cuda", dtype=dt, requires_grad=True)
    V = torch.randn((R // nch, 16, N) if per_position else (nch, 16), device="cuda", dtype=dt, requires_grad=True)
    ls = m.log_scales.detach().clone().requires_grad_()
    w0 = torch.tensor(m.w0_value, device="cuda", dtype=dt, requires_grad=True)
    gy = torch.randn(R, N, device="cuda", dtype=dt)
    fn = lambda x, V, ls, w0: spectral_softmax_params(x, m.plan, ls, w0, V, nch, 0.3)
    fdwt.config.sweep_backward = "auto"
    got = _grads(fn, x, V, ls, w0, gy)
    fdwt.config.sweep_backward = False
    ref = _grads(fn, x, V, ls, w0, gy)
    for a, b in zip(got, ref):
        torch.testing.assert_close(a, b, atol=1e-10 * b.abs().max().item(), rtol=0)


@cuda
@pytest.mark.parametrize("shrink", [2.0, 1.0])  # one / two carried input tiles across segment boundaries
@pytest.mark.parametrize("per_position", [False, True])
def test_sweep_backward_segments(shrink, per_position, config_restore, monkeypatch):
    # rows split into many one-tile segments: every boundary tile is shared by two programs
    monkeypatch.setattr(KS, "SWEEP_MIN_PROGRAMS_PER_SM", 10**6)
    monkeypatch.setattr(KS, "SWEEP_MIN_SEG_TILES", 1)
    torch.manual_seed(2)
    m = _module(shrink=shrink, w0=0.7).double()
    m.replan()
    dt = torch.float64
    R, N, nch = 64, 400, 32
    x = torch.randn(R, N, device="cuda", dtype=dt, requires_grad=True)
    V = torch.randn((R // nch, 16, N) if per_position else (nch, 16), device="cuda", dtype=dt, requires_grad=True)
    ls = m.log_scales.detach().clone().requires_grad_()
    w0 = torch.tensor(m.w0_value, device="cuda", dtype=dt, requires_grad=True)
    gy = torch.randn(R, N, device="cuda", dtype=dt)
    fn = lambda x, V, ls, w0: spectral_softmax_params(x, m.plan, ls, w0, V, nch, 0.3)
    fdwt.config.sweep_backward = "auto"
    got = _grads(fn, x, V, ls, w0, gy)
    fdwt.config.sweep_backward = False
    ref = _grads(fn, x, V, ls, w0, gy)
    for a, b in zip(got, ref):
        torch.testing.assert_close(a, b, atol=1e-10 * b.abs().max().item(), rtol=0)


@cuda
@pytest.mark.parametrize("case", ["zero_tail", "tiny_rows", "tiny_tail"])
def test_sweep_backward_fp32_degenerate_rows(case, config_restore):
    # fp32 takes the fast-math sweep (rsqrt, exp2): zero-padded or tiny inputs give subnormal
    # |W|^2, which rsqrt flushes to 0; the gradients must stay finite and match the grad-x /
    # grad-h path
    torch.manual_seed(3)
    m = _module(shrink=1.0, w0=0.7)
    m.replan()
    R, N, nch = 96, 400, 32
    x = 0.3 * torch.randn(R, N, device="cuda")
    if case == "zero_tail":
        x[::3, N // 2:] = 0
    elif case == "tiny_rows":
        x[::3] *= 1e-20
    else:
        x[::3, N // 2:] *= 1e-22
    x.requires_grad_()
    V = torch.randn(R // nch, 16, N, device="cuda", requires_grad=True)
    ls = m.log_scales.detach().clone().requires_grad_()
    w0 = torch.tensor(m.w0_value, device="cuda", requires_grad=True)
    gy = torch.randn(R, N, device="cuda")
    fn = lambda x, V, ls, w0: spectral_softmax_params(x, m.plan, ls, w0, V, nch, 0.1)
    fdwt.config.sweep_backward = "auto"
    got = _grads(fn, x, V, ls, w0, gy)
    fdwt.config.sweep_backward = False
    ref = _grads(fn, x, V, ls, w0, gy)
    for a, b in zip(got, ref):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=1e-4 * b.abs().max().item(), rtol=0)


@pytest.mark.parametrize("norm,complex_", [("l2", True), ("l1", True), ("l2", False)])
def test_filter_tangents_match_jvp(norm, complex_):
    # closed-form dH/dlog s and dH/dw0 against forward-mode autodiff of the filters
    torch.manual_seed(4)
    m = fdwt.CWT1D(freqs=torch.logspace(0, 1.2, 16).tolist(), fs=64, learnable=True, learnable_w0=True,
                   norm=norm, complex=complex_).double()
    p = m.plan
    ls = m.log_scales.detach() + 0.1 * torch.randn(16, dtype=torch.float64)
    w = torch.tensor(4.7, dtype=torch.float64)
    fn = lambda a, b: p.filters(a.exp(), b, torch.float64)
    ref = [torch.func.jvp(fn, (ls, w), (torch.ones_like(ls), torch.zeros_like(w)))[1],
           torch.func.jvp(fn, (ls, w), (torch.zeros_like(ls), torch.ones_like(w)))[1]]
    for a, b in zip(p.filter_tangents(ls, w, torch.float64), ref):
        torch.testing.assert_close(a, b, atol=1e-14 * b.abs().max().item(), rtol=0)


@cuda
@pytest.mark.parametrize("banks,heads", [(4, None), (1, 4), (1, 3), (2, 2)])
@pytest.mark.parametrize("per_position,nch", [(False, 1), (False, 8), (True, 8), (True, 64)])
@pytest.mark.parametrize("replan,sweep", [(False, "auto"), (True, False), (True, "auto")])
def test_banks_and_value_heads_match_cpu(banks, heads, per_position, nch, replan, sweep, config_restore):
    # filter banks (own scales and w0 per bank, one launch per octave) and value heads (one
    # softmax, several value sets) against the plain PyTorch path; nch=64 takes the grouped
    # dL/dV atomics, heads=3 the zero-padded head axis
    torch.manual_seed(0)
    m = fdwt.CWT1D(freqs=torch.logspace(0, 1.2, 16).tolist(), fs=64, learnable=True, learnable_w0=True,
                   banks=banks)
    with torch.no_grad():  # banks that drifted apart, as after training
        m.log_scales.add_(0.1 * torch.randn_like(m.log_scales) - (2.0 if replan else 0.0))
        m.w0.copy_(torch.linspace(5.5, 6.5, banks) if banks > 1 else torch.tensor(0.7 if replan else 6.0))
    m = m.cuda()
    if replan:
        m.replan()
    assert m.plan.banks == banks
    S, hv = m.plan.num_scales, heads or 1
    R, N = 2 * nch if per_position else 48, 300
    dt = torch.float64
    x = torch.randn(R, N, device="cuda", dtype=dt, requires_grad=True)
    vshape = (R // nch, hv, S, N) if per_position else (hv, nch, S)
    if heads is None:
        vshape = vshape[:1] + vshape[2:] if per_position else vshape[1:]
    V = torch.randn(vshape, device="cuda", dtype=dt, requires_grad=True)
    ls = m.log_scales.detach().to(dt).clone().requires_grad_()
    w0 = m.w0.detach().to(dt).clone().requires_grad_()
    gy = torch.randn((hv, banks, R, N) if (heads or banks > 1) else (R, N), device="cuda", dtype=dt)

    fdwt.config.sweep_backward = sweep
    got = _grads(lambda x, V, ls, w0: spectral_softmax_params(x, m.plan, ls, w0, V, nch, 0.1, heads),
                 x, V, ls, w0, gy)
    cpu = [t.detach().cpu().requires_grad_() for t in (x, V, ls, w0)]
    ref = _grads(lambda x, V, ls, w0: spectral_softmax_params(x, m.plan, ls, w0, V, nch, 0.1, heads),
                 *cpu, gy.cpu())
    for a, b in zip(got, ref):
        torch.testing.assert_close(a.cpu(), b, atol=1e-10 * b.abs().max().item(), rtol=0)


def test_bank_plan_shares_octaves():
    # scale k of every bank sits in one octave (the finest), stages list the banks in order
    scales = [1.0, 3.0, 9.0, 27.0]
    plan = MorletPlan(scales + [s * 1.9 for s in scales], banks=2)
    oct_ = plan.octaves
    assert oct_[:4] == oct_[4:]
    assert oct_[:4] == MorletPlan(scales).octaves
    for st in plan.stages:
        k = len(st.scale_idx) // 2
        assert all(i < 4 for i in st.scale_idx[:k]) and all(i >= 4 for i in st.scale_idx[k:])
