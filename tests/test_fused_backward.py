import itertools

import pytest
import torch

import fdwt
from fdwt.cwt import spectral_softmax, spectral_softmax_params
from fdwt.kernels import softmax as KS

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def config_restore():
    saved = (fdwt.config.fused_input_grad, fdwt.config.fused_param_grad, KS.USE_WINDOW)
    yield
    fdwt.config.fused_input_grad, fdwt.config.fused_param_grad, KS.USE_WINDOW = saved


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
@pytest.mark.parametrize("replan", [False, True])
def test_param_path_matches_filter_autograd(per_position, nch, fused_x, fused_p, replan, config_restore):
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
    got = _grads(lambda x, V, ls, w0: spectral_softmax_params(x, m.plan, ls, w0, V, nch, 0.1), x, V, ls, w0, gy)
    fdwt.config.fused_input_grad, fdwt.config.fused_param_grad = False, False
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
