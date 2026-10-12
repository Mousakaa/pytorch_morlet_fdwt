import math

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

import fdwt
from fdwt.cwt import _apply_plan

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
FREQS = np.geomspace(0.5, 100, 24)


def _naive_cwt(x, scales, w0=6.0, truncate=4.0):
    """Literal definition: W[s, n] = sum_m x[m] conj(psi((m - n)/s)) / sqrt(s)."""
    N = x.shape[-1]
    m = torch.arange(N, dtype=torch.float64)
    out = []
    for s in scales:
        t = (m[None, :] - m[:, None]) / s  # [n, m]
        env = torch.exp(-0.5 * t ** 2) * ((m[None, :] - m[:, None]).abs() <= math.ceil(truncate * s))
        psi = math.pi ** -0.25 * (torch.exp(1j * w0 * t) - math.exp(-w0 ** 2 / 2)) * env
        out.append((x.to(torch.complex128) @ psi.conj().T) / math.sqrt(s))
    return torch.stack(out, -2)


def test_direct_matches_definition():
    torch.manual_seed(0)
    x = torch.randn(2, 300, dtype=torch.float64)
    scales = [1.0, 2.5, 7.0, 30.0]
    ref = _naive_cwt(x, scales)
    for device in DEVICES:
        for method in ["direct", "fft"]:
            out = fdwt.cwt(x.to(device), scales, method=method).cpu()
            torch.testing.assert_close(out, ref, atol=1e-12, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [7, 500, 3000])
def test_methods_agree(device, n):
    torch.manual_seed(0)
    x = torch.randn(3, n, dtype=torch.float64, device=device)
    ref = fdwt.cwt(x, freqs=FREQS, fs=250, method="direct")
    scale = ref.abs().max()
    torch.testing.assert_close(fdwt.cwt(x, freqs=FREQS, fs=250, method="fft"), ref,
                               atol=1e-12 * scale, rtol=0)
    fast = fdwt.cwt(x, freqs=FREQS, fs=250, method="fast")
    assert ((fast - ref).abs().max() / scale) < 3e-4
    # octave 0 (the finest scales) is computed exactly
    torch.testing.assert_close(fast[..., -4:, :], ref[..., -4:, :], atol=1e-12 * scale, rtol=0)


def test_fast_accuracy_with_wider_truncation():
    torch.manual_seed(0)
    x = torch.randn(2, 4000, dtype=torch.float64)
    ref = fdwt.cwt(x, freqs=FREQS, fs=250, method="direct", truncate=5.0)
    fast = fdwt.cwt(x, freqs=FREQS, fs=250, method="fast", truncate=5.0)
    assert ((fast - ref).abs().max() / ref.abs().max()) < 2e-5


@cuda
@pytest.mark.parametrize("method", ["direct", "fast"])
@pytest.mark.parametrize("cplx", [True, False])
def test_triton_matches_torch_reference(method, cplx):
    torch.manual_seed(0)
    x = torch.randn(3, 2, 3000, dtype=torch.float64)
    g = torch.randn(3, 2, len(FREQS), 3000, 2 if cplx else 1, dtype=torch.float64)
    res = []
    for device in ["cuda", "cpu"]:
        m = fdwt.CWT1D(freqs=FREQS, fs=250, method=method, complex=cplx, learnable=True,
                       learnable_w0=True).to(device).double()
        xd = x.to(device).requires_grad_()
        y = m(xd)
        yr = torch.view_as_real(y) if cplx else y[..., None]
        (yr * g.to(device)).sum().backward()
        res.append([t.detach().cpu() for t in (yr, xd.grad, m.log_scales.grad, m.w0.grad)])
    for a, b in zip(*res):
        torch.testing.assert_close(a, b, atol=1e-10 * b.abs().max().item(), rtol=0)


@cuda
@pytest.mark.parametrize("method", ["direct", "fast"])
@pytest.mark.parametrize("cplx", [True, False])
def test_gradcheck(method, cplx):
    torch.manual_seed(0)
    scales = [1.5, 3.0, 6.0, 9.0, 14.0, 23.0]
    m = fdwt.CWT1D(scales=scales, method=method, complex=cplx, learnable=True,
                   learnable_w0=True).cuda().double()
    for n in [5, 37, 150]:
        x = torch.randn(2, n, dtype=torch.float64, device="cuda", requires_grad=True)
        ls = m.log_scales.detach().clone().requires_grad_()
        w0 = torch.tensor(6.0, dtype=torch.float64, device="cuda", requires_grad=True)

        def f(ls, w0, x):
            y = _apply_plan(x, m.plan, m.plan.filters(ls.exp(), w0, torch.float64))
            return torch.view_as_real(y) if cplx else y

        assert gradcheck(f, (ls, w0, x))


@cuda
def test_fp32_accuracy_and_dtypes():
    torch.manual_seed(0)
    x = torch.randn(8, 2500, device="cuda")
    ref = fdwt.cwt(x.double(), freqs=FREQS, fs=250, method="fast")
    out = fdwt.cwt(x, freqs=FREQS, fs=250, method="fast")
    assert out.dtype == torch.complex64
    assert ((out - ref).abs().max() / ref.abs().max()) < 1e-5
    out16 = fdwt.cwt(x.half(), freqs=FREQS, fs=250, method="fast")
    assert out16.dtype == torch.complex64
    assert ((out16 - ref).abs().max() / ref.abs().max()) < 1e-2


def test_frequency_localisation():
    fs, f0 = 250.0, 10.0
    t = torch.arange(5000, dtype=torch.float64) / fs
    x = torch.sin(2 * math.pi * f0 * t)
    freqs = np.geomspace(1, 60, 40)
    for norm in ["l1", "l2"]:
        w = fdwt.cwt(x, freqs=freqs, fs=fs, norm=norm).abs()[:, 2000:3000].mean(-1)
        assert abs(freqs[int(w.argmax())] - f0) / f0 < 0.06
    # l1 normalisation gives the same amplitude to sinusoids of any frequency
    amps = []
    for f in [3.0, 12.0, 40.0]:
        x = torch.sin(2 * math.pi * f * t)
        amps.append(fdwt.cwt(x, freqs=[f], fs=fs, norm="l1").abs()[0, 2000:3000].mean())
    assert max(amps) / min(amps) < 1.01


def test_module_api():
    m = fdwt.CWT1D(freqs=[2.0, 10.0, 40.0], fs=250)
    x = torch.randn(4, 5, 200)
    y = m(x)
    assert y.shape == (4, 5, 3, 200) and y.is_complex()
    np.testing.assert_allclose(m.freqs.numpy(), [2.0, 10.0, 40.0])
    assert not any(True for _ in m.parameters())
    with pytest.raises(ValueError):
        fdwt.cwt(x, scales=[1.0], freqs=[1.0])
    with pytest.raises(ValueError):
        fdwt.cwt(x, [1.0], method="nope")
