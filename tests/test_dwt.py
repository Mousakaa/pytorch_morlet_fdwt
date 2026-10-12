import numpy as np
import pytest
import pywt
import torch
from torch.autograd import gradcheck, gradgradcheck

import fdwt

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
MODES = ["zero", "periodic", "symmetric", "reflect", "periodization"]
WAVELETS = ["haar", "db2", "db4", "sym5", "coif3", "bior2.2", "bior3.5"]


def _pywt_dwt(x, w, mode):
    try:
        return pywt.dwt(x, w, mode=mode)
    except ValueError:
        return None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("wavelet", WAVELETS)
@pytest.mark.parametrize("mode", MODES)
def test_dwt_idwt_match_pywt(device, wavelet, mode):
    torch.manual_seed(0)
    for n in [1, 2, 3, 7, 16, 33, 100, 257]:
        x = torch.randn(3, n, dtype=torch.float64, device=device)
        ref = _pywt_dwt(x.cpu().numpy(), wavelet, mode)
        if ref is None:
            continue
        a, d = fdwt.dwt(x, wavelet, mode)
        np.testing.assert_allclose(a.cpu().numpy(), ref[0], atol=1e-12, rtol=0)
        np.testing.assert_allclose(d.cpu().numpy(), ref[1], atol=1e-12, rtol=0)
        try:
            y_ref = pywt.idwt(ref[0], ref[1], wavelet, mode=mode)
        except ValueError:
            continue
        y = fdwt.idwt(a, d, wavelet, mode)
        np.testing.assert_allclose(y.cpu().numpy(), y_ref, atol=1e-12, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["symmetric", "periodization", "zero"])
def test_wavedec_waverec(device, mode):
    torch.manual_seed(0)
    for wavelet in ["db4", "sym8"]:
        for n in [64, 1000, 4097]:
            x = torch.randn(2, 3, n, dtype=torch.float64, device=device)
            coeffs = fdwt.wavedec(x, wavelet, mode)
            ref = pywt.wavedec(x.cpu().numpy(), wavelet, mode=mode)
            assert len(coeffs) == len(ref)
            for c, r in zip(coeffs, ref):
                np.testing.assert_allclose(c.cpu().numpy(), r, atol=1e-11, rtol=0)
            rec = fdwt.waverec(coeffs, wavelet, mode)[..., :n]
            np.testing.assert_allclose(rec.cpu().numpy(), x.cpu().numpy(), atol=1e-10, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", MODES)
def test_gradients(device, mode):
    torch.manual_seed(0)
    for wavelet in ["haar", "db3", "bior3.5"]:
        for n in [1, 2, 5, 8, 13, 40]:
            x = torch.randn(2, n, dtype=torch.float64, device=device, requires_grad=True)
            if _pywt_dwt(x.detach().cpu().numpy(), wavelet, mode) is None:
                continue
            assert gradcheck(lambda x: fdwt.dwt(x, wavelet, mode), (x,))
            assert gradgradcheck(lambda x: fdwt.dwt(x, wavelet, mode), (x,))
            a, d = (t.detach().requires_grad_() for t in fdwt.dwt(x, wavelet, mode))
            L = fdwt.get_filter_bank(wavelet).length
            if mode == "periodization" or 2 * a.shape[-1] - L + 2 > 0:
                assert gradcheck(lambda a, d: fdwt.idwt(a, d, wavelet, mode), (a, d))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_low_precision_dtypes(dtype):
    torch.manual_seed(0)
    x = torch.randn(4, 1000, device="cuda")
    ref = fdwt.wavedec(x.double(), "db4", level=4)
    out = fdwt.wavedec(x.to(dtype), "db4", level=4)
    tol = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 6e-2}[dtype]
    for o, r in zip(out, ref):
        assert o.dtype == dtype
        torch.testing.assert_close(o.double(), r, atol=tol * r.abs().max().item(), rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_matches_cpu_large_batch():
    torch.manual_seed(0)
    x = torch.randn(5, 7, 2049)
    xg = x.cuda().requires_grad_()
    xc = x.clone().requires_grad_()
    outs = [fdwt.wavedec(t, "sym4", "reflect") for t in (xg, xc)]
    for a, b in zip(*outs):
        torch.testing.assert_close(a.cpu(), b, atol=1e-5, rtol=1e-5)
    for t, o in zip((xg, xc), outs):
        sum((c * c).sum() for c in o).backward()
    torch.testing.assert_close(xg.grad.cpu(), xc.grad, atol=1e-4, rtol=1e-4)


def test_modules_and_errors():
    x = torch.randn(2, 128)
    m, im = fdwt.DWT1D("db2", level=3), fdwt.IDWT1D("db2")
    torch.testing.assert_close(im(m(x))[..., :128], x, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError):
        fdwt.dwt(x, "morl")
    with pytest.raises(ValueError):
        fdwt.dwt(x, "db2", mode="smooth")
    custom = fdwt.get_filter_bank(pywt.Wavelet("db2").filter_bank)
    torch.testing.assert_close(fdwt.dwt(x, custom)[0], fdwt.dwt(x, "db2")[0])
    assert fdwt.dwt_max_level(1000, "db4") == pywt.dwt_max_level(1000, 8)
