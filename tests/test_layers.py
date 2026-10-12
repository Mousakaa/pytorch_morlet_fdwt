import numpy as np
import pytest
import torch

import fdwt
from torch.autograd import gradcheck

from fdwt.cwt import _MorletMixFunction, _SpectralSoftmaxFunction, morlet_mix, spectral_softmax

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _mix_reference(x, plan, H, a, b):
    w = fdwt.cwt.__globals__["_apply_plan"](x, plan, H)
    re = w.real if plan.complex else w
    return torch.einsum("bdsn,ds->bdn", w.abs(), a) + torch.einsum("bdsn,ds->bdn", re, b)


@cuda
@pytest.mark.parametrize("method", ["fast", "direct"])
@pytest.mark.parametrize("cplx", [True, False])
def test_fused_mix_matches_unfused(method, cplx, monkeypatch):
    torch.manual_seed(0)
    m = fdwt.CWT1D(freqs=np.geomspace(1, 60, 12), fs=250, method=method, complex=cplx,
                   learnable=True).cuda().double()
    B, D, N = 2, 3, 1500
    x = torch.randn(B, D, N, dtype=torch.float64, device="cuda")
    a = torch.randn(D, 12, dtype=torch.float64, device="cuda")
    b = torch.randn(D, 12, dtype=torch.float64, device="cuda")
    gy = torch.randn(B, D, N, dtype=torch.float64, device="cuda")
    # force several row chunks in the backward pass
    monkeypatch.setattr(_MorletMixFunction, "MIX_CHUNK_BYTES", 4 * 12 * N * 16)
    res = []
    for fn in (morlet_mix, _mix_reference):
        m.log_scales.grad = None
        xx, aa, bb = (t.clone().requires_grad_() for t in (x, a, b))
        H = m.plan.filters(m.log_scales.exp(), m.w0, torch.float64)
        y = fn(xx, m.plan, H, aa, bb)
        (y * gy).sum().backward()
        res.append([t.detach() for t in (y, xx.grad, aa.grad, bb.grad, m.log_scales.grad)])
    for p, q in zip(*res):
        torch.testing.assert_close(p, q, atol=1e-11 * q.abs().max().item(), rtol=0)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_token_mixers(device):
    torch.manual_seed(0)
    x = torch.randn(2, 300, 16, device=device, requires_grad=True)
    for mixer in (fdwt.MorletTokenMixer(16, freqs=[2.0, 8.0, 30.0], fs=250),
                  fdwt.MorletTokenMixer(16, use_phase=False),
                  fdwt.WaveletTokenMixer(16, "db4", hidden=32),
                  fdwt.WaveletTokenMixer(16, "sym4", mode="symmetric", level=3)):
        mixer = mixer.to(device)
        y = mixer(x)
        assert y.shape == x.shape
        y.square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        for p in mixer.parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all()
        x.grad = None


@cuda
def test_token_mixer_gpu_matches_cpu():
    torch.manual_seed(0)
    mixer = fdwt.MorletTokenMixer(8, freqs=np.geomspace(1, 40, 6).tolist(), fs=250).double()
    x = torch.randn(3, 700, 8, dtype=torch.float64)
    outs = []
    for device in ["cpu", "cuda"]:
        mixer.zero_grad()
        mixer = mixer.to(device)
        xx = x.detach().to(device).requires_grad_()
        y = mixer(xx)
        y.pow(2).sum().backward()
        outs.append([t.detach().cpu() for t in (y, xx.grad, mixer.mag_weight.grad,
                                                mixer.cwt.log_scales.grad)])
    for p, q in zip(*outs):
        torch.testing.assert_close(p, q, atol=1e-10 * q.abs().max().item(), rtol=0)


def _channel_mix_reference(m, x):
    """Literal MorletChannelMixAttention, with the explicit C x C map."""
    B, H, C, T, D = x.shape
    w = m.cwt(x.reshape(B, H, C, T * D))                       # (B, H, C, f, N)
    p = torch.softmax(w.abs() / m.temperature, dim=-2)
    p = p.permute(0, 1, 4, 2, 3)                               # (B, H, N, C, f)
    a = p @ m.freq_to_channel.to(p.dtype)                      # (B, H, N, C, C)
    v = x.reshape(B, H, C, T * D).permute(0, 1, 3, 2)[..., None].to(p.dtype)
    out = (a @ v)[..., 0]                                      # (B, H, N, C)
    return out.permute(0, 1, 3, 2).reshape(B, H, C, T, D)


def _attention_reference(m, x):
    """Literal MorletAttention: softmax over f, then a (C, f, 1) matrix per channel."""
    B, H, C, T, D = x.shape
    w = m.cwt(x.reshape(B, H, C, T * D))                       # (B, H, C, f, N)
    p = torch.softmax(w.abs() / m.temperature, dim=-2)
    p = p.permute(0, 1, 4, 2, 3)                               # (B, H, N, C, f)
    out = (p.unsqueeze(-2) @ m.freq_weight[..., None].to(p.dtype))[..., 0, 0]   # (B, H, N, C)
    return out.permute(0, 1, 3, 2).reshape(B, H, C, T, D)


ATTENTIONS = [
    (fdwt.MorletAttention, _attention_reference, "freq_weight"),
    (fdwt.MorletChannelMixAttention, _channel_mix_reference, "freq_to_channel"),
]
DEVICES_ = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES_)
@pytest.mark.parametrize("cls,ref,weight", ATTENTIONS)
def test_attention_matches_definition(device, cls, ref, weight, monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(_SpectralSoftmaxFunction, "CHUNK_BYTES", 7 * 3 * 16 * 384)  # many chunks
    m = cls(num_channels=5, hidden_dim=64, temperature=0.5).to(device).double()
    x = torch.randn(2, 3, 5, 6, 64, dtype=torch.float64, device=device)
    res = []
    for fn in (m, lambda t: ref(m, t)):
        m.zero_grad()
        xx = x.clone().requires_grad_()
        y = fn(xx)
        assert y.shape == x.shape and y.dtype == x.dtype
        y.square().sum().backward()
        res.append([y.detach(), xx.grad, getattr(m, weight).grad, m.cwt.log_scales.grad,
                    m.cwt.w0.grad])
    for p, q in zip(*res):
        torch.testing.assert_close(p, q, atol=1e-10 * q.abs().max().item(), rtol=0)


@cuda
@pytest.mark.parametrize("cls,ref,weight", ATTENTIONS)
def test_attention_gpu_matches_cpu(cls, ref, weight):
    torch.manual_seed(0)
    m = cls(num_channels=4, hidden_dim=128, freqs=[2.0, 6.0, 12.0, 25.0])
    x = torch.randn(2, 2, 4, 5, 128)
    outs = []
    for device in ["cpu", "cuda"]:
        m = m.to(device).double()
        m.zero_grad()
        xx = x.double().to(device).requires_grad_()
        y = m(xx)
        y.square().mean().backward()
        outs.append([t.detach().cpu() for t in (y, xx.grad, getattr(m, weight).grad,
                                                m.cwt.log_scales.grad, m.cwt.w0.grad)])
    for p, q in zip(*outs):
        torch.testing.assert_close(p, q, atol=1e-10 * q.abs().max().item(), rtol=0)
    m = m.float()
    for dt in (torch.float32, torch.bfloat16):
        y = m(x.cuda().to(dt))
        assert y.dtype == dt and torch.isfinite(y.float()).all()
    np.testing.assert_allclose(m.freqs.cpu().numpy(), [2.0, 6.0, 12.0, 25.0], rtol=1e-5)


@cuda
@pytest.mark.parametrize("per_position", [False, True])
def test_spectral_softmax_gradcheck(per_position):
    torch.manual_seed(0)
    m = fdwt.CWT1D(scales=[1.5, 3.0, 7.0, 13.0, 26.0], learnable=True, learnable_w0=True)
    m = m.cuda().double()
    nch, R, N = 3, 6, 80
    x = torch.randn(R, N, dtype=torch.float64, device="cuda", requires_grad=True)
    shape = (R // nch, 5, N) if per_position else (nch, 5)
    V = torch.randn(shape, dtype=torch.float64, device="cuda", requires_grad=True)
    ls = m.log_scales.detach().clone().requires_grad_()
    w0 = torch.tensor(6.0, dtype=torch.float64, device="cuda", requires_grad=True)
    f = lambda x, V, ls, w0: spectral_softmax(x, m.plan, m.plan.filters(ls.exp(), w0, torch.float64),
                                              V, nch, 0.7)
    assert gradcheck(f, (x, V, ls, w0))


@cuda
def test_spectral_softmax_large_amplitudes():
    # the online softmax must stay finite where a naive exp(|W| / tau) overflows
    torch.manual_seed(0)
    m = fdwt.MorletAttention(num_channels=2, hidden_dim=100, temperature=0.01).cuda()
    x = (1e3 * torch.randn(1, 1, 2, 10, 100, device="cuda")).requires_grad_()
    y = m(x)
    y.sum().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    ref = _attention_reference(m.cpu().double(), x.detach().cpu().double())
    torch.testing.assert_close(y.detach().cpu().double(), ref, atol=1e-4 * ref.abs().max().item(), rtol=0)


@pytest.mark.parametrize("cls", [fdwt.MorletAttention, fdwt.MorletChannelMixAttention])
def test_attention_errors(cls):
    m = cls(num_channels=3, hidden_dim=32)
    with pytest.raises(ValueError):
        m(torch.randn(2, 3, 4, 32))
    with pytest.raises(ValueError):
        m(torch.randn(1, 1, 4, 2, 32))
    with pytest.raises(ValueError):
        cls(num_channels=3, hidden_dim=32, freqs=[20.0])  # Nyquist = 16 Hz


def test_channel_mix_attention_map():
    m = fdwt.MorletChannelMixAttention(num_channels=5, hidden_dim=64)
    assert m.attention_map(torch.randn(2, 3, 5, 6, 64)).shape == (2, 3, 6 * 64, 5, 5)
