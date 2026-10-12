"""Throughput benchmarks: fdwt vs. reference PyTorch implementations.

    python benchmarks/bench.py [--quick]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import triton

import fdwt
from fdwt.cwt import MorletPlan, _torch_cwt


def bench(fn, backward=False):
    if backward:
        def run():
            out = fn()
            out = torch.view_as_real(out) if out.is_complex() else out
            out.backward(torch.ones_like(out))
    else:
        run = fn
    return triton.testing.do_bench(run, warmup=10, rep=50)


def gbps(nbytes, ms):
    return nbytes / ms / 1e6


def dwt_suite(rows, n):
    print(f"\n== DWT  ({rows} rows x {n} samples, fp32, wavedec to max level) ==")
    x = torch.randn(rows, n, device="cuda")
    xg = x.clone().requires_grad_()
    try:
        from pytorch_wavelets import DWT1DForward
    except ImportError:
        DWT1DForward = None
    print(f"{'wavelet':10s}{'mode':15s}{'fdwt fwd':>12s}{'fdwt f+b':>12s}{'ptwt fwd':>12s}{'ptwt f+b':>12s}{'GB/s fwd':>10s}")
    for w in ["haar", "db4", "sym8", "coif5"]:
        for mode in ["periodization", "symmetric", "zero"]:
            lvl = fdwt.dwt_max_level(n, w)
            t_f = bench(lambda: fdwt.wavedec(x, w, mode, lvl))
            t_b = bench(lambda: sum(c.sum() for c in fdwt.wavedec(xg, w, mode, lvl)), backward=True)
            nbytes = sum(c.numel() for c in fdwt.wavedec(x, w, mode, lvl)) * 4 + x.numel() * 4 * 2
            row = f"{w:10s}{mode:15s}{t_f:10.3f}ms{t_b:10.3f}ms"
            if DWT1DForward is not None:
                ptmode = {"periodization": "periodization", "symmetric": "symmetric", "zero": "zero"}[mode]
                m = DWT1DForward(J=lvl, wave=w, mode=ptmode).cuda()
                x3, x3g = x[:, None], xg[:, None]
                t_pf = bench(lambda: m(x3))
                t_pb = bench(lambda: (lambda yl, yh: yl.sum() + sum(h.sum() for h in yh))(*m(x3g)),
                             backward=True)
                row += f"{t_pf:10.3f}ms{t_pb:10.3f}ms"
            else:
                row += f"{'-':>12s}{'-':>12s}"
            print(row + f"{gbps(nbytes, t_f):10.0f}")


def cwt_suite(rows, n, n_freqs, fs=250.0):
    freqs = np.geomspace(0.5, 100, n_freqs)
    print(f"\n== Morlet CWT  ({rows} rows x {n} samples, {n_freqs} freqs 0.5-100 Hz @ {fs:g} Hz, fp32) ==")
    x = torch.randn(rows, n, device="cuda")
    xg = x.clone().requires_grad_()
    out_bytes = rows * n_freqs * n * 8
    print(f"output {out_bytes / 1e9:.2f} GB (complex64); "
          f"write-only roofline ~{out_bytes / 1e6 / 700:.2f} ms at 700 GB/s")
    print(f"{'method':22s}{'fwd':>12s}{'fwd+bwd':>12s}")
    for method in ["fast", "direct", "fft"]:
        t_f = bench(lambda: fdwt.cwt(x, freqs=freqs, fs=fs, method=method))
        t_b = bench(lambda: fdwt.cwt(xg, freqs=freqs, fs=fs, method=method), backward=True)
        print(f"{'fdwt ' + method:22s}{t_f:10.3f}ms{t_b:10.3f}ms")
    plan = MorletPlan(fdwt.freqs_to_scales(freqs, fs).tolist(), method="direct")
    h = plan.filters(torch.tensor(plan.scales, device="cuda", dtype=torch.float64), 6.0, torch.float32)
    t_f = bench(lambda: _torch_cwt(x, plan, h))
    t_b = bench(lambda: torch.view_as_complex(_torch_cwt(xg, plan, h)), backward=True)
    print(f"{'torch conv1d (direct)':22s}{t_f:10.3f}ms{t_b:10.3f}ms")
    m = fdwt.CWT1D(freqs=freqs, fs=fs, learnable=True).cuda()
    t_b = bench(lambda: m(xg), backward=True)
    print(f"{'fdwt fast, learnable':22s}{'':>12s}{t_b:10.3f}ms")


def mixer_suite(batch=8, dim=64, lengths=(1024, 4096, 16384, 65536)):
    print(f"\n== Token mixers vs attention  (batch {batch}, dim {dim}, fp32, fwd+bwd) ==")
    print(f"{'L':>7s}{'Morlet (16 sc.)':>18s}{'DWT db4 (L=4)':>16s}{'SDPA attention':>16s}")
    morlet = fdwt.MorletTokenMixer(dim, scales=np.geomspace(2, 256, 16).tolist()).cuda()
    dwtm = fdwt.WaveletTokenMixer(dim, "db4", level=4).cuda()
    attn = torch.nn.MultiheadAttention(dim, 4, batch_first=True).cuda()

    def cell(f, x):
        try:
            torch.cuda.reset_peak_memory_stats()
            t = bench(lambda: f(x), backward=True)
            mem = torch.cuda.max_memory_allocated() / 2**20
            return f"{t:7.2f}ms {mem:5.0f}MiB"
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return f"{'OOM':>16s}"

    for L in lengths:
        x = torch.randn(batch, L, dim, device="cuda", requires_grad=True)
        row = f"{L:7d}" + cell(morlet, x) + cell(dwtm, x)
        row += cell(lambda x: attn(x, x, x, need_weights=False)[0], x)
        print(row)
        del x


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    print(torch.cuda.get_device_name())
    if a.quick:
        dwt_suite(512, 4096)
        cwt_suite(256, 2500, 32)
        mixer_suite(lengths=(1024, 8192))
    else:
        dwt_suite(2048, 4096)
        dwt_suite(64, 2 ** 18)
        cwt_suite(512, 2500, 40)
        cwt_suite(64, 15000, 40)
        mixer_suite()
