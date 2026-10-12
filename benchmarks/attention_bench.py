"""MorletAttention / MorletChannelMixAttention vs a FlashAttention block.

Input (B, H, C, T, D) = (8, 4, 19, T, 200); the attention block treats the C*T patches as
tokens with head dimension D.  For each precision mode of fdwt.config:

* fwd / fwd+bwd time,
* "saved": activation memory held between forward and backward (what accumulates
  across the layers of a deep network),
* "peak": extra peak memory during the whole step.

    python benchmarks/attention_bench.py [--T 30 120 480]
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
import triton
from torch.nn.attention import SDPBackend, sdpa_kernel

import fdwt

B, H, C, D = 8, 4, 19, 200


class FlashBlock(torch.nn.Module):
    """Q/K/V/out projections + FlashAttention over the C*T patch tokens."""

    def __init__(self):
        super().__init__()
        self.qkv, self.o = torch.nn.Linear(D, 3 * D), torch.nn.Linear(D, D)

    def forward(self, x):
        b, h, c, t, d = x.shape
        q, k, v = self.qkv(x.reshape(b, h, c * t, d)).chunk(3, -1)
        pad = lambda z: F.pad(z, (0, 256 - d))  # FlashAttention can't train head_dim 200 on sm86-89
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            o = F.scaled_dot_product_attention(pad(q), pad(k), pad(v), scale=d ** -0.5)[..., :d]
        return self.o(o).reshape(x.shape)


def measure(f, x):
    def step():
        f(x).float().square().mean().backward()

    step()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    y = f(x)
    saved = (torch.cuda.memory_allocated() - base - y.numel() * y.element_size()) / 2**20
    del y
    torch.cuda.reset_peak_memory_stats()
    step()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    return triton.testing.do_bench(lambda: f(x)), triton.testing.do_bench(step), saved, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, nargs="+", default=[30, 120, 480])
    a = ap.parse_args()
    print(torch.cuda.get_device_name())
    print(f"{'T':>4s}  {'module':27s} {'mode':5s} {'fwd':>8s} {'fwd+bwd':>9s} {'saved':>9s} {'peak':>9s}")
    for T in a.T:
        x32 = torch.randn(B, H, C, T, D, device="cuda", requires_grad=True)
        for cls in (fdwt.MorletAttention, fdwt.MorletChannelMixAttention):
            m = cls(C, D).cuda()
            for mode in ("ieee", "tf32", "bf16"):
                fdwt.config.allow_tf32, fdwt.config.allow_bf16 = mode == "tf32", mode == "bf16"
                tf, tb, saved, peak = measure(m, x32)
                print(f"{T:4d}  {cls.__name__:27s} {mode:5s} {tf:6.2f}ms {tb:7.2f}ms "
                      f"{saved:6.0f}MiB {peak:6.0f}MiB", flush=True)
            fdwt.config.allow_tf32 = fdwt.config.allow_bf16 = False
        x16 = x32.detach().bfloat16().requires_grad_()
        tf, tb, saved, peak = measure(FlashBlock().cuda().bfloat16(), x16)
        print(f"{T:4d}  {'FlashAttention block':27s} {'bf16':5s} {tf:6.2f}ms {tb:7.2f}ms "
              f"{saved:6.0f}MiB {peak:6.0f}MiB", flush=True)


if __name__ == "__main__":
    main()
