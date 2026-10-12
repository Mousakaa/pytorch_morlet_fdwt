"""Train a tiny transformer whose attention is replaced by a wavelet token mixer.

Synthetic "biosignal" task: each sequence is 1/f noise plus a transient burst whose
oscillation frequency (6, 10, 18 or 30 Hz) is the class.  The burst position is random,
so the model has to look for a localised time-frequency pattern.

    python examples/token_mixer_demo.py [--mixer morlet|dwt|attention] [--steps 300]
"""

from __future__ import annotations

import argparse
import math
import time

import torch
from torch import nn

import fdwt

FS = 250.0
CLASS_FREQS = [6.0, 10.0, 18.0, 30.0]


def make_batch(batch: int, length: int, device):
    t = torch.arange(length, device=device) / FS
    y = torch.randint(len(CLASS_FREQS), (batch,), device=device)
    f = torch.tensor(CLASS_FREQS, device=device)[y]
    center = torch.rand(batch, device=device) * (t[-1] - 1.0) + 0.5
    env = torch.exp(-0.5 * ((t[None] - center[:, None]) / 0.25) ** 2)
    burst = env * torch.sin(2 * math.pi * (f[:, None] * t[None] + torch.rand(batch, 1, device=device)))
    pink = torch.cumsum(torch.randn(batch, length, device=device), -1)
    pink = (pink - pink.mean(-1, keepdim=True)) / pink.std(-1, keepdim=True)
    return 0.6 * burst + 0.5 * pink + 0.3 * torch.randn(batch, length, device=device), y


class Block(nn.Module):
    def __init__(self, dim: int, mixer: str):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        if mixer == "morlet":
            self.mix = fdwt.MorletTokenMixer(dim, freqs=torch.logspace(0, math.log10(60), 16).tolist(), fs=FS)
        elif mixer == "dwt":
            self.mix = fdwt.WaveletTokenMixer(dim, "db4", hidden=dim)
        else:
            self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
            self.mix = lambda x: self.attn(x, x, x, need_weights=False)[0]
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        return x + self.ffn(self.n2(x))


class Model(nn.Module):
    def __init__(self, dim: int, depth: int, mixer: str):
        super().__init__()
        self.embed = nn.Linear(1, dim)
        self.blocks = nn.Sequential(*(Block(dim, mixer) for _ in range(depth)))
        self.head = nn.Linear(dim, len(CLASS_FREQS))

    def forward(self, x):
        h = self.blocks(self.embed(x[..., None]))
        return self.head(h.mean(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixer", default="morlet", choices=["morlet", "dwt", "attention"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--length", type=int, default=1000)
    ap.add_argument("--dim", type=int, default=32)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = Model(a.dim, 2, a.mixer).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    warmup = 20  # excluded from the timing: first calls JIT-compile the Triton kernels
    for step in range(1, a.steps + 1):
        if step == warmup + 1:
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        x, y = make_batch(32, a.length, dev)
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss.item():.3f}")
    if dev == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else 0
    model.eval()
    with torch.no_grad():
        correct = 0
        for _ in range(8):
            x, y = make_batch(250, a.length, dev)
            correct += (model(x).argmax(-1) == y).sum().item()
        acc = correct / 2000
    print(f"{a.mixer}: test accuracy {acc:.3f}  ({elapsed / (a.steps - warmup) * 1e3:.1f} ms/step, "
          f"peak training mem {peak:.0f} MiB)" if dev == "cuda"
          else f"{a.mixer}: test accuracy {acc:.3f}")
    if a.mixer == "morlet":
        print("learned centre frequencies (Hz):",
              " ".join(f"{f:.1f}" for f in model.blocks[0].mix.cwt.freqs.tolist()))


if __name__ == "__main__":
    main()
