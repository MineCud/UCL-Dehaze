"""CUDA-free 2D Mamba-style blocks for UCL-Dehaze.

Replaces SCBottleneck with a CNN local branch + bidirectional axial SSM branch.
No mamba-ssm / custom CUDA dependency.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BiSSM2D(nn.Module):
    """Mamba-inspired 2D block: pointwise gate + DWConv + bidirectional axial mixing.

    Axial Conv1d scans along H/W approximate long-range 1D state mixing without
    unstable selective-scan cumsums, and stay shape-compatible with SCBottleneck.
    """

    def __init__(self, channels: int, kernel_size: int = 11):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        padding = kernel_size // 2

        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.in_proj = nn.Conv2d(channels, channels * 2, 1, bias=False)
        self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.scan_w = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, groups=channels, bias=False
        )
        self.scan_h = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, groups=channels, bias=False
        )
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.act = nn.SiLU()

    def _scan_w(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        t = x.permute(0, 2, 1, 3).reshape(b * h, c, w)
        y = self.scan_w(t) + self.scan_w(torch.flip(t, dims=[-1])).flip(dims=[-1])
        return y.reshape(b, h, c, w).permute(0, 2, 1, 3)

    def _scan_h(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        t = x.permute(0, 3, 1, 2).reshape(b * w, c, h)
        y = self.scan_h(t) + self.scan_h(torch.flip(t, dims=[-1])).flip(dims=[-1])
        return y.reshape(b, w, c, h).permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        u, g = self.in_proj(x).chunk(2, dim=1)
        u = self.act(self.dw(u)) * torch.sigmoid(g)
        y = self._scan_w(u) + self._scan_h(u)
        y = self.out_proj(self.act(y))
        return y + residual


class MambaBottleneck(nn.Module):
    """Drop-in replacement for SCBottleneck: local conv branch + BiSSM2D branch."""

    def __init__(self, in_planes: int, planes: int):
        super().__init__()
        mid = int(planes / 2)

        self.conv1_a = nn.Conv2d(in_planes, mid, 1, 1)
        self.k1 = nn.Sequential(
            nn.Conv2d(mid, mid, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

        self.conv1_b = nn.Conv2d(in_planes, mid, 1, 1)
        self.ssm = BiSSM2D(mid)

        self.conv3 = nn.Conv2d(mid * 2, mid * 2, 1, 1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out_a = self.relu(self.conv1_a(x))
        out_a = self.k1(out_a)

        out_b = self.relu(self.conv1_b(x))
        out_b = self.ssm(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.relu(out + residual)
        return out
