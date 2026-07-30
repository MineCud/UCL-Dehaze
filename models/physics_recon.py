"""Physical reconstruction closed-loop for atmospheric scattering.

I = J * t + A * (1 - t)

Training uses a lightweight TransmissionNet to predict t from the hazy image,
estimates airlight A, then enforces:
  1) forward recon:  I_hat = J*t + A*(1-t)  ≈ I
  2) inverse consistency: J_phy = (I-A)/t + A  ≈ J   (optional)
  3) chromaticity fidelity: chroma(J) ≈ chroma(I)     (optional)

Inference still only needs netG; TransmissionNet is training-only.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_01(x: torch.Tensor) -> torch.Tensor:
    """Map UCL tensors [-1,1] -> [0,1]."""
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def to_norm(x: torch.Tensor) -> torch.Tensor:
    """Map [0,1] -> [-1,1]."""
    return (x * 2.0 - 1.0).clamp(-1.0, 1.0)


def estimate_airlight(I: torch.Tensor, ratio: float = 0.001) -> torch.Tensor:
    """Differentiable airlight from brightest pixels (B,3,1,1), I in [0,1]."""
    b, c, h, w = I.shape
    flat = I.reshape(b, c, -1)
    lum = I.mean(dim=1).reshape(b, -1)
    k = max(1, int(ratio * h * w))
    _, idx = torch.topk(lum, k=k, dim=1)
    A = []
    for i in range(b):
        A.append(flat[i, :, idx[i]].mean(dim=1))
    return torch.stack(A, dim=0).view(b, c, 1, 1).clamp(0.0, 1.0)


def dark_channel_t(
    I: torch.Tensor,
    A: torch.Tensor,
    omega: float = 0.95,
    t_min: float = 0.1,
    pool: int = 5,
) -> torch.Tensor:
    """Heuristic transmission (no grad needed for soft prior target)."""
    ratio = (I / (A + 1e-6)).clamp(0.0, 1.0)
    dark = ratio.min(dim=1, keepdim=True).values
    pad = pool // 2
    dark = -F.max_pool2d(-dark, kernel_size=pool, stride=1, padding=pad)
    return (1.0 - omega * dark).clamp(t_min, 1.0)


def chromaticity(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-pixel chromaticity x / sum(x), shape (B,3,H,W)."""
    s = x.sum(dim=1, keepdim=True).clamp_min(eps)
    return x / s


class TransmissionNet(nn.Module):
    """Lightweight CNN: hazy RGB [-1,1] -> transmission map in [t_min, 1]."""

    def __init__(self, in_ch: int = 3, base: int = 32, t_min: float = 0.1):
        super().__init__()
        self.t_min = float(t_min)
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 1, 3, 1, 1),
        )

    def forward(self, hazy: torch.Tensor) -> torch.Tensor:
        h, w = hazy.shape[-2:]
        feat = self.enc(hazy)
        feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)
        raw = torch.sigmoid(self.dec(feat))
        # Map (0,1) -> [t_min, 1]
        t = self.t_min + (1.0 - self.t_min) * raw
        return t


def define_T(opt, gpu_ids=None):
    """Build and init TransmissionNet."""
    from . import networks

    t_min = float(getattr(opt, "phys_t_min", 0.1))
    base = int(getattr(opt, "phys_t_nf", 32))
    net = TransmissionNet(in_ch=opt.input_nc, base=base, t_min=t_min)
    return networks.init_net(
        net,
        init_type=getattr(opt, "init_type", "normal"),
        init_gain=getattr(opt, "init_gain", 0.02),
        gpu_ids=gpu_ids if gpu_ids is not None else getattr(opt, "gpu_ids", []),
    )


def physical_closed_loop_loss(
    hazy: torch.Tensor,
    dehazed: torch.Tensor,
    t: torch.Tensor,
    *,
    omega: float = 0.95,
    t_min: float = 0.1,
    recon_w: float = 1.0,
    inv_w: float = 0.5,
    chroma_w: float = 0.5,
    t_prior_w: float = 0.1,
    airlight_ratio: float = 0.001,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Full physical closed-loop loss.

    Args:
        hazy, dehazed: tensors in [-1, 1]
        t: predicted transmission (B,1,H,W) in [t_min, 1]
    Returns:
        total loss, dict of unweighted components (for logging)
    """
    I = to_01(hazy)
    J = to_01(dehazed)
    if t.shape[-2:] != I.shape[-2:]:
        t = F.interpolate(t, size=I.shape[-2:], mode="bilinear", align_corners=False)
    t = t.clamp(t_min, 1.0)

    A = estimate_airlight(I, ratio=airlight_ratio)

    # 1) Forward reconstruction: synthesize haze from clear + t + A
    I_hat = J * t + A * (1.0 - t)
    loss_recon = F.l1_loss(I_hat, I)

    # 2) Inverse consistency: physics-derived clear should match G(I)
    J_phy = ((I - A) / t.clamp_min(t_min) + A).clamp(0.0, 1.0)
    loss_inv = F.l1_loss(J_phy, J)

    # 3) Chromaticity fidelity: discourage semantic-neighbor color hijack
    loss_chroma = F.l1_loss(chromaticity(J), chromaticity(I))

    # 4) Soft DCP prior on t (stabilize early training; stop-grad target)
    with torch.no_grad():
        t_dcp = dark_channel_t(I, A, omega=omega, t_min=t_min)
    loss_t_prior = F.l1_loss(t, t_dcp)

    total = (
        recon_w * loss_recon
        + inv_w * loss_inv
        + chroma_w * loss_chroma
        + t_prior_w * loss_t_prior
    )
    parts = {
        "phys_recon": loss_recon.detach(),
        "phys_inv": loss_inv.detach(),
        "phys_chroma": loss_chroma.detach(),
        "phys_t_prior": loss_t_prior.detach(),
        "t_map": t.detach(),
        "A": A.detach(),
        "I_hat": I_hat.detach(),
    }
    return total, parts
