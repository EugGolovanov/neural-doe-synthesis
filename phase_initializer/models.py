"""Checkpoint-compatible U-Net used only for two-step initialization."""

import torch
from torch import nn
from torch.nn import functional as F

from .algorithms import gs_muon
from .physics import fresnel_factors


def convolution_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Build the checkpoint's two bias-free convolutions with SiLU activations."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.SiLU(),
    )


class PhysicsAdapter(nn.Module):
    """Mix decoder features with two GS+Muon steps on each latent phase."""

    def __init__(self, channels: int, rank: int, alpha: float, epsilon: float, muon: dict):
        """Create the exact parameter layout expected by the trained checkpoint."""
        super().__init__()
        self.base = nn.Conv2d(channels, channels, 1, bias=False)
        self.A = nn.Conv2d(channels, rank, 1, bias=False)
        self.B = nn.Conv2d(rank, channels, 1, bias=False)
        self.scale = alpha / rank
        self.epsilon = epsilon
        self.muon = muon

    def forward(self, features: torch.Tensor, target: torch.Tensor, factors: tuple) -> torch.Tensor:
        """Solve each latent phase with fresh momentum and combine the resulting features."""
        phases = self.A(features)
        solved = []
        for batch_index in range(target.shape[0]):
            maps = [gs_muon(phi, target[batch_index], factors, 2, self.epsilon, self.muon)
                    for phi in phases[batch_index]]
            solved.append(torch.stack(maps))
        return self.base(features) + self.scale * self.B(torch.stack(solved))


class PhysicsAdapterUNet(nn.Module):
    """Map four target intensity channels to one initial phase without training weights."""

    def __init__(self, config: dict):
        """Construct the original two-level U-Net using the checkpoint dimensions."""
        super().__init__()
        width = config['width']
        self.encoder = convolution_block(4, width)
        self.middle = convolution_block(width, 2 * width)
        self.bottom = convolution_block(2 * width, 4 * width)
        self.decoder_middle = convolution_block(6 * width, 2 * width)
        self.decoder = convolution_block(3 * width, width)
        self.adapter = PhysicsAdapter(width, config['rank'], config['alpha'],
                                      config['loss_epsilon'], config['phase_muon'])
        self.head = nn.Conv2d(width, 1, 1, bias=False)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        """Return detached phase [B,128,128] while retaining the adapter's required autograd."""
        with torch.enable_grad():
            factors = fresnel_factors(target.dtype)
            first = self.encoder(target)
            second = self.middle(F.avg_pool2d(first, 2))
            bottom = self.bottom(F.avg_pool2d(second, 2))
            up = F.interpolate(bottom, scale_factor=2, mode='nearest')
            middle = self.decoder_middle(torch.cat((second, up), dim=1))
            up = F.interpolate(middle, scale_factor=2, mode='nearest')
            features = self.decoder(torch.cat((first, up), dim=1))
            features = self.adapter(features, target, factors)
            phi = self.head(features)[:, 0]
        return phi.detach()
