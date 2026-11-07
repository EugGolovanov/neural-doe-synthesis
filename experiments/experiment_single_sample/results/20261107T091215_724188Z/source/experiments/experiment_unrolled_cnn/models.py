"""Общая компактная CNN-коррекция для физической развёртки."""

import torch
from torch import nn


class CorrectionNet(nn.Module):
    """Кодирует четыре цели и предсказывает остаточную фазовую коррекцию."""

    def __init__(self, width: int):
        """Создаёт свёртки без смещений с нулевой выходной головой."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, width, 3, padding=1, bias=False), nn.SiLU(),
        )
        self.correction = nn.Sequential(
            nn.Conv2d(width + 3, width, 3, padding=1, bias=False), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False), nn.SiLU(),
            nn.Conv2d(width, 1, 3, padding=1, bias=False),
        )
        nn.init.zeros_(self.correction[-1].weight)

    def forward(self, v: torch.Tensor, gradient: torch.Tensor,
                context: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Возвращает ограниченную коррекцию из периодических каналов и RMS-градиента."""
        rms = (gradient.square().mean(dim=(-2, -1), keepdim=True) + epsilon**2).sqrt()
        state = torch.cat((v.cos(), v.sin(), gradient / rms, context), dim=1)
        return torch.tanh(self.correction(state))
