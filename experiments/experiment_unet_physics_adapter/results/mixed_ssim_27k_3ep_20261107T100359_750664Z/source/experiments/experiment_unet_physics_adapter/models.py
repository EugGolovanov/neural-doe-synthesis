"""U-Net с полноразмерным физическим адаптером A → GS + Muon → B."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from experiments.experiment_gs_muon.physics import fresnel_factors
from .algorithms import gs_muon


def convolution_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Создаёт две свёртки с нулевым padding признаков и гладкими активациями."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.SiLU(),
    )


class PhysicsAdapter(nn.Module):
    """Смешивает основной путь W и низкоранговую физическую ветку B S(Ah)."""

    def __init__(self, channels: int, rank: int, alpha: float, steps: int,
                 loss_epsilon: float, muon: dict):
        """Инициализирует матрицы смешивания и неизменные параметры солвера."""
        super().__init__()
        if rank < 1 or steps < 1:
            raise ValueError('rank и steps должны быть положительными.')
        self.base = nn.Conv2d(channels, channels, 1, bias=False)
        self.A = nn.Conv2d(channels, rank, 1, bias=False)
        self.B = nn.Conv2d(rank, channels, 1, bias=False)
        nn.init.dirac_(self.base.weight)
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / rank
        self.steps = steps
        self.loss_epsilon = loss_epsilon
        self.muon = dict(muon)
        self.activation_checkpointing = False

    def forward(self, features: torch.Tensor, target: torch.Tensor,
                factors: tuple) -> torch.Tensor:
        """Решает физическую задачу для каждой латентной фазы и смешивает каналы."""
        phases = self.A(features)
        solved = []
        for batch_index in range(target.shape[0]):
            maps = []
            for phi in phases[batch_index]:
                if self.activation_checkpointing:
                    solved_phi = checkpoint(gs_muon, phi, target[batch_index], factors,
                                            self.steps, self.loss_epsilon, self.muon,
                                            use_reentrant=False)
                else:
                    solved_phi = gs_muon(phi, target[batch_index], factors, self.steps,
                                         self.loss_epsilon, self.muon)
                maps.append(solved_phi)
            solved.append(torch.stack(maps))
        physical_features = torch.stack(solved)
        return self.base(features) + self.scale * self.B(physical_features)


class PhysicsAdapterUNet(nn.Module):
    """Предсказывает общую фазу с GS + Muon внутри адаптера декодера U-Net."""

    def __init__(self, width: int, rank: int, alpha: float, steps: int,
                 loss_epsilon: float, muon: dict):
        """Строит двухуровневый U-Net и физическую ветку на сетке ДОЭ."""
        super().__init__()
        self.encoder = convolution_block(4, width)
        self.middle = convolution_block(width, 2 * width)
        self.bottom = convolution_block(2 * width, 4 * width)
        self.decoder_middle = convolution_block(6 * width, 2 * width)
        self.decoder = convolution_block(3 * width, width)
        self.adapter = PhysicsAdapter(width, rank, alpha, steps, loss_epsilon, muon)
        self.head = nn.Conv2d(width, 1, 1, bias=False)
        self.activation_checkpointing = False

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        """Возвращает непрерывную фазу [batch,128,128] из интенсивностей [batch,4,128,128]."""
        if target.ndim != 4 or target.shape[1:] != (4, 128, 128):
            raise ValueError('Ожидается target [batch,4,128,128].')
        if not torch.isfinite(target).all() or torch.any((target < 0) | (target > 1)):
            raise ValueError('Целевые интенсивности должны быть конечными в [0,1].')
        if torch.any(target.sum(dim=(-2, -1)) == 0):
            raise ValueError('GS не определён для полностью чёрной целевой плоскости.')
        if torch.is_inference_mode_enabled():
            raise ValueError('Используйте torch.no_grad(), а не inference_mode: Muon требует autograd.')
        outer_grad = torch.is_grad_enabled()

        def block(module: nn.Module, values: torch.Tensor) -> torch.Tensor:
            """Evaluate a convolution block with optional exact activation recomputation."""
            if self.activation_checkpointing and outer_grad:
                return checkpoint(module, values, use_reentrant=False)
            return module(values)

        with torch.enable_grad():
            factors = tuple(value.to(target.device) for value in fresnel_factors(target.dtype))
            first = block(self.encoder, target)
            second = block(self.middle, F.avg_pool2d(first, 2))
            bottom = block(self.bottom, F.avg_pool2d(second, 2))
            up = F.interpolate(bottom, scale_factor=2, mode='nearest')
            middle = block(self.decoder_middle, torch.cat((second, up), dim=1))
            up = F.interpolate(middle, scale_factor=2, mode='nearest')
            features = block(self.decoder, torch.cat((first, up), dim=1))
            features = self.adapter(features, target, factors)
            phi = self.head(features)[:, 0]
        return phi if outer_grad else phi.detach()
