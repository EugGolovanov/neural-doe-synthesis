"""Сменные коррекции U-Net, FNO и axial selective SSM для общей развёртки."""

import torch
from torch import nn
from torch.nn import functional as F


def state_channels(v, gradient, context, epsilon):
    """Объединяет периодическую фазу, RMS-градиент и условные признаки целей."""
    rms = (gradient.square().mean((-2, -1), keepdim=True) + epsilon**2).sqrt()
    return torch.cat((v.cos(), v.sin(), gradient / rms, context), dim=1)


class UNetCorrection(nn.Module):
    """Использует двухуровневую U-Net с одной skip-связью."""

    def __init__(self, width):
        """Создаёт свёртки без bias и нулевую выходную голову."""
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(4, width, 3, padding=1, bias=False), nn.SiLU())
        self.high = nn.Sequential(nn.Conv2d(width + 3, width, 3, padding=1, bias=False), nn.SiLU())
        self.low = nn.Sequential(nn.Conv2d(width, 2 * width, 3, stride=2, padding=1, bias=False),
                                 nn.SiLU(), nn.Conv2d(2 * width, 2 * width, 3, padding=1, bias=False), nn.SiLU())
        self.decode = nn.Sequential(nn.Conv2d(3 * width, width, 3, padding=1, bias=False), nn.SiLU())
        self.head = nn.Conv2d(width, 1, 3, padding=1, bias=False)
        nn.init.zeros_(self.head.weight)

    def forward(self, v, gradient, context, epsilon):
        """Декодирует коррекцию после bilinear upsampling признаков без изменения физической сетки."""
        high = self.high(state_channels(v, gradient, context, epsilon))
        low = F.interpolate(self.low(high), size=high.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(self.decode(torch.cat((high, low), dim=1))).tanh()


class SpectralLayer(nn.Module):
    """Смешивает каналы в усечённом rFFT-спектре с вещественным выходом."""

    def __init__(self, width, modes):
        """Хранит комплексные веса как пары вещественных компонент для Muon."""
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(width, width, 2 * modes - 1, modes, 2) / width)
        self.local = nn.Conv2d(width, width, 1, bias=False)

    def forward(self, x):
        """Проецирует нулевой столбец на эрмитову симметрию перед обратной rFFT."""
        height, width = x.shape[-2:]
        m = self.modes
        if 2 * m - 1 > height or m > width // 2:
            raise ValueError('Моды должны быть неперекрывающимися и исключать Nyquist-столбец.')
        rows = torch.cat((torch.arange(m, device=x.device), torch.arange(height - m + 1, height, device=x.device)))
        spectrum = torch.fft.rfft2(x, norm='ortho')
        selected = spectrum[:, :, rows, :m]
        mixed = torch.einsum('bihw,oihw->bohw', selected, torch.view_as_complex(self.weight.contiguous()))
        output = torch.zeros_like(spectrum)
        output[:, :, rows, :m] = mixed
        reverse = (-torch.arange(height, device=x.device)) % height
        column = output[..., 0]
        symmetric = (column + column[:, :, reverse].conj()) / 2
        output = torch.cat((symmetric.unsqueeze(-1), output[..., 1:]), dim=-1)
        return F.gelu(torch.fft.irfft2(output, s=(height, width), norm='ortho') + self.local(x))


class FNOCorrection(nn.Module):
    """Применяет FNO к состоянию на каждом шаге физической развёртки."""

    def __init__(self, width, modes, blocks):
        """Создаёт спектральные блоки без padding и без координатных каналов."""
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(4, width, 3, padding=1, bias=False), nn.SiLU())
        self.lift = nn.Conv2d(width + 3, width, 1, bias=False)
        self.blocks = nn.Sequential(*(SpectralLayer(width, modes) for _ in range(blocks)))
        self.head = nn.Conv2d(width, 1, 1, bias=False)
        nn.init.zeros_(self.head.weight)

    def forward(self, v, gradient, context, epsilon):
        """Возвращает ограниченную спектральную коррекцию фазы."""
        state = state_channels(v, gradient, context, epsilon)
        return self.head(self.blocks(self.lift(state))).tanh()


class SelectiveScan(nn.Module):
    """Вычисляет входозависимую диагональную SSM с последовательной рекурсией."""

    def __init__(self, width, states):
        """Задаёт отрицательный спектр A и входозависимые delta, B и C без bias."""
        super().__init__()
        self.log_a = nn.Parameter(torch.arange(1, states + 1).float().log().repeat(width, 1))
        self.skip = nn.Parameter(torch.ones(width, 1))
        self.delta = nn.Linear(width, width, bias=False)
        self.bc = nn.Linear(width, 2 * states, bias=False)

    def forward(self, u):
        """Сканирует [batch,length,channels] с нулевым начальным состоянием."""
        delta = F.softplus(self.delta(u) - 3.0)
        b, c = self.bc(u).chunk(2, dim=-1)
        a = -self.log_a.exp()
        hidden = u.new_zeros(u.shape[0], *a.shape)
        outputs = []
        for t in range(u.shape[1]):
            dt = delta[:, t, :, None]
            hidden = torch.exp(dt * a) * hidden + dt * b[:, t, None, :] * u[:, t, :, None]
            outputs.append((hidden * c[:, t, None, :]).sum(-1) + self.skip[:, 0] * u[:, t])
        return torch.stack(outputs, dim=1)


class AxialMambaBlock(nn.Module):
    """Сочетает четыре осевых selective scan с локальной свёрткой и SiLU-gate."""

    def __init__(self, width, states):
        """Создаёт общий SSM для четырёх направлений без fused CUDA kernels."""
        super().__init__()
        self.input = nn.Conv2d(width, 2 * width, 1, bias=False)
        self.local = nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False)
        self.scan = SelectiveScan(width, states)
        self.output = nn.Conv2d(width, width, 1, bias=False)

    def forward(self, x):
        """Усредняет независимые проходы вдоль строк и столбцов в обоих направлениях."""
        u, gate = self.input(x).chunk(2, dim=1)
        u = F.silu(self.local(u))
        batch, channels, height, width = u.shape
        rows = u.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
        columns = u.permute(0, 3, 2, 1).reshape(batch * width, height, channels)
        horizontal = self.scan(rows) + self.scan(rows.flip(1)).flip(1)
        vertical = self.scan(columns) + self.scan(columns.flip(1)).flip(1)
        horizontal = horizontal.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        vertical = vertical.reshape(batch, width, height, channels).permute(0, 3, 2, 1)
        return x + self.output((horizontal + vertical) * 0.25 * F.silu(gate))


class MambaCorrection(nn.Module):
    """Реализует исследовательскую axial Mamba-like коррекцию, не Vim или HoloMamba."""

    def __init__(self, width, states, blocks):
        """Создаёт осевые SSM-блоки с нулевой выходной головой."""
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(4, width, 3, padding=1, bias=False), nn.SiLU())
        self.lift = nn.Conv2d(width + 3, width, 1, bias=False)
        self.blocks = nn.Sequential(*(AxialMambaBlock(width, states) for _ in range(blocks)))
        self.head = nn.Conv2d(width, 1, 1, bias=False)
        nn.init.zeros_(self.head.weight)

    def forward(self, v, gradient, context, epsilon):
        """Предсказывает коррекцию из периодического состояния и осевых SSM-признаков."""
        state = state_channels(v, gradient, context, epsilon)
        return self.head(self.blocks(self.lift(state))).tanh()
