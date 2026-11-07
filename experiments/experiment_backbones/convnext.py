"""ConvNeXt-Tiny для плотной коррекции фазы с матричным хранением affine-параметров Muon."""

import torch
from torch import nn
from torch.nn import functional as F

from .models import state_channels


class AffineConv(nn.Module):
    """Хранит bias свёртки как матрицу [out,1] для существующего Muon."""

    def __init__(self, inputs, outputs, kernel, stride=1, padding=0, groups=1):
        """Инициализирует ядро truncated normal со стандартным отклонением 0,02 и нулевой bias."""
        super().__init__()
        self.conv = nn.Conv2d(inputs, outputs, kernel, stride=stride, padding=padding,
                              groups=groups, bias=False)
        self.bias = nn.Parameter(torch.zeros(outputs, 1))
        nn.init.trunc_normal_(self.conv.weight, std=0.02)

    def forward(self, x):
        """Применяет свёртку и поканальное смещение без изменения его математического смысла."""
        return self.conv(x) + self.bias.view(1, -1, 1, 1)


class ChannelNorm(nn.Module):
    """Вычисляет LayerNorm по каналам с epsilon=1e-6 и матричными affine-параметрами."""

    def __init__(self, channels):
        """Задаёт единичный scale и нулевую bias в форме [channels,1]."""
        super().__init__()
        self.channels = channels
        self.weight = nn.Parameter(torch.ones(channels, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x):
        """Нормирует каналы каждого пространственного отсчёта NCHW независимо."""
        normalized = F.layer_norm(x.permute(0, 2, 3, 1), (self.channels,),
                                  self.weight.flatten(), self.bias.flatten(), eps=1e-6)
        return normalized.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """Реализует depthwise 7×7, LayerNorm, MLP с expansion=4 и LayerScale residual."""

    def __init__(self, channels):
        """Создаёт ConvNeXt-блок с LayerScale=1e-6 без stochastic depth."""
        super().__init__()
        self.depthwise = AffineConv(channels, channels, 7, padding=3, groups=channels)
        self.norm = ChannelNorm(channels)
        self.expand = AffineConv(channels, 4 * channels, 1)
        self.contract = AffineConv(4 * channels, channels, 1)
        self.layer_scale = nn.Parameter(torch.full((channels, 1), 1e-6))

    def forward(self, x):
        """Добавляет масштабированную нелинейную ветвь к входным признакам."""
        features = self.norm(self.depthwise(x))
        features = self.contract(F.gelu(self.expand(features)))
        return x + self.layer_scale.view(1, -1, 1, 1) * features


class ConvNeXtTiny(nn.Module):
    """Возвращает четыре масштаба ConvNeXt-Tiny с глубинами [3,3,9,3] и ширинами [96,192,384,768]."""

    def __init__(self, input_channels):
        """Создаёт случайно инициализированный backbone с адаптированным stem без классификатора."""
        super().__init__()
        channels = (96, 192, 384, 768)
        self.downsample = nn.ModuleList([
            nn.Sequential(AffineConv(input_channels, 96, 4, stride=4), ChannelNorm(96)),
            *(nn.Sequential(ChannelNorm(before), AffineConv(before, after, 2, stride=2))
              for before, after in zip(channels[:-1], channels[1:])),
        ])
        self.stages = nn.ModuleList([
            nn.Sequential(*(ConvNeXtBlock(width) for _ in range(depth)))
            for width, depth in zip(channels, (3, 3, 9, 3))
        ])

    def forward(self, x):
        """Выдаёт NCHW-признаки 32×32, 16×16, 8×8 и 4×4 для входа 128×128."""
        features = []
        for downsample, stage in zip(self.downsample, self.stages):
            x = stage(downsample(x))
            features.append(x)
        return features


class ConvNeXtCorrection(nn.Module):
    """Предсказывает коррекцию из последнего масштаба ConvNeXt-Tiny без skip-связей."""

    def __init__(self, width):
        """Создаёт encoder целей, ConvNeXt-Tiny и простой декодер с нулевой головой."""
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(4, width, 3, padding=1, bias=False), nn.SiLU())
        self.backbone = ConvNeXtTiny(width + 3)
        self.decode = nn.Sequential(nn.Conv2d(768, width, 1, bias=False), nn.SiLU())
        self.head = nn.Conv2d(width, 1, 3, padding=1, bias=False)
        nn.init.zeros_(self.head.weight)

    def forward(self, v, gradient, context, epsilon):
        """Декодирует последний масштаб в ограниченную коррекцию на исходной сетке фазы."""
        features = self.backbone(state_channels(v, gradient, context, epsilon))
        decoded = self.decode(features[-1])
        decoded = F.interpolate(decoded, size=v.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(decoded).tanh()


class ConvNeXtUNetCorrection(nn.Module):
    """Декодирует четыре масштаба ConvNeXt-Tiny с U-Net skip-связями."""

    def __init__(self, width):
        """Создаёт латеральные проекции, три skip-fusion блока и нулевую выходную голову."""
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(4, width, 3, padding=1, bias=False), nn.SiLU())
        self.backbone = ConvNeXtTiny(width + 3)
        self.lateral = nn.ModuleList(nn.Conv2d(channels, width, 1, bias=False)
                                     for channels in (96, 192, 384, 768))
        self.fusion = nn.ModuleList(nn.Sequential(
            nn.Conv2d(2 * width, width, 3, padding=1, bias=False), nn.SiLU()) for _ in range(3))
        self.head = nn.Conv2d(width, 1, 3, padding=1, bias=False)
        nn.init.zeros_(self.head.weight)

    def forward(self, v, gradient, context, epsilon):
        """Объединяет масштабы через concatenation и возвращает коррекцию 128×128."""
        features = self.backbone(state_channels(v, gradient, context, epsilon))
        projected = [layer(feature) for layer, feature in zip(self.lateral, features)]
        decoded = projected[-1]
        for fusion, skip in zip(self.fusion, reversed(projected[:-1])):
            decoded = F.interpolate(decoded, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            decoded = fusion(torch.cat((decoded, skip), dim=1))
        decoded = F.interpolate(decoded, size=v.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(decoded).tanh()
