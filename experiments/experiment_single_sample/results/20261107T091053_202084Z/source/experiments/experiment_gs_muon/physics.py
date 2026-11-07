"""Одно-FFT модель из docs/doe_physics.md для фиксированной сетки 128×128."""

import math

import torch


N = 128
WAVELENGTH = 532e-9
PITCH = 1e-5
DISTANCES = (0.0241, 0.0253, 0.0265, 0.0277)


def fresnel_factors(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Возвращает входную квадратичную фазу и масштаб поля для четырёх плоскостей."""
    coordinates = (torch.arange(N, dtype=dtype) - N / 2) * PITCH
    radius_squared = coordinates[:, None] ** 2 + coordinates[None, :] ** 2
    z = torch.tensor(DISTANCES, dtype=dtype)[:, None, None]
    chirp = torch.exp(1j * math.pi * radius_squared / (WAVELENGTH * z))
    scale = PITCH**2 / (WAVELENGTH * z)
    return chirp, scale


def intensity(phi: torch.Tensor, factors: tuple) -> torch.Tensor:
    """Вычисляет интенсивности [4,128,128] из фазы [128,128] в радианах."""
    chirp, scale = factors
    field = torch.exp(1j * phi) * chirp
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1)), norm="backward"),
        dim=(-2, -1),
    )
    return (spectrum * scale).abs().square()


def physical_loss(phi: torch.Tensor, target: torch.Tensor, factors: tuple,
                  epsilon: float) -> torch.Tensor:
    """Вычисляет средний квадрат ошибки интенсивностей с нормировкой по максимуму."""
    values = intensity(phi, factors)
    normalized = values / (values.amax(dim=(-2, -1), keepdim=True) + epsilon)
    return (normalized - target).square().mean()
