"""Одно-FFT модель из phase_initializer/docs/doe_physics.md для фиксированной сетки 128×128."""

import math

import torch
from torch.nn import functional as F


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


def local_ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute valid-window Gaussian SSIM with separable filtering and fixed unit data range."""
    coordinates = torch.arange(-5, 6, dtype=a.dtype, device=a.device)
    weights = torch.exp(-coordinates.square() / (2 * 1.5**2))
    weights = weights / weights.sum()
    images = torch.cat((a, b, a.square(), b.square(), a * b))[:, None]
    horizontal = F.conv2d(images, weights[None, None, None, :])
    moments = F.conv2d(horizontal, weights[None, None, :, None])[:, 0]
    mean_a, mean_b, square_a, square_b, product = moments.chunk(5)
    variance_a = square_a - mean_a.square()
    variance_b = square_b - mean_b.square()
    covariance = product - mean_a * mean_b
    numerator = (2 * mean_a * mean_b + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (mean_a.square() + mean_b.square() + 0.01**2) * (variance_a + variance_b + 0.03**2)
    return (numerator / denominator).mean(dim=(-2, -1))

def calibration(target: torch.Tensor, factors: tuple) -> torch.Tensor:
    """Calibrate target intensity using the exact total intensity of the unnormalized FFT model."""
    power = N**4 * factors[1].square()
    return power / target.sum(dim=(-2, -1), keepdim=True)

def amplitude_loss(values: torch.Tensor, target: torch.Tensor, scale: torch.Tensor, config: dict) -> torch.Tensor:
    """Return the power-calibrated per-plane amplitude MSE with explicit square-root smoothing."""
    amplitude = torch.sqrt(values / scale + config['amplitude_epsilon'])
    target_amplitude = torch.sqrt(target + config['amplitude_epsilon'])
    return (amplitude - target_amplitude).square().mean(dim=(-2, -1))

def structural_loss(values: torch.Tensor, target: torch.Tensor, scale: torch.Tensor, config: dict) -> torch.Tensor:
    """Return per-plane one-minus-SSIM with differentiated exact maximum normalization."""
    normalized = values / (values.amax(dim=(-2, -1), keepdim=True) + config['maximum_epsilon'])
    return 1 - local_ssim(normalized, target)

def amplitude_ssim(values: torch.Tensor, target: torch.Tensor, scale: torch.Tensor, config: dict) -> torch.Tensor:
    """Add a fixed structural weight to the same calibrated amplitude objective used in branch B."""
    return amplitude_loss(values, target, scale, config) + config['ssim_weight'] * structural_loss(values, target, scale, config)
