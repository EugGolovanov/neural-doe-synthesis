"""Finite hardware pixels and finite detector pixels on one common physical grid."""

import math

import numpy as np
import torch
from torch.nn import functional as F


N = 128
PITCH = 10e-6
WAVELENGTH = 532e-9
DISTANCES = (0.0241, 0.0253, 0.0265, 0.0277)
INCIDENT_POWER = (N * PITCH)**2


def pixel_kernels(output_pitch: float, detector_samples: int, pixel_order: int) -> tuple:
    """Integrate the separable Fresnel kernel over constant-phase hardware pixels by Gauss quadrature."""
    nodes, weights = np.polynomial.legendre.leggauss(pixel_order)
    centers = (torch.arange(N, dtype=torch.float64) - N / 2) * PITCH
    source = centers[:, None] + torch.tensor(nodes)[None, :] * PITCH / 2
    offsets = (torch.arange(detector_samples, dtype=torch.float64) + 0.5) / detector_samples - 0.5
    output = ((torch.arange(N, dtype=torch.float64) - N / 2)[:, None] + offsets) * output_pitch
    z = torch.tensor(DISTANCES, dtype=torch.float64)
    difference = output.flatten()[None, :, None, None] - source[None, None, :, :]
    phase = math.pi * difference.square() / (WAVELENGTH * z[:, None, None, None])
    kernel = (torch.exp(1j * phase) * torch.tensor(weights)).sum(dim=-1) * PITCH / 2
    scale = 1 / (1j * WAVELENGTH * z[:, None, None])
    return kernel, scale


def propagate(field: torch.Tensor, factors: tuple) -> torch.Tensor:
    """Propagate a complex hardware-pixel field to the four detector quadrature grids."""
    kernel, scale = factors
    return (kernel @ field @ kernel.transpose(-1, -2)) * scale


def adjoint(field: torch.Tensor, factors: tuple) -> torch.Tensor:
    """Apply the Euclidean adjoint and sum the four quadrature-grid backprojections."""
    kernel, scale = factors
    return (kernel.conj().transpose(-1, -2) @ (field * scale.conj()) @ kernel.conj()).sum(dim=0)


def intensity(phi: torch.Tensor, factors: tuple) -> torch.Tensor:
    """Average squared propagated fields over each detector pixel without renormalizing power."""
    field = propagate(torch.exp(1j * phi), factors)
    samples = field.shape[-1] // N
    return field.abs().square().reshape(4, N, samples, N, samples).mean(dim=(2, 4))


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute per-plane valid-window Gaussian SSIM with a permanently fixed data range of one."""
    coordinate = torch.arange(-5, 6, dtype=a.dtype, device=a.device)
    weights = torch.exp(-coordinate.square() / (2 * 1.5**2))
    weights = weights / weights.sum()
    kernel = (weights[:, None] * weights[None, :])[None, None]
    images = torch.cat((a, b, a.square(), b.square(), a * b))[:, None]
    moments = F.conv2d(images, kernel)[:, 0]
    mean_a, mean_b, square_a, square_b, product = moments.chunk(5)
    variance_a = square_a - mean_a.square()
    variance_b = square_b - mean_b.square()
    covariance = product - mean_a * mean_b
    numerator = (2 * mean_a * mean_b + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (mean_a.square() + mean_b.square() + 0.01**2) * (variance_a + variance_b + 0.03**2)
    return (numerator / denominator).mean(dim=(-2, -1))


def spectral_weights(output_pitch: float, band_weights: list) -> torch.Tensor:
    """Define three fixed radial frequency bands in cycles per metre relative to detector Nyquist."""
    frequency = torch.fft.fftfreq(N, d=output_pitch, dtype=torch.float64)
    radius = torch.sqrt(frequency[:, None].square() + frequency[None, :].square())
    nyquist = 0.5 / output_pitch
    bands = (radius < 0.15 * nyquist, (radius >= 0.15 * nyquist) & (radius < 0.5 * nyquist), radius >= 0.5 * nyquist)
    return sum(weight * band for weight, band in zip(band_weights, bands))


def loss_terms(values: torch.Tensor, target: torch.Tensor, regions: dict, config: dict) -> dict:
    """Compute the five physical objective terms using immutable calibration and signal masks."""
    normalized = values / regions['intensity_scale'][:, None, None]
    epsilon = config['epsilon']
    amplitude = (torch.sqrt(normalized + epsilon) - torch.sqrt(target + epsilon)).square().mean(dim=(-2, -1))
    structure = ssim(normalized, target)
    signal = regions['signal']
    signal_sum = (values * signal).sum(dim=(-2, -1))
    efficiency = signal_sum * config['output_pitch_m']**2 / INCIDENT_POWER
    efficiency_loss = torch.relu(regions['eta_required'] - efficiency).square()
    signal_mean = signal_sum / signal.sum(dim=(-2, -1))
    relative = values / (signal_mean[:, None, None] + config['guard_epsilon'])
    guard = (relative.square() * ~signal).sum(dim=(-2, -1)) / (~signal).sum(dim=(-2, -1))
    residual = torch.fft.fft2(normalized - target, norm='ortho')
    frequency = (residual.abs().square() * regions['frequency_weights']).mean(dim=(-2, -1))
    return {'amplitude': amplitude, 'ssim_loss': 1 - structure, 'efficiency_loss': efficiency_loss,
            'guard': guard, 'frequency': frequency, 'ssim': structure, 'efficiency': efficiency}


def combined_loss(values: torch.Tensor, target: torch.Tensor, regions: dict, config: dict, step: int) -> torch.Tensor:
    """Combine plane losses with a scheduled structural ramp and stable soft worst-plane penalty."""
    terms = loss_terms(values, target, regions, config)
    ramp = min(1.0, max(0.0, (step - config['warmup_steps'] + 1) / config['ramp_steps']))
    weights = config['loss_weights']
    plane_loss = weights['amplitude'] * terms['amplitude'] + weights['efficiency'] * terms['efficiency_loss']
    plane_loss = plane_loss + ramp * (weights['ssim'] * terms['ssim_loss'] + weights['guard'] * terms['guard'] + weights['frequency'] * terms['frequency'])
    beta = config['worst_beta']
    worst = (torch.logsumexp(beta * plane_loss, dim=0) - math.log(4)) / beta
    return plane_loss.mean() + ramp * weights['worst'] * worst


def original_loss(values: torch.Tensor, target: torch.Tensor, regions: dict, config: dict, step: int) -> torch.Tensor:
    """Retain the original per-plane maximum-normalized MSE formula under the new physical operator."""
    normalized = values / (values.amax(dim=(-2, -1), keepdim=True) + config['original_epsilon'])
    return (normalized - target).square().mean()
