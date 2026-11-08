"""Original maximum-normalized MSE+SSIM training objective."""

import torch
from torch.nn import functional as F
from phase_initializer.physics import intensity


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


def structural_losses(phi: torch.Tensor, target: torch.Tensor, factors: tuple,
                      epsilon: float, ssim_weight: float) -> torch.Tensor:
    """Return per-sample per-plane maximum-normalized MSE plus weighted one-minus-SSIM."""
    values = intensity(phi[:, None], factors)
    normalized = values / (values.amax(dim=(-2, -1), keepdim=True) + epsilon)
    mse = (normalized - target).square().mean(dim=(-2, -1))
    structure = ssim(normalized.flatten(0, 1), target.flatten(0, 1)).reshape_as(mse)
    return mse + ssim_weight * (1 - structure)
