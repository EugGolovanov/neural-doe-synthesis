"""SSIM-informed outer objective with the unchanged single-FFT Fresnel operator."""

import torch

from experiments.experiment_gs_muon.physics import intensity
from experiments.experiment_single_sample.physics import ssim


def structural_losses(phi: torch.Tensor, target: torch.Tensor, factors: tuple,
                      epsilon: float, ssim_weight: float) -> torch.Tensor:
    """Return per-sample per-plane maximum-normalized MSE plus weighted one-minus-SSIM."""
    values = intensity(phi[:, None], factors)
    normalized = values / (values.amax(dim=(-2, -1), keepdim=True) + epsilon)
    mse = (normalized - target).square().mean(dim=(-2, -1))
    structure = ssim(normalized.flatten(0, 1), target.flatten(0, 1)).reshape_as(mse)
    return mse + ssim_weight * (1 - structure)
