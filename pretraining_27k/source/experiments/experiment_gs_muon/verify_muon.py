"""Малые численные проверки GS-проекции и Muon-коррекции без ассертов."""

import numpy as np
import torch

from algorithms import gs_step, muon_direction
from evaluate import image_metrics
from experiment import CONFIG, synthetic_targets
from physics import N, fresnel_factors, intensity, physical_loss


def main() -> None:
    """Печатает ошибки согласованности алгоритма и завершает проверку при превышении допусков."""
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(101)
    factors = fresnel_factors(torch.float64)
    phi = torch.randn((N, N), generator=generator, dtype=torch.float64)
    reachable = intensity(phi, factors)
    projected = gs_step(phi, reachable, factors)
    gs_error = (torch.exp(1j * projected) - torch.exp(1j * phi)).abs().max().item()
    print(f"GS fixed-point field error: {gs_error:.3e}; tolerance=1e-10")
    target = torch.tensor(synthetic_targets('spots').astype(np.float64) / 255)
    phi.requires_grad_()
    loss = physical_loss(phi, target, factors, CONFIG['loss_epsilon'])
    gradient, = torch.autograd.grad(loss, phi)
    direction, _ = muon_direction(gradient, torch.zeros_like(phi), CONFIG['muon'])
    analytic = (gradient * direction).sum().item()
    errors = []
    for h in (1e-3, 1e-4, 1e-5):
        plus = physical_loss(phi + h * direction, target, factors, CONFIG['loss_epsilon'])
        minus = physical_loss(phi - h * direction, target, factors, CONFIG['loss_epsilon'])
        numerical = ((plus - minus) / (2 * h)).item()
        error = abs(numerical - analytic) / max(abs(analytic), 1e-15)
        errors.append(error)
        print(f"directional derivative h={h:g}: relative error={error:.3e}; tolerance=1e-5")
    zero_direction, _ = muon_direction(torch.zeros_like(phi), torch.zeros_like(phi), CONFIG['muon'])
    zero_error = zero_direction.abs().max().item()
    print(f"Muon zero-gradient error: {zero_error:.3e}; expected=0")
    image = synthetic_targets('dense')[0]
    identity = image_metrics(image, image)
    constant = image_metrics(np.zeros_like(image), image)
    ssim_error = abs(identity['ssim'] - 1)
    print(f"SSIM identity error: {ssim_error:.3e}; tolerance=1e-12")
    print(f"constant-image Pearson: {constant['pearson']}; expected=None")
    if (gs_error > 1e-10 or not np.isfinite(errors).all() or min(errors) > 1e-5
            or zero_error != 0 or not torch.isfinite(direction).all()
            or ssim_error > 1e-12 or constant['pearson'] is not None):
        raise RuntimeError('Численная проверка не пройдена.')


if __name__ == '__main__':
    main()
