"""Numerical checks for the structural objective and checkpointed MPS training path."""

import json

import torch
from torch.utils.checkpoint import checkpoint

from experiments.experiment_gs_muon.evaluate import image_metrics
from experiments.experiment_gs_muon.physics import fresnel_factors
from experiments.experiment_single_sample.physics import ssim
from .experiment import CONFIG, build_model
from .physics import structural_losses


def main() -> None:
    """Check SSIM agreement, finite-difference gradients and checkpoint equivalence with explicit tolerances."""
    torch.set_num_threads(1)
    torch.manual_seed(17)
    a = torch.rand(4, 128, 128, dtype=torch.float64)
    b = torch.rand_like(a)
    reference = torch.tensor([image_metrics((x * 255).numpy(), (y * 255).numpy())['ssim']
                              for x, y in zip(a, b)], dtype=torch.float64)
    ssim_error = float((ssim(a, b) - reference).abs().max())
    phi = torch.randn(1, 128, 128, dtype=torch.float64, requires_grad=True)
    target = a[None]
    factors = fresnel_factors(torch.float64)
    loss = structural_losses(phi, target, factors, 1e-12, 0.5).mean()
    gradient, = torch.autograd.grad(loss, phi)
    direction = torch.randn_like(phi)
    direction /= direction.norm()
    step = 1e-4
    difference = (structural_losses(phi + step * direction, target, factors, 1e-12, 0.5).mean()
                  - structural_losses(phi - step * direction, target, factors, 1e-12, 0.5).mean()) / (2 * step)
    derivative_error = abs(float(difference.detach() - (gradient * direction).sum()))
    model = build_model({**CONFIG, 'device': 'mps', 'dtype': 'float32'})
    target = target.float().to('mps')
    factors = tuple(x.to('mps') for x in fresnel_factors(torch.float32))
    direct = structural_losses(model(target), target, factors, 1e-12, 0.5).mean()
    direct.backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    model.activation_checkpointing = True
    model.adapter.activation_checkpointing = True
    checked = checkpoint(structural_losses, model(target), target, factors,
                         1e-12, 0.5, use_reentrant=False).mean()
    checked.backward()
    checkpoint_error = max(float((p.grad - g).abs().max()) for p, g in zip(model.parameters(), gradients))
    finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters())
    report = {'ssim_reference_absolute_error': ssim_error, 'ssim_tolerance': 1e-10,
              'directional_derivative_absolute_error': derivative_error, 'derivative_tolerance': 1e-7,
              'checkpoint_gradient_max_absolute_error': checkpoint_error, 'checkpoint_tolerance': 1e-5,
              'finite_mps_gradients': finite}
    print(json.dumps(report, indent=2))
    if ssim_error > 1e-10 or derivative_error > 1e-7 or checkpoint_error > 1e-5 or not finite:
        raise RuntimeError('Structural objective verification failed.')


if __name__ == '__main__':
    main()
