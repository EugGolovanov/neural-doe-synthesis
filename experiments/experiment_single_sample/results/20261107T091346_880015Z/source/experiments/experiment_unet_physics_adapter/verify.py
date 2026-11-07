"""Численные проверки адаптера без обучения на датасете и без ассертов."""

import json
import platform

import torch

from experiments.experiment_gs_muon.algorithms import run_gs_muon
from experiments.experiment_gs_muon.physics import fresnel_factors
from experiments.experiment_unrolled_cnn.algorithms import initial_momenta
from .algorithms import gs_muon
from .experiment import CONFIG, build_model, train_step


def main() -> None:
    """Проверяет совпадение солвера, производную развёртки и обучение матриц смешивания."""
    torch.set_num_threads(1)
    torch.manual_seed(CONFIG['seed'])
    torch.use_deterministic_algorithms(True)
    dtype = torch.float64
    factors = fresnel_factors(dtype)
    target = 0.1 + 0.9 * torch.rand(1, 4, 128, 128, dtype=dtype)
    phi = torch.randn(128, 128, dtype=dtype, requires_grad=True)
    result = gs_muon(phi, target[0], factors, CONFIG['steps'],
                     CONFIG['loss_epsilon'], CONFIG['phase_muon'])
    reference, _ = run_gs_muon(phi.detach(), target[0], factors, CONFIG['steps'],
                              CONFIG['loss_epsilon'], CONFIG['phase_muon'])
    baseline_error = (reference - result).abs().max().item()
    probe = torch.randn_like(phi)
    direction = torch.randn_like(phi)
    direction = direction / direction.norm()
    gradient, = torch.autograd.grad((result * probe).mean(), phi)
    analytical = (gradient * direction).sum().item()
    derivative_errors = {}
    for h in (1e-4, 1e-5, 1e-6):
        plus = gs_muon(phi + h * direction, target[0], factors, CONFIG['steps'],
                       CONFIG['loss_epsilon'], CONFIG['phase_muon'])
        minus = gs_muon(phi - h * direction, target[0], factors, CONFIG['steps'],
                        CONFIG['loss_epsilon'], CONFIG['phase_muon'])
        numerical = (((plus - minus) * probe).mean() / (2 * h)).item()
        derivative_errors[str(h)] = abs(numerical - analytical)
    model = build_model(CONFIG)
    momenta = initial_momenta(model)
    with torch.no_grad():
        first = model.encoder(target)
        second = model.middle(torch.nn.functional.avg_pool2d(first, 2))
        bottom = model.bottom(torch.nn.functional.avg_pool2d(second, 2))
        up = torch.nn.functional.interpolate(bottom, scale_factor=2, mode='nearest')
        middle = model.decoder_middle(torch.cat((second, up), dim=1))
        up = torch.nn.functional.interpolate(middle, scale_factor=2, mode='nearest')
        features = model.decoder(torch.cat((first, up), dim=1))
        plain = model.head(model.adapter.base(features))[:, 0]
        zero_B_error = (model(target) - plain).abs().max().item()
    history = [train_step(model, target, momenta, CONFIG) for _ in range(2)]
    gradient_norms = {name: parameter.grad.norm().item()
                      for name, parameter in model.named_parameters()}
    finite = all(torch.isfinite(parameter.grad).all().item() for parameter in model.parameters())
    with torch.no_grad():
        predicted = model(target)
    batch_error = (model(target.repeat(2, 1, 1, 1)).detach() - predicted).abs().max().item()
    passed = (baseline_error < 1e-12 and min(derivative_errors.values()) < 1e-7
              and zero_B_error < 1e-12 and batch_error < 1e-10 and finite
              and gradient_norms['adapter.A.weight'] > 0
              and gradient_norms['adapter.B.weight'] > 0)
    print(json.dumps({'passed': passed, 'python': platform.python_version(),
                      'torch': torch.__version__, 'config': CONFIG,
                      'baseline_max_error': baseline_error,
                      'directional_derivative_absolute_errors': derivative_errors,
                      'zero_B_max_error': zero_B_error, 'batch_max_error': batch_error,
                      'all_gradients_finite': finite, 'gradient_norms': gradient_norms,
                      'two_smoke_losses_not_quality_evidence': history}, indent=2))
    if not passed:
        raise RuntimeError('Численная проверка адаптера не пройдена.')


if __name__ == '__main__':
    main()
