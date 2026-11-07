"""Численные проверки развёртки без обучения исследовательской модели."""

import copy
import json
import math
import tempfile
from pathlib import Path

import torch

from experiments.experiment_gs_muon.physics import N, PITCH, WAVELENGTH, DISTANCES, fresnel_factors, intensity
from .algorithms import initial_momenta, losses, unroll, update_muon
from .experiment import CONFIG, run_experiment
from .models import CorrectionNet


def verify() -> None:
    """Проверяет производные, checkpointing, batch-независимость и короткий полный запуск."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(123)
    config = copy.deepcopy(CONFIG)
    config.update(steps=2, width=2)
    factors = fresnel_factors(torch.float64)
    initial = (2 * torch.rand(1, 1, N, N, dtype=torch.float64) - 1) * math.pi
    target = torch.rand(1, 4, N, N, dtype=torch.float64)
    model = CorrectionNet(config['width']).double()
    torch.nn.init.normal_(model.correction[-1].weight, std=0.02)
    phi = unroll(model, target, initial, factors, config, training=True, use_checkpoint=False)
    loss = losses(phi, target, factors, config['loss_epsilon']).mean()
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    checked = unroll(model, target, initial, factors, config, training=True)
    checked_loss = losses(checked, target, factors, config['loss_epsilon']).mean()
    checked_gradients = torch.autograd.grad(checked_loss, tuple(model.parameters()))
    checkpoint_error = max((a - b).abs().max().item() for a, b in zip(gradients, checked_gradients))
    parameter = model.correction[0].weight
    parameter_index = next(i for i, p in enumerate(model.parameters()) if p is parameter)
    direction = torch.randn_like(parameter)
    direction /= direction.norm()
    derivative = (gradients[parameter_index] * direction).sum().item()
    original = parameter.detach().clone()
    difference_errors = []
    for delta in (1e-3, 1e-4, 1e-5):
        values = []
        for sign in (1, -1):
            with torch.no_grad():
                parameter.copy_(original + sign * delta * direction)
            shifted = unroll(model, target, initial, factors, config, training=False)
            values.append(losses(shifted, target, factors, config['loss_epsilon']).mean().item())
        finite_difference = (values[0] - values[1]) / (2 * delta)
        difference_errors.append(abs(finite_difference - derivative) / max(abs(derivative), 1e-12))
    with torch.no_grad():
        parameter.copy_(original)
    single = unroll(model, target, initial, factors, config, training=False)
    batch = unroll(model, target.repeat(2, 1, 1, 1), initial.repeat(2, 1, 1, 1),
                   factors, config, training=False)
    batch_error = (batch - single).abs().max().item()
    values = intensity(initial, factors)
    pitches = torch.tensor(DISTANCES, dtype=torch.float64) * WAVELENGTH / (N * PITCH)
    power_error = (values.sum(dim=(-2, -1)) * pitches.square() / (N * PITCH)**2 - 1).abs().max().item()
    phase = initial.clone().requires_grad_(True)
    gradient, = torch.autograd.grad(losses(phase, target, factors, config['loss_epsilon']).mean(), phase)
    global_phase_error = gradient.sum().abs().item() / gradient.abs().sum().item()
    for parameter, gradient in zip(model.parameters(), gradients):
        parameter.grad = gradient
    update_muon(model, initial_momenta(model), config['muon'])
    finite_weights = all(torch.isfinite(p).all().item() for p in model.parameters())
    report = {
        'checkpoint_gradient_max_abs_error': checkpoint_error, 'checkpoint_tolerance': 1e-11,
        'full_unroll_derivative_relative_errors': difference_errors, 'derivative_tolerance': 1e-4,
        'batch_phase_max_abs_error': batch_error, 'batch_tolerance': 1e-11,
        'power_relative_error': power_error, 'power_tolerance': 1e-12,
        'global_phase_derivative_relative_error': global_phase_error, 'phase_tolerance': 1e-12,
        'muon_weights_finite': finite_weights,
    }
    print(json.dumps(report, indent=2), flush=True)
    if (checkpoint_error > 1e-11 or min(difference_errors) > 1e-4 or batch_error > 1e-11
            or power_error > 1e-12 or global_phase_error > 1e-12 or not finite_weights):
        raise RuntimeError('Численная проверка не пройдена; см. невязки выше.')
    config.update(seeds=[17], epochs=1, baseline_steps=1,
                  split_sizes={'train': 3, 'validation': 1, 'test': 1})
    with tempfile.TemporaryDirectory(prefix='doe_cnn_verify_') as temporary:
        output = Path(temporary) / 'smoke'
        run_experiment(config, output)
        summary = json.loads((output / 'summary.json').read_text())
        if len(summary) != 2 or any(len(row['metrics']) != 4 for row in summary):
            raise RuntimeError('Неполный отчёт сквозного запуска.')
        print(json.dumps({'smoke': 'passed', 'methods': [r['method'] for r in summary],
                          'artifact_directory': 'temporary; removed after verification'}))


if __name__ == '__main__':
    verify()
