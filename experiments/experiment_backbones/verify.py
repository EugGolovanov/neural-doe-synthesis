"""Численные проверки новых блоков и полного обратного прохода без обучения датасета."""

import json
from copy import deepcopy

import torch

from experiments.experiment_gs_muon.physics import N, fresnel_factors
from experiments.experiment_unrolled_cnn.algorithms import initial_momenta, losses, unroll, update_muon
from .experiment import CONFIG
from .models import FNOCorrection, MambaCorrection, SelectiveScan, SpectralLayer, UNetCorrection


def directional_error(module, x):
    """Сравнивает autograd по входу с центральной разностью для квадратичного функционала."""
    x = x.requires_grad_(True)
    direction = torch.randn_like(x)
    direction = direction / direction.norm()
    value = module(x).square().mean()
    gradient, = torch.autograd.grad(value, x)
    analytical = (gradient * direction).sum()
    step = 1e-5
    numerical = (module(x + step * direction).square().mean()
                 - module(x - step * direction).square().mean()) / (2 * step)
    return float((analytical - numerical).abs().detach())


def main():
    """Печатает ошибки разностей и checkpoint, проверяя конечность обновления Muon."""
    torch.set_num_threads(1)
    torch.manual_seed(10)
    report = {
        'spectral_input_fd_absolute_error': directional_error(SpectralLayer(2, 3).double(), torch.randn(1, 2, 8, 8, dtype=torch.float64)),
        'ssm_input_fd_absolute_error': directional_error(SelectiveScan(2, 3).double(), torch.randn(2, 8, 2, dtype=torch.float64)),
        'fd_tolerance': 1e-7,
        'checkpoint_relative_tolerance': 1e-5,
        'models': {},
    }
    config = deepcopy(CONFIG)
    config['steps'] = 2
    initial = torch.randn(1, 1, N, N)
    target = torch.rand(1, 4, N, N)
    factors = fresnel_factors(torch.float32)
    for name, model in [('unet', UNetCorrection(2)), ('fno', FNOCorrection(2, 3, 1)),
                        ('mamba_axial', MambaCorrection(2, 2, 1))]:
        context = model.encoder(target)
        zero_error = model(initial, initial, context, 1e-8).abs().max().item()
        torch.nn.init.normal_(model.head.weight, std=0.01)
        gradients = []
        for checkpointed in (False, True):
            model.zero_grad(set_to_none=True)
            phi = unroll(model, target, initial, factors, config, training=True, use_checkpoint=checkpointed)
            loss = losses(phi, target, factors, config['loss_epsilon']).mean()
            loss.backward()
            gradients.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
        error = ((gradients[0] - gradients[1]).norm() / (gradients[0].norm() + 1e-12)).item()
        before = model.head.weight.detach().clone()
        update_muon(model, initial_momenta(model), config['muon'])
        finite = all(torch.isfinite(p).all().item() for p in model.parameters())
        entry = {'zero_head_max_error': zero_error, 'checkpoint_gradient_relative_error': error,
                 'finite_gradients': bool(torch.isfinite(gradients[1]).all()),
                 'finite_parameters_after_muon': finite,
                 'head_update_norm': (model.head.weight.detach() - before).norm().item()}
        report['models'][name] = entry
    print(json.dumps(report, indent=2, allow_nan=False))
    if max(report['spectral_input_fd_absolute_error'], report['ssm_input_fd_absolute_error']) > 1e-7:
        raise RuntimeError('Конечная разность не прошла допуск float64.')
    for entry in report['models'].values():
        if (entry['zero_head_max_error'] != 0 or entry['checkpoint_gradient_relative_error'] > 1e-5
                or not entry['finite_gradients'] or not entry['finite_parameters_after_muon']
                or entry['head_update_norm'] == 0):
            raise RuntimeError('Проверка развёртки не пройдена.')


if __name__ == '__main__':
    main()
