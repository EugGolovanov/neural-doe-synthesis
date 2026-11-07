"""Проверки полного ConvNeXt-Tiny в физической развёртке без обучения датасета."""

import gc
import json
from copy import deepcopy

import torch

from experiments.experiment_gs_muon.physics import N, fresnel_factors
from experiments.experiment_unrolled_cnn.algorithms import initial_momenta, losses, unroll, update_muon
from .convnext import ConvNeXtBlock, ConvNeXtCorrection, ConvNeXtUNetCorrection
from .experiment import CONFIG
from .verify import directional_error


def main():
    """Проверяет размерности, производные блока, checkpoint-градиенты и обновление всех весов Muon."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(123)
    error = directional_error(ConvNeXtBlock(8).double(), torch.randn(1, 8, 8, 8, dtype=torch.float64))
    report = {'block_fd_absolute_error': error, 'fd_tolerance': 1e-7,
              'checkpoint_relative_tolerance': 1e-5, 'models': {}}
    if error > 1e-7:
        raise RuntimeError(f'Ошибка производной ConvNeXt-блока: {error}.')
    config = deepcopy(CONFIG)
    config['steps'] = 2
    initial = torch.randn(1, 1, N, N)
    target = torch.rand(1, 4, N, N)
    factors = fresnel_factors(torch.float32)
    for name, constructor in [('cnn_convnext_tiny', ConvNeXtCorrection),
                              ('unet_convnext_tiny', ConvNeXtUNetCorrection)]:
        model = constructor(8)
        with torch.no_grad():
            context = model.encoder(target)
            state = torch.cat((initial.cos(), initial.sin(), initial, context), dim=1)
            shapes = [list(feature.shape) for feature in model.backbone(state)]
            zero_error = model(initial, initial, context, 1e-8).abs().max().item()
        torch.nn.init.normal_(model.head.weight, std=0.01)
        model.zero_grad(set_to_none=True)
        phi = unroll(model, target, initial, factors, config, training=True, use_checkpoint=False)
        losses(phi, target, factors, config['loss_epsilon']).mean().backward()
        reference = [parameter.grad.clone() for parameter in model.parameters()]
        model.zero_grad(set_to_none=True)
        phi = unroll(model, target, initial, factors, config, training=True, use_checkpoint=True)
        losses(phi, target, factors, config['loss_epsilon']).mean().backward()
        difference = sum((parameter.grad - gradient).square().sum().item()
                         for parameter, gradient in zip(model.parameters(), reference))
        denominator = sum(gradient.square().sum().item() for gradient in reference)
        relative = (difference / max(denominator, 1e-24)) ** 0.5
        finite_gradients = all(torch.isfinite(parameter.grad).all().item() for parameter in model.parameters())
        del reference
        before = model.head.weight.detach().clone()
        update_muon(model, initial_momenta(model), config['muon'])
        entry = {
            'parameters': sum(parameter.numel() for parameter in model.parameters()),
            'feature_shapes': shapes, 'phase_shape': list(phi.shape),
            'zero_head_max_error': zero_error, 'checkpoint_gradient_relative_error': relative,
            'finite_gradients': finite_gradients,
            'finite_parameters_after_muon': all(torch.isfinite(parameter).all().item() for parameter in model.parameters()),
            'head_update_norm': float((model.head.weight.detach() - before).norm()),
        }
        report['models'][name] = entry
        print(json.dumps({name: entry}, indent=2), flush=True)
        if (shapes != [[1, 96, 32, 32], [1, 192, 16, 16], [1, 384, 8, 8], [1, 768, 4, 4]]
                or list(phi.shape) != [1, 1, N, N] or zero_error != 0 or relative > 1e-5
                or not finite_gradients or not entry['finite_parameters_after_muon']
                or entry['head_update_norm'] == 0):
            raise RuntimeError(f'Проверка {name} не пройдена: {entry}.')
        del model, phi, context
        gc.collect()
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
