"""Явная конфигурация и один шаг совместного обучения U-Net и адаптера."""

import torch

from experiments.experiment_gs_muon.physics import fresnel_factors, physical_loss
from experiments.experiment_unrolled_cnn.algorithms import update_muon
from .models import PhysicsAdapterUNet


CONFIG = {
    'seed': 17, 'device': 'cpu', 'dtype': 'float64', 'batch_size': 1,
    'width': 8, 'rank': 2, 'alpha': 2.0, 'steps': 2, 'loss_epsilon': 1e-12,
    'phase_muon': {'lr': 0.1, 'momentum': 0.95, 'epsilon': 1e-12,
                   'ns_steps': 5, 'coefficients': [3.4445, -4.7750, 2.0315], 'scale': 1.0},
    'weight_muon': {'lr': 0.001, 'momentum': 0.95, 'epsilon': 1e-12,
                    'ns_steps': 5, 'coefficients': [3.4445, -4.7750, 2.0315], 'scale': 1.0},
}


def build_model(config: dict) -> PhysicsAdapterUNet:
    """Создаёт совместно обучаемую основу и адаптер с явно заданными параметрами."""
    return PhysicsAdapterUNet(config['width'], config['rank'], config['alpha'],
                              config['steps'], config['loss_epsilon'],
                              config['phase_muon']).to(device=config['device'],
                                                       dtype=getattr(torch, config['dtype']))


def train_step(model: PhysicsAdapterUNet, target: torch.Tensor,
               momenta: list, config: dict) -> float:
    """Вычисляет физический loss и обновляет все матрицы сети внешним Muon."""
    model.train()
    model.zero_grad(set_to_none=True)
    phi = model(target)
    factors = tuple(value.to(target.device) for value in fresnel_factors(target.dtype))
    loss = torch.stack([physical_loss(p, t, factors, config['loss_epsilon'])
                        for p, t in zip(phi, target)]).mean()
    loss.backward()
    update_muon(model, momenta, config['weight_muon'])
    return loss.item()
