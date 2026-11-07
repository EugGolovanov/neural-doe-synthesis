"""Дифференцируемая версия существующего GS + Muon для физического адаптера."""

import torch

from experiments.experiment_gs_muon.algorithms import gs_step, muon_direction
from experiments.experiment_gs_muon.physics import physical_loss


def gs_muon(phi: torch.Tensor, target: torch.Tensor, factors: tuple,
            steps: int, loss_epsilon: float, muon: dict) -> torch.Tensor:
    """Разворачивает GS и Muon с полным градиентом и локальным состоянием импульса."""
    momentum = torch.zeros_like(phi)
    for _ in range(steps):
        phi = gs_step(phi, target, factors)
        loss = physical_loss(phi, target, factors, loss_epsilon)
        gradient, = torch.autograd.grad(loss, phi, create_graph=True)
        direction, momentum = muon_direction(gradient, momentum, muon)
        phi = phi - muon['lr'] * direction
    return phi
