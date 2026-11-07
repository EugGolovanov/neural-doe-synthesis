"""Полная физическая развёртка и Muon для матричных весов CNN."""

import torch
from torch.utils.checkpoint import checkpoint

from experiments.experiment_gs_muon.algorithms import muon_direction
from experiments.experiment_gs_muon.physics import intensity


def losses(phi: torch.Tensor, target: torch.Tensor, factors: tuple,
           epsilon: float) -> torch.Tensor:
    """Возвращает физический MSE по примерам и плоскостям в форме [B,4]."""
    values = intensity(phi, factors)
    normalized = values / (values.amax(dim=(-2, -1), keepdim=True) + epsilon)
    return (normalized - target).square().mean(dim=(-2, -1))


def unroll(model, target: torch.Tensor, initial: torch.Tensor, factors: tuple,
           config: dict, training: bool, use_checkpoint: bool = True) -> torch.Tensor:
    """Выполняет K градиентных шагов с общей CNN и полными производными при обучении."""
    with torch.enable_grad():
        context = model.encoder(target)
        phi = initial.clone().requires_grad_(True)

        def step(phi: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
            """Вычисляет физический градиент и добавляет обучаемую остаточную коррекцию."""
            loss = losses(phi, target, factors, config['loss_epsilon']).mean(dim=1).sum()
            gradient, = torch.autograd.grad(loss, phi, create_graph=training)
            v = phi - config['step_size'] * gradient
            correction = model(v, gradient, context, config['gradient_epsilon'])
            return v + config['correction_scale'] * correction

        for _ in range(config['steps']):
            if training and use_checkpoint:
                phi = checkpoint(step, phi, context, use_reentrant=False)
            else:
                phi = step(phi, context)
            if not training:
                phi = phi.detach().requires_grad_(True)
    return phi if training else phi.detach()


def update_muon(model, momenta: list, config: dict) -> None:
    """Обновляет свёртки Muon после представления ядра матрицей [out,in·ky·kx]."""
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            gradient = parameter.grad.flatten(1)
            transposed = gradient.shape[0] > gradient.shape[1]
            if transposed:
                gradient = gradient.T
            direction, momenta[index] = muon_direction(gradient, momenta[index], config)
            if transposed:
                direction = direction.T
            parameter.add_(direction.reshape_as(parameter), alpha=-config['lr'])


def initial_momenta(model) -> list:
    """Создаёт нулевой импульс Muon в ориентации с меньшим числом строк."""
    momenta = []
    for parameter in model.parameters():
        matrix = parameter.detach().flatten(1)
        if matrix.shape[0] > matrix.shape[1]:
            matrix = matrix.T
        momenta.append(torch.zeros_like(matrix))
    return momenta
