"""Muon updates for the pretrained U-Net weights."""

import torch
from phase_initializer.algorithms import muon_direction


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
