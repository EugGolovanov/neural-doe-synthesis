"""GS+Muon initialization and direct phase refinement."""

import torch

from .physics import N, physical_loss


def gs_step(phi: torch.Tensor, target: torch.Tensor, factors: tuple) -> torch.Tensor:
    """Объединяет четыре обратных поля после замены амплитуд и проецирует на фазу."""
    chirp, scale = factors
    field = torch.exp(1j * phi)
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(field * chirp, dim=(-2, -1))),
        dim=(-2, -1),
    ) * scale
    target_sum = target.sum(dim=(-2, -1), keepdim=True)
    if torch.any(target_sum == 0):
        raise ValueError("GS с сохранением мощности не определён для чёрной цели.")
    power = N**4 * scale.square()
    amplitude = (target * power / target_sum).sqrt()
    projected = amplitude * torch.exp(1j * torch.angle(spectrum))
    back = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(projected / scale, dim=(-2, -1))),
        dim=(-2, -1),
    ) * chirp.conj()
    combined = back.mean(dim=0)
    # Ближайшая к текущей фазе ветвь сохраняет локальную непрерывность состояния.
    delta = torch.angle(combined * field.conj())
    return phi + delta

def muon_direction(gradient: torch.Tensor, momentum: torch.Tensor,
                   config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Строит Nesterov-направление с приближённой поляризацией Newton–Schulz."""
    beta = config["momentum"]
    momentum = beta * momentum + (1 - beta) * gradient
    direction = (1 - beta) * gradient + beta * momentum
    direction = direction / (torch.linalg.vector_norm(direction) + config["epsilon"])
    a, b, c = config["coefficients"]
    for _ in range(config["ns_steps"]):
        gram = direction @ direction.T
        direction = a * direction + (b * gram + c * (gram @ gram)) @ direction
    return direction * config["scale"], momentum


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
