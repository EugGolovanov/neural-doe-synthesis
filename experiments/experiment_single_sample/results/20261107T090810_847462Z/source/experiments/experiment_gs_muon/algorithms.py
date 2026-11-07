"""Многоплоскостной GS и исследовательские гибриды с шагом по физическому loss."""

import torch

if __package__:
    from .physics import N, physical_loss
else:
    from physics import N, physical_loss


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


def run_gs(phi: torch.Tensor, target: torch.Tensor, factors: tuple,
           steps: int, epsilon: float) -> tuple[torch.Tensor, list]:
    """Выполняет GS без градиентной коррекции с записью loss после каждой итерации."""
    history = [{"iteration": 0, "loss": physical_loss(phi, target, factors, epsilon).item()}]
    with torch.no_grad():
        for iteration in range(1, steps + 1):
            phi = gs_step(phi, target, factors)
            history.append({"iteration": iteration,
                            "loss": physical_loss(phi, target, factors, epsilon).item()})
    return phi, history


def run_gs_muon(phi: torch.Tensor, target: torch.Tensor, factors: tuple,
                steps: int, epsilon: float, config: dict) -> tuple[torch.Tensor, list]:
    """Чередует GS и Muon с сохранением momentum внутри одного примера."""
    momentum = torch.zeros_like(phi)
    history = [{"iteration": 0, "loss": physical_loss(phi, target, factors, epsilon).item()}]
    for iteration in range(1, steps + 1):
        phi = gs_step(phi.detach(), target, factors).requires_grad_()
        loss = physical_loss(phi, target, factors, epsilon)
        gradient, = torch.autograd.grad(loss, phi)
        with torch.no_grad():
            direction, momentum = muon_direction(gradient, momentum, config)
            alignment = (gradient * direction).sum().item()
            phi = phi - config["lr"] * direction
            final_loss = physical_loss(phi, target, factors, epsilon).item()
        history.append({"iteration": iteration, "after_gs": loss.item(),
                        "loss": final_loss, "gradient_dot_direction": alignment,
                        "gradient_rms": gradient.square().mean().sqrt().item(),
                        "correction_rms": (config["lr"] * direction).square().mean().sqrt().item()})
    return phi.detach(), history


def run_gs_adam(phi: torch.Tensor, target: torch.Tensor, factors: tuple,
                steps: int, epsilon: float, config: dict) -> tuple[torch.Tensor, list]:
    """Чередует GS и Adam с сохранением моментов внутри одного примера."""
    phi = phi.clone().requires_grad_()
    optimizer = torch.optim.Adam([phi], lr=config["lr"], betas=tuple(config["betas"]),
                                 eps=config["epsilon"], weight_decay=0)
    history = [{"iteration": 0, "loss": physical_loss(phi, target, factors, epsilon).item()}]
    for iteration in range(1, steps + 1):
        with torch.no_grad():
            phi.copy_(gs_step(phi, target, factors))
        optimizer.zero_grad()
        loss = physical_loss(phi, target, factors, epsilon)
        loss.backward()
        before = phi.detach().clone()
        gradient = phi.grad.detach().clone()
        optimizer.step()
        with torch.no_grad():
            correction = before - phi
            final_loss = physical_loss(phi, target, factors, epsilon).item()
        history.append({"iteration": iteration, "after_gs": loss.item(),
                        "loss": final_loss,
                        "gradient_dot_direction": (gradient * correction).sum().item(),
                        "correction_rms": correction.square().mean().sqrt().item()})
    return phi.detach(), history
