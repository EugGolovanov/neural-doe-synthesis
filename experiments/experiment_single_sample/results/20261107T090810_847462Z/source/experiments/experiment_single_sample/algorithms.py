"""Single-tuple weight fitting and finite-pixel adjoint-GS control."""

import time
from pathlib import Path

import torch

from .evaluate import measure, relative_error, save_json, selection_key
from .physics import adjoint, intensity, loss_terms, propagate


def plateau(records: list, config: dict) -> bool:
    """Detect a declared numerical plateau after the full objective ramp without claiming optimality."""
    window = config['plateau_evaluations']
    if len(records) < window or records[-window]['step'] < config['minimum_steps']:
        return False
    recent = records[-window:]
    losses = [r['validation_objective'] for r in recent]
    relative_span = (max(losses) - min(losses)) / max(abs(sum(losses) / len(losses)), config['epsilon'])
    scores = [r['selection_key'][2] for r in recent]
    return (relative_span < config['loss_plateau_tolerance']
            and max(scores) - min(scores) < config['score_plateau_tolerance']
            and max(r['intensity_change'] for r in recent) < config['intensity_plateau_tolerance'])


def optimize_weights(model: torch.nn.Module, target: torch.Tensor, regions: dict, factors: tuple,
                     validation_factors: tuple, config: dict, objective, directory: Path) -> tuple:
    """Fit all checkpoint weights to one tuple with fresh Adam and independent physical score selection."""
    directory.mkdir(exist_ok=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], betas=tuple(config['adam_betas']),
                                 eps=config['adam_epsilon'], weight_decay=0)
    records = []
    history = []
    best_key = None
    best_phi = None
    previous = None
    start = time.monotonic()
    status = 'budget_exhausted'
    model.train()
    for step in range(config['max_steps'] + 1):
        optimizer.zero_grad(set_to_none=True)
        phi = model(target[None])[0]
        values = intensity(phi, factors)
        loss = objective(values, target, regions, config, step)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at step {step}.')
        history.append({'step': step, 'loss': float(loss.detach())})
        if step % config['evaluate_every'] == 0 or step == config['max_steps']:
            with torch.no_grad():
                validation = intensity(phi.detach(), validation_factors)
                measured = measure(validation, target, regions, config)
                key = selection_key(measured, regions['efficiency_floor'])
                change = max(relative_error(validation, previous)) if previous is not None else 1.0
                previous = validation.clone()
                fixed_loss = objective(validation, target, regions, config, config['warmup_steps'] + config['ramp_steps'])
                terms = loss_terms(values.detach(), target, regions, config)
                records.append({'step': step, 'validation_objective': float(fixed_loss), 'metrics': measured,
                                'training_terms': {name: value.tolist() for name, value in terms.items()},
                                'selection_key': key, 'intensity_change': change, 'elapsed_seconds': time.monotonic() - start})
            if best_key is None or key > best_key:
                best_key = key
                best_phi = phi.detach().clone()
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                            'phase': best_phi, 'selection_key': key}, directory / 'best_checkpoint.pt')
            save_json(directory / 'evaluations.json', records)
            save_json(directory / 'history.json', history)
            if plateau(records, config):
                status = 'numerical_plateau'
                break
        if step == config['max_steps']:
            break
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError(f'Nonfinite or missing gradients at step {step}.')
        history[-1]['gradient_l2'] = float(sum(g.square().sum() for g in gradients).sqrt())
        optimizer.step()
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                'phase': phi.detach(), 'torch_rng': torch.get_rng_state()}, directory / 'last_checkpoint.pt')
    save_json(directory / 'history.json', history)
    result = {'status': status, 'updates': step, 'best_selection_key': best_key,
              'elapsed_seconds': time.monotonic() - start, 'best_is_not_necessarily_last': True}
    save_json(directory / 'status.json', result)
    return best_phi, result


def run_adjoint_gs(initial_phi: torch.Tensor, target: torch.Tensor, regions: dict, factors: tuple,
                   validation_factors: tuple, config: dict, directory: Path) -> tuple:
    """Alternate amplitude replacement and a stable adjoint correction for the nonunitary finite-pixel operator."""
    directory.mkdir(exist_ok=False)
    kernel, scale = factors
    samples = kernel.shape[1] // 128
    desired = torch.sqrt(target * regions['intensity_scale'][:, None, None])
    desired = desired.repeat_interleave(samples, 1).repeat_interleave(samples, 2)
    operator_norm = torch.linalg.matrix_norm(kernel, ord=2).square() * scale.flatten().abs()
    step_size = float(1 / operator_norm.square().mean())
    phi = initial_phi.detach().clone()
    records = []
    best_phi = phi.clone()
    best_key = None
    previous = None
    start = time.monotonic()
    status = 'budget_exhausted'
    with torch.no_grad():
        for step in range(config['gs_steps'] + 1):
            if step % config['evaluate_every'] == 0 or step == config['gs_steps']:
                validation = intensity(phi, validation_factors)
                measured = measure(validation, target, regions, config)
                key = selection_key(measured, regions['efficiency_floor'])
                change = max(relative_error(validation, previous)) if previous is not None else 1.0
                previous = validation.clone()
                records.append({'step': step, 'metrics': measured, 'selection_key': key,
                                'validation_objective': sum(p['amplitude_mse_epsilon'] for p in measured['planes']) / 4,
                                'intensity_change': change, 'elapsed_seconds': time.monotonic() - start})
                if best_key is None or key > best_key:
                    best_key, best_phi = key, phi.clone()
                save_json(directory / 'evaluations.json', records)
                if plateau(records, config):
                    status = 'numerical_plateau'
                    break
            if step == config['gs_steps']:
                break
            field = torch.exp(1j * phi)
            propagated = propagate(field, factors)
            projected = desired * torch.exp(1j * torch.angle(propagated))
            gradient_correction = adjoint(projected - propagated, factors) / 4
            updated = field + step_size * gradient_correction
            phi = torch.angle(updated)
    torch.save({'phase': phi, 'step': step}, directory / 'last_phase.pt')
    result = {'status': status, 'updates': step, 'step_size': step_size,
              'best_selection_key': best_key, 'elapsed_seconds': time.monotonic() - start}
    save_json(directory / 'status.json', result)
    return best_phi, result
