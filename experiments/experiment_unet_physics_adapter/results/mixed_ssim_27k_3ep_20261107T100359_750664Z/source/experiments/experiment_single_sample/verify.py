"""Numerical residual checks and a bounded end-to-end single-tuple smoke experiment."""

import json
import numpy as np
import torch

from experiments.experiment_gs_muon.evaluate import image_metrics
from .data import load_sample
from .evaluate import relative_error, save_json
from .experiment import CONFIG, ROOT, run_experiment
from .physics import adjoint, combined_loss, intensity, pixel_kernels, propagate, ssim


def numerical_checks() -> dict:
    """Check adjointness, quadrature, physical invariances, SSIM and directional objective gradients."""
    torch.set_num_threads(1)
    torch.manual_seed(17)
    target, regions, _ = load_sample(ROOT / CONFIG['sample_directory'], CONFIG)
    factors = pixel_kernels(CONFIG['output_pitch_m'], 4, 8)
    refined_source = pixel_kernels(CONFIG['output_pitch_m'], 4, 16)
    refined_detector = pixel_kernels(CONFIG['output_pitch_m'], 8, 16)
    phi = (torch.rand((128, 128), dtype=torch.float64) * 2 - 1) * np.pi
    field = torch.exp(1j * phi)
    propagated = propagate(field, factors)
    probe = torch.randn(propagated.shape, dtype=torch.complex128)
    left = torch.vdot(propagated.flatten(), probe.flatten())
    right = torch.vdot(field.flatten(), adjoint(probe, factors).flatten())
    errors = {'adjoint_relative': float((left - right).abs() / left.abs())}
    values = intensity(phi, factors)
    errors['source_quadrature_relative'] = max(relative_error(values, intensity(phi, refined_source)))
    errors['detector_quadrature_relative'] = max(relative_error(values, intensity(phi, refined_detector)))
    errors['global_phase_relative'] = max(relative_error(intensity(phi + 0.7, factors), values))
    errors['periodicity_relative'] = max(relative_error(intensity(phi + 2 * np.pi, factors), values))
    a = torch.rand((4, 128, 128), dtype=torch.float64)
    reference = [image_metrics((x * 255).numpy(), (y * 255).numpy())['ssim'] for x, y in zip(a, target)]
    errors['ssim_absolute'] = float((ssim(a, target) - torch.tensor(reference, dtype=torch.float64)).abs().max())
    phase = phi.clone().requires_grad_()
    loss = combined_loss(intensity(phase, factors), target, regions, CONFIG, 200)
    gradient, = torch.autograd.grad(loss, phase)
    direction = torch.randn_like(phase)
    direction = direction / torch.linalg.vector_norm(direction)
    analytic = float((gradient * direction).sum())
    differences = []
    for step_size in (1e-3, 1e-4, 1e-5):
        positive = combined_loss(intensity(phi + step_size * direction, factors), target, regions, CONFIG, 200)
        negative = combined_loss(intensity(phi - step_size * direction, factors), target, regions, CONFIG, 200)
        finite = float((positive - negative) / (2 * step_size))
        differences.append({'step': step_size, 'analytic': analytic, 'finite': finite,
                            'relative_error': abs(finite - analytic) / max(abs(analytic), 1e-12)})
    errors['directional_gradient_relative'] = min(row['relative_error'] for row in differences)
    errors['global_phase_gradient_relative'] = float(gradient.sum().abs() / torch.linalg.vector_norm(gradient))
    tolerances = {'adjoint_relative': 1e-10, 'source_quadrature_relative': 1e-10,
                  'detector_quadrature_relative': 0.01, 'global_phase_relative': 1e-10,
                  'periodicity_relative': 1e-10, 'ssim_absolute': 1e-8,
                  'directional_gradient_relative': 1e-5, 'global_phase_gradient_relative': 1e-10}
    return {'errors': errors, 'tolerances': tolerances,
            'basis': '1e-10 complex128 algebra; 1% integration refinement; 1e-5 finite-difference truncation/cancellation; SSIM 1e-8.',
            'directional_differences': differences,
            'passed': all(value < tolerances[name] for name, value in errors.items())}


def main() -> None:
    """Save measured numerical checks and run exactly two updates per network without claiming convergence."""
    report = numerical_checks()
    print(json.dumps(report, indent=2), flush=True)
    if not report['passed']:
        raise RuntimeError('Numerical checks failed; no optimization was launched.')
    output = run_experiment({**CONFIG, 'max_steps': 2, 'gs_steps': 2, 'evaluate_every': 1,
                             'warmup_steps': 0, 'ramp_steps': 1, 'purpose': 'two-update smoke only'})
    initial = torch.load(output / 'initial_weights.pt', weights_only=True)
    updates = {}
    for name in ('original_loss', 'full_loss'):
        last = torch.load(output / name / 'last_checkpoint.pt', weights_only=True)
        difference = sum((last['model'][key] - value).square().sum() for key, value in initial['model'].items())
        optimizer_steps = [float(state['step']) for state in last['optimizer']['state'].values()]
        updates[name] = {'weight_change_l2': float(difference.sqrt()), 'optimizer_step_range': [min(optimizer_steps), max(optimizer_steps)],
                         'passed': float(difference) > 0 and min(optimizer_steps) == 2 and max(optimizer_steps) == 2}
    comparison = json.loads((output / 'comparison.json').read_text())
    report['smoke'] = updates
    report['export_refinement_passed'] = comparison['numerical_refinement_passed']
    report['passed'] = report['passed'] and all(value['passed'] for value in updates.values()) and report['export_refinement_passed']
    save_json(output / 'verification.json', report)
    print(f'VERIFICATION={output / "verification.json"}', flush=True)
    if not report['passed']:
        raise RuntimeError('Smoke verification failed; inspect the saved report.')


if __name__ == '__main__':
    main()
