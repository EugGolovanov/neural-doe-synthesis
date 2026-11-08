"""Small numerical checks for the standalone network-initialized phase solver."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from synthesize_phase import CHECKPOINT, CONFIG, TARGETS, refine_phase
from phase_initializer.evaluate import export_result, image_metrics
from phase_initializer.models import PhysicsAdapterUNet
from phase_initializer.physics import (
    N, amplitude_ssim, calibration, fresnel_factors, intensity, local_ssim,
)


def main() -> None:
    """Report float64 residuals against explicit tolerances without training the network."""
    torch.set_num_threads(1)
    torch.manual_seed(17)
    factors = fresnel_factors(torch.float64)
    pixels = np.stack([np.array(Image.open(path)) for path in TARGETS])
    target = torch.from_numpy(pixels).double() / 255
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model = PhysicsAdapterUNet(state['config']).double().eval()
    model.load_state_dict(state['model'], strict=True)
    with torch.no_grad():
        phi = model(target[None])[0]
    phase = phi.clone().requires_grad_(True)
    scale = calibration(target, factors)
    loss = amplitude_ssim(intensity(phase, factors), target, scale, CONFIG).mean()
    gradient, = torch.autograd.grad(loss, phase)
    direction = torch.randn_like(phase)
    direction /= direction.norm()
    h = 1e-4
    plus = amplitude_ssim(intensity(phi + h * direction, factors), target, scale, CONFIG).mean()
    minus = amplitude_ssim(intensity(phi - h * direction, factors), target, scale, CONFIG).mean()
    finite_difference = (plus - minus) / (2 * h)
    analytic = (gradient * direction).sum()
    derivative_error = (finite_difference - analytic).abs() / torch.maximum(analytic.abs(), finite_difference.abs())
    values = intensity(phi, factors)
    power = N**4 * factors[1].flatten().square()
    power_error = ((values.sum((-2, -1)) - power).abs() / power).max().item()
    normalized = values / values.amax((-2, -1), keepdim=True)
    independent = torch.tensor([image_metrics(pixels[j], normalized[j].numpy() * 255)['ssim']
                                for j in range(4)], dtype=torch.float64)
    ssim_error = (local_ssim(normalized, target) - independent).abs().max().item()
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        refined = refine_phase(phi, target, factors, {**CONFIG, 'steps': 1, 'save_every': 1}, output)
        metrics = export_result(output, refined, pixels, factors)
        phase_report = json.loads((output / 'phase_evaluation.json').read_text())
    checks = {
        'power_relative_error': (power_error, 1e-12),
        'directional_derivative_relative_error': (derivative_error.item(), 1e-5),
        'SSIM_absolute_error': (ssim_error, 1e-12),
        'phase_quantization_error_rad': (phase_report['phase_quantization_max_circular_error_rad'], np.pi / 255 + 1e-12),
    }
    for name, (error, tolerance) in checks.items():
        print(f'{name}: {error:.6e}; tolerance={tolerance:.6e}')
        if not np.isfinite(error) or error > tolerance:
            raise ValueError(f'Failed check: {name}')
    if not phase_report['npy_roundtrip_exact'] or len(metrics) != 4:
        raise ValueError('Incomplete export or inexact phase roundtrip.')
    print('Exact NPY roundtrip and four-plane PNG replay passed; one external update completed.')


if __name__ == '__main__':
    main()
