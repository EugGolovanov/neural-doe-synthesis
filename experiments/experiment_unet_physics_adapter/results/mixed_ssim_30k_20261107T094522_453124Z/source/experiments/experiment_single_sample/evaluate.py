"""Fixed-calibration optical metrics, independent selection, and phase export."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .physics import INCIDENT_POWER, intensity, loss_terms


def save_json(path: Path, value: object) -> None:
    """Write readable finite JSON without replacing an unrelated run directory."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def measure(values: torch.Tensor, target: torch.Tensor, regions: dict, config: dict) -> dict:
    """Report per-plane shape and optical power metrics on the fixed common detector grid."""
    terms = loss_terms(values, target, regions, config)
    normalized = values / regions['intensity_scale'][:, None, None]
    rows = []
    for k in range(4):
        a = normalized[k] - normalized[k].mean()
        b = target[k] - target[k].mean()
        denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
        pearson = float((a * b).sum() / denominator) if denominator > 0 else None
        mse = float((normalized[k] - target[k]).square().mean())
        fov_power = float(values[k].sum()) * config['output_pitch_m']**2
        eta = float(terms['efficiency'][k])
        rows.append({'plane': k + 1, 'ssim': float(terms['ssim'][k]), 'pearson': pearson,
                     'pearson_status': 'defined' if pearson is not None else 'constant_image',
                     'amplitude_mse_epsilon': float(terms['amplitude'][k]),
                     'amplitude_rmse': float((torch.sqrt(normalized[k]) - torch.sqrt(target[k])).square().mean().sqrt()),
                     'intensity_mse': mse, 'intensity_nrmse': math.sqrt(mse / float(target[k].square().mean())),
                     'useful_efficiency': eta, 'useful_power_A0_squared_m2': eta * INCIDENT_POWER,
                     'fov_efficiency': fov_power / INCIDENT_POWER,
                     'guard_efficiency': fov_power / INCIDENT_POWER - eta,
                     'guard_power_A0_squared_m2': fov_power - eta * INCIDENT_POWER,
                     'spectral_residual': float(terms['frequency'][k]),
                     'guard_loss': float(terms['guard'][k]),
                     'fixed_scale_maximum': float(normalized[k].max())})
    return {'planes': rows, 'mean_ssim': float(terms['ssim'].mean()), 'min_ssim': float(terms['ssim'].min()),
            'mean_eta': float(terms['efficiency'].mean()), 'min_eta': float(terms['efficiency'].min())}


def selection_key(metrics: dict, efficiency_floor: torch.Tensor) -> tuple:
    """Rank by power feasibility first and then by a non-training structural and efficiency score."""
    deficits = [max(0.0, float(floor) - row['useful_efficiency']) for floor, row in zip(efficiency_floor, metrics['planes'])]
    score = 0.5 * metrics['mean_ssim'] + 0.5 * metrics['min_ssim'] + 0.1 * metrics['mean_eta']
    return (max(deficits) == 0, -max(deficits), score)


def relative_error(a: torch.Tensor, b: torch.Tensor) -> list:
    """Return per-plane intensity relative L2 differences against a nonzero reference."""
    return [float(torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(y)) for x, y in zip(a, b)]


def export_result(directory: Path, phi: torch.Tensor, target: torch.Tensor, regions: dict,
                  training_factors: tuple, factors: tuple, refined_factors: tuple, config: dict) -> dict:
    """Export continuous and decoded 255-interval phases with absolute-scale maps and refinement checks."""
    directory.mkdir(exist_ok=False)
    continuous = phi.detach().cpu().double()
    np.save(directory / 'phase_continuous_rad.npy', continuous.numpy())
    wrapped = (continuous + math.pi) % (2 * math.pi) - math.pi
    codes = torch.floor(255 * (wrapped + math.pi) / (2 * math.pi) + 0.5).to(torch.uint8).numpy()
    Image.fromarray(codes).save(directory / 'doe.png')
    with Image.open(directory / 'doe.png') as image:
        loaded = np.array(image, dtype=np.float64)
    decoded = torch.tensor(-math.pi + 2 * math.pi * loaded / 255)
    report = {'phase_rounding_rms_rad': float(torch.angle(torch.exp(1j * (decoded - continuous))).square().mean().sqrt())}
    for name, phase in (('continuous', continuous), ('quantized_reimported', decoded)):
        with torch.no_grad():
            values = intensity(phase, factors)
            refined = intensity(phase, refined_factors)
            report[name] = measure(values, target, regions, config)
            errors = relative_error(values, refined)
            training_errors = relative_error(intensity(phase, training_factors), values)
            power_bound = all(0 <= row['fov_efficiency'] <= 1 + config['physical_tolerance'] for row in report[name]['planes'])
            report[name]['refinement_relative_l2'] = errors
            report[name]['training_vs_validation_relative_l2'] = training_errors
            report[name]['fov_power_bound_passed'] = power_bound
            report[name]['numerically_converged'] = max(errors + training_errors) < config['physical_tolerance'] and power_bound
        np.save(directory / f'{name}_intensity.npy', values.numpy())
        np.save(directory / f'{name}_refined_intensity.npy', refined.numpy())
        for k, value in enumerate(values.numpy(), 1):
            Image.fromarray(value.astype(np.float32)).save(directory / f'{name}_plane_{k}_absolute.tiff')
    save_json(directory / 'metrics.json', report)
    return report
