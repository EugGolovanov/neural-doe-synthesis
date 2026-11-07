"""Диагностика сохранённого MNIST-запуска на validation без повторного обучения."""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
from PIL import Image

from experiments.experiment_gs_muon.algorithms import run_gs, run_gs_muon
from experiments.experiment_gs_muon.physics import fresnel_factors, intensity
from experiments.experiment_unrolled_cnn.algorithms import losses, unroll
from experiments.experiment_unrolled_cnn.models import CorrectionNet
from .models import FNOCorrection, UNetCorrection


def statistics(phi, target, factors, epsilon):
    """Измеряет непрерывные MSE, Pearson и пиковость интенсивности по четырём плоскостям."""
    with torch.no_grad():
        values = intensity(phi, factors)
        normalized = values / (values.amax((-2, -1), keepdim=True) + epsilon)
        centered = normalized - normalized.mean((-2, -1), keepdim=True)
        reference = target - target.mean((-2, -1), keepdim=True)
        pearson = (centered * reference).sum((-2, -1)) / (
            centered.square().sum((-2, -1)) * reference.square().sum((-2, -1))).sqrt()
        return {
            'loss': float((normalized - target).square().mean()),
            'pearson': pearson.flatten().tolist(),
            'peak_to_mean': (values.amax((-2, -1)) / values.mean((-2, -1))).flatten().tolist(),
        }


def gradient_descent(initial, target, factors, epsilon, step_size, steps):
    """Выполняет фиксированный градиентный спуск по исходному max-нормированному MSE."""
    phi = initial.clone()
    for _ in range(steps):
        phi = phi.detach().requires_grad_(True)
        loss = losses(phi, target, factors, epsilon).mean()
        gradient, = torch.autograd.grad(loss, phi)
        phi = phi - step_size * gradient
    return phi.detach()


def diagnose(directory):
    """Сопоставляет обученные модели с физическими контролями на сохранённых validation-целях."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(123)
    config = json.loads((directory / 'parameters.json').read_text())['config']
    seed_directory = directory / 'seed_17'
    initial = torch.from_numpy(np.load(seed_directory / 'initial_phi.npy'))
    targets = []
    inputs = {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (directory / 'parameters.json', seed_directory / 'initial_phi.npy')
    }
    for example in sorted((directory / 'data/validation').iterdir()):
        planes = []
        for path in sorted(example.glob('target_*.png')):
            inputs[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as image:
                planes.append(np.array(image))
        targets.append(torch.tensor(np.stack(planes), dtype=torch.float32)[None] / 255)
    factors = fresnel_factors(torch.float32)
    epsilon = config['loss_epsilon']
    settings = config['backbones']
    models = {
        'cnn': CorrectionNet(settings['cnn_width']),
        'unet': UNetCorrection(settings['unet_width']),
        'fno': FNOCorrection(settings['fno_width'], settings['fno_modes'], settings['fno_blocks']),
    }
    for name, model in models.items():
        path = seed_directory / name / 'best.pt'
        inputs[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
        model.load_state_dict(torch.load(path, weights_only=True)['model'])
        model.eval()
    rows = []
    for index, target in enumerate(targets):
        phase = initial.clone().requires_grad_(True)
        values = intensity(phase, factors)
        maximum = values.amax((-2, -1), keepdim=True)
        full_loss = ((values / (maximum + epsilon) - target).square()).mean()
        fixed_loss = ((values / (maximum.detach() + epsilon) - target).square()).mean()
        full_gradient, = torch.autograd.grad(full_loss, phase, retain_graph=True)
        fixed_gradient, = torch.autograd.grad(fixed_loss, phase)
        maximum_gradient = full_gradient - fixed_gradient
        gradient_info = {
            'physical_step_rms_rad': float((config['step_size'] * full_gradient).square().mean().sqrt()),
            'max_term_to_fixed_denominator_gradient_norm': float(maximum_gradient.norm() / fixed_gradient.norm()),
            'full_vs_fixed_gradient_cosine': float((full_gradient * fixed_gradient).sum() / (full_gradient.norm() * fixed_gradient.norm())),
        }
        candidates = {'initial': initial}
        for steps in (3, 10, 100):
            candidates[f'gd_{steps}'] = gradient_descent(initial, target, factors, epsilon, config['step_size'], steps)
        for eta in (100.0, 10000.0):
            candidates[f'gd_3_eta_{eta:g}'] = gradient_descent(initial, target, factors, epsilon, eta, 3)
        for steps in (1, 3, 10, 100):
            candidates[f'gs_{steps}'], _ = run_gs(initial[0, 0], target[0], factors, steps, epsilon)
        candidates['gs_muon_100'], _ = run_gs_muon(initial[0, 0], target[0], factors, 100, epsilon, config['baseline_muon'])
        for name, model in models.items():
            candidates[name] = unroll(model, target, initial, factors, config, training=False)
        results = {name: statistics(phi, target, factors, epsilon) for name, phi in candidates.items()}
        for name in models:
            results[name]['distance_to_gd_3_rms_rad'] = float((candidates[name] - candidates['gd_3']).square().mean().sqrt())
        rows.append({'validation_example': index, 'gradient': gradient_info, 'methods': results})
    aggregates = {}
    for name in rows[0]['methods']:
        entries = [row['methods'][name] for row in rows]
        aggregates[name] = {
            'mean_loss': float(np.mean([entry['loss'] for entry in entries])),
            'mean_pearson': float(np.mean([entry['pearson'] for entry in entries])),
            'mean_peak_to_mean': float(np.mean([entry['peak_to_mean'] for entry in entries])),
        }
    return {
        'dataset_directory': str(directory), 'split': 'validation', 'training': 'none',
        'python': platform.python_version(), 'platform': platform.platform(),
        'torch': torch.__version__, 'numpy': np.__version__, 'pillow': PIL.__version__,
        'threads': 1, 'device': 'cpu', 'seed': 123, 'deterministic_algorithms': True,
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True),
        'diagnostic_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'evaluation': 'continuous float32, no PNG export; not final protocol metrics',
        'input_sha256': inputs, 'config': config, 'aggregates': aggregates, 'examples': rows,
    }


if __name__ == '__main__':
    print(json.dumps(diagnose(Path(sys.argv[1])), indent=2, allow_nan=False))
