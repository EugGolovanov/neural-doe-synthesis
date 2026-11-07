"""Обучение компактной CNN-развёртки и сравнение с GS + Muon на отложенных целях."""

import hashlib
import json
import math
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
import torch
from PIL import Image

from experiments.experiment_gs_muon.algorithms import run_gs_muon
from experiments.experiment_gs_muon.evaluate import export_result
from experiments.experiment_gs_muon.physics import DISTANCES, N, PITCH, WAVELENGTH, fresnel_factors
from .algorithms import initial_momenta, losses, unroll, update_muon
from .data import prepare_data
from .models import CorrectionNet


CONFIG = {
    'seeds': [17, 29, 43], 'data_seed': 2026,
    'split_sizes': {'train': 24, 'validation': 6, 'test': 6},
    'epochs': 10, 'batch_size': 1, 'width': 8, 'steps': 3,
    'step_size': 1000.0, 'correction_scale': 0.1,
    'loss_epsilon': 1e-12, 'gradient_epsilon': 1e-8,
    'device': 'cpu', 'dtype': 'float32', 'threads': 1,
    'baseline_steps': 100,
    'muon': {'lr': 0.01, 'momentum': 0.95, 'epsilon': 1e-12,
             'ns_steps': 5, 'coefficients': [3.4445, -4.7750, 2.0315], 'scale': 1.0},
    'baseline_muon': {'lr': 0.1, 'momentum': 0.95, 'epsilon': 1e-12,
                      'ns_steps': 5, 'coefficients': [3.4445, -4.7750, 2.0315], 'scale': 1.0},
}


def save_json(path: Path, value: object) -> None:
    """Сохраняет JSON с явным отказом при нечисловых результатах."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def snapshot_sources(output: Path, config: dict) -> None:
    """Сохраняет исполняемые исходники, документы, ревизию и фактическое окружение."""
    root = Path(__file__).resolve().parents[2]
    snapshot = output / 'source'
    paths = list(Path(__file__).parent.glob('*.py')) + list(Path(__file__).parent.glob('*.md'))
    baseline = root / 'experiments' / 'experiment_gs_muon'
    paths += [baseline / name for name in ('physics.py', 'algorithms.py', 'evaluate.py')]
    paths += list((root / 'docs').glob('*.md')) + [root / 'AGENTS.md']
    hashes = {}
    for path in paths:
        relative = path.relative_to(root)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True)
    diff = subprocess.check_output(['git', 'diff', 'HEAD', '--binary'], cwd=root)
    (snapshot / 'working_tree.diff').write_bytes(diff)
    save_json(output / 'parameters.json', {
        'config': config, 'revision': revision, 'git_status': status, 'source_sha256': hashes,
        'python': platform.python_version(), 'platform': platform.platform(),
        'processor': platform.processor(), 'torch': torch.__version__,
        'numpy': np.__version__, 'pillow': PIL.__version__,
        'physics': {'N': N, 'wavelength_m': WAVELENGTH, 'pitch_m': PITCH, 'z_m': DISTANCES},
        'deterministic_algorithms': True, 'export_dtype': 'float64/complex128',
        'rng': 'NumPy PCG64 (dataset-specific rule in README/config); torch CPU manual_seed(seed)',
        'timing': 'CPU batch=1; full CNN warmup and one baseline iteration; export excluded',
        'peak_memory': None, 'peak_memory_status': 'not_measured_on_cpu',
    })


def train_model(data: dict, initial: torch.Tensor, factors: tuple, config: dict,
                directory: Path) -> tuple[CorrectionNet, float]:
    """Обучает общую CNN с Muon и выбирает checkpoint только по validation loss."""
    model = CorrectionNet(config['width'])
    momenta = initial_momenta(model)
    history = []
    best_validation = math.inf
    started = time.perf_counter()
    for epoch in range(config['epochs'] + 1):
        model.train()
        training_losses, gradient_norms = [], []
        if epoch > 0:
            for index in torch.randperm(len(data['train'])):
                target = data['train'][index:index + 1]
                model.zero_grad(set_to_none=True)
                phi = unroll(model, target, initial, factors, config, training=True)
                per_plane = losses(phi, target, factors, config['loss_epsilon'])
                per_plane.mean().backward()
                norm = torch.stack([p.grad.square().sum() for p in model.parameters()]).sum().sqrt()
                gradient_norms.append(norm.item())
                training_losses.append(per_plane.detach()[0].tolist())
                update_muon(model, momenta, config['muon'])
        model.eval()
        validation = []
        for target in data['validation']:
            phi = unroll(model, target[None], initial, factors, config, training=False)
            validation.append(losses(phi, target[None], factors, config['loss_epsilon'])[0].tolist())
        mean_validation = float(np.mean(validation))
        if mean_validation < best_validation:
            best_validation = mean_validation
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'validation_loss': mean_validation}, directory / 'best.pt')
        history.append({'epoch': epoch, 'training_loss_per_plane': training_losses,
                        'gradient_norms': gradient_norms, 'validation_loss_per_plane': validation})
        save_json(directory / 'training_history.json', history)
        print(f'{directory.name} epoch={epoch}: validation={mean_validation:.6g}', flush=True)
    elapsed = time.perf_counter() - started
    torch.save({'model': model.state_dict(), 'momenta': momenta, 'rng': torch.get_rng_state(),
                'config': config, 'epoch': config['epochs']}, directory / 'last.pt')
    best = torch.load(directory / 'best.pt', weights_only=True)
    model.load_state_dict(best['model'])
    model.eval()
    save_json(directory / 'training_summary.json', {
        'seconds_including_validation_and_checkpoints': elapsed,
        'selected_epoch': best['epoch'], 'validation_loss': best['validation_loss'],
        'parameters': sum(p.numel() for p in model.parameters()),
    })
    return model, elapsed


def compare_test(model, data: dict, initial: torch.Tensor, factors: tuple,
                 config: dict, directory: Path) -> list:
    """Сравнивает CNN и GS + Muon по отдельным тестовым целям после PNG-экспорта."""
    evaluation_factors = fresnel_factors(torch.float64)
    summary = []
    for index, target in enumerate(data['test']):
        unroll(model, target[None], initial, factors, config, training=False)
        started = time.perf_counter()
        predicted = unroll(model, target[None], initial, factors, config, training=False)[0, 0]
        cnn_seconds = time.perf_counter() - started
        run_gs_muon(initial[0, 0], target, factors, 1,
                    config['loss_epsilon'], config['baseline_muon'])
        started = time.perf_counter()
        baseline, history = run_gs_muon(initial[0, 0], target, factors, config['baseline_steps'],
                                        config['loss_epsilon'], config['baseline_muon'])
        baseline_seconds = time.perf_counter() - started
        codes = torch.floor(target * 255 + 0.5).to(torch.uint8).numpy()
        for method, phi, seconds in [('cnn', predicted, cnn_seconds),
                                     ('gs_muon', baseline, baseline_seconds)]:
            path = directory / 'test' / f'{index:04d}' / method
            path.mkdir(parents=True, exist_ok=False)
            phi = phi.double()
            np.save(path / 'continuous_phi.npy', phi.numpy())
            metrics = export_result(path, phi, codes, evaluation_factors)
            with Image.open(path / 'doe.png') as image:
                decoded = torch.tensor(np.array(image), dtype=torch.float64)
            quantized = -math.pi + 2 * math.pi * decoded / 255
            continuous_loss = losses(phi[None, None], target[None].double(), evaluation_factors,
                                     config['loss_epsilon'])[0].tolist()
            quantized_loss = losses(quantized[None, None], target[None].double(), evaluation_factors,
                                    config['loss_epsilon'])[0].tolist()
            if method == 'gs_muon':
                save_json(path / 'history.json', history)
            entry = {'example': index, 'method': method, 'seconds': seconds, 'metrics': metrics,
                     'continuous_loss_per_plane': continuous_loss,
                     'quantized_loss_per_plane': quantized_loss,
                     'optimization_propagations': 8 * config['steps'] if method == 'cnn'
                     else 16 * config['baseline_steps'],
                     'diagnostic_propagations': 0 if method == 'cnn'
                     else 4 * (config['baseline_steps'] + 1)}
            save_json(path / 'metrics.json', entry)
            summary.append(entry)
    return summary


def run_experiment(config: dict, output: Path) -> None:
    """Запускает воспроизводимый эксперимент с фиксированным CPU-контрактом и новым каталогом."""
    if config['device'] != 'cpu' or config['dtype'] != 'float32' or config['batch_size'] != 1:
        raise ValueError('Этот эксперимент определён для CPU, float32, batch_size=1.')
    if config['epochs'] < 1 or config['steps'] < 1 or min(config['split_sizes'].values()) < 1:
        raise ValueError('Нужны положительные число эпох, шагов и размеры всех выборок.')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    snapshot_sources(output, config)
    data, manifest = prepare_data(output / 'data', config['split_sizes'], config['data_seed'])
    save_json(output / 'data_manifest.json', manifest)
    factors = fresnel_factors(torch.float32)
    summary = []
    for seed in config['seeds']:
        torch.manual_seed(seed)
        directory = output / f'seed_{seed}'
        directory.mkdir()
        initial = (2 * torch.rand(1, 1, N, N) - 1) * math.pi
        np.save(directory / 'initial_phi.npy', initial.numpy())
        model, training_seconds = train_model(data, initial, factors, config, directory)
        for entry in compare_test(model, data, initial, factors, config, directory):
            summary.append({'seed': seed, 'training_seconds': training_seconds, **entry})
        save_json(output / 'summary.json', summary)


if __name__ == '__main__':
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run_experiment(CONFIG, Path(__file__).resolve().parent / 'results' / f'cnn_{stamp}')
