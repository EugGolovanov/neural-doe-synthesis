"""Explicit single-instance protocol and four matched-geometry experimental controls."""

import hashlib
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
import torch

from experiments.experiment_unet_physics_adapter.experiment import build_model
from .algorithms import optimize_weights, run_adjoint_gs
from .data import load_sample
from .evaluate import export_result, measure, relative_error, save_json
from .physics import DISTANCES, INCIDENT_POWER, combined_loss, intensity, original_loss, pixel_kernels


ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = 'experiments/experiment_unet_physics_adapter/results/mnist_10k_20261107T065810_115710Z'
CONFIG = {
    'checkpoint': f'{SOURCE_RUN}/checkpoint_50_040.pt',
    'sample_directory': f'{SOURCE_RUN}/data/validation/0000',
    'output_root': 'experiments/experiment_single_sample/results',
    'seed': 17, 'threads': 1, 'device': 'cpu', 'dtype': 'float64',
    'output_pitch_m': 10e-6, 'pixel_order': 8, 'detector_samples': 4,
    'validation_pixel_order': 16, 'validation_detector_samples': 8,
    'refined_pixel_order': 32, 'refined_detector_samples': 16,
    'physical_tolerance': 0.01, 'support_threshold': 0.1,
    'target_useful_efficiency': 0.5, 'minimum_efficiency': 0.35,
    'allowed_efficiency_drop': 0.005,
    'epsilon': 1e-8, 'guard_epsilon': 1e-8, 'original_epsilon': 1e-12,
    'frequency_band_weights': [0.25, 1.0, 2.0],
    'loss_weights': {'amplitude': 1.0, 'ssim': 0.5, 'efficiency': 5.0,
                     'guard': 0.02, 'frequency': 0.1, 'worst': 0.25},
    'worst_beta': 10.0, 'warmup_steps': 50, 'ramp_steps': 150,
    'learning_rate': 1e-5, 'adam_betas': [0.9, 0.999], 'adam_epsilon': 1e-8,
    'max_steps': 3000, 'gs_steps': 3000, 'evaluate_every': 20,
    'minimum_steps': 400, 'plateau_evaluations': 10,
    'loss_plateau_tolerance': 1e-4, 'score_plateau_tolerance': 1e-4,
    'intensity_plateau_tolerance': 1e-4,
    'success_mean_ssim_gain': 0.05, 'success_min_ssim_gain': 0.03,
}


def save_provenance(output: Path, config: dict, manifest: list) -> None:
    """Snapshot the actual implementation, inputs, environment, and existing code revision for this run."""
    source = output / 'source'
    source.mkdir()
    for name in ('experiment_single_sample', 'experiment_unet_physics_adapter', 'experiment_gs_muon', 'experiment_unrolled_cnn'):
        destination = source / 'experiments' / name
        destination.mkdir(parents=True)
        for path in (ROOT / 'experiments' / name).glob('*.py'):
            shutil.copy2(path, destination / path.name)
    shutil.copytree(ROOT / 'docs', source / 'docs', ignore=shutil.ignore_patterns('papers'))
    shutil.copy2(Path(__file__).with_name('README.md'), source / 'README.md')
    (source / 'working_tree.diff').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD', '--binary'], cwd=ROOT))
    checkpoint = ROOT / config['checkpoint']
    save_json(output / 'provenance.json', {
        'python': platform.python_version(), 'platform': platform.platform(),
        'torch': torch.__version__, 'numpy': np.__version__, 'pillow': PIL.__version__,
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'working_tree_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'inputs': manifest,
        'source_sha256': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*.py'))},
        'rng': f'torch.manual_seed({config["seed"]}); no sampling, augmentation, or shuffling',
        'deterministic_algorithms': True, 'device': 'cpu', 'dtype': 'float64/complex128',
        'limitations': 'No cross-platform bitwise guarantee; this tuple is deliberately optimized, not held-out validation.'})


def run_experiment(config: dict) -> Path:
    """Run checkpoint, original-loss, combined-loss, and adjoint-GS controls for one fixed physical tuple."""
    if config['device'] != 'cpu' or config['dtype'] != 'float64':
        raise ValueError('This verified finite-pixel experiment uses CPU float64 without device fallback.')
    if config['ramp_steps'] <= 0 or config['evaluate_every'] <= 0 or config['max_steps'] < 0 or config['gs_steps'] < 0:
        raise ValueError('Invalid ramp, evaluation interval, or update budget.')
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config['seed'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = ROOT / config['output_root'] / stamp
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'config.json', config)
    save_json(output / 'status.json', {'status': 'initializing'})
    target, regions, manifest = load_sample(ROOT / config['sample_directory'], config)
    save_provenance(output, config, manifest)
    target_directory = output / 'targets'
    target_directory.mkdir()
    for entry in manifest:
        shutil.copy2(entry['path'], target_directory / f'target_{entry["plane"]}.png')
    factors = pixel_kernels(config['output_pitch_m'], config['detector_samples'], config['pixel_order'])
    validation_factors = pixel_kernels(config['output_pitch_m'], config['validation_detector_samples'], config['validation_pixel_order'])
    refined_factors = pixel_kernels(config['output_pitch_m'], config['refined_detector_samples'], config['refined_pixel_order'])
    checkpoint = torch.load(ROOT / config['checkpoint'], map_location='cpu', weights_only=True)
    model_config = {**checkpoint['config'], 'device': 'cpu', 'dtype': 'float64'}
    model = build_model(model_config)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    torch.save({'model': model.state_dict(), 'config': model_config}, output / 'initial_weights.pt')
    with torch.no_grad():
        initial_phi = model(target[None])[0]
        initial_values = intensity(initial_phi, validation_factors)
        initial_metrics = measure(initial_values, target, regions, config)
        efficiency = torch.tensor([row['useful_efficiency'] for row in initial_metrics['planes']], dtype=torch.float64)
        regions['efficiency_floor'] = torch.maximum(torch.zeros(4, dtype=torch.float64), efficiency - config['allowed_efficiency_drop'])
        regions['eta_required'] = torch.maximum(regions['eta_required'], regions['efficiency_floor'])
        errors = relative_error(intensity(initial_phi, factors), initial_values)
    np.save(output / 'signal_regions.npy', regions['signal'].numpy())
    np.save(output / 'target_normalized.npy', target.numpy())
    save_json(output / 'physical_contract.json', {
        'shape': [4, 128, 128], 'axis_order': 'plane,y,x', 'distances_m': DISTANCES,
        'coordinates_m': '(index - 64) * output_pitch_m at every plane; finite detector cells centered here',
        'D': 'entire 128x128 common FOV', 'S': 'target > support_threshold', 'guard': 'D minus S',
        'target_interpretation': 'Source PNGs reinterpreted on the same common 10um grid, not their historical z-dependent grids.',
        'intensity_scale_per_plane': regions['intensity_scale'].tolist(),
        'eta_required': regions['eta_required'].tolist(), 'selection_efficiency_floor': regions['efficiency_floor'].tolist(),
        'incident_power_A0_squared_m2': INCIDENT_POWER, 'absolute_watts': 'undefined without physical A0 calibration',
        'initial_train_vs_validation_relative_l2': errors,
        'legacy_adapter': 'Original architecture and internal GS+Muon transform unchanged; only external physical propagation and objectives replaced.'})
    if max(errors) >= config['physical_tolerance']:
        save_json(output / 'status.json', {'status': 'failed_initial_quadrature_check', 'errors': errors})
        raise ValueError(f'Training quadrature is not converged at tolerance {config["physical_tolerance"]}: {errors}')
    results = {}
    results['checkpoint'] = export_result(output / 'checkpoint', initial_phi, target, regions, factors, validation_factors, refined_factors, config)
    budgets = {}
    for name, objective in (('original_loss', original_loss), ('full_loss', combined_loss)):
        save_json(output / 'status.json', {'status': 'optimizing', 'control': name})
        model = build_model(model_config)
        model.load_state_dict(checkpoint['model'], strict=True)
        phi, budgets[name] = optimize_weights(model, target, regions, factors, validation_factors, config, objective, output / name)
        results[name] = export_result(output / name / 'best', phi, target, regions, factors, validation_factors, refined_factors, config)
    save_json(output / 'status.json', {'status': 'optimizing', 'control': 'adjoint_gs'})
    phi, budgets['adjoint_gs'] = run_adjoint_gs(initial_phi, target, regions, factors, validation_factors, config, output / 'adjoint_gs')
    results['adjoint_gs'] = export_result(output / 'adjoint_gs' / 'best', phi, target, regions, factors, validation_factors, refined_factors, config)
    baseline = results['checkpoint']['continuous']
    candidate = results['full_loss']['continuous']
    power_preserved = all(row['useful_efficiency'] >= float(floor) for row, floor in zip(candidate['planes'], regions['efficiency_floor']))
    numerical = all(result[phase]['numerically_converged'] for result in results.values() for phase in ('continuous', 'quantized_reimported'))
    success = (candidate['mean_ssim'] >= baseline['mean_ssim'] + config['success_mean_ssim_gain']
               and candidate['min_ssim'] >= baseline['min_ssim'] + config['success_min_ssim_gain'] and power_preserved and numerical)
    save_json(output / 'comparison.json', {'results': results, 'budgets': budgets, 'success': success,
                                         'numerical_refinement_passed': numerical,
                                         'success_scope': 'Predeclared single-instance continuous-phase criterion, not generalization or exact scalar-wave validation.'})
    save_json(output / 'status.json', {'status': 'completed', 'success': success, 'budgets': budgets})
    print(output, flush=True)
    return output


if __name__ == '__main__':
    run_experiment(CONFIG)
