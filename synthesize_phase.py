"""Initialize phase with the pretrained two-step U-Net and refine amplitude+SSIM."""

import hashlib
import json
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

from phase_initializer.algorithms import muon_direction
from phase_initializer.evaluate import export_result
from phase_initializer.models import PhysicsAdapterUNet
from phase_initializer.physics import (
    DISTANCES, N, PITCH, WAVELENGTH, amplitude_ssim, calibration, fresnel_factors, intensity,
)

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'inputs/network_phase/checkpoint_003.pt'
TARGETS = tuple(ROOT / f'inputs/network_phase/target_{j}.png' for j in range(1, 5))
CONFIG = {
    'seed': 17, 'steps': 3000, 'save_every': 100,
    'maximum_epsilon': 1e-12, 'amplitude_epsilon': 1e-8, 'ssim_weight': 0.5,
    'muon': {'lr': 0.1, 'momentum': 0.95, 'epsilon': 1e-12, 'ns_steps': 5,
             'coefficients': [3.4445, -4.775, 2.0315], 'scale': 1.0},
}


def save_json(path: Path, value) -> None:
    """Save readable finite JSON metadata for the numerical run."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def refine_phase(initial: torch.Tensor, target: torch.Tensor, factors: tuple,
                 config: dict, output: Path) -> torch.Tensor:
    """Update only the phase with fresh Muon momentum and the calibrated amplitude+SSIM loss."""
    phi = initial.detach().clone()
    momentum = torch.zeros_like(phi)
    scale = calibration(target, factors)
    with (output / 'history.jsonl').open('x') as history:
        for iteration in range(1, config['steps'] + 1):
            phi = phi.detach().requires_grad_(True)
            values = intensity(phi, factors)
            plane_loss = amplitude_ssim(values, target, scale, config)
            loss = plane_loss.mean()
            gradient, = torch.autograd.grad(loss, phi)
            with torch.no_grad():
                direction, momentum = muon_direction(gradient, momentum, config['muon'])
                phi = phi - config['muon']['lr'] * direction
            if not torch.isfinite(phi).all() or not torch.isfinite(loss):
                raise ValueError(f'Nonfinite phase or loss at update {iteration}.')
            history.write(json.dumps({'iteration': iteration, 'loss_before_update': loss.item(),
                                      'plane_loss_before_update': plane_loss.detach().tolist()},
                                     allow_nan=False) + '\n')
            if iteration % config['save_every'] == 0 or iteration == config['steps']:
                np.save(output / f'phase_{iteration:04d}.npy', phi.detach().numpy())
                print(f'{iteration}: loss_before_update={loss.item():.8f}', flush=True)
    np.save(output / 'momentum.npy', momentum.numpy())
    return phi.detach()


def run(checkpoint: Path = CHECKPOINT, targets: tuple = TARGETS, config: dict = CONFIG) -> Path:
    """Run one fixed scene with two internal GS+Muon steps and no network training."""
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    pixels = np.stack([np.array(Image.open(path)) for path in targets])
    if pixels.shape != (4, N, N) or pixels.dtype != np.uint8 or np.any(pixels.sum(axis=(1, 2)) == 0):
        raise ValueError('Expected four nonblack grayscale uint8 PNGs of size 128x128.')
    target = torch.from_numpy(pixels).double() / 255
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['config']['steps'] != 2:
        raise ValueError('This initializer requires a checkpoint trained with two internal steps.')
    model = PhysicsAdapterUNet(state['config']).double().eval()
    model.load_state_dict(state['model'], strict=True)
    factors = fresnel_factors(torch.float64)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = ROOT / 'results' / f'network_phase_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    source.mkdir()
    shutil.copy2(Path(__file__), source)
    shutil.copytree(ROOT / 'phase_initializer', source / 'phase_initializer',
                    ignore=shutil.ignore_patterns('__pycache__'))
    data = output / 'inputs'
    data.mkdir()
    for index, path in enumerate(targets, 1):
        shutil.copy2(path, data / f'target_{index}.png')
    shutil.copy2(checkpoint, data / 'checkpoint.pt')
    save_json(output / 'parameters.json', {
        **config, 'checkpoint_config': state['config'], 'internal_steps': 2,
        'network_updates': 0, 'device': 'cpu', 'dtype': 'float64', 'threads': 1,
        'N': N, 'pitch_m': PITCH, 'wavelength_m': WAVELENGTH, 'distances_m': DISTANCES,
        'incident_amplitude': 1, 'selection': 'Final iteration only; no best-checkpoint selection',
        'calibration': calibration(target, factors).flatten().tolist(),
        'inputs': [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in (checkpoint, *targets)],
    })
    save_json(output / 'environment.json', {
        'python': platform.python_version(), 'platform': platform.platform(),
        'torch': torch.__version__, 'numpy': np.__version__, 'pillow': PIL.__version__,
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
        'deterministic_algorithms': True, 'rng': 'torch.manual_seed; NumPy RNG unused',
        'limitations': 'No cross-platform bitwise guarantee; no physical robustness certification',
    })
    (source / 'working_tree.diff').write_bytes(
        subprocess.check_output(['git', 'diff', 'HEAD', '--binary'], cwd=ROOT))
    start = time.perf_counter()
    with torch.no_grad():
        initial = model(target[None])[0]
    inference_seconds = time.perf_counter() - start
    del model
    initial_dir = output / 'initial'
    initial_dir.mkdir()
    initial_metrics = export_result(initial_dir, initial, pixels, factors)
    start = time.perf_counter()
    phi = refine_phase(initial, target, factors, config, output)
    refinement_seconds = time.perf_counter() - start
    final_dir = output / 'final'
    final_dir.mkdir()
    final_metrics = export_result(final_dir, phi, pixels, factors)
    save_json(output / 'summary.json', {
        'initial_metrics': initial_metrics, 'final_metrics': final_metrics,
        'inference_seconds': inference_seconds, 'refinement_seconds': refinement_seconds,
        'timing': 'CPU, no warmup; refinement includes history and phase snapshots, excludes final export',
        'final_continuous_loss': amplitude_ssim(intensity(phi, factors), target,
                                               calibration(target, factors), config).mean().item(),
    })
    print(output, flush=True)
    return output


if __name__ == '__main__':
    run()
