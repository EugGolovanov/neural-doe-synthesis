"""Diagnose the fixed completed MNIST run without changing its evaluation protocol."""

import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import scipy
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import map_coordinates
import torch
from PIL import Image

from experiments.experiment_gs_muon.physics import N, PITCH, WAVELENGTH, DISTANCES, fresnel_factors, intensity
from experiments.experiment_gs_muon.evaluate import image_metrics
from experiments.experiment_gs_muon.algorithms import run_gs, run_gs_muon
from .experiment import build_model

RUN = Path(__file__).resolve().parent / 'results/mnist_10k_20261107T065810_115710Z'
OUTPUT = RUN / 'investigation_v1'


def load_targets(split: str, index: int) -> np.ndarray:
    """Read the original target intensities in plane-row-column order."""
    return np.stack([np.asarray(Image.open(RUN / f'data/{split}/{index:04d}/target_{k}.png'), dtype=float) / 255 for k in range(1, 5)])


def metrics(values: np.ndarray, target: np.ndarray) -> list:
    """Measure continuous max-normalized fidelity and explicitly post-hoc signal support."""
    rows = []
    for k, (value, goal) in enumerate(zip(values, target)):
        normalized = value / value.max()
        roi = goal > 0.1
        pitch = WAVELENGTH * DISTANCES[k] / (N * PITCH)
        efficiency = float(value[roi].sum() * pitch**2 / (N * PITCH)**2)
        rows.append({**image_metrics(goal * 255, normalized * 255),
                     'mse': float(np.mean((normalized - goal)**2)),
                     'amplitude_rmse': float(np.sqrt(np.mean((np.sqrt(normalized) - np.sqrt(goal))**2))),
                     'signal_efficiency': efficiency,
                     'window_power_over_incident': float(value.sum() * pitch**2 / (N * PITCH)**2),
                     'background_fraction': float(value[~roi].sum() / value.sum()),
                     'roi_area_fraction': float(roi.mean()),
                     'roi_intensity_cv_not_pure_speckle': float(value[roi].std() / value[roi].mean()),
                     'peak_over_mean': float(value.max() / value.mean())})
    return rows


def pixel_fresnel(phi: np.ndarray, subdivision: int) -> np.ndarray:
    """Integrate a piecewise-constant pixel mask by midpoint quadrature on the original output coordinates."""
    s = subdivision
    dx = PITCH / s
    x = (np.arange(N * s) - N * s / 2 - (s - 1) / 2) * dx
    field = np.repeat(np.repeat(np.exp(1j * phi), s, axis=0), s, axis=1)
    result = []
    for z in DISTANCES:
        out = (np.arange(N) - N / 2) * WAVELENGTH * z / (N * PITCH)
        kernel = np.exp(1j * np.pi * (out[:, None] - x[None, :])**2 / (WAVELENGTH * z))
        propagated = kernel @ field @ kernel.T * dx**2 / (WAVELENGTH * z)
        result.append(np.abs(propagated)**2)
    return np.stack(result)


def pixel_asm(phi: np.ndarray, subdivision: int, padding: int) -> np.ndarray:
    """Propagate the same subdivided pixel mask with padded exact ASM and sample the original output grids."""
    s = subdivision
    dx = PITCH / s
    size = N * s * padding
    field = np.zeros((size, size), dtype=np.complex128)
    start = (size - N * s) // 2
    field[start:start + N * s, start:start + N * s] = np.repeat(np.repeat(np.exp(1j * phi), s, axis=0), s, axis=1)
    spectrum = fft2(field)
    frequency = fftfreq(size, dx)
    kz = 2 * np.pi * np.sqrt((1 / WAVELENGTH**2 - frequency[:, None]**2 - frequency[None, :]**2).astype(complex))
    result = []
    for z in DISTANCES:
        propagated = ifft2(spectrum * np.exp(1j * z * kz))
        out = (np.arange(N) - N / 2) * WAVELENGTH * z / (N * PITCH)
        indices = out / dx + size / 2 + (s - 1) / 2
        yy, xx = np.meshgrid(indices, indices, indexing='ij')
        result.append(map_coordinates(np.abs(propagated)**2, [yy, xx], order=3))
    return np.stack(result)


def relative_errors(a: np.ndarray, b: np.ndarray) -> list:
    """Return per-plane relative intensity L2 differences with the second argument as reference."""
    return [float(np.linalg.norm(x - y) / np.linalg.norm(y)) for x, y in zip(a, b)]


def main() -> None:
    """Save run-wide audits and bounded validation diagnostics without retraining or overwriting results."""
    OUTPUT.mkdir(exist_ok=False)
    shutil.copy2(__file__, OUTPUT / 'investigate_latest.py')
    torch.set_num_threads(1)
    torch.manual_seed(17)
    factors = fresnel_factors(torch.float64)
    parameters = json.loads((RUN / 'parameters.json').read_text())
    report = {'run': str(RUN), 'python': platform.python_version(), 'numpy': np.__version__,
              'torch': torch.__version__, 'scipy': scipy.__version__, 'device': 'cpu',
              'precision': 'float64/complex128; network replay float32 CPU', 'seed': 17,
              'roi': 'post-hoc diagnostic T > 0.1, not a training ROI',
              'baseline': 'validation 0,1,2; seed 17 per example; 100 iterations; original grids; continuous metrics',
              'hashes': {}, 'summary': {}, 'audit': [], 'controls': [], 'replay': []}
    for path in [RUN / 'parameters.json', RUN / 'checkpoint_50_040.pt', *sorted((RUN / 'source/experiments').rglob('*.py'))]:
        report['hashes'][str(path.relative_to(RUN))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for split in ('validation', 'test'):
        saved = json.loads((RUN / f'{split}_metrics.json').read_text())
        report['summary'][split] = {name: {'mean': float(np.mean([[p[name] for p in e['planes']] for e in saved])),
                                               'per_plane': np.mean([[p[name] for p in e['planes']] for e in saved], axis=0).tolist()}
                                    for name in ('ssim', 'pearson')}
    for index in range(100):
        target = load_targets('test', index)
        directory = RUN / f'evaluation/test/{index:04d}'
        values = np.load(directory / 'intensity.npy')
        codes = np.asarray(Image.open(directory / 'doe.png'))
        phi = torch.tensor(-np.pi + 2 * np.pi * codes.astype(float) / 255)
        recalculated = intensity(phi, factors).numpy()
        png_error = 0
        for k in range(4):
            expected = np.floor(values[k] / values[k].max() * 255 + 0.5).astype(np.uint8)
            actual = np.asarray(Image.open(directory / f'restored_{k + 1}.png'))
            png_error = max(png_error, int(np.max(np.abs(expected.astype(int) - actual.astype(int)))))
        cross = np.corrcoef(np.concatenate([values.reshape(4, -1), target.reshape(4, -1)]))[:4, 4:]
        report['audit'].append({'example': index, 'planes': metrics(values, target),
                                'repropagation_relative_error': relative_errors(recalculated, values),
                                'restored_png_max_code_error': png_error,
                                'cross_target_pearson_index_grid_only': cross.tolist()})
    checkpoint = torch.load(RUN / 'checkpoint_50_040.pt', map_location='cpu', weights_only=False)
    model = build_model({**parameters, 'device': 'cpu'})
    model.load_state_dict(checkpoint['model'])
    model.eval()
    for index in range(3):
        target = load_targets('validation', index)
        with torch.no_grad():
            phi = model(torch.tensor(target[None], dtype=torch.float32))[0].double()
        wrapped = (phi.numpy() + np.pi) % (2 * np.pi) - np.pi
        codes = np.floor(255 * (wrapped + np.pi) / (2 * np.pi) + 0.5).astype(np.uint8)
        quantized = torch.tensor(-np.pi + 2 * np.pi * codes.astype(float) / 255)
        saved_codes = np.asarray(Image.open(RUN / f'evaluation/validation/{index:04d}/doe.png'))
        report['replay'].append({'example': index, 'continuous': metrics(intensity(phi, factors).numpy(), target),
                                 'quantized': metrics(intensity(quantized, factors).numpy(), target),
                                 'saved_mask_code_match_fraction': float(np.mean(codes == saved_codes)),
                                 'circular_quantization_rms_rad': float(np.sqrt(np.mean(np.angle(np.exp(1j * (quantized.numpy() - phi.numpy())))**2)))})
        for planes in range(1, 5):
            torch.manual_seed(17)
            initial = (torch.rand((N, N), dtype=torch.float64) * 2 - 1) * np.pi
            subset = tuple(f[:planes] for f in factors)
            goal = torch.tensor(target[:planes])
            for name, solver in [('gs', run_gs), ('gs_muon', run_gs_muon)]:
                args = (parameters['phase_muon'],) if name == 'gs_muon' else ()
                solved, history = solver(initial.clone(), goal, subset, 100, 1e-12, *args)
                np.save(OUTPUT / f'{index}_{planes}_{name}_phi.npy', solved.numpy())
                report['controls'].append({'example': index, 'planes': planes, 'method': name,
                                           'metrics': metrics(intensity(solved, subset).numpy(), target[:planes]), 'history': history})
    codes = np.asarray(Image.open(RUN / 'evaluation/validation/0000/doe.png'), dtype=float)
    phi = -np.pi + 2 * np.pi * codes / 255
    reference = intensity(torch.tensor(phi), factors).numpy()
    quadrature = {s: pixel_fresnel(phi, s) for s in (1, 2, 4, 8)}
    asm = {(s, p): pixel_asm(phi, s, p) for s, p in ((2, 2), (2, 4), (4, 2), (4, 4))}
    report['propagation'] = {
        'point_sum_vs_fft': relative_errors(quadrature[1], reference),
        'pixel_quadrature_vs_point': {str(s): relative_errors(v, reference) for s, v in quadrature.items()},
        'quadrature_4_vs_8': relative_errors(quadrature[4], quadrature[8]),
        'asm_padding_2_vs_4': {str(s): relative_errors(asm[s, 2], asm[s, 4]) for s in (2, 4)},
        'asm_subdivision_2_vs_4': relative_errors(asm[2, 4], asm[4, 4]),
        'asm_4_pad4_vs_pixel_fresnel_8': relative_errors(asm[4, 4], quadrature[8]),
        'asm_4_pad4_vs_original_fft': relative_errors(asm[4, 4], reference)}
    report['axial_dof_m'] = [8 * WAVELENGTH * z**2 / (N * PITCH)**2 for z in DISTANCES]
    history = json.loads((RUN / 'history.json').read_text())
    report['epoch_loss'] = {str(epoch): np.average([r['loss_per_plane'] for r in history if r['epoch'] == epoch], axis=0,
                                                 weights=[r['quadruples'] for r in history if r['epoch'] == epoch]).tolist()
                            for epoch in (1, 10, 25, 50)}
    (OUTPUT / 'diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(OUTPUT)


if __name__ == '__main__':
    main()
