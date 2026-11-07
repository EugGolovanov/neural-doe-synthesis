"""Check padding convergence and independently verify the midpoint Fresnel calculation."""

import json
import shutil

import numpy as np
from PIL import Image

from .investigate_latest import RUN, OUTPUT, pixel_asm, pixel_fresnel, relative_errors
import torch
from experiments.experiment_gs_muon.physics import N, PITCH, WAVELENGTH, DISTANCES


def main() -> None:
    """Save independent matrix-product and larger-padding checks for the first validation mask."""
    destination = OUTPUT / 'convergence.json'
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(__file__, OUTPUT / 'investigate_convergence.py')
    torch.set_num_threads(1)
    codes = np.asarray(Image.open(RUN / 'evaluation/validation/0000/doe.png'), dtype=float)
    phi = -np.pi + 2 * np.pi * codes / 255
    s = 8
    dx = PITCH / s
    x = (torch.arange(N * s, dtype=torch.float64) - N * s / 2 - (s - 1) / 2) * dx
    field = torch.exp(1j * torch.tensor(phi)).repeat_interleave(s, 0).repeat_interleave(s, 1)
    values = []
    for z in DISTANCES:
        out = (torch.arange(N, dtype=torch.float64) - N / 2) * WAVELENGTH * z / (N * PITCH)
        kernel = torch.exp(1j * np.pi * (out[:, None] - x[None, :])**2 / (WAVELENGTH * z))
        propagated = kernel @ field @ kernel.T * dx**2 / (WAVELENGTH * z)
        values.append(propagated.abs().square().numpy())
    independent = np.stack(values)
    quadrature = pixel_fresnel(phi, 8)
    asm4 = pixel_asm(phi, 4, 4)
    asm8 = pixel_asm(phi, 4, 8)
    report = {'numpy_matmul_warning_check_torch_vs_numpy': relative_errors(independent, quadrature),
              'asm_s4_padding4_vs8': relative_errors(asm4, asm8),
              'asm_s4_padding8_vs_fresnel_s8': relative_errors(asm8, quadrature),
              'tolerances': {'point_sum_fft_relative_l2': 1e-10,
                             'independent_matrix_product_relative_l2': 1e-10,
                             'successive_physical_refinement_relative_l2': 0.01},
              'tolerance_basis': '1e-10 allows floating-point accumulation; 1% is a diagnostic numerical convergence target, not physical validation.'}
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
