"""One immutable target tuple interpreted on a shared physical detector grid."""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .physics import INCIDENT_POWER, spectral_weights


def load_sample(directory: Path, config: dict) -> tuple:
    """Read four fixed PNG targets and define physical support and calibration before optimization."""
    arrays = []
    manifest = []
    for plane in range(1, 5):
        path = directory / f'target_{plane}.png'
        with Image.open(path) as image:
            if image.mode != 'L' or image.size != (128, 128):
                raise ValueError(f'{path}: expected a 128x128 uint8 L PNG.')
            arrays.append(np.array(image, dtype=np.float64) / 255)
        manifest.append({'plane': plane, 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    target = torch.tensor(np.stack(arrays), dtype=torch.float64)
    signal = target > config['support_threshold']
    counts = signal.sum(dim=(-2, -1))
    if torch.any(counts == 0) or torch.any(counts == 128**2):
        raise ValueError('Each target must define both a nonempty signal region and a nonempty guard region.')
    useful_target = (target * signal).sum(dim=(-2, -1)) * config['output_pitch_m']**2
    scale = config['target_useful_efficiency'] * INCIDENT_POWER / useful_target
    regions = {'signal': signal, 'intensity_scale': scale,
               'eta_required': torch.full((4,), config['minimum_efficiency'], dtype=torch.float64),
               'frequency_weights': spectral_weights(config['output_pitch_m'], config['frequency_band_weights'])}
    return target, regions, manifest
