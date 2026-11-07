"""Фиксированный диагностический набор непересекающихся синтетических объектов."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.experiment_gs_muon.physics import N


def prepare_data(directory: Path, split_sizes: dict, seed: int) -> tuple[dict, list]:
    """Сохраняет и повторно читает PNG независимых объектов с явными параметрами и SHA-256."""
    y, x = np.mgrid[:N, :N] / N
    data, manifest = {}, []
    object_id = 0
    for split, count in split_sizes.items():
        examples = []
        for example in range(count):
            planes = []
            for plane in range(4):
                generator = np.random.default_rng(np.random.SeedSequence([seed, object_id]))
                cx, cy = generator.uniform(0.2, 0.8, 2)
                width = float(generator.uniform(0.04, 0.15))
                angle = float(generator.uniform(0, 2 * np.pi))
                frequency = float(generator.uniform(2, 8))
                coordinate = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
                kind = example % 3
                if kind == 0:
                    image = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * width**2))
                elif kind == 1:
                    image = ((np.abs(coordinate) < width)
                             & ((x - cx)**2 + (y - cy)**2 < 0.3**2)).astype(float)
                else:
                    image = (1 + np.cos(2 * np.pi * frequency * coordinate)) / 2
                codes = np.floor(255 * image + 0.5).astype(np.uint8)
                relative = Path(split) / f'{example:04d}' / f'target_{plane + 1}.png'
                path = directory / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(codes).save(path)
                with Image.open(path) as saved:
                    if saved.mode != 'L' or saved.size != (N, N):
                        raise ValueError(f'Ожидался одноканальный PNG 128×128: {path}')
                    planes.append(np.array(saved))
                manifest.append({'split': split, 'example': example, 'plane': plane + 1,
                                 'object_id': object_id, 'path': str(relative),
                                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'kind': kind, 'cx': float(cx), 'cy': float(cy),
                                 'width': width, 'angle': angle, 'frequency': frequency})
                object_id += 1
            examples.append(torch.tensor(np.stack(planes), dtype=torch.float32) / 255)
        data[split] = torch.stack(examples)
    hashes = [item['sha256'] for item in manifest]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Набор содержит повторяющиеся PNG объектов.')
    return data, manifest
