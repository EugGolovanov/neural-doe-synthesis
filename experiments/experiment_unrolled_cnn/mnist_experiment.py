"""Предварительный запуск CNN и GS + Muon для четырёх полутоновых объектов MNIST."""

import copy
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from experiments.experiment_gs_muon.physics import N, fresnel_factors
from .experiment import CONFIG, compare_test, save_json, snapshot_sources, train_model
from .mnist_data import download_mnist, prepare_mnist_data


MNIST_CONFIG = copy.deepcopy(CONFIG)
MNIST_CONFIG.update({
    'seeds': [17], 'split_sizes': {'train': 24, 'validation': 6, 'test': 1},
    'dataset': 'MNIST', 'test_digits': [0, 1, 2, 3],
    'preprocessing': 'Pillow L uint8; 28x28 -> 128x128 BILINEAR; no threshold/gamma/renormalization',
    'data_rng': 'numpy.default_rng(data_seed).permutation(60000); disjoint train/validation slices',
    'test_selection': 'first official test occurrence of each digit in test_digits',
})


def save_preview(output: Path, directory: Path) -> None:
    """Собирает исходные цели и повторно прочитанные восстановления без изменения яркости."""
    preview = Image.new('RGB', (4 * 144, 3 * 160), 'white')
    draw = ImageDraw.Draw(preview)
    for row, method in enumerate(('target', 'cnn', 'gs_muon')):
        for plane in range(4):
            if method == 'target':
                path = output / 'data' / 'test' / '0000' / f'target_{plane + 1}.png'
            else:
                path = directory / 'test' / '0000' / method / f'restored_{plane + 1}.png'
            draw.text((plane * 144 + 8, row * 160 + 4), f'{method} z{plane + 1}', fill='black')
            with Image.open(path) as image:
                preview.paste(image, (plane * 144 + 8, row * 160 + 24))
    preview.save(directory / 'preview.png')


def run_mnist_experiment(output: Path) -> None:
    """Обучает CNN на MNIST и экспортирует четырёхплоскостной результат для отдельной тестовой четвёрки."""
    config = MNIST_CONFIG
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    snapshot_sources(output, config)
    arrays, sources = download_mnist(output / 'raw_mnist')
    save_json(output / 'dataset_sources.json', sources)
    data, manifest = prepare_mnist_data(output / 'data', arrays, config)
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
        save_preview(output, directory)
    print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run_mnist_experiment(Path(__file__).resolve().parent / 'results' / f'mnist_{stamp}')
