"""Запуск всех backbone на двух непересекающихся наборах MNIST с одним seed моделей."""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.experiment_unrolled_cnn.experiment import save_json
from experiments.experiment_unrolled_cnn.mnist_data import download_mnist
from .experiment import CONFIG, run_experiment


def run_mnist_suite(output: Path) -> None:
    """Фиксирует две непересекающиеся выборки исходных объектов и последовательно сравнивает методы."""
    output.mkdir(parents=True, exist_ok=False)
    arrays, sources = download_mnist(output / 'raw_mnist')
    save_json(output / 'dataset_sources.json', sources)
    generator = np.random.default_rng(2026)
    train_order = generator.permutation(len(arrays['train_labels']))
    test_order = generator.permutation(len(arrays['t10k_labels']))
    selected_hashes = set()
    configs = []
    for dataset in range(2):
        config = deepcopy(CONFIG)
        config.update({
            'seeds': [17], 'dataset': 'MNIST', 'dataset_id': dataset,
            'preprocessing': 'Pillow L uint8; 28x28 -> 128x128 BILINEAR; no threshold/gamma/renormalization',
            'data_rng': 'one numpy PCG64(2026), train permutation then test permutation; disjoint slices',
            'test_selection': 'disjoint slices of official test permutation, no label selection',
        })
        train_count = 4 * config['split_sizes']['train']
        validation_count = 4 * config['split_sizes']['validation']
        test_count = 4 * config['split_sizes']['test']
        offset = dataset * (train_count + validation_count)
        indices = {
            'train': train_order[offset:offset + train_count],
            'validation': train_order[offset + train_count:offset + train_count + validation_count],
            'test': test_order[dataset * test_count:(dataset + 1) * test_count],
        }
        for split, values in indices.items():
            source = 't10k' if split == 'test' else 'train'
            for index in values:
                pixels = arrays[f'{source}_images'][index].tobytes()
                if pixels in selected_hashes:
                    raise ValueError('Исходные пиксели MNIST повторяются между выборками или наборами.')
                selected_hashes.add(pixels)
        config['mnist_indices'] = {split: values.tolist() for split, values in indices.items()}
        configs.append(config)
    save_json(output / 'suite_configurations.json', configs)
    for config in configs:
        directory = output / f"dataset_{config['dataset_id'] + 1}"
        print(f'Start {directory}', flush=True)
        run_experiment(config, directory, mnist_arrays=arrays)
    save_json(output / 'status.json', {'status': 'completed', 'datasets': 2, 'model_seed': 17})
    print(f'Completed: {output}', flush=True)


if __name__ == '__main__':
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run_mnist_suite(Path(__file__).resolve().parent / 'results' / f'mnist_two_datasets_{stamp}')
