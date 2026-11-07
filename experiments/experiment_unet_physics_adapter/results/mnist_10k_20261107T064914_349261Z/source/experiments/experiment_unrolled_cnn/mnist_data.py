"""Настоящий MNIST с явным преобразованием полутоновых объектов в четыре PNG 128×128."""

import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image


MNIST_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'


def download_mnist(directory: Path) -> tuple[dict, list]:
    """Загружает исходные IDX MNIST и проверяет размеры массивов без изменения значений."""
    directory.mkdir(parents=True, exist_ok=False)
    arrays, sources = {}, []
    for split, count in [('train', 60000), ('t10k', 10000)]:
        for kind, magic, shape in [('images', 2051, (count, 28, 28)), ('labels', 2049, (count,))]:
            suffix = 'idx3-ubyte.gz' if kind == 'images' else 'idx1-ubyte.gz'
            name = f'{split}-{kind}-{suffix}'
            url = MNIST_URL + name
            path = directory / name
            print(f'Download {url}', flush=True)
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
            content = gzip.decompress(path.read_bytes())
            header_size = 4 * (1 + len(shape))
            header = struct.unpack('>' + 'I' * (1 + len(shape)), content[:header_size])
            if header != (magic, *shape) or len(content) != header_size + int(np.prod(shape)):
                raise ValueError(f'Некорректный IDX MNIST: {name}')
            arrays[f'{split}_{kind}'] = np.frombuffer(content, dtype=np.uint8,
                                                     offset=header_size).reshape(shape)
            sources.append({'url': url, 'filename': name,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'shape': shape})
    return arrays, sources


def prepare_mnist_data(directory: Path, arrays: dict, config: dict) -> tuple[dict, list]:
    """Делит исходные объекты до группировки и сохраняет билинейно увеличенные полутоновые PNG."""
    generator = np.random.default_rng(config['data_seed'])
    shuffled = generator.permutation(len(arrays['train_labels']))
    training_count = config['split_sizes']['train'] * 4
    validation_count = config['split_sizes']['validation'] * 4
    test_indices = [int(np.flatnonzero(arrays['t10k_labels'] == digit)[0])
                    for digit in config.get('test_digits', [])]
    indices = {'train': shuffled[:training_count],
               'validation': shuffled[training_count:training_count + validation_count],
               'test': np.array(test_indices)}
    if 'mnist_indices' in config:
        indices = {split: np.asarray(values, dtype=np.int64)
                   for split, values in config['mnist_indices'].items()}
    for split, selected in indices.items():
        if len(selected) != 4 * config['split_sizes'][split]:
            raise ValueError(f'Число исходных MNIST-объектов не совпадает с размером {split}.')
    data, manifest = {}, []
    for split, selected in indices.items():
        source_split = 't10k' if split == 'test' else 'train'
        examples = []
        for example, group in enumerate(selected.reshape(-1, 4)):
            planes = []
            for plane, index in enumerate(group):
                original = arrays[f'{source_split}_images'][index]
                resized = Image.fromarray(original).resize((128, 128), Image.Resampling.BILINEAR)
                relative = Path(split) / f'{example:04d}' / f'target_{plane + 1}.png'
                path = directory / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                resized.save(path)
                with Image.open(path) as image:
                    if image.mode != 'L' or image.size != (128, 128):
                        raise ValueError(f'Ожидался одноканальный PNG 128×128: {path}')
                    planes.append(np.array(image))
                manifest.append({'split': split, 'example': example, 'plane': plane + 1,
                                 'source_split': source_split, 'source_index': int(index),
                                 'label': int(arrays[f'{source_split}_labels'][index]),
                                 'source_pixels_sha256': hashlib.sha256(original.tobytes()).hexdigest(),
                                 'path': str(relative),
                                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            examples.append(torch.tensor(np.stack(planes), dtype=torch.float32) / 255)
        data[split] = torch.stack(examples)
    hashes = [entry['sha256'] for entry in manifest]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Выбранные объекты MNIST содержат повторяющиеся PNG.')
    return data, manifest
