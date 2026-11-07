"""Balanced fixed quadruples from official CIFAR-10, Fashion-MNIST and MNIST splits."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST


class MixedTargets:
    """Read prepared grayscale PNG quadruples without keeping the dataset in RAM."""

    def __init__(self, paths: list):
        """Store the ordered four-plane PNG paths for each sample."""
        self.paths = paths

    def __len__(self) -> int:
        """Return the number of four-image samples."""
        return len(self.paths)

    def __getitem__(self, indices: torch.Tensor) -> torch.Tensor:
        """Read selected quadruples as float32 intensities in [0,1] with shape [B,4,128,128]."""
        images = []
        for index in indices.tolist():
            planes = []
            for path in self.paths[index]:
                with Image.open(path) as image:
                    planes.append(np.array(image))
            images.append(np.stack(planes))
        return torch.from_numpy(np.stack(images)).float() / 255


def prepare_mixed(output: Path, config: dict) -> tuple:
    """Download official datasets and prepare disjoint balanced PNG splits with recorded hashes."""
    generator = np.random.default_rng(config['data_seed'])
    cache = output.parents[1] / 'datasets'
    cache.mkdir(exist_ok=True)
    constructors = {'cifar10': CIFAR10, 'fashion_mnist': FashionMNIST, 'mnist': MNIST}
    pools = {split: [] for split in config['split_sizes']}
    manifest = []
    seen = set()
    for name, constructor in constructors.items():
        print(f'DOWNLOAD/PREPARE {name}', flush=True)
        training = constructor(str(cache), train=True, download=True)
        testing = constructor(str(cache), train=False, download=True)
        train_order = iter(generator.permutation(len(training)).tolist())
        test_order = iter(generator.permutation(len(testing)).tolist())
        for split, samples in config['split_sizes'].items():
            count = samples * 4 // 3
            source = testing if split == 'test' else training
            order = test_order if split == 'test' else train_order
            directory = output / 'data' / split / name
            directory.mkdir(parents=True)
            accepted = 0
            while accepted < count:
                index = next(order)
                image, label = source[index]
                image = image.convert('L').resize((128, 128), Image.Resampling.BILINEAR)
                pixels = np.array(image)
                digest = hashlib.sha256(pixels.tobytes()).hexdigest()
                if digest in seen or not pixels.any():
                    continue
                seen.add(digest)
                path = directory / f'{index:05d}.png'
                image.save(path)
                pools[split].append(str(path))
                manifest.append({'dataset': name, 'official_split': 'test' if split == 'test' else 'train',
                                 'index': index, 'label': int(label), 'split': split,
                                 'path': str(path.relative_to(output)), 'pixels_sha256': digest,
                                 'png_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                accepted += 1
    datasets = {}
    quadruples = {}
    for split, paths in pools.items():
        generator.shuffle(paths)
        groups = [paths[i:i + 4] for i in range(0, len(paths), 4)]
        datasets[split] = MixedTargets(groups)
        quadruples[split] = [[str(Path(p).relative_to(output)) for p in group] for group in groups]
    return datasets, {'images': manifest, 'quadruples': quadruples,
                      'sources': {name: {'class': constructor.__name__,
                                         'resources': getattr(constructor, 'resources', None),
                                         'url': getattr(constructor, 'url', None),
                                         'mirrors': getattr(constructor, 'mirrors', None)}
                                  for name, constructor in constructors.items()}}
