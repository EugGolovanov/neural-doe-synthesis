"""Сравнение общих backbone в физической развёртке с итеративным GS + Muon."""

import hashlib
import math
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiments.experiment_gs_muon.physics import N, fresnel_factors
from experiments.experiment_unrolled_cnn.data import prepare_data
from experiments.experiment_unrolled_cnn.experiment import (
    CONFIG as CNN_CONFIG, compare_test, save_json, snapshot_sources, train_model,
)
from experiments.experiment_unrolled_cnn.mnist_data import prepare_mnist_data
from .models import FNOCorrection, MambaCorrection
from .convnext import ConvNeXtCorrection, ConvNeXtUNetCorrection


CONFIG = deepcopy(CNN_CONFIG)
CONFIG.update({
    'backbones': {'cnn_backbone': 'convnext_tiny', 'unet_backbone': 'convnext_tiny',
                  'convnext_depths': [3, 3, 9, 3], 'convnext_channels': [96, 192, 384, 768],
                  'convnext_pretrained': False, 'convnext_layer_scale': 1e-6,
                  'convnext_norm_epsilon': 1e-6, 'convnext_drop_path': 0.0,
                  'convnext_affine_muon': 'vectors stored as [channels,1]; same matrix Muon',
                  'cnn_width': 8, 'unet_width': 8, 'fno_width': 4,
                  'fno_modes': 4, 'fno_blocks': 2,
                  'mamba_width': 8, 'mamba_states': 4, 'mamba_blocks': 2},
    'comparison': 'fixed K/data/epochs/lr; unequal parameter counts; CPU reference implementations',
    'training_order_seed_offset': 100000,
})


def run_experiment(config: dict, output: Path, mnist_arrays: dict | None = None) -> None:
    """Обучает четыре корректора на общем наборе и один раз оценивает GS + Muon для каждого seed."""
    if config['device'] != 'cpu' or config['dtype'] != 'float32' or config['batch_size'] != 1:
        raise ValueError('Поддерживается только CPU float32 batch_size=1.')
    if min(config['split_sizes'].values()) < 1 or config['epochs'] < 1 or config['steps'] < 1:
        raise ValueError('Размеры выборок, число эпох и шагов должны быть положительными.')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    snapshot_sources(output, config)
    hashes = {}
    for path in Path(__file__).parent.iterdir():
        if path.suffix not in ('.py', '.md'):
            continue
        relative = Path('experiments/experiment_backbones') / path.name
        destination = output / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_json(output / 'comparison_source_sha256.json', hashes)
    if mnist_arrays is None:
        data, manifest = prepare_data(output / 'data', config['split_sizes'], config['data_seed'])
    else:
        data, manifest = prepare_mnist_data(output / 'data', mnist_arrays, config)
    save_json(output / 'data_manifest.json', manifest)
    factors = fresnel_factors(torch.float32)
    summary = []
    settings = config['backbones']
    for seed in config['seeds']:
        torch.manual_seed(seed)
        initial = (2 * torch.rand(1, 1, N, N) - 1) * math.pi
        seed_directory = output / f'seed_{seed}'
        seed_directory.mkdir()
        np.save(seed_directory / 'initial_phi.npy', initial.numpy())
        torch.manual_seed(seed)
        cnn = ConvNeXtCorrection(settings['cnn_width'])
        torch.manual_seed(seed)
        unet = ConvNeXtUNetCorrection(settings['unet_width'])
        torch.manual_seed(seed)
        fno = FNOCorrection(settings['fno_width'], settings['fno_modes'], settings['fno_blocks'])
        torch.manual_seed(seed)
        mamba = MambaCorrection(settings['mamba_width'], settings['mamba_states'], settings['mamba_blocks'])
        for name, model in [('cnn_convnext_tiny', cnn), ('unet_convnext_tiny', unet),
                            ('fno', fno), ('mamba_axial', mamba)]:
            directory = seed_directory / name
            directory.mkdir()
            torch.manual_seed(seed + config['training_order_seed_offset'])
            model, seconds = train_model(data, initial, factors, config, directory, model=model)
            entries = compare_test(model, data, initial, factors, config, directory,
                                   method_name=name, include_baseline=name == 'cnn_convnext_tiny')
            for entry in entries:
                baseline = entry['method'] == 'gs_muon'
                summary.append({'seed': seed, 'training_seconds': 0.0 if baseline else seconds,
                                'parameters': 0 if baseline else sum(p.numel() for p in model.parameters()),
                                'result_directory': str(directory.relative_to(output)), **entry})
            save_json(output / 'summary.json', summary)


if __name__ == '__main__':
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run_experiment(CONFIG, Path(__file__).resolve().parent / 'results' / f'backbones_{stamp}')
