"""Обучение на 10000 PNG MNIST с эффективным батчем 256 изображений."""

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

from experiments.experiment_gs_muon.evaluate import export_result
from experiments.experiment_gs_muon.physics import fresnel_factors, physical_loss
from experiments.experiment_unrolled_cnn.algorithms import initial_momenta, losses, update_muon
from experiments.experiment_unrolled_cnn.mnist_data import download_mnist, prepare_mnist_data
from .experiment import CONFIG, build_model


RUN = {**CONFIG, 'epochs': 1, 'batch_size': 64, 'microbatch_size': 1,
       'batch_individual_images': 256, 'training_individual_images': 10000,
       'split_sizes': {'train': 2500, 'validation': 100, 'test': 100},
       'data_seed': 2026, 'threads': 1,
       'preprocessing': 'MNIST uint8 28x28 -> Pillow BILINEAR 128x128 L PNG; divide by 255',
       'batch_convention': '64 quadruples = 256 separate PNG; final batch has 4 quadruples',
       'selection': 'one fixed epoch; no test tuning or hyperparameter selection'}


def save_json(path: Path, value: object) -> None:
    """Сохраняет параметры или результаты в читаемый JSON без NaN."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def unique_indices(images: np.ndarray, order: np.ndarray, count: int,
                   seen: set) -> list:
    """Выбирает неповторяющиеся исходные изображения до формирования четвёрок."""
    selected = []
    for index in order:
        resized = Image.fromarray(images[index]).resize((128, 128), Image.Resampling.BILINEAR)
        digest = hashlib.sha256(np.asarray(resized).tobytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            selected.append(int(index))
        if len(selected) == count:
            return selected
    raise ValueError('Недостаточно уникальных изображений MNIST.')


def main() -> None:
    """Готовит независимые выборки, обучает сеть и оценивает экспортированные маски."""
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = Path(__file__).resolve().parent / 'results' / f'mnist_10k_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    print(f'OUTPUT={output}', flush=True)
    save_json(output / 'status.json', {'status': 'preparing', 'config': RUN})
    torch.set_num_threads(RUN['threads'])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(RUN['seed'])
    root = Path(__file__).resolve().parents[2]
    snapshot = output / 'source'
    for experiment in ('experiment_unet_physics_adapter', 'experiment_unrolled_cnn', 'experiment_gs_muon'):
        destination = snapshot / 'experiments' / experiment
        destination.mkdir(parents=True)
        for path in (root / 'experiments' / experiment).glob('*.py'):
            shutil.copy2(path, destination / path.name)
    shutil.copytree(root / 'docs', snapshot / 'docs', ignore=shutil.ignore_patterns('papers'))
    (snapshot / 'working_tree.diff').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD', '--binary'], cwd=root))
    save_json(output / 'environment.json', {
        'python': platform.python_version(), 'platform': platform.platform(),
        'torch': torch.__version__, 'numpy': np.__version__, 'pillow': PIL.__version__,
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True),
        'rng': 'torch CPU manual_seed(17), numpy PCG64(2026) for splits; torch Generator(2026) for epoch order',
        'deterministic_algorithms': True, 'device': 'cpu', 'dtype': 'float64',
        'reproducibility': 'No cross-platform bitwise guarantee; source snapshot includes untracked implementation.',
    })
    arrays, sources = download_mnist(output / 'raw_mnist')
    save_json(output / 'dataset_sources.json', sources)
    generator = np.random.default_rng(RUN['data_seed'])
    seen = set()
    selected = unique_indices(arrays['train_images'], generator.permutation(60000), 10400, seen)
    test = unique_indices(arrays['t10k_images'], generator.permutation(10000), 400, seen)
    config = {**RUN, 'mnist_indices': {'train': selected[:10000],
                                      'validation': selected[10000:], 'test': test}}
    save_json(output / 'parameters.json', config)
    data, manifest = prepare_mnist_data(output / 'data', arrays, config)
    save_json(output / 'inputs.json', manifest)
    model = build_model(config)
    momenta = initial_momenta(model)
    factors = fresnel_factors(torch.float64)
    shuffle = torch.Generator().manual_seed(config['data_seed'])
    history = []
    start = time.monotonic()
    for epoch in range(config['epochs']):
        order = torch.randperm(len(data['train']), generator=shuffle)
        save_json(output / f'epoch_{epoch + 1}_order.json', order.tolist())
        for batch_number, indices in enumerate(order.split(config['batch_size']), 1):
            model.train()
            model.zero_grad(set_to_none=True)
            plane_losses = torch.zeros(4, dtype=torch.float64)
            for index in indices.tolist():
                target = data['train'][index:index + 1].to(torch.float64)
                phi = model(target)
                loss = physical_loss(phi[0], target[0], factors, config['loss_epsilon'])
                (loss / len(indices)).backward()
                with torch.no_grad():
                    plane_losses += losses(phi[:, None], target, factors, config['loss_epsilon'])[0]
            if not all(torch.isfinite(p.grad).all().item() for p in model.parameters()):
                raise RuntimeError('Неконечные градиенты: обучение остановлено без подмены алгоритма.')
            update_muon(model, momenta, config['weight_muon'])
            record = {'epoch': epoch + 1, 'batch': batch_number, 'quadruples': len(indices),
                      'loss_per_plane': (plane_losses / len(indices)).tolist(),
                      'elapsed_seconds': time.monotonic() - start}
            history.append(record)
            save_json(output / 'history.json', history)
            torch.save({'model': model.state_dict(), 'momenta': momenta, 'config': config,
                        'epoch': epoch + 1, 'batch': batch_number, 'torch_rng': torch.get_rng_state(),
                        'shuffle_rng': shuffle.get_state()}, output / f'checkpoint_{epoch + 1}_{batch_number:03d}.pt')
            save_json(output / 'status.json', {'status': 'training', **record})
            print(json.dumps(record), flush=True)
    model.eval()
    for split in ('validation', 'test'):
        metrics = []
        for index, target in enumerate(data[split]):
            with torch.no_grad():
                phi = model(target[None].to(torch.float64))[0]
            directory = output / 'evaluation' / split / f'{index:04d}'
            directory.mkdir(parents=True)
            targets = torch.round(target * 255).to(torch.uint8).numpy()
            metrics.append({'example': index, 'planes': export_result(directory, phi, targets, factors)})
        save_json(output / f'{split}_metrics.json', metrics)
    save_json(output / 'status.json', {'status': 'completed', 'elapsed_seconds': time.monotonic() - start})
    print(f'COMPLETED={output}', flush=True)


if __name__ == '__main__':
    main()
