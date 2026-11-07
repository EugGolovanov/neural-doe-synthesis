"""Pretrain the physics-adapter U-Net on 27000 mixed quadruples for three epochs."""

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
import torchvision
from torch.utils.checkpoint import checkpoint

from experiments.experiment_gs_muon.evaluate import export_result
from experiments.experiment_gs_muon.physics import fresnel_factors
from experiments.experiment_unrolled_cnn.algorithms import initial_momenta, update_muon
from .experiment import CONFIG, build_model
from .mixed_data import prepare_mixed
from .physics import structural_losses
from .train_mnist import save_json


RUN = {**CONFIG, 'epochs': 3, 'device': 'mps', 'dtype': 'float32',
       'batch_size': 256, 'microbatch_size': 256, 'ssim_weight': 0.5,
       'split_sizes': {'train': 27000, 'validation': 99, 'test': 99},
       'data_seed': 2026, 'threads': 1, 'checkpoint_every_epochs': 5,
       'activation_checkpointing': True,
       'preprocessing': 'Pillow L grayscale then BILINEAR 128x128 uint8 PNG; intensity=uint8/255',
       'mixture': 'Equal individual-image counts from CIFAR-10, Fashion-MNIST and MNIST; shuffled into fixed quadruples',
       'batch_convention': '256 quadruples = 1024 images; final batch 120 quadruples',
       'loss': 'mean over examples and planes of max-normalized MSE + 0.5*(1-SSIM); epsilon=1e-12',
       'inner_solver': 'Unchanged GS+Muon MSE steps with full higher-order gradients',
       'selection': 'Fixed 3 epochs from seed 17; final checkpoint; no test tuning'}


def main() -> None:
    """Prepare mixed datasets, train with Muon, checkpoint and evaluate reread exported masks."""
    if not torch.backends.mps.is_available():
        raise RuntimeError('This experiment requires MPS; no automatic device fallback.')
    torch.set_num_threads(RUN['threads'])
    torch.manual_seed(RUN['seed'])
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = Path(__file__).resolve().parent / 'results' / f'mixed_ssim_27k_3ep_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    print(f'OUTPUT={output}', flush=True)
    save_json(output / 'parameters.json', RUN)
    save_json(output / 'status.json', {'status': 'preparing'})
    for name in ('experiment_unet_physics_adapter', 'experiment_unrolled_cnn',
                 'experiment_gs_muon', 'experiment_single_sample'):
        destination = output / 'source' / 'experiments' / name
        destination.mkdir(parents=True)
        for path in (root / 'experiments' / name).iterdir():
            if path.suffix in ('.py', '.md'):
                shutil.copy2(path, destination / path.name)
    shutil.copytree(root / 'docs', output / 'source' / 'docs', ignore=shutil.ignore_patterns('papers'))
    (output / 'source' / 'working_tree.diff').write_bytes(
        subprocess.check_output(['git', 'diff', 'HEAD', '--binary'], cwd=root))
    save_json(output / 'environment.json', {
        'python': platform.python_version(), 'platform': platform.platform(),
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'numpy': np.__version__, 'pillow': PIL.__version__,
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True),
        'device': 'mps', 'dtype': 'float32', 'deterministic_algorithms': True,
        'rng': 'torch seed 17; split PCG64 and epoch-order torch Generator seed 2026',
        'reproducibility': 'No cross-platform bitwise guarantee; source snapshot includes untracked code.'})
    data, manifest = prepare_mixed(output, RUN)
    save_json(output / 'inputs.json', manifest)
    model = build_model(RUN)
    model.activation_checkpointing = True
    model.adapter.activation_checkpointing = True
    momenta = initial_momenta(model)
    factors = tuple(value.to('mps') for value in fresnel_factors(torch.float32))
    shuffle = torch.Generator().manual_seed(RUN['data_seed'])
    start = time.monotonic()
    save_json(output / 'status.json', {'status': 'training', 'epoch': 1, 'batch': 0})
    for epoch in range(1, RUN['epochs'] + 1):
        model.train()
        order = torch.randperm(len(data['train']), generator=shuffle)
        save_json(output / f'epoch_{epoch:03d}_order.json', order.tolist())
        for batch_number, indices in enumerate(order.split(RUN['batch_size']), 1):
            model.zero_grad(set_to_none=True)
            plane_losses = torch.zeros(4, device='mps')
            for micro_indices in indices.split(RUN['microbatch_size']):
                target = data['train'][micro_indices].to('mps')
                phi = model(target)
                per_plane = checkpoint(structural_losses, phi, target, factors,
                                       RUN['loss_epsilon'], RUN['ssim_weight'], use_reentrant=False)
                loss = per_plane.mean(dim=1).sum() / len(indices)
                loss.backward()
                plane_losses += per_plane.detach().sum(dim=0)
                del target, phi, per_plane, loss
            if not all(torch.isfinite(p.grad).all().item() for p in model.parameters()):
                raise RuntimeError('Nonfinite gradients; stopping without changing the algorithm.')
            update_muon(model, momenta, RUN['weight_muon'])
            torch.mps.synchronize()
            record = {'epoch': epoch, 'batch': batch_number, 'quadruples': len(indices),
                      'loss_per_plane': (plane_losses / len(indices)).tolist(),
                      'elapsed_seconds': time.monotonic() - start,
                      'mps_allocated_bytes': torch.mps.current_allocated_memory(),
                      'mps_driver_bytes': torch.mps.driver_allocated_memory()}
            with (output / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            save_json(output / 'status.json', {'status': 'training', **record})
            print(json.dumps(record), flush=True)
        model.eval()
        validation = []
        with torch.no_grad():
            for indices in torch.arange(len(data['validation'])).split(RUN['microbatch_size']):
                target = data['validation'][indices].to('mps')
                validation.extend(structural_losses(model(target), target, factors,
                                                   RUN['loss_epsilon'], RUN['ssim_weight']).cpu().tolist())
        save_json(output / f'validation_epoch_{epoch:03d}.json', validation)
        if epoch % RUN['checkpoint_every_epochs'] == 0 or epoch in (1, RUN['epochs']):
            torch.save({'model': model.state_dict(), 'momenta': momenta, 'config': RUN,
                        'epoch': epoch, 'torch_rng': torch.get_rng_state(),
                        'shuffle_rng': shuffle.get_state(), 'mps_rng': torch.mps.get_rng_state()},
                       output / f'checkpoint_{epoch:03d}.pt')
    export_factors = fresnel_factors(torch.float64)
    for split in ('validation', 'test'):
        metrics = []
        for index in range(len(data[split])):
            target = data[split][torch.tensor([index])]
            with torch.no_grad():
                phi = model(target.to('mps'))[0].cpu().double()
            directory = output / 'evaluation' / split / f'{index:04d}'
            directory.mkdir(parents=True)
            targets = torch.round(target[0] * 255).to(torch.uint8).numpy()
            metrics.append({'example': index, 'planes': export_result(directory, phi, targets, export_factors)})
        save_json(output / f'{split}_metrics.json', metrics)
    save_json(output / 'status.json', {'status': 'completed', 'elapsed_seconds': time.monotonic() - start})


if __name__ == '__main__':
    main()
