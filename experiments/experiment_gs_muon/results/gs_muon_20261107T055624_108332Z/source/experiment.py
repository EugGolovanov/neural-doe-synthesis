"""Предварительное сравнение GS, GS + Muon и GS + Adam на синтетических целях."""

import hashlib
import json
import math
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

from algorithms import run_gs, run_gs_adam, run_gs_muon
from evaluate import export_result
from physics import DISTANCES, N, PITCH, WAVELENGTH, fresnel_factors


CONFIG = {
    "steps": 100,
    "seeds": [17, 29, 43],
    "cases": ["spots", "bars", "dense"],
    "device": "cpu",
    "dtype": "float64",
    "threads": 1,
    "loss_epsilon": 1e-12,
    "muon": {"lr": 0.1, "momentum": 0.95, "epsilon": 1e-12,
             "ns_steps": 5, "coefficients": [3.4445, -4.7750, 2.0315], "scale": 1.0},
    "adam": {"lr": 0.01, "betas": [0.9, 0.999], "epsilon": 1e-8},
}


def synthetic_targets(case: str) -> np.ndarray:
    """Генерирует фиксированную четвёрку uint8-целей без случайности и гамма-коррекции."""
    y, x = np.mgrid[:N, :N] / N
    targets = []
    for j in range(4):
        if case == "spots":
            cx, cy = 0.3 + 0.12 * j, 0.35 + 0.08 * (j % 2)
            image = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * 0.065**2))
        elif case == "bars":
            image = ((np.abs(x - (0.25 + 0.15 * j)) < 0.04)
                     & (np.abs(y - 0.5) < 0.3)).astype(float)
        elif case == "dense":
            image = (1 + np.cos(2 * math.pi * ((j + 2) * x + (5 - j) * y))) / 2
        else:
            raise ValueError(f"Неизвестный набор целей: {case}")
        targets.append(np.floor(255 * image + 0.5).astype(np.uint8))
    return np.stack(targets)


def save_json(path: Path, data: object) -> None:
    """Сохраняет JSON без нечисловых значений и неявной подмены неопределённых метрик."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def run_experiment(config: dict, output: Path) -> None:
    """Запускает парное сравнение трёх алгоритмов и сохраняет условия и результаты."""
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    factors = fresnel_factors(torch.float64)
    source = Path(__file__).resolve().parent
    snapshot = output / "source"
    snapshot.mkdir()
    for name in ("physics.py", "algorithms.py", "evaluate.py", "experiment.py", "verify_muon.py"):
        shutil.copy2(source / name, snapshot / name)
    shutil.copy2(source / "docs" / "muon_experiment.md", snapshot / "muon_experiment.md")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True)
    diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=source)
    (snapshot / "working_tree.diff").write_bytes(diff)
    save_json(output / "parameters.json", {
        "config": config, "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "torch": torch.__version__, "numpy": np.__version__,
        "pillow": PIL.__version__, "revision": revision, "git_status": status,
        "physics": {"N": N, "wavelength_m": WAVELENGTH, "pitch_m": PITCH, "z_m": DISTANCES},
        "deterministic_algorithms": True, "rng": "torch.Generator CPU, manual_seed per initialization",
        "split": "exploratory_only; no training, validation selection or held-out test",
        "timing": "CPU, batch=1, one untimed iteration warmup per algorithm; optimization and history only",
    })
    summary = []
    for case in config["cases"]:
        targets = synthetic_targets(case)
        case_path = output / case
        case_path.mkdir()
        checksums = {}
        for j in range(4):
            path = case_path / f"target_{j + 1}.png"
            Image.fromarray(targets[j]).save(path)
            checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        save_json(case_path / "inputs.json", checksums)
        target = torch.tensor(targets.astype(np.float64) / 255)
        for seed in config["seeds"]:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            initial = (2 * torch.rand((N, N), generator=generator, dtype=torch.float64) - 1) * math.pi
            seed_path = case_path / f"seed_{seed}"
            seed_path.mkdir()
            np.save(seed_path / "initial_phi.npy", initial.numpy())
            runs = [("gs", run_gs, ()), ("gs_muon", run_gs_muon, (config["muon"],)),
                    ("gs_adam", run_gs_adam, (config["adam"],))]
            for name, algorithm, extra in runs:
                directory = seed_path / name
                directory.mkdir()
                algorithm(initial.clone(), target, factors, 1, config["loss_epsilon"], *extra)
                started = time.perf_counter()
                phi, history = algorithm(initial.clone(), target, factors, config["steps"],
                                         config["loss_epsilon"], *extra)
                elapsed = time.perf_counter() - started
                save_json(directory / "history.json", history)
                metrics = export_result(directory, phi, targets, factors)
                save_json(directory / "metrics.json", metrics)
                np.save(directory / "continuous_phi.npy", phi.numpy())
                # GS: 4 прямых + 4 обратных; коррекция: ещё 4 прямых + 4 сопряжённых.
                steps = config["steps"]
                propagations = steps * (8 if name == "gs" else 16)
                entry = {"case": case, "seed": seed, "method": name,
                         "seconds": elapsed, "optimization_propagations": propagations,
                         "diagnostic_propagations": 4 * (steps + 1),
                         "initial_loss": history[0]["loss"], "final_loss": history[-1]["loss"],
                         "metrics": metrics}
                summary.append(entry)
                save_json(output / "summary.json", summary)
                print(f"{case} seed={seed} {name}: loss={history[-1]['loss']:.6g}, {elapsed:.3f}s", flush=True)


if __name__ == "__main__":
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_experiment(CONFIG, Path("results") / f"gs_muon_{stamp}")
