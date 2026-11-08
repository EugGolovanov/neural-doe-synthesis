"""Экспорт и оценка по протоколу phase_initializer/docs/doe_physics.md без skimage."""

import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .physics import DISTANCES, N, PITCH, WAVELENGTH, intensity


def image_metrics(target: np.ndarray, restored: np.ndarray) -> dict:
    """Вычисляет Pearson и SSIM с гауссовым окном 11×11 на полных окнах."""
    a = target.astype(np.float64) / 255
    b = restored.astype(np.float64) / 255
    centered_a, centered_b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(centered_a) * np.linalg.norm(centered_b)
    pearson = float(np.sum(centered_a * centered_b) / denominator) if denominator > 0 else None
    coordinates = np.arange(-5, 6, dtype=np.float64)
    weights = np.exp(-coordinates**2 / (2 * 1.5**2))
    weights /= weights.sum()
    kernel = weights[:, None] * weights[None, :]
    windows_a = np.lib.stride_tricks.sliding_window_view(a, (11, 11))
    windows_b = np.lib.stride_tricks.sliding_window_view(b, (11, 11))
    mean_a = np.einsum("ijkl,kl->ij", windows_a, kernel)
    mean_b = np.einsum("ijkl,kl->ij", windows_b, kernel)
    variance_a = np.einsum("ijkl,kl->ij", windows_a**2, kernel) - mean_a**2
    variance_b = np.einsum("ijkl,kl->ij", windows_b**2, kernel) - mean_b**2
    covariance = np.einsum("ijkl,kl->ij", windows_a * windows_b, kernel) - mean_a * mean_b
    ssim = ((2 * mean_a * mean_b + 0.01**2) * (2 * covariance + 0.03**2)
            / ((mean_a**2 + mean_b**2 + 0.01**2) * (variance_a + variance_b + 0.03**2)))
    return {"pearson": pearson, "pearson_status": "constant_image" if pearson is None else "defined",
            "ssim": float(ssim.mean())}


def export_result(directory: Path, phi: torch.Tensor, targets: np.ndarray,
                  factors: tuple) -> list:
    """Сохраняет точную NPY-фазу и отдельно оценивает непрерывное и протокольное PNG-восстановления."""
    original_phase = phi.detach().numpy()
    if not np.isfinite(original_phase).all():
        raise ValueError("Фаза содержит нечисловые значения.")
    phase_path = directory / "phase.npy"
    with phase_path.open("xb") as stream:
        np.save(stream, original_phase, allow_pickle=False)
    phase = np.load(phase_path, allow_pickle=False)
    if phase.dtype != original_phase.dtype or not np.array_equal(phase, original_phase):
        raise ValueError("Повторное чтение NPY изменило фазу или её dtype.")
    continuous_values = intensity(torch.from_numpy(phase), factors).detach().numpy()
    np.save(directory / "intensity_from_phase.npy", continuous_values, allow_pickle=False)
    wrapped = (phase + math.pi) % (2 * math.pi) - math.pi
    codes = np.clip(np.floor(255 * (wrapped + math.pi) / (2 * math.pi) + 0.5), 0, 255).astype(np.uint8)
    Image.fromarray(codes).save(directory / "doe.png")
    decoded = np.array(Image.open(directory / "doe.png"), dtype=np.float64)
    quantized_phi = torch.tensor(-math.pi + 2 * math.pi * decoded / 255, dtype=phi.dtype)
    values = intensity(quantized_phi, factors).detach().numpy()
    np.save(directory / "intensity.npy", values)
    metrics = []
    continuous_metrics = []
    quantized_phase_metrics = []
    for j, z in enumerate(DISTANCES):
        maximum = float(values[j].max())
        if maximum <= 0 or not np.isfinite(values[j]).all():
            raise ValueError("Восстановление имеет нулевую мощность или нечисловые значения.")
        continuous_maximum = float(continuous_values[j].max())
        if continuous_maximum <= 0 or not np.isfinite(continuous_values[j]).all():
            raise ValueError("Восстановление из NPY имеет нулевую мощность или нечисловые значения.")
        continuous_normalized = continuous_values[j] / continuous_maximum
        continuous_metrics.append({"plane": j + 1, "z_m": z,
                                   "output_pitch_m": WAVELENGTH * z / (N * PITCH),
                                   "maximum": continuous_maximum,
                                   **image_metrics(targets[j], 255 * continuous_normalized)})
        quantized_phase_metrics.append({"plane": j + 1,
                                        **image_metrics(targets[j], 255 * values[j] / maximum)})
        restored = np.floor(255 * values[j] / maximum + 0.5).astype(np.uint8)
        path = directory / f"restored_{j + 1}.png"
        Image.fromarray(restored).save(path)
        restored = np.array(Image.open(path))
        metrics.append({"plane": j + 1, "z_m": z,
                        "output_pitch_m": WAVELENGTH * z / (N * PITCH),
                        "maximum": maximum, **image_metrics(targets[j], restored)})
    circular_error = np.angle(np.exp(1j * (quantized_phi.numpy() - phase)))
    relative_intensity_error = np.linalg.norm((values - continuous_values).reshape(4, -1), axis=1) / np.linalg.norm(continuous_values.reshape(4, -1), axis=1)
    phase_report = {"phase_file": "phase.npy", "phase_units": "radians", "phase_axes": ["y", "x"],
                    "phase_dtype": str(phase.dtype), "phase_shape": list(phase.shape),
                    "phase_range": "unwrapped optimizer output; no clipping or quantization",
                    "npy_roundtrip_exact": True,
                    "continuous_intensity_file": "intensity_from_phase.npy",
                    "continuous_metrics": continuous_metrics,
                    "png_phase_continuous_intensity_metrics": quantized_phase_metrics,
                    "png_protocol_metrics": metrics,
                    "phase_quantization_max_circular_error_rad": float(np.abs(circular_error).max()),
                    "phase_quantization_relative_intensity_error": relative_intensity_error.tolist(),
                    "metric_convention": "Per-plane max normalization; continuous metrics without image rounding; PNG protocol unchanged",
                    "png_note": "PNG compression is lossless; phase changes arise from uint8 quantization, not interpolation"}
    (directory / "phase_evaluation.json").write_text(json.dumps(phase_report, indent=2, allow_nan=False) + "\n")
    return metrics
