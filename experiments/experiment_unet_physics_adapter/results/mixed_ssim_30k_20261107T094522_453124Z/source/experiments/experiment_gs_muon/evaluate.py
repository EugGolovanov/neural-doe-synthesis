"""Экспорт и оценка по протоколу docs/doe_physics.md без skimage."""

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__:
    from .physics import DISTANCES, N, PITCH, WAVELENGTH, intensity
else:
    from physics import DISTANCES, N, PITCH, WAVELENGTH, intensity


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
    """Сохраняет маску и оценивает четыре восстановления после повторного чтения PNG."""
    wrapped = (phi.detach().numpy() + math.pi) % (2 * math.pi) - math.pi
    codes = np.clip(np.floor(255 * (wrapped + math.pi) / (2 * math.pi) + 0.5), 0, 255).astype(np.uint8)
    Image.fromarray(codes).save(directory / "doe.png")
    decoded = np.array(Image.open(directory / "doe.png"), dtype=np.float64)
    quantized_phi = torch.tensor(-math.pi + 2 * math.pi * decoded / 255, dtype=phi.dtype)
    values = intensity(quantized_phi, factors).detach().numpy()
    np.save(directory / "intensity.npy", values)
    metrics = []
    for j, z in enumerate(DISTANCES):
        maximum = float(values[j].max())
        if maximum <= 0 or not np.isfinite(values[j]).all():
            raise ValueError("Восстановление имеет нулевую мощность или нечисловые значения.")
        restored = np.floor(255 * values[j] / maximum + 0.5).astype(np.uint8)
        path = directory / f"restored_{j + 1}.png"
        Image.fromarray(restored).save(path)
        restored = np.array(Image.open(path))
        metrics.append({"plane": j + 1, "z_m": z,
                        "output_pitch_m": WAVELENGTH * z / (N * PITCH),
                        "maximum": maximum, **image_metrics(targets[j], restored)})
    return metrics
