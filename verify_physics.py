"""Проверка дискретной модели; запуск: python3 verify_physics.py (NumPy)."""
import numpy as np

N = 128
DELTA = 10e-6
WAVELENGTH = 532e-9
DISTANCES = np.array([24.1, 25.3, 26.5, 27.7]) * 1e-3


def propagate(field, z):
    """Одно-FFT Френель; общая несущая exp(ikz) опущена."""
    x = (np.arange(N) - N // 2) * DELTA
    xx, yy = np.meshgrid(x, x)
    output_step = WAVELENGTH * z / (N * DELTA)
    X, Y = np.meshgrid((np.arange(N) - N // 2) * output_step,
                       (np.arange(N) - N // 2) * output_step)
    chirped = field * np.exp(1j * np.pi * (xx**2 + yy**2) / (WAVELENGTH * z))
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(chirped)))
    result = (DELTA**2 / (1j * WAVELENGTH * z) * spectrum
              * np.exp(1j * np.pi * (X**2 + Y**2) / (WAVELENGTH * z)))
    return result, output_step


def encode(phase):
    return np.clip(np.floor(255 * (phase + np.pi) / (2 * np.pi) + 0.5),
                   0, 255).astype(np.uint8)


def decode(codes):
    return -np.pi + 2 * np.pi * codes.astype(np.float64) / 255


def main():
    rng = np.random.default_rng(2026)
    phase = rng.uniform(-np.pi, np.pi, (N, N))
    field = np.exp(1j * phase)
    x = (np.arange(N) - N // 2) * DELTA
    xx, yy = np.meshgrid(x, x)
    expected_steps = [10.0165625, 10.5153125, 11.0140625, 11.5128125]
    for z, expected in zip(DISTANCES, expected_steps):
        out, step = propagate(field, z)
        np.testing.assert_allclose(step * 1e6, expected, rtol=1e-13)
        input_power = np.sum(abs(field)**2) * DELTA**2
        output_power = np.sum(abs(out)**2) * step**2
        np.testing.assert_allclose(output_power, input_power, rtol=1e-13, atol=0)
        # Независимая сумма по исходному ядру, без разложения квадрата и FFT.
        errors = []
        for q, p in [(0, 0), (23, 97), (64, 64), (127, 127), (100, 45)]:
            X, Y = (p - N // 2) * step, (q - N // 2) * step
            direct = DELTA**2 / (1j * WAVELENGTH * z) * np.sum(
                field * np.exp(1j * np.pi * ((X - xx)**2 + (Y - yy)**2)
                               / (WAVELENGTH * z)))
            np.testing.assert_allclose(out[q, p], direct, rtol=1e-10, atol=1e-11)
            errors.append(abs(out[q, p] - direct))
        shifted, _ = propagate(field * np.exp(0.73j), z)
        np.testing.assert_allclose(abs(shifted)**2, abs(out)**2,
                                   rtol=1e-10, atol=1e-12)
        rho = np.sqrt(2) * (N * DELTA + N * step) / 2
        phase_error = np.pi * rho**4 / (4 * WAVELENGTH * z**3)
        print(f'z={z*1e3:.1f} mm; step={step*1e6:.7f} um; '
              f'window={N*step*1e3:.5f} mm; '
              f'Fresnel phase bound~{phase_error:.3f} rad; '
              f'direct sum error={max(errors):.2e}')
    codes = np.arange(256, dtype=np.uint8)
    np.testing.assert_array_equal(encode(decode(codes)), codes)
    np.testing.assert_array_equal(encode(np.array([-np.pi, np.pi])), [0, 255])
    assert np.max(abs(decode(encode(phase)) - phase)) <= np.pi / 255 + 1e-14
    np.testing.assert_allclose(np.exp(1j * decode(codes[[0, 255]])), [-1, -1], atol=1e-14)
    print(f'Chirp sampling threshold: {N*DELTA**2/WAVELENGTH*1e3:.8f} mm')
    print('PASS: scales, direct sum, power, global phase, phase quantization')


if __name__ == '__main__':
    main()
