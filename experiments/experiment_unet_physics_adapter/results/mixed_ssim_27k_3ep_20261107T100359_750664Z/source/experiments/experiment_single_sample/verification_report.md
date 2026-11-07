# Verification: single-sample feature

Command from repository root:

```bash
python -m experiments.experiment_single_sample.verify
```

Verified environment: Python 3.12.12, PyTorch 2.10.0, NumPy 2.2.6, Pillow 10.4.0; CPU float64/complex128, one torch thread, seed 17. No expensive full optimization was run.

Numerical and end-to-end smoke artifacts: `results/20261107T091215_724188Z/verification.json` and its adjacent `comparison.json`, config, source snapshot and per-arm histories. Preceding smoke runs are retained separately; no run was overwritten.

## Numerical properties

| Check | Measured error | Tolerance | Result |
|---|---:|---:|---|
| Euclidean propagation adjoint, relative complex inner-product error | 2.47e-14 | 1e-10 | pass |
| Source-pixel Gauss quadrature 8 vs 16, relative intensity L2 | 2.54e-12 | 1e-10 | pass |
| Detector quadrature 4×4 vs 8×8, random phase, relative intensity L2 | 8.22e-3 | 1e-2 | pass |
| Global phase invariance, relative intensity L2 | 4.66e-16 | 1e-10 | pass |
| 2π periodicity, relative intensity L2 | 4.37e-16 | 1e-10 | pass |
| Differentiable SSIM vs existing independent NumPy formula, absolute error | 2.60e-16 | 1e-8 | pass |
| Full combined objective autograd vs central directional difference | 6.21e-10 | 1e-5 | pass |
| Global-phase gradient component / gradient norm | 6.41e-16 | 1e-10 | pass |

Finite-difference relative errors for steps 1e-3, 1e-4, 1e-5: 6.21e-10, 5.16e-9, 5.27e-8. The 1e-10 algebra tolerances allow complex128 accumulation error; finite differences allow truncation/cancellation; the 1% integration target is an explicit numerical diagnostic, not an optical accuracy guarantee.

## Two-update smoke experiment

Loaded the real checkpoint and fixed validation tuple 0000; both network arms reload the same weights, independently create fresh Adam state and perform two updates. The smoke schedule uses warmup=0, ramp=1, evaluation every update; it is deliberately not the full optimization schedule.

- Original-loss weight change L2: 0.002990; every populated Adam state has step=2.
- Combined-loss weight change L2: 0.003430; every populated Adam state has step=2.
- Both gradients are finite, including backpropagation through the existing higher-order GS+Muon adapter.
- Adjoint-GS performs two updates and saves its history and selected phase.
- All four arms export continuous phase, 8-bit phase PNG, reimported-mask intensities, absolute TIFFs, and all requested per-plane metrics.
- All selected continuous and reimported masks pass training/scoring and scoring/refined quadrature checks and the FOV power bound.
- Maximum scoring/refined relative intensity difference across these exports: 6.53e-4; maximum training/scoring difference: 2.67e-3 (both below 1%).
- All iterative arms correctly report `budget_exhausted`, not convergence; the declared combined-loss success criterion is **false** after this smoke test.

The two-update numbers establish execution and numerical consistency, not reconstruction-quality superiority. Full optimization, plateau behavior on a long trajectory, hyperparameter quality, independent ASM checking of newly optimized masks, and physical-device validation remain untested. Final optional direct phase refinement is not implemented.

AST inspection found one-line docstrings on all functions and no `assert` statements in the new Python files. Existing unrelated code and original result artifacts were left unchanged.
