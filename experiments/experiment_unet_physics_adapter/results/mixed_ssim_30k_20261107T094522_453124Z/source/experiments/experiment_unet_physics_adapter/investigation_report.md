# Investigation of the latest completed MNIST run

Run: `results/mnist_10k_20261107T065810_115710Z` (50 epochs, seed 17, MPS float32, 2,500 training quadruples / 10,000 images, 100 validation and 100 test quadruples).

## Conclusion

**The observed SSIM is not an established four-plane physical limit.** Small, same-grid GS controls produce much higher SSIM than the trained network. The evidence points to a combination of poor learned reconstruction/background suppression and mismatch between the trained maximum-normalized MSE and SSIM. Multi-plane difficulty is real in these controls, but does not explain SSIM near 0.087 by itself. Export corruption and 8-bit phase quantization are not supported as explanations.

There is also a measurable **point-sample versus finite-pixel physical-model discrepancy**. It affects physical predictions, but cannot explain the entire reported network-versus-target gap: those reported metrics were calculated with the same point-sampled operator used for training.

No training, loss, target preprocessing, or original result files were changed.

## 1. Verified original results

| Split | Mean SSIM | Mean Pearson |
|---|---:|---:|
| Validation, 400 plane pairs | 0.087839 | 0.566643 |
| Test, 400 plane pairs | 0.087240 | 0.573907 |

Test per-plane SSIM: **0.088563, 0.086439, 0.082066, 0.091894**.
Test per-plane Pearson: **0.626714, 0.602466, 0.554151, 0.512298**.

Training loss continued decreasing: epoch-50 mean per-plane losses were 0.05871, 0.05601, 0.05942, 0.06261. Validation losses were 0.05767, 0.05369, 0.05803, 0.06328. These values do not suggest a large train/validation generalization gap, but do not establish optimizer convergence.

For **all 100 test examples**:

- Repropagation of the decoded saved phase exactly matches the saved raw intensity arrays (relative L2 error 0).
- Regenerating the restored PNGs gives maximum code difference 0.
- Discrete integrated power differs from incident power by at most 2.22e-16, below the 1e-10 floating-point diagnostic tolerance.

These are internal consistency checks, not experimental optical validation.

## 2. Physical sampling and protocol differences from the proposed plan

The original protocol explicitly assigns targets to each plane's native grid (`docs/doe_physics.md`, sections 3–6). Output pitches are 10.0165625, 10.5153125, 11.0140625, 11.5128125 micrometres; the field-of-view ratio is 1.14938. The input targets are four independently selected MNIST images, not repetitions of one image.

Thus the same digital extent has different physical sizes, but this is **documented behavior, not an implementation inconsistency**. Equal physical target extent requires a separately specified common-coordinate target and resampling experiment. The original paired SSIM already compares target and reconstruction on the same native physical grid for that plane.

Additional distinctions:

- Export has **255 intervals**, with optically equivalent endpoint codes, not 256 intervals. Expected uniform-rounding RMS is about 0.00711 rad.
- Four propagation planes are cross-sections of one field, not four independent power allocations. Do not sum their efficiencies as consumed incident power. Include pixel area in power sums.
- Zero-padding alone does not refine the physical pixels. Subdivide the same 128×128 constant-phase pixels while keeping the 1.28 mm aperture and 10 micrometre hardware pitch fixed.
- A Fresnel/ASM difference alone does not distinguish approximation error, quadrature, finite-pixel modeling, boundary wraparound, and coordinate mismatch.
- The architecture document currently favors Muon for further development; the GS-only runs here are explicitly requested diagnostic controls, not an unannounced optimizer replacement.

## 3. Background energy and cross-target confusion

Using the **post-hoc diagnostic support** `T > 0.1`, averaged over all 400 test planes:

- Support occupies 20.16% of pixels.
- Signal efficiency is **39.61%** of incident power in the discrete model.
- Background contains **60.39%** of plane power.
- Peak/mean intensity is 6.65.

Signal efficiency per plane: 37.35%, 39.45%, 39.76%, 41.90%. This threshold was not a predefined training ROI and its sensitivity has not been tested. Raw absolute watts are not available because illumination is normalized.

Mean Pearson cross-target matrix (rows = reconstructed plane, columns = target):

```
0.6267  0.5805  0.5268  0.5000
0.6360  0.6025  0.5582  0.5351
0.6071  0.5839  0.5542  0.5342
0.5630  0.5479  0.5311  0.5123
```

This matrix compares **digital indices**, not common physical coordinates across planes. Shared centered MNIST structure also confounds off-diagonal correlations. It indicates poor target specificity but is not an isolated physical cross-talk measurement.

The JSON contains amplitude RMSE and signal-region intensity coefficient of variation. The latter is deliberately **not called pure speckle contrast**, because MNIST intensity varies within the support. Resolution-matched SSIM and detrended speckle statistics were not computed.

## 4. Continuous phase versus export

Replayed the final checkpoint on validation examples 0–2 using CPU float32, then evaluated in float64. Exported replay masks match 99.988–99.994% of original MPS mask codes; the original continuous phase was not saved, so this is a near-identical replay, not exact recovery of it.

Measured circular phase rounding RMS: 0.00710–0.00713 rad.

| Example | Continuous SSIM | Quantized-phase SSIM |
|---|---:|---:|
| 0 | 0.067381 | 0.067284 |
| 1 | 0.071938 | 0.071913 |
| 2 | 0.087079 | 0.087041 |

These SSIM values use continuous normalized intensities, without restored-image uint8 rounding, to isolate phase quantization. Phase quantization is far too small an effect here to explain the low SSIM. The separate 100-example audit checks actual saved PNG reconstruction consistency.

## 5. Small single-/multi-plane controls

Validation examples 0–2 only; first 1, 2, 3, or 4 targets; original native output grids; continuous phases; identical random initial phase per example and method, seed 17; 100 iterations. GS uses power-matched target amplitudes. GS+Muon adds the original max-normalized intensity-MSE gradient update with the run's phase-Muon parameters. Every update history and final phase is saved.

| Planes | Method | Mean SSIM | Mean Pearson | Mean normalized MSE |
|---:|---|---:|---:|---:|
| 1 | GS | 0.7920 | 0.9857 | 0.00974 |
| 1 | GS+Muon | 0.7791 | 0.9860 | 0.00672 |
| 2 | GS | 0.6355 | 0.7492 | 0.04307 |
| 2 | GS+Muon | 0.6287 | 0.7494 | 0.03975 |
| 3 | GS | 0.5664 | 0.6689 | 0.06861 |
| 3 | GS+Muon | 0.5557 | 0.6701 | 0.06273 |
| 4 | GS | 0.5635 | 0.6356 | 0.06903 |
| 4 | GS+Muon | 0.5496 | 0.6370 | 0.06436 |
| 4 | Trained network replay | 0.0755 | 0.5714 | 0.06253 |

Interpretation:

- Single-plane reconstruction works well under the original discrete operator.
- Adding planes substantially reduces attainable fidelity at this budget.
- Four-plane GS SSIM remains far above the network's SSIM on these examples.
- **The network has lower MSE than either four-plane control despite much worse SSIM.** This is strong evidence that the trained objective does not track the desired image quality sufficiently; it is not proof that the weight optimizer failed to minimize its actual objective.
- GS+Muon improves MSE over GS while slightly reducing SSIM, another indication of objective/metric tension.

Limitations: three examples, one seed, nested plane subsets, unequal compute versus amortized inference, no hyperparameter search or convergence guarantee. GS is an amplitude-projection method, not a solver explicitly minimizing the same MSE. The network contains two GS+Muon steps in latent feature branches, followed by learned mixing and a phase head; its output is not the result of 100 direct phase-space GS+Muon iterations. This does not isolate Muon versus another weight optimizer, nor the adapter versus the backbone.

## 6. Fixed-mask physical propagation checks

Validation example 0, saved quantized phase; constant 1.28 mm square aperture; no optimization. All methods sampled on the original plane-specific 128×128 output coordinates. Physical pixels are piecewise constant with 100% fill factor. Subcells are centered symmetrically on original pixel centers.

1. Direct separable point-sample Fresnel sum versus original FFT: relative intensity L2 errors **1.56e-14–2.39e-14** (passes 1e-10 tolerance).
2. Midpoint integration with 2, 4, 8 subcells per pixel axis: 256², 512², 1024² integration grids, same phase parameters and aperture.
3. Fresnel subdivision 4 versus 8: **0.149–0.170%** difference (passes the declared 1% numerical diagnostic criterion).
4. Subdivision-8 finite-pixel Fresnel versus original point-sampled FFT: **17.85%, 11.73%, 8.62%, 6.81%** per-plane differences.
5. Exact ASM with subdivisions 2 and 4, padding factors 2 and 4; further padding-8 check at subdivision 4. Output intensity is cubic-interpolated to the original physical coordinates.
6. ASM subdivision-4 padding 2 versus 4 differs by **3.59–4.14%**: padding 2 is insufficient.
7. ASM subdivision-4 padding 4 versus 8 differs by **0.00136–0.00229%**: padding is stable at this subdivision.
8. ASM subdivision 2 versus 4, padding 4: **0.615–1.061%**; plane 1 narrowly misses the 1% refinement criterion. Full ASM subpixel convergence remains unproven.
9. ASM subdivision-4/padding-8 versus Fresnel subdivision-8: **0.412%, 0.389%, 0.355%, 0.328%**.

For this mask and sampled ROI, the much larger discrepancy is consistent with **finite-pixel integration versus point sampling**, not a gross error in the Fresnel formula or a large Fresnel-versus-exact-ASM difference. The small residual still mixes ASM subpixel discretization, intensity interpolation, and Fresnel approximation; it is not a clean measurement of the paraxial approximation error. This is one-mask evidence, not validation for every possible DOE.

NumPy emitted divide/overflow/invalid warnings in complex matrix products despite finite results. An independent PyTorch complex128 subdivision-8 calculation agrees to **2.31e-15** relative L2, below 1e-10. Warnings were not suppressed. A first follow-up process aborted on duplicate OpenMP initialization; changing import order to load the existing SciPy diagnostic module before torch resolved it, without the unsafe duplicate-runtime environment override.

## 7. Axial separability

Using D = 1.28 mm and the proposed approximate formula, DOF estimates are **1.509, 1.663, 1.824, 1.993 mm**. The 1.2 mm spacing is 0.60–0.80 of this scale, making cross-talk plausible. This order-of-magnitude estimate uses the square aperture side as D; it is not a measured axial correlation length. An optimized single-target axial scan has not been run.

## 8. Recommended next experiment and remaining gaps

1. Keep original protocol results as baseline; define common physical targets, ROI, field of view, and detector sampling explicitly for the new experiment.
2. Compare amplitude-domain, power-calibrated fidelity plus background loss against current max-normalized MSE on validation only. Document target power normalization and zero-intensity derivative handling before implementation; do not silently replace final PNG metrics.
3. Complete the proposed 1/4-plane × Fresnel/ASM × method experiment, with consistent physical pixels and independently converged propagation. The four-input trained network cannot simply be called a one-plane solver without specifying its adaptation/retraining.
4. Include direct phase-space baselines, final physical refinement of the network phase, axial scans, multiple seeds, matched-budget reporting, and resolution-matched SSIM.

**Not completed:** the full common-coordinate 2×2×2 study; one-plane network training; ASM-based optimization; Adam/L-BFGS comparisons; amplitude-loss training; axial correlation scan; resolution-matched metrics; multiple seeds. No claim of an intrinsic four-plane limit or universal optimizer superiority follows from this investigation.

## Reproduction and artifacts

New scripts:

- `experiments/experiment_unet_physics_adapter/investigate_latest.py`
- `experiments/experiment_unet_physics_adapter/investigate_convergence.py`

From repository root, using the existing NumPy/SciPy/PyTorch/Pillow environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.experiment_unet_physics_adapter.investigate_latest
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.experiment_unet_physics_adapter.investigate_convergence
```

Outputs are in the run's `investigation_v1/`: `diagnostics.json`, `convergence.json`, all 24 control phases, per-iteration loss histories, per-example/per-plane metrics, and copies of both scripts. Existing outputs cause an explicit refusal to overwrite; archive/rename that investigation directory before rerunning. Scripts intentionally target this specific run and do not select a changing 'latest' directory.

`diagnostics.json` records dependency versions, checkpoint/parameter/source hashes, seed and precision. Original run metadata and exact data manifests remain alongside it. Original `inputs.json` SHA256: `cae29a5de2dcd8f665c9acea51eed59777d33c4efe31177cf16599f4414111e3`.

Current imported physics/model/algorithm/evaluation files match the run snapshot. The training entry point differs only in checkpoint-save frequency; this diagnostic does not invoke it. Existing unrelated working-tree changes were left untouched. CPU replay does not promise bitwise identity with MPS.
