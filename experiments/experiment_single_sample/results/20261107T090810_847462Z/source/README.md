# Single-sample four-plane fine-tuning

This is a **new research protocol**, not a replacement for the original single-FFT/maximum-normalized PNG protocol. It deliberately overfits one tuple; there is no generalization claim. It uses the existing U-Net checkpoint without changing its architecture or parameter shapes.

## Run

From repository root, in the existing environment with Python, NumPy, Pillow and PyTorch:

```bash
python -m experiments.experiment_single_sample.verify
python -m experiments.experiment_single_sample.experiment
```

The first command runs numerical checks and a two-update-per-network end-to-end smoke experiment. The second runs the full budget (up to 3,000 updates per network control plus 3,000 adjoint-GS steps); it can be expensive. CPU float64/complex128 and one torch thread are intentional; there is no automatic device fallback. No new dependencies, downloads, or CLI are introduced.

Default inputs are repository-relative:

- Checkpoint: `experiments/experiment_unet_physics_adapter/results/mnist_10k_20261107T065810_115710Z/checkpoint_50_040.pt`.
- Tuple: the same run's `data/validation/0000/target_{1,2,3,4}.png`.

The selected tuple is no longer held out: “validation” in its historical directory name does not imply independence. The “validation score” below is an independent **criterion and quadrature refinement**, not independent data.

All experimental parameters are explicit in `experiment.py:CONFIG`. A documented programmatic override avoids editing sources, for example:

```bash
python - <<'PY'
from experiments.experiment_single_sample.experiment import CONFIG, run_experiment
run_experiment({**CONFIG, 'max_steps': 100, 'gs_steps': 100})
PY
```

Supply `checkpoint` and `sample_directory` in this dictionary to choose another checkpoint/tuple; paths are resolved relative to repository root (absolute paths also work). The checkpoint must have the existing `model` and `config` structure. Targets must be four nonconstant-support 128×128 grayscale uint8 PNGs; they are not silently resized, reordered, or corrected.

## Physical and tensor contract

- Phase: `[128,128]` radians, unconstrained continuous values during optimization, interpreted modulo 2π.
- Target and intensity arrays: `[4,128,128]`, axes `(plane,y,x)`, float64. Target `T=B/255`, no gamma conversion.
- Hardware: 128×128 constant-phase, fully filled 10 µm square pixels; incident amplitude A0=1; zero outside their square aperture. Hardware centers are `(index-64)*10 µm`.
- Wavelength 532 nm; z=(24.1,25.3,26.5,27.7) mm.
- **Common coordinates:** all four targets and detector grids have pitch 10 µm and centers `(index-64)*10 µm`. Source PNGs are explicitly reinterpreted as detector-cell averages on this common physical grid. No native-grid-to-common-grid resampling is implied: these are newly defined physical targets, not the historical unequal-size objects.
- Controlled domain D is the full 1.28 mm square FOV. Signal S is `T > 0.1` and guard is D minus S, frozen before optimization. Masks are saved.
- Intensities are detector-cell averages, not point samples. All powers use the cell area `(10 µm)^2`; Pin=`(128*10 µm)^2` in normalized `A0² m²` units. Watts cannot be inferred without illumination calibration. FOV power need not equal Pin, and powers at successive z are not additive allocations.

### Finite-pixel operator

For hardware pixel m and output quadrature coordinate X:

`K_z[X,m] = integral_pixel_m exp(i*pi*(X-x)^2/(lambda*z)) dx`.

Then `U_z = K_z @ exp(i*phi) @ K_z.T / (i*lambda*z)`, omitting only the global phase exp(ikz). This includes both Fresnel quadratic phases and the aperture-pixel area. The integral is computed by Gauss–Legendre quadrature (8 nodes per hardware-pixel axis by default). Since phase is constant inside each hardware pixel, integration can be factored into K; there are still only 16,384 phase parameters.

Detector integration uses uniform midpoint samples inside each detector cell, 4×4 for training, 8×8 for candidate scoring, and 16×16 for final numerical refinement. Source quadrature is simultaneously refined to 16 and 32 nodes. These all represent the **same physical detector cells**, not different targets. No periodic propagation boundary, FFT wraparound, or physical zero-padding is used by this direct separable operator. The CNN's original feature padding is unchanged.

Initial training versus scoring intensities must agree to 1% relative L2 per plane before optimization starts. Selected continuous and exported masks receive the same scoring-versus-refined check. A failed final check disqualifies the run's success criterion. This is a numerical convergence check, not proof of experimental accuracy or exact nonparaxial validity. The earlier fixed-mask ASM comparison does not automatically validate every new optimized mask; independent ASM validation remains a follow-up.

### Architecture compatibility

The original network's embedded two-step GS+Muon latent adapter is **unchanged**, including its historical point-sampled Fresnel surrogate and higher-order gradients. It is treated here as part of the checkpoint's feature transform, not the authoritative physical output model. All external optimization objectives, selection, controls and reported optical metrics use the new finite-pixel model. No detach is inserted inside the adapter; the original `create_graph=True` path remains active.

Replacing that internal transform too would change the checkpoint's forward algorithm and is **not** implemented under the requirement to keep it unchanged. This distinction is saved in every run's physical contract.

## Fixed normalization and objective

For each plane a fixed positive intensity calibration is determined **only from the target**:

`c_k = 0.5 * Pin / sum_S(T_k * detector_area)`.

Thus the desired absolute intensity `c_k*T_k` has useful efficiency 0.5 (its total FOV efficiency may exceed 0.5). This is a declared target calibration, not a claim of attainable efficiency. Normalized image intensity is `Itilde=I/c_k`; target is `Ttilde=T`. No current-image maximum, min/max rescaling, clipping, or fitted exposure is used in the combined objective or primary metrics. Itilde can exceed one; SSIM's reference data range stays **one** throughout.

- Amplitude MSE: mean_D `(sqrt(Itilde+1e-8)-sqrt(T+1e-8))²`.
- SSIM: Gaussian 11×11 window, sigma=1.5, population moments, valid-window mean, K1=.01, K2=.03, data_range=1. Loss=1-SSIM.
- Useful efficiency: `eta=sum_S(I)*area/Pin`.
- Requirement: `eta_req=max(0.35, eta_initial-0.005)` per plane, fixed from the initial refined evaluation. Hinge-squared penalty on its deficit.
- Guard: mean_(D\S) `(I/(mean_S I+1e-8))²`. Guard epsilon is in normalized A0² intensity units, separate from the amplitude epsilon.
- Spectral residual: unshifted orthonormal 2D FFT of `(Itilde-T)`, squared modulus weighted in three radial bands and divided by 128². Bands relative to detector-axis Nyquist are `[0,.15)`, `[.15,.5)`, `[.5,sqrt(2)]`; weights are .25, 1, 2. This is an explicit normalized version of the sum of band-weighted squared Fourier norms. The periodic FFT is only a loss feature, not a propagation boundary.
- Plane weights: all 1/4.
- Final term coefficients: amplitude=1, SSIM=.5, efficiency=5, guard=.02, frequency=.1, worst=.25.
- Worst-plane loss is `(logsumexp(10*ell_k)-log(4))/10`, where ell_k is that step's scheduled weighted plane loss.

First 50 updates use only amplitude and efficiency. SSIM, guard, frequency and worst weights then ramp linearly over 150 updates. These starting coefficients are hypotheses, not validated optimal hyperparameters. The original-loss control retains its original formula `mean((I/(max(I)+1e-12)-T)^2)` but uses the **same corrected physical operator and targets** as every other arm; it deliberately retains dynamic normalization only inside that historical loss.

## Controls and optimizer

1. Unmodified checkpoint before fitting, evaluated under the new physical contract.
2. Same checkpoint, all weights optimized with original loss.
3. Same checkpoint, all weights optimized with the combined objective.
4. Finite-pixel **adjoint-GS** amplitude-projection control initialized from the same checkpoint phase.

Each network arm reloads identical initial weights and creates **fresh Adam** state, lr=1e-5 (100× below the original weight-Muon numerical lr), betas=(.9,.999), eps=1e-8, no weight decay, no clipping. Adam and Muon learning rates are not directly equivalent. This explicitly overrides the earlier project Muon-only convention for this user-requested experiment. The internal adapter's fixed Muon algorithm is still unchanged.

GS cannot use the old inverse FFT: a finite-pixel/cropped detector operator is not unitary. The control instead applies the explicit relaxed projection:

```
u = exp(i*phi)
v_k = P_k u
v'_k = sqrt(c_k*T_k) * exp(i*angle(v_k))
u' = u + tau * (1/4) sum_k P_k^H(v'_k-v_k)
phi' = angle(u')
```

Target intensity is constant inside each detector cell when imposing subcell amplitudes. `tau=1/mean_k(||P_k||_2²)`, using `||P_k||_2=|1/(lambda*z_k)|*||K_k||_2²`; it bounds the averaged normal-operator step. This is a documented adjoint-GS variant, **not canonical inverse-propagation GS**, and uses the same physical operator/ROI/calibration. No claim of its optimality is made.

Network arms and GS have equal default update limits but different per-update costs. Actual steps and wall times are saved; these are end-to-end CPU timings, not isolated inference benchmarks. Final direct phase refinement is optional in the proposal and not implemented in this first feature.

## Selection, stopping and success

Every 20 steps, candidates are scored on the more finely integrated operator. Selection is lexicographic:

1. All per-plane useful efficiencies at least `max(0,eta_initial-0.005)`.
2. If none are feasible, smallest worst efficiency deficit.
3. Highest `.5*mean_SSIM + .5*min_SSIM + .1*mean_eta`.

This score is independent of the training-loss sum and its ramp. Initial phase is eligible, so failure to improve without losing power can legitimately select step 0. Best and last weights/phases are distinguished and saved. Checkpoint selection occurs only at the documented evaluation cadence.

After at least 400 updates, a numerical plateau requires ten consecutive evaluations with relative full-weight objective span <1e-4, score span <1e-4, and every successive intensity relative L2 change <1e-4. This is an operational stopping condition, not a convergence theorem or early stopping for generalization. Hitting the 3,000-update ceiling is recorded as `budget_exhausted`, not convergence. Increase the explicit budget for a new run if needed; automatic resume is not implemented.

Declared success: combined-loss best continuous phase gains at least .05 mean SSIM and .03 minimum-plane SSIM over the initial checkpoint, preserves every initial useful efficiency to within .005 absolute, and all exported arms pass final quadrature refinement. These thresholds are transparent experiment criteria, not external acceptance requirements. The original-loss and GS results must still be examined to attribute an improvement to the new objective rather than extra computation. Success does not mean a global optimum, validated real-device performance, or generalization.

## Outputs and reproducibility

Unique timestamped `results/<UTC>/` directories never reuse other runs. Each contains:

- Actual config, fixed calibration, region masks, target PNG copies, hashes, initial weights, dependency/platform versions, seed, git revision/status/diff and source snapshot (including untracked implementation).
- Per-arm loss history, per-plane objective terms, refined validation metrics, selection key, update count, stopping status, elapsed time, best/last network and Adam states.
- Best **continuous** phase in radians; `doe.png` with original **255-interval** phase encoding and half-up rounding; evaluation after PNG reimport.
- Continuous and reimported intensity arrays, refined arrays, and per-plane **float32 TIFF absolute-intensity maps without contrast rescaling**. These are relative to A0², not calibrated watts.
- SSIM, Pearson (null/status for constant images), amplitude MSE and RMSE, normalized intensity MSE, NRMSE (normalized by target RMS), useful/FOV/guard efficiencies, guard powers, spectral residuals, mean/min SSIM and mean/min eta, and quantization RMS.
- `comparison.json` collects all four arms and the explicit success decision; individual files expose per-plane results rather than only averages.

All final metrics use unrounded detector intensities from the decoded mask with the fixed calibration; they do **not** reuse historical maximum-normalized restored-PNG SSIM. No plotting exposure enters metrics. Reproduction is statistical/numerical across platforms, not guaranteed bitwise; recorded source and environment are the authority. No completed full optimization or quality improvement is claimed merely by adding this implementation.
