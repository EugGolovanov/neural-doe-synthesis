# Mixed-dataset SSIM-informed pretraining

Run from the repository root:

```bash
python3 -m pip install torch==2.10.0 torchvision==0.25.0 numpy Pillow
python3 -m experiments.experiment_unet_physics_adapter.pretrain_mixed
```

Requires Apple MPS and Python 3.12; no device or batch-size fallback. The requested microbatch is 256 quadruples (1024 images), also one optimizer batch; the final batch has 48 quadruples. Activation checkpointing recomputes the model without truncating its higher-order gradients. Memory feasibility at the full batch must be checked on the actual machine; the implementation does not disable MPS memory limits.

## Experiment

Hypothesis: adding structural supervision improves exported SSIM relative to the existing MNIST-only MSE trainer. This run changes both data and loss, so it is not a controlled loss ablation and cannot alone establish an advantage. No published method is claimed reproduced. The architecture and inner GS+Muon solver remain unchanged, including the MSE inner objective and full differentiation. The outer objective is the documented single-FFT maximum-normalized MSE plus `0.5*(1-SSIM)`, averaged over samples and planes; no phase regularizer, SSIM ramp, or quantization surrogate. This is not the separate finite-pixel single-sample experiment's five-term objective.

SSIM uses the existing Gaussian 11x11 valid-window implementation, sigma 1.5, C1=0.01², C2=0.03², population covariance, data range 1. Maximum normalization uses epsilon 1e-12, with gradients through the maximum. Physical geometry, FFT normalization and exported PNG evaluation are unchanged from `docs/doe_physics.md`.

## Data and parameters

30,000 samples mean 120,000 individual training images: 40,000 each from the official training sets of CIFAR-10, Fashion-MNIST and MNIST. Validation contains 99 quadruples (132 images per source), drawn disjointly from official training sets; test contains 99 quadruples from official test sets. Source indices are selected with NumPy PCG64 seed 2026. Prepared-pixel hashes exclude exact duplicates across all splits and fully black images; exclusions consume the next shuffled index. Independent images are shuffled into fixed quadruples, so an individual quadruple need not contain every dataset.

Torchvision downloads/checks official archives into `experiments/experiment_unet_physics_adapter/datasets`. Pillow converts images to L grayscale and resizes with BILINEAR to 128x128 before saving uint8 PNG. This explicit preprocessing is applied only to the source datasets; training reads those PNGs as float32 intensity/255 with shape [B,4,128,128]. Labels do not enter training. Data are read per microbatch, not held as a full float32 dataset in RAM.

50 epochs, seed 17 initialization, float32 MPS, one CPU thread, width 8, adapter rank 2, alpha 2, two inner solver steps. Outer Muon learning rate 0.001, momentum 0.95; inner phase Muon learning rate 0.1. Remaining fixed Muon settings and all actual parameters are in `parameters.json`. Epoch shuffle uses a separate Torch generator seeded 2026. Deterministic algorithms are requested; cross-platform bitwise reproducibility is not promised. This is one preliminary seed, not a stochastic comparison.

## Outputs

Every launch creates a unique `results/mixed_ssim_30k_<UTC>` directory: parameters, dependency/device/git metadata, source snapshot and tracked diff, input PNG hashes and ordered quadruples, per-epoch shuffle indices, per-batch plane loss JSONL, validation plane losses, model/optimizer/RNG checkpoints after epoch 1 and every 5 epochs, and final validation/test export metrics. Final evaluation rereads quantized DOE masks and reconstruction PNGs, retaining per-plane metrics and undefined Pearson statuses. Training, validation and final export are separate phases; logged elapsed time is wall time since training began, not a controlled inference benchmark.

Watch the launch log and `status.json`; an uncaught exception is recorded in the launch log and means the run did not complete even if the last status was `training` or `preparing`. No existing result directory is overwritten.

## Checks

```bash
python3 -m experiments.experiment_unet_physics_adapter.verify
python3 -m experiments.experiment_unet_physics_adapter.verify_structural
```

Before launch, the existing adapter verification passed. Structural verification found SSIM reference error 3.82e-17 (tolerance 1e-10), directional derivative absolute error 4.95e-13 at step 1e-4 (tolerance 1e-7), and checkpoint/direct MPS gradient maximum difference 0 (tolerance 1e-5), with finite gradients. These tolerances check floating-point formula consistency, not physical accuracy or image quality. Full-batch feasibility and final training quality are not established by these small checks.
