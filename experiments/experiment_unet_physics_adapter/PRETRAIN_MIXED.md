# Mixed-dataset SSIM-informed pretraining

Run from the repository root:

```bash
python3 -m pip install torch==2.10.0 torchvision==0.25.0 numpy Pillow
python3 -m experiments.experiment_unet_physics_adapter.pretrain_mixed
```

Requires Apple MPS and Python 3.12; no device or batch-size fallback. The requested microbatch is 256 quadruples (1024 images), also one optimizer batch; the final batch has 120 quadruples. Activation checkpointing recomputes individual convolution blocks, each latent GS+Muon solve, and the outer loss without truncating higher-order gradients. Memory feasibility at the full batch must be checked on the actual machine; the implementation does not disable MPS memory limits.

## Experiment

Hypothesis: adding structural supervision improves exported SSIM relative to the existing MNIST-only MSE trainer. This run changes both data and loss, so it is not a controlled loss ablation and cannot alone establish an advantage. No published method is claimed reproduced. The architecture and inner GS+Muon solver remain unchanged, including the MSE inner objective and full differentiation. The outer objective is the documented single-FFT maximum-normalized MSE plus `0.5*(1-SSIM)`, averaged over samples and planes; no phase regularizer, SSIM ramp, or quantization surrogate. This is not the separate finite-pixel single-sample experiment's five-term objective.

SSIM uses the existing Gaussian 11x11 valid-window implementation, sigma 1.5, C1=0.01², C2=0.03², population covariance, data range 1. Maximum normalization uses epsilon 1e-12, with gradients through the maximum. Physical geometry, FFT normalization and exported PNG evaluation are unchanged from `docs/doe_physics.md`.

## Data and parameters

27,000 samples mean 108,000 individual training images: 36,000 each from the official training sets of CIFAR-10, Fashion-MNIST and MNIST. Validation contains 99 quadruples (132 images per source), drawn disjointly from official training sets; test contains 99 quadruples from official test sets. Source indices are selected with NumPy PCG64 seed 2026. Prepared-pixel hashes exclude exact duplicates across all splits and fully black images; exclusions consume the next shuffled index. Independent images are shuffled into fixed quadruples, so an individual quadruple need not contain every dataset.

Torchvision downloads/checks official archives into `experiments/experiment_unet_physics_adapter/datasets`. Pillow converts images to L grayscale and resizes with BILINEAR to 128x128 before saving uint8 PNG. This explicit preprocessing is applied only to the source datasets; training reads those PNGs as float32 intensity/255 with shape [B,4,128,128]. Labels do not enter training. Data are read per microbatch, not held as a full float32 dataset in RAM.

3 epochs, seed 17 initialization, float32 MPS, one CPU thread, width 8, adapter rank 2, alpha 2, two inner solver steps. Outer Muon learning rate 0.001, momentum 0.95; inner phase Muon learning rate 0.1. Remaining fixed Muon settings and all actual parameters are in `parameters.json`. Epoch shuffle uses a separate Torch generator seeded 2026. Deterministic algorithms are requested; cross-platform bitwise reproducibility is not promised. This is one preliminary seed, not a stochastic comparison.

## Runtime budget revision

The previous 30,000-quadruple, 50-epoch run was stopped during epoch 1 without a checkpoint. This variant restarts from seed 17 with 27,000 quadruples and 3 epochs; architecture, batch size, loss and solver are unchanged. At the measured 63.34 seconds per full batch, 106 batches per epoch give approximately 5 hours 36 minutes for training. Preparation, validation, checkpointing and final export add time; this is an estimate, not a hard six-hour deadline. Changing split size regenerates the seeded source selection and quadruples, including validation/test membership; use each run's recorded manifest rather than assuming identical evaluation examples.

## Outputs

Every launch creates a unique `results/mixed_ssim_27k_3ep_<UTC>` directory: parameters, dependency/device/git metadata, source snapshot and tracked diff, input PNG hashes and ordered quadruples, per-epoch shuffle indices, per-batch plane loss JSONL, validation plane losses, model/optimizer/RNG checkpoints after epoch 1, every 5 epochs, and the final epoch, and final validation/test export metrics. Final evaluation rereads quantized DOE masks and reconstruction PNGs, retaining per-plane metrics and undefined Pearson statuses. Training, validation and final export are separate phases; logged elapsed time is wall time since training began, not a controlled inference benchmark.

Watch the launch log and `status.json`; an uncaught exception is recorded in the launch log and means the run did not complete even if the last status was `training` or `preparing`. No existing result directory is overwritten.

## Checks

```bash
python3 -m experiments.experiment_unet_physics_adapter.verify
python3 -m experiments.experiment_unet_physics_adapter.verify_structural
```

Before launch, the existing adapter verification passed. Structural verification found SSIM reference error 3.82e-17 (tolerance 1e-10), directional derivative absolute error 4.95e-13 at step 1e-4 (tolerance 1e-7), and checkpoint/direct MPS gradient maximum difference 0 (tolerance 1e-5), with finite gradients. These tolerances check floating-point formula consistency, not physical accuracy or image quality. Full-batch feasibility and final training quality are not established by these small checks. The first launch with whole-model checkpointing exhausted MPS memory before its first optimizer step (20.00 GiB allocated); that failed run is preserved. The runner now uses finer-grained checkpointing, verified against direct MPS gradients with maximum absolute difference zero on one sample; the requested batch size remains unchanged.
