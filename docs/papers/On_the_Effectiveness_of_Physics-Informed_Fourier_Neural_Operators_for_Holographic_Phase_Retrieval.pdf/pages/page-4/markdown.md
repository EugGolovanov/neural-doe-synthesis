CoDIT 2026 - Technical Co-Sponsors: IEEE CSS, IEEE SMC & IEEE RAS

where $f_x, f_y$ are the spatial frequencies conjugate to $x, y$, $H(f_x, f_y; \Delta z) = \exp\left(j2\pi\sqrt{k^2 - f_x^2 - f_y^2} \cdot \Delta z\right)$ is the transfer function, $k = n/\lambda$ is the wavenumber ($\lambda = 530$ nm, $n = 1.0$ for air), and $\Delta z$ is the propagation distance. The simulated holograms are computed at axial distances $z \in \{300, 375\}$ $\mu$m with a pixel pitch of 0.3733 $\mu$m, considering 10X magnification provided by microscope objective lens in DHM geometry.

The total loss function combines three complementary objectives:

$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{MAE}}^{\text{spectral}} + \beta \mathcal{L}_{\text{MSE}}^{\text{spatial}} + \gamma \mathcal{L}_{\text{TV}}, \tag{5}$$

where the spectral loss operates on the windowed Fourier transform (using a 2D Hann window to suppress spectral leakage), the spatial loss enforces pixel-wise consistency, and the total variation (TV) regularizer promotes piece-wise smooth reconstructions. The weights are set to $\alpha = 0.1$, $\beta = 1.0$, and $\gamma = 20.0$.

### D. Two-Phase Training Strategy

To ensure stable convergence of the boosting pipeline, we employ a two-phase training schedule:

- Phase 1 (Epochs 0–1000): Only the Generator $G_a$ is trained while the Refiner $G_b$ remains frozen with random initialization. This ensures that $G_a$ first learns a reasonable initial estimate before the Refiner attempts to correct it.
- Phase 2 (Epochs 1000–2000): $G_a$ is frozen with its learned weights, and only $G_b$ is trained. During this phase, the Refiner receives gradient signals specifically targeting the errors left by the Generator, enabling it to specialize in residual correction.

Both phases use the Adam optimizer with an initial learning rate of 0.002 and a CosineAnnealingWarmRestarts scheduler with $T_0 = 2$ and multiplicative factor $T_{\text{mult}} = 2$. Gradient clipping with a maximum norm of 1.0 is applied to prevent training instability.

### IV. EXPERIMENTAL SETUP AND RESULTS

### A. Dataset and Preprocessing

Following the GedankenNet methodology [5], we use two synthetically generated datasets, each comprising 100,000 samples at $512 \times 512$ resolution: (1) Random Noise Images (100,000 samples) generated from random Gaussian noise modulated by random phase masks drawn uniformly in $[-\pi, \pi]$, with amplitude clipped to $[0, 1]$; and (2) COCO Images (100,000 samples) derived from MS-COCO 2017 [10] grayscale images randomly cropped to $512 \times 512$ and linearly mapped to phase values in $[-\pi, \pi]$. For both, the ASM propagator (Eq. 4) simulates holographic recordings at the specified axial distances. The data is split 80%/10%/10% for training/validation/testing.

### B. Implementation Details

For reproducibility: both $G_a$ and $G_b$ are FNO2d backbones with $w = 48$ channels, modes $m_1 = m_2 = 256$, 8 multi-scale blocks of scales $\{1, 1, 2, 2, 2, 4, 4, 4\}$ with 2 sublayers each, PReLU activations, and a three-level residual skip hierarchy ($\sim$54M total parameters). Training uses Adam (lr = 0.002, weight decay $10^{-4}$, grad-clip norm 1), batch size 2, 250 minibatches/epoch, and CosineAnnealingWarmRestarts ($T_0 = 2$, $T_{\text{mult}} = 2$ for PhysSpec-FNO; $T_0 = 50$ for Hybrid). Loss weights: $\alpha = 0.1$, $\beta = 1.0$, $\gamma = 20.0$, $\lambda = 0.5$. ASM: $\lambda_{\text{light}} = 530$ nm, $n = 1.0$, pixel pitch 0.3733 $\mu$m, $z \in \{300, 375\}$ $\mu$m.

### C. Baseline: Hybrid SpecBoost (Supervised)

We compare against a supervised baseline (Hybrid SpecBoost) using the same SpecBoost architecture [8] but trained with ground-truth supervision. The loss combines MSE supervision with a physics-consistency term ($\lambda = 0.5$):

$$\mathcal{L}_{\text{hybrid}} = \text{MSE}(\hat{y}_{\text{pred}}, y_{\text{GT}}) + \lambda \cdot \mathcal{L}_{\text{physics}}. \tag{6}$$

Both stages are trained sequentially for 2 epochs each, following the protocol of the original SpecBoost work [8]. This baseline requires ground-truth complex fields during training. On training-budget asymmetry. Each method follows the schedule of its source publication: the Hybrid baseline converges within a handful of supervised epochs due to the direct per-pixel MSE target, whereas GedankenNet's indirect ASM-mediated signal requires the longer 2,000-epoch schedule of [5]. Budgets are thus matched to the convergence behaviour of each loss rather than to wall-clock. We treat an explicitly compute-matched ablation as future work; the qualitative trends in Sections IV-C–IV-D reflect properties of the loss formulations and are expected to persist regardless.

### D. Quantitative Results

We report fidelity via the Structural Similarity Index (SSIM), which compares local luminance, contrast, and structural statistics; SSIM is computed independently on the reconstructed amplitude $|\hat{y}_{\text{final}}|$ and phase $\arg(\hat{y}_{\text{final}})$ after global-phase normalisation, then averaged. We evaluate three frameworks across four domain configurations (Table I): PhysSpec-FNO (ours), the original GedankenNet single-FNO baseline [5], and the supervised Hybrid SpecBoost. In matched-domain settings (Random→Random and COCO→COCO), PhysSpec-FNO substantially outperforms both baselines, achieving +26% over GedankenNet on Random→Random (0.846 vs. 0.669) and +5% on COCO→COCO (0.794 vs. 0.755). Adding the Refiner stage therefore contributes a measurable gain over the single-FNO predecessor. Under domain shift, however, the picture inverts: GedankenNet is the most robust of the three, particularly in COCO→Random where PhysSpec-FNO collapses to 0.194 and Hybrid to 0.374, while GedankenNet retains 0.606. This is consistent with the specialization–generalization trade-off discussed in Section V: the Refiner specialises $G_b$ to the training-domain spectral statistics, sacrificing transfer.

2202

Authorized licensed use limited to: National Research Nuclear University - MEPhi. Downloaded on November 06,2026 at 18:40:50 UTC from IEEE Xplore. Restrictions apply.