2026 12th International Conference on Control, Decision and Information Technologies (CoDIT 2026)

Bari, Italy - July 13-16, 2026
CoDIT 2026 - Technical Co-Sponsors: IEEE CSS, IEEE SMC & IEEE RAS

# On the Effectiveness of Physics-Informed Fourier Neural Operators for Holographic Phase Retrieval

Mohammad Saim Khan, Stéphane Cuenat, Rémi Kieber, Maxime Jacquot and Raphaël Couturier
Université Marie et Louis Pasteur, CNRS, Institut FEMTO-ST

Besançon, France

Emails: {mohammad_saim.khan, stephane.cuenat, remi.kieber, maxime.jacquot, raphael.couturier}@umlp.fr

**Abstract**—Digital holography enables marker-free, quantitative phase imaging of transparent biological specimens, yet retrieving phase from intensity-only measurements remains fundamentally ill-posed. We investigate the effectiveness of physics-informed Fourier Neural Operators (FNOs) for holographic phase retrieval through PhysSpec-FNO, a self-supervised spectral boosting framework that derives its entire training signal from the Angular Spectrum Method (ASM) propagator—requiring no paired ground-truth data. The architecture employs a two-stage Generator-Refiner pipeline: the Generator captures dominant low-frequency wavefield structure, while the Refiner corrects high-frequency residuals via concatenated input conditioning. We systematically evaluate PhysSpec-FNO against a supervised Hybrid SpecBoost baseline across four domain configurations, using synthetically generated Random Noise and MS-COCO datasets (100,000 images at 512×512 resolution each). In matched-domain settings, PhysSpec-FNO achieves a peak Structural Similarity Index (SSIM) of 0.846—a 33% relative improvement over the supervised baseline (0.636)—while the supervised method demonstrates superior cross-domain robustness. Validation on real lung tissue specimens further confirms the practical advantage of the physics-informed approach (SSIM 0.509 vs. 0.484). These results illuminate a fundamental specialization-generalization trade-off inherent to physics-informed learning: FNOs constrained by physical propagation laws excel when training and deployment domains are aligned, but sacrifice distributional flexibility. Our findings position physics-informed FNOs as an effective solution for dedicated biological microscopy pipelines where ground-truth annotations are unavailable and domain alignment can be assumed.

**Index Terms**—holographic phase retrieval, Fourier Neural Operator, self-supervised learning, spectral boosting, physics-informed deep learning, digital holography, biomedical imaging

## I. INTRODUCTION

Digital Holography (DH) is a powerful imaging modality that captures both the amplitude and phase of a light wave scattered by a specimen. Unlike conventional microscopy, which records only intensity, imaging in digital holographic microscopy (DHM) provides quantitative phase maps that encode morphological and refractive index information about transparent biological samples—such as live cells, tissue sections, and pathological specimens—without the need for fluorescent labeling or staining [1], [2]. This label-free, non-invasive capability has made DHM increasingly important in biomedical applications including cell counting, disease diagnosis, and real-time monitoring of cellular dynamics.

However, practical holographic sensors record only intensity measurements, discarding the phase component of the complex optical wavefield. Recovering this lost phase—the so-called *phase retrieval problem*—is inherently ill-posed, as multiple complex fields can produce identical intensity patterns [3]. Traditional iterative algorithms, such as the Gerchberg–Saxton algorithm and its variants, rely on alternating projections between spatial and Fourier domains but suffer from slow convergence, stagnation at local minima, and sensitivity to noise [4].

Recent advances in deep learning have produced diverse strategies for the phase retrieval problem. Supervised methods [11] train convolutional networks on paired hologram/phase data, while untrained physics-driven techniques such as the deep image prior [12] optimize a randomly initialized network on a single hologram, exploiting implicit regularization rather than a learned prior. Physics-informed neural operators [9] embed governing wave equations into the training objective, and broader surveys of deep learning for computational imaging [13] catalogue further variants. Among these, the GedankenNet framework [5] stands out by training a single Fourier Neural Operator (FNO) [6] entirely without ground-truth data, leveraging the Angular Spectrum Method (ASM) as a differentiable physics propagator that maps a complex wavefield from the sample plane to the sensor plane via the FFT. The off-axis DHM geometry underlying this propagation model—wherein object and reference beams interfere at a prescribed angle to encode the complex wavefield in a single interferogram—is described in detail in [7]. Independently, Xiao et al. [8] proposed SpecBoost, an ensemble framework that sequentially trains multiple FNOs to mitigate the inherent low-frequency bias of these operators, originally designed for solving partial differential equations (PDEs). While SpecBoost improves prediction accuracy on PDE tasks by up to 71%, its potential for holographic reconstruction—and, more critically, for a self-supervised regime where no residual ground truth is available—has not previously been explored.

In this paper, we propose **PhysSpec-FNO**, a framework that brings the SpecBoost architecture into the domain of self-supervised holographic phase retrieval. Our key contributions are:

1) We adapt the SpecBoost ensemble architecture [8]—originally designed for supervised PDE solving—to

979-8-3195-2077-7/26/$31.00 ©2026 IEEE

2199

Authorized licensed use limited to: National Research Nuclear University - MEPhi. Downloaded on November 06, 2026 at 18:40:50 UTC from IEEE Xplore. Restrictions apply.

2026 12th International Conference on Control, Decision and Information Technologies (CoDIT) | 979-8-3195-2077-7/26/$31.00 ©2026 IEEE | DOI: 10.1109/CoDIT.70676.2026.11631208