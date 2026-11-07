2025 IEEE International Conference on Multimedia and Expo (ICME) | 979-8-3315-9495-4/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICME59968.2025.11210165

# Unfolding Framework with Complex-Valued Deformable Attention for High-Quality Computer-Generated Hologram Generation

Haomiao Zhang \( ^{1,2} \)  Zhangyuan Li \( ^{1,2} \)  Yanling Piao \( ^{3} \)  Zhi Li \( ^{4,5} \) 
Xiaodong Wang \( ^{1,2} \)  Miao Cao \( ^{1,2} \)  Xiongfei Su \( ^{1,2} \)  Qiang Song \( ^{4} \)  Xin Yuan \( ^{2*} \) 

 \( ^{1} \) Zhejiang University, Hangzhou, Zhejiang 310027, China.

\( ^{2} \) School of Engineering, Westlake University, Hangzhou, Zhejiang 310030, China.

\( ^{3} \) Westlake Institute for Optoelectronics, Hangzhou, Zhejiang 311421, China.

\( ^{4} \) Greater Bay Area Institute for Innovation, Hunan University, Guangzhou 511300, China.

\( ^{5} \) School of Physics, Sun Yat-sen University, Guangzhou 510275, China.

xyuan@westlake.edu.cn

Abstract—Computer-generated holography (CGH) has gained wide attention with deep learning-based algorithms. However, due to its nonlinear and ill-posed nature, challenges remain in achieving accurate and stable reconstruction. Specifically, (i) the widely used end-to-end networks treat the reconstruction model as a black box, ignoring underlying physical relationships, which reduces interpretability and flexibility. (ii) CNN-based CGH algorithms have limited receptive fields, hindering their ability to capture long-range dependencies and global context. (iii) Angular spectrum method (ASM)-based models are constrained to finite near-fields. In this paper, we propose a Deep Unfolding Network (DUN) that decomposes gradient descent into two modules: an adaptive bandwidth-preserving model (ABPM) and a phase-domain complex-valued denoiser (PCD), providing more flexibility. ABPM allows for wider working distances compared to ASM-based methods. At the same time, PCD leverages its complex-valued deformable self-attention module to capture global features and enhance performance, achieving a PSNR over 35 dB. Experiments on simulated and real data show state-of-the-art results. Code is available at https://github.com/HannahZhang1926/Complex-Valued-Deformable-Transformer-for-CGH.

Index Terms—Computer-generated holography, Convolutional neural network, Transformer, Deep unfolding network

## I. INTRODUCTION

Computer-generated holography (CGH) is a prominent display technology, which has attracted wide interest. Traditional CGH algorithms [1]–[3] suffer from poor performance and long reconstruction time. Deep learning methods, especially end-to-end (E2E) networks, often surpass traditional model-based approaches in both efficiency and effectiveness, significantly enhancing CGH quality while reducing computational complexity [4], [5]. However, E2E algorithms lack flexibility. For example, if the wavelength or propagation distance (optical parameters) change, a new network has to be trained. In contrast, deep unfolding networks (DUNs) offer strong interpretability and flexibility, while achieving strong performance in vision tasks [6]. Additionally, CNN-based algorithms have limited receptive fields, losing detailed information due to

![img-0.jpeg](img-0.jpeg)

Fig. 1. (a) Schematic of CGH model. (b) Experiment setup of our CGH system (BS: beam splitter, CCD: charge-coupled device, P: polarizer). (c) Comparison between the reconstruction results of several representative CGH algorithms in 256×256 resolution.

challenges in capturing long-range relationships. In contrast, Transformer models excel at capturing long-range, non-local relationships, improving reconstruction quality by preserving more global and detailed information [7], [8].

Meanwhile, in diffractive optics, angular spectrum method (ASM)-based algorithms are limited by the transfer function [9] and require equal-sized object and image planes. Previous research [10] showed that applying ASM beyond its working distances leads to significant image distortion, such as spectrum aliasing. Efforts have been made to ensure accurate propagation across varying distances, including band-limited angular spectrum method [10], non-uniform sampling [11], adaptive sampling [12], etc.

In summary, the motivations behind our proposed methods are: (i) E2E-CNN algorithms suffer from a lack of flexibility and interpretability and are limited by the respective field [4], [5]. (ii) Global attentional mechanisms in existing Complex-valued Transformers have higher memory consumption, making them inefficient for CGH reconstruction [13]–[15]. (iii) ASM model has a limited propagation distance [9]. Bearing these concerns in mind, we propose a novel approach for high-quality CGH generation. Our main contributions are summarized as follows.

- We propose a DUN framework incorporating global im-

Authorized licensed use limited to: National Research Nuclear University - MEPhi. Downloaded on November 06,2026 at 18:16:46 UTC from IEEE Xplore. Restrictions apply.

![img-1.jpeg](img-1.jpeg)

Fig. 2. The overall framework of our method, containing a deep-unfolding structure with N stages; GD and PCD blocks represent the operation in Eq.(4) and Eq.(5) respectively.

age features and physical constraints in CGH reconstruction, which consists of a gradient descent (GD) module and a deep denoising module to enhance both flexibility and interpretability.

- In the GD module, we replace the widely-used ASM model with an adaptive bandwidth-preserving method (ABPM) based on the sampling theory, enabling a wider working distance.
- In the denoising module, we introduce a Phase-domain Complex-valued Denoiser (PCD) with a Complex-valued Deformable Self-Attention (CDSA) mechanism, leveraging spatial self-attention to achieve more efficient and high-quality reconstruction.
- We built a holographic display system using a phase-only spatial light modulator (SLM) to demonstrate the wide working distance and high performance of our method.

## II. RELATED WORK

Model-based Algorithms for CGH: Model-based CGH algorithms can be broadly classified into iterative and non-iterative methods. Iterative techniques include the Gerchberg-Saxton (GS) method [1], Wirtinger holography [3], and gradient descent [2]. Non-iterative approaches contain point-based [16], double phase-amplitude coding [17] methods et al. Although these techniques are robust and training-free, their limited performance hinders their application in high-quality holographic displays.

Deep Learning Algorithms for CGH: Deep learning-based methods for CGH can be categorized into data-driven and physics-driven models. Data-driven CGHs rely on labeled data to learn the mapping between input scenes and output holograms, but they do not fully incorporate the physical model, limiting their performance [18]–[21]. In contrast, model-based E2E-CNN algorithms [4], [5], [22]–[25] integrate physical diffraction directly into the network, achieving high-fidelity large-scale reconstruction but lacking flexibility and struggling with long-term dependencies, leading to reconstructions below 35 dB. Furthermore, the existing Transformer [26] for CGH reconstruction remains under-explored and suboptimal. In contrast, physics-driven DUNs offer strong performance, clear interpretability, and outperform E2E algorithms in many tasks. [27], [28].

### III. PROBLEM STATEMENT

As depicted in Fig. 1(a), the SLM displays a phase-only hologram \( U_0(x_0, y_0, 0) = \exp(\mathrm{i}\phi(x_0, y_0, 0)) \). After propagating a distance \( z \), the complex amplitude \( U_z(x_z, y_z, z) \) is recorded on the camera plane, containing the information from

\(U_{0}\). This free propagation process can be represented in the matrix form: \(U_{z}(x_{z},y_{z},z)=\mathbf{A}U_{0}(x_{0},y_{0},0)\), where matrix \(\mathbf{A}\) models the discrete propagation from the SLM plane \(z=0\) to the camera plane \(z=d\). Due to CMOS camera limitations, only the intensity information can be captured. Specifically, the captured digital images are closer to the amplitude \(|U_{z}(x_{z},y_{z},z)|\) than to the squared intensity \(|U_{z}(x_{z},y_{z},z)|^{2}\), as noted in [23], [29]. Suppose \(\boldsymbol{y}=U_{z}(x_{z},y_{z},z)\) and \(\boldsymbol{x}=U_{z}(x_{z},y_{z},z)\), and the forward model is formulated as \(\boldsymbol{y}=|\boldsymbol{\Phi}\boldsymbol{x}|\), where \(\boldsymbol{x}\in\mathbb{C}^{W\times H}\), \(\boldsymbol{y}\in\mathbb{R}^{W\times H}\). \(\boldsymbol{\Phi}\in\mathbb{C}^{W\times H}\) denotes the propagation matrix, which corresponds to the ABPM model introduced in our article.

## IV. PROPOSED METHOD

### A. Overall Deep Unfolding Structure

The general structure of our DUN framework is shown in Fig.2. To solve the aforementioned ill-posed equation, a regularizer is introduced to constrain the solution space. This regularization term \( R(\pmb{x}) \) is used to find an estimate \( \hat{\pmb{x}} \) of \( \pmb{x} \) by solving the following problem:

\[
\hat {\boldsymbol {x}} = \arg \min _ {\boldsymbol {x}} \frac {1}{2} \| \boldsymbol {y} - | \boldsymbol {\Phi} \boldsymbol {x} | \| _ {2} ^ {2} + \lambda R (\boldsymbol {x}), \tag {1}
\]

\(\lambda\) balances the fidelity term \(\frac{1}{2}\| \pmb {y} - |\Phi \pmb {x}||_2^2\) and the regularization term. Since the regularization term may be not differentiable or even unknown, it is hard to find a closed-form solution directly. Alternatively, the Half Quadratic Splitting (HQS) method is adopted, which introduces an auxiliary variable \(\pmb {v}\in \mathbb{C}^{W\times H}\) and turns the unconstrained optimization into an augmented Lagrangian function:

\[
(\hat {\boldsymbol {x}}, \hat {\boldsymbol {v}}) = \underset {\boldsymbol {x}, \boldsymbol {v}} {\arg \min} \frac {1}{2} \| \boldsymbol {y} - | \boldsymbol {\Phi} \boldsymbol {v} | \| _ {2} ^ {2} + \lambda R (\boldsymbol {x}) + \frac {\eta}{2} \| \boldsymbol {v} - \boldsymbol {x} \| _ {2} ^ {2}. \tag {2}
\]

Previous research [30] indicated that optimization-based algorithms include : i) Gradient descent and ii) Projection to the signal domain. The first step refines the current estimate by incorporating additional information from the input. The second step constrains the result to the target signal space, typically using a denoiser. Thus Eq.(2) can be solved by decoupling x and v into two iterative sub-problems: woy

\[
\boldsymbol {v} ^ {(k + 1)} = \underset {\boldsymbol {v}} {\arg \min} \left\| \boldsymbol {y} - | \boldsymbol {\Phi} \boldsymbol {v} | \right\| _ {2} ^ {2} + \eta \left\| \boldsymbol {v} - \boldsymbol {x} ^ {(k)} \right\| _ {2} ^ {2}, \tag {3a}
\]

\[
\boldsymbol {x} ^ {(k + 1)} = \underset {\boldsymbol {x}} {\arg \min} \frac {1}{2} \| \boldsymbol {x} - \boldsymbol {v} ^ {(k + 1)} \| _ {2} ^ {2} + \lambda R (\boldsymbol {x}). \tag {3b}
\]

For the \(v\)-subproblem, Eq. (3a) has a closed-form solution [29]:

\[
\begin{array}{l} \boldsymbol {v} ^ {(k + 1)} = \boldsymbol {x} ^ {(k)} - \rho \nabla_ {\boldsymbol {x}} F (\boldsymbol {x} ^ {(k)}), \\ = \boldsymbol {x} ^ {(k)} - \boldsymbol {\Phi} ^ {H} \operatorname{diag} \left(\frac {\boldsymbol {\Phi} \boldsymbol {x} ^ {(k)}}{| \boldsymbol {\Phi} \boldsymbol {x} ^ {(k)} |}\right) (\boldsymbol {y} - | \boldsymbol {\Phi} \boldsymbol {x} ^ {(k)} |), \tag {4} \\ \end{array}
\]

Authorized licensed use limited to: National Research Nuclear University - MEPhi. Downloaded on November 06,2026 at 18:16:46 UTC from IEEE Xplore. Restrictions apply.

![img-2.jpeg](img-2.jpeg)

Fig. 3. (a) The diagram of CDAT block. (b) Complex-valued Deformable Self-Attention (CDSA) module. (c) An illustration of the difference between Hermitian inner production and dot production.

where $\nabla_{\boldsymbol{x}}F(\boldsymbol{x}^{(k)})$ denotes the gradient of $F(\boldsymbol{x}) = |\Phi \boldsymbol{x}|$ at the point $\boldsymbol{x} = \boldsymbol{x}^{(k)}$. $\nabla_{\boldsymbol{x}}F(\boldsymbol{x}^{(k)})$ is derived as $\nabla_{\boldsymbol{x}}F(\boldsymbol{x}) = \frac{1}{2}\Phi^{H}\mathrm{diag}\left(\frac{\Phi\boldsymbol{x}}{|\Phi\boldsymbol{x}|}\right)(\boldsymbol{y} - |\Phi\boldsymbol{x}|)$, where $(\cdot)^H$ denotes the conjugate transpose operator. Notice that $\frac{1}{2}$ is absorbed into the constant $\rho$, and $\rho$ is set to 1.

For the $\boldsymbol{x}$-subproblem, Eq. (3b) aims to find $\boldsymbol{x}^{(k+1)}$ that is as close as possible to the updated $\boldsymbol{v}^{(k+1)}$ while maintaining consistency with previous iteration. Since modeling a proper prior for complex amplitude fields is nontrivial, we implement the mapping from $\boldsymbol{v}^{(k+1)}$ to $\boldsymbol{x}^{(k+1)}$ using a deep neural network. This module is referred to as the Phase-domain Complex-valued Denoiser (PCD):

$$\boldsymbol{x}^{(k+1)} = \mathrm{PCD}(\boldsymbol{v}^{(k+1)}). \tag{5}$$

Eq. (4) and Eq. (5) form one iteration (or stage) to solve Eq.(1). In our DUN framework, this iteration is applied a few times, allowing explicit unfolding and independent network parameters for PCDs at each stage. Learnable PCDs, derived from real data, are more efficient and accurate than traditional hand-crafted priors. By combining the optimization algorithm with the denoising network, our method is both effective and interpretable. Next, we provide a detailed explanation of the propagation matrix $\Phi$ and PCD blocks.

### B. Adaptive Bandwidth-Preserving Method

To ensure diffraction calculations are performed accurately at all working distances, we rewrite the ASM propagation model as follows:

$$U_z(x_z, y_z, z) = \begin{cases} \mathcal{F}^{-1}\left\{H(f_x, f_y, z)\mathcal{F}\left\{U_0(x_0, y_0, 0)\right\}\right\}, & z \le z_1. \\ \mathcal{F}^{-1}\left\{\mathcal{F}\left\{h(x_z, y_z, z)\right\}\mathcal{F}\left\{U_0(x_0, y_0, 0)\right\}\right\}, & z_1 < z \le z_2. \\ \mathcal{F}^{-1}\left\{\mathcal{F}\left\{h'(x_z, y_z, z)\right\}\mathcal{F}\left\{U_0(x_0, y_0, 0)\right\}\right\}, & z > z_2. \end{cases} \tag{6}$$

where $z_1 = N\delta x / \lambda \sqrt{(\delta x)^2 - (\lambda/2)^2}$ is the ASM threshold, $z_2 = (25N^4(\delta x)^4/\lambda)^{\frac{1}{3}}$ is the far field threshold, and $h(x_z, y_z, z)$ and $h'(x_z, y_z, z)$ are the impulse response at various propagation distance. We name this propagation model the adaptive bandwidth-preserving method (ABPM). In far-field propagation, common models such as Fresnel or Fraunhofer increase the image plane size, requiring target image prescaling, which reduces fidelity. The band-limited ASM extends propagation slightly further but may lose high-frequency information. The two-step Fresnel model, suitable only for

long distances, may lack accuracy. Our ABPM model ensures rigorous sampling across general cases, with the image plane's pixel size matching that of the SLM, avoiding scaling issues. Details are provided in the Supplementary Material (SM).

### C. Phase-domain Complex-valued Denoiser

As shown in Fig. 2, the PCD module consists of three parts: i) Feature Extraction Module (FEM), ii) Complex-valued Deformable Attention Transformer (CDAT) blocks and iii) Phase Image Recovery Module (PIRM). The FEM and PIRM are explained in detail in the SM.

In Fig. 3(a), the CDAT module consists of a Complex-valued Deformable Self-Attention (CDSA) module and a Complex-valued Feed-Forward Network (CFFN), both using complex-valued layer normalization for stable training and residual shortcuts for improved efficiency. The CDSA module will be discussed next. The CFFN enhances CDAT's non-linearity and representation ability, featuring two complex-valued linear layers (please refer to the SM) with ReLU activation applied separately to the real and imaginary parts of the input.

### D. Complex-valued Deformable Self-Attention module

Previous works [14], [15] adopt the global scaled dot-product self-attention mechanism, which suffers from high time complexity, making it impractical for large-scale high-resolution CGH generation. The computational bottleneck occurs mainly at the softmax function, thus reducing the input size of the softmax function, i.e. the size of the attention matrix $\boldsymbol{Q}\boldsymbol{K}^T$, is crucial. In contrast, we employ a local, deformable self-attention mechanism [31] that adaptively selects the attention regions, which alleviates the computational burden concerning image size.

The structure of our proposed CDSA module is shown in Fig. 3(b). The CDSA module receives the feature map $\boldsymbol{f} \in \mathbb{C}^{H/4 \times W/4 \times C}$ from FEM (the FEM will down-sample the size of the input feature map to a quarter). For simplicity, we denote $\tilde{H} = H/4$ and $\tilde{W} = W/4$. To shrink the size of the attention matrix, the deformable attention mechanism chooses to down-sample $\boldsymbol{f}$ before calculating $\boldsymbol{K}$ and $\boldsymbol{V}$. The down-sampled points' coordinates are the sum of a fixed reference point and a Query-dependent offset. The reference points are evenly distributed across the original feature map, and their density determines the down-sample rate. In our case, the rate is 64 (8 on each side), reducing the attention matrix by a factor

Authorized licensed use limited to: National Research Nuclear University - MEPhi. Downloaded on November 06,2026 at 18:16:46 UTC from IEEE Xplore. Restrictions apply.

![img-3.jpeg](img-3.jpeg)

Fig. 4. Simulation:(a) Numerical reconstructions of CGH generated by different algorithms on four benchmark datasets at 1920×1080 resolution. (b) CGH reconstruction results of our algorithm at 8 cm, 20 cm, 40 cm, and 60 cm. Zoom in for a better view.

of 64. The offset is generated from Q. As shown in Fig. 3(b), a 1 × 1 convolution layer converts f into the Q ∈ ℂᴺᴼˣᶜ (N_Q = H̄W), and then a sub-network transforms the Q into coordinate offsets. To be specific, the sub-network consists of a depth-wise convolution layer, a layer normalization, a GELU activation function, and a 1 × 1 regular convolution layer. The kernel size, stride, and padding size of depth-wise convolution are set to 9 × 9, 8, and 4, which is consistent with the down-sample rate setting. After the last 1 × 1 convolution, the channel dimension of the feature map is reduced to 2, meaning offset value on the x and y axes respectively. Note that the bias in the 1 × 1 convolution is omitted to prevent uniform shifts across locations. With the coordinates of the reference points and their offsets, a down-sampled feature map can be interpolated from the original input f. After this, two 1 × 1 convolution layers convert this feature map into the K ∈ ℂᴺᴷˣᶜ and V ∈ ℂᴺᴼˣᶜ (N_K = N_V = H̄W/64) separately.

When calculating the similarity between Q and K, the dot product operation QKᵀ fails to capture the phase similarity of the complex-valued query-key pairs. To address this, we use the Hermitian inner product to measure their similarity. Given two complex vectors x, y ∈ ℂⁿ, the Hermitian inner product is defined as ⟨x, y⟩ = ∑ⱼ₌₁ⁿ xⱼȳⱼ = ∑ⱼ₌₁ⁿ |xⱼ||yⱼ|exp[i(φₓⱼ - φᵧⱼ)], while the dot product is xyᵀ = ∑ⱼ₌₁ⁿ xⱼyⱼ = ∑ⱼ₌₁ⁿ |xⱼ||yⱼ|exp[i(φₓⱼ + φᵧⱼ)]. Although the amplitudes of ⟨x, y⟩ and xyᵀ are identical, the phase angle of ⟨x, y⟩ can reflect the phase discrepancy between x and y, while xyᵀ cannot. This proves the rotation invariance of the Hermitian inner product. As depicted in Fig. 3(c), given two complex inputs x and y with fixed phase angle discrepancy, the results of the dot product will change when we rotate x and y synchronously while the results of Hermitian inner product remain the same.

The Hermitian inner product similarity ⟨Q, K⟩ ∈ ℂᴺᴼˣᴺᴷ is scaled by √C and combined with a relative position bias matrix to form a complex attention matrix. During softmax normalization, only the real part is used. Since the similarity dominates the attention matrix (due to initially small values in T), the real part of Hermitian inner product similarity, i.e., ∑ⱼ₌₁ⁿ |xⱼ||yⱼ|cos(φₓⱼ - φᵧⱼ), maximizes when cos(φₓⱼ - φᵧⱼ) = 1, equaling to φₓⱼ = φᵧⱼ. It strictly decreases while

|φₓⱼ - φᵧⱼ| grows from 0 to π and minimizes at φₓⱼ = -φᵧⱼ. Thus the real part of the complex attention matrix effectively captures the correlation between Q and K.

The overall CDSA mechanism can be written as:

$$\mathrm{CDSA}(\boldsymbol{f}) = \sigma \left( \mathcal{R} \left( \frac{\langle \boldsymbol{W}_Q \boldsymbol{f}, \boldsymbol{W}_K \phi(\boldsymbol{f}) \rangle}{\sqrt{C}} + \boldsymbol{B} \right) \right) \boldsymbol{W}_V \phi(\boldsymbol{f}), \tag{7}$$

where W_Q, W_K, and W_V refer to the 1 × 1 convolution layers and φ(·) is the deformable down-sample function, σ(·) denotes the softmax function, R refers to extracting the real part. In conclusion, the per-layer complexity of the CDSA mechanism is O(N_Q N_K C + N_Q N_V C), which is much lower than the complexity O(2N_Q² C) of scaled dot production self-attention since N_K, N_V ≪ N_Q in our setting.

## V. EXPERIMENTAL RESULTS

### A. Implementation Details

Our model is trained on the DIV2K super-resolution dataset [32], which contains 800 images, and evaluated on 100 randomly selected images from the DIV2K and Flickr2K [33] datasets. Adam optimizer is used with a learning rate of 1e-4. The model is implemented in PyTorch 2.0 and Python 3.9, trained for 30 epochs on an NVIDIA RTX 8000 GPU. Optical parameters are set to a pixel pitch of 8 μm, wavelength of 520 nm, and a maximum resolution of 1920×1080.

### B. Simulation Results

Our method is superior to previous CGH methods in terms of both performance and interpretability by integrating a physics-based DUN structure and achieves a better balance of accuracy and flexibility. To verify the performance of our model, we compare our method with the traditional GS algorithm [1], HoloNet [23] based on CNN with real values, and CCNN-CGH [25] based on CNN with complex values. All models are trained under identical conditions, including the same database, epoch number, and optical setup. The main focus of the comparison is to assess the quality of real-valued images generated from the recovered phase distribution. Peak-Signal-to-Noise-Ratio (PSNR) and Structured Similarity Index Metrics (SSIM) [34] are used to evaluate the results.

Authorized licensed use limited to: National Research Nuclear University - MEPhi. Downloaded on November 06,2026 at 18:16:46 UTC from IEEE Xplore. Restrictions apply.

![img-4.jpeg](img-4.jpeg)

Fig. 5. Real data results: Reconstructed results of various algorithms at different propagation distances with ASM threshold. (b) Reconstructed results of our algorithm beyond the ASM threshold. Zoom in for a better view.

We compare the performance of various algorithms in the near field. According to Eq. (6), the threshold of ASM is 23.62 cm. Therefore, we set the diffraction distance at 20 cm. Tab. I shows a comparison of models trained with various methods at a resolution of  \( 1920 \times 1080 \) . The average PSNR and SSIM values were calculated from 100 validation samples using numerical reconstruction. Our method outperforms others, achieving a PSNR improvement of 4.44-15.19 dB and an SSIM increase of 0.03 to 0.4. Moreover, it requires fewer parameters and lower computational complexity (FLOPs), demonstrating superior performance in CGH tasks. Fig. 4(a) shows the simulated results of two chosen benchmark images from the test data set (Istanbul and Car). The results of the GS algorithm are excluded due to its unsatisfactory performance. Our method produces reconstruction results closer to the ground truth, recovering more detail and exhibiting less noise compared with previous algorithms.

TABLE I PERFORMANCE AT 1920×1080 RESOLUTION CGH GENERATION.

|  Method | Params | FLOPs(G) | Time(ms) | Avg.PSNR | Avg.SSIM  |
| --- | --- | --- | --- | --- | --- |
|  GS | - | - | 10640 | 21.26 | 0.55  |
|  HoloNet | 2,868,754 | 328.53 | 60 | 29.69 | 0.90  |
|  CCNN-CGH | 42,260 | 7.21 | 16 | 32.01 | 0.92  |
|  Our Method | 19,430 | 2.63 | 140 | 36.45 | 0.95  |

We also performed a comprehensive evaluation of our proposed ABPM method at different propagation distances from 8 to 60 cm in Fig.4(b). The results demonstrate the high quality of reconstruction attributed to the inherent effectiveness of the ABPM model, both in the near- and far-fields.

### C. Real Data Results

Our optical system is illustrated in Fig. 1 (b) and more details can be found in SM. The first experiment is designed to test the reconstructed quality under the ASM threshold, with a specific propagation distance of 15 to 23 cm. Due to the poor performance of the GS algorithm, its results are omitted. Real data results are shown in Fig. 5(a), HoloNet produces blurry edges, while CCNN-CGH over-smooths some areas. In contrast, our method yields sharper edges, finer details, and higher resolution, closely matching the actual values,

demonstrating superior reconstruction performance. Fig. 5(b) further validates our algorithm's performance across distances from \(10\mathrm{cm}\) to \(60~\mathrm{cm}\) for two scenes, showing its effectiveness in both near- and far-fields.

## VI. ABLATION STUDY

### A. The Denoising Block

TABLE II ABLATION STUDY ON THE DENOISER BLOCK.

|  Denoiser Type | Resolution | Params | FLOPs(G) | Avg.PSNR(dB) | Avg.SSIM  |
| --- | --- | --- | --- | --- | --- |
|  w/o | 1920×1080 | - | - | 28.56 | 0.79  |
|  Complex TV | 1920×1080 | - | - | 29.54 | 0.85  |
|  Complex CNN | 1920×1080 | 22,368 | 4.46 | 32.12 | 0.90  |
|  CVTF | 256×256 | 114,150 | 1.12 | 29.09 | 0.88  |
|  CDAT | 1920×1080 | 19,430 | 2.63 | 36.45 | 0.95  |

To validate the effectiveness of our algorithms, we conduct an ablation study on the denoising module, comparing our method with direct removal, model-based complex total variation (TV) algorithm  \( [29] \) , learning-based complex CNN algorithm and the existing Complex-Valued Transformer (denoted as CVTF)  \( [15] \) . Due to high computational cost, the resolution in the CVTF comparison is limited to  \( 256 \times 256 \) . As shown in Tab. II, our algorithm achieves an average PSNR improvement of 4.33-7.89 dB and SSIM enhancement of 0.05-0.16, outperforming other methods and demonstrating that the CDAT in our denoising module enables high-quality reconstruction results.

TABLE III ABLATION STUDY ON THE EMBEDDING CHANNELS.

|  Embedding Channels | Params(M) | Flops(G) | Avg.PSNR(dB)  |
| --- | --- | --- | --- |
|  96 | 87,473 | 26.8 | 36.87  |
|  32 | 19,430 | 2.6 | 36.45  |

### B. Embedding Channel

In CGH applications, maintaining a lightweight design is essential. As discussed in Sec. IV-D, the number of embedding channels C is closely related to the computational complexity. As shown in Tab. III, when the embedding dimension is set to 32, the reconstruction achieves excellent results while keeping the computational cost relatively low.

Authorized licensed use limited to: National Research Nuclear University - MEPhi. Downloaded on November 06,2026 at 18:16:46 UTC from IEEE Xplore. Restrictions apply.

## VII. CONCLUSION

To achieve high-quality CGH display, previous CNN-based methods are limited in propagation distance and performance, while fixed global self-attention in complex-valued Transformers is impractical. Towards this end, this paper introduces a deep unfolding network with a Complex-valued Deformable Self Attention mechanism on an Adaptive Bandwidth-Preserving Method. Simulation and optical experiments show that our method significantly outperforms previous approaches in both near- and far-fields. This work paves the way for high-quality CGH generation with potential applications in optical information storage, tomography, and display technology.

## ACKNOWLEDGMENT

This work was supported by National Key R&D Program of China (2024YFF0505603), the National Natural Science Foundation of China (grant number 62271414), Zhejiang Provincial Distinguished Young Scientist Foundation (grant number LR23F010001), Zhejiang “Pioneer” and “Leading Goose” R&D Program (grant number 2024SDXHDX0006, 2024C03182), the Key Project of Westlake Institute for Optoelectronics (grant number 2023GD007), the 2023 International Sci-tech Cooperation Projects under the purview of the “Innovation Yongjiang 2035” Key R&D Program (grant number 2024Z126).

## REFERENCES

[1] R. W. Gerchberg, “A practical algorithm for the determination of plane from image and diffraction pictures,” Optik, vol. 35, no. 2, pp. 237–246, 1972.
[2] Y. Chen, Y. Chi, J. Fan, and C. Ma, “Gradient descent with random initialization: Fast global convergence for nonconvex phase retrieval,” Mathematical Programming, vol. 176, pp. 5–37, 2019.
[3] E. J. Candès, X. Li, and M. Soltanolkotabi, “Phase retrieval via wirtinger flow: Theory and algorithms,” IEEE Transactions on Information Theory, vol. 61, no. 4, pp. 1985–2007, 2015.
[4] X. Shui, H. Zheng, X. Xia, F. Yang, W. Wang, and Y. Yu, “Diffraction model-informed neural network for unsupervised layer-based computer-generated holography,” Optics Express, vol. 30, no. 25, pp. 44814–44826, 2022.
[5] K. Liu, J. Wu, Z. He, and L. Cao, “4k-dmdnet: diffraction model-driven network for 4k computer-generated holography,” Opto-Electronic Advances, pp. 220135–1, 2023.
[6] K. Gregor and Y. LeCun, “Learning fast approximations of sparse coding,” in Proceedings of the 27th international conference on international conference on machine learning, 2010, pp. 399–406.
[7] C. Li, C.-L. Guo, M. Zhou, Z. Liang, S. Zhou, R. Feng, and C. C. Loy, “Embedding fourier for ultra-high-definition low-light image enhancement,” arXiv preprint arXiv:2302.11831, 2023.
[8] M. Zhou, J. Huang, C.-L. Guo, and C. Li, “Fourmer: An efficient global modeling paradigm for image restoration,” International conference on machine learning, pp. 42589–42601, 2023.
[9] J. W. Goodman, Introduction to Fourier optics. Roberts and Company publishers, 2005.
[10] K. Matsushima and T. Shimobaba, “Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields,” Optics express, vol. 17, no. 22, pp. 19662–19673, 2009.
[11] Y.-H. Kim, C.-W. Byun, H. Oh, J. Lee, J.-E. Pi, G. H. Kim, M.-L. Lee, H. Ryu, H.-Y. Chu, and C.-S. Hwang, “Non-uniform sampling and wide range angular spectrum method,” Journal of Optics, vol. 16, no. 12, p. 125710, 2014.
[12] W. Zhang, H. Zhang, and G. Jin, “Adaptive-sampling angular spectrum method with full utilization of space-bandwidth product,” Optics Letters, vol. 45, no. 16, pp. 4416–4419, 2020.

[13] Y. Dong, Y. Peng, M. Yang, S. Lu, and Q. Shi, “Signal transformer: Complex-valued attention and meta-learning for signal recognition,” arXiv preprint arXiv:2106.04392, 2021.
[14] F. Eilers and X. Jiang, “Building blocks for a complex-valued transformer architecture,” in ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023, pp. 1–5.
[15] M. Yang, M. Q. Ma, D. Li, Y.-H. H. Tsai, and R. Salakhutdinov, “Complex transformer: A framework for modeling complex-valued sequence,” in ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020, pp. 4232–4236.
[16] P. Tsang, T.-C. Poon, and Y. Wu, “Review of fast methods for point-based computer-generated holography,” Photonics Research, vol. 6, no. 9, pp. 837–846, 2018.
[17] A. Maimone, A. Georgiou, and J. S. Kollin, “Holographic near-eye displays for virtual and augmented reality,” ACM Transactions on Graphics (Tog), vol. 36, no. 4, pp. 1–16, 2017.
[18] S.-C. Liu and D. Chu, “Deep learning for hologram generation,” Optics Express, vol. 29, no. 17, pp. 27373–27395, 2021.
[19] L. Shi, B. Li, C. Kim, P. Kellhofer, and W. Matusik, “Towards real-time photorealistic 3d holography with deep neural networks,” Nature, vol. 591, no. 7849, pp. 234–239, 2021.
[20] L. Shi, B. Li, and W. Matusik, “End-to-end learning of 3d phase-only holograms for holographic display,” Light: Science & Applications, vol. 11, no. 1, p. 247, 2022.
[21] C. Chang, B. Dai, D. Zhu, J. Li, J. Xia, D. Zhang, L. Hou, and S. Zhuang, “From picture to 3d hologram: end-to-end learning of real-time 3d photorealistic hologram generation from 2d image input,” Optics Letters, vol. 48, no. 4, pp. 851–854, 2023.
[22] J. Wu, K. Liu, X. Sui, and L. Cao, “High-speed computer-generated holography using an autoencoder-based deep neural network,” Optics Letters, vol. 46, no. 12, pp. 2908–2911, 2021.
[23] Y. Peng, S. Choi, N. Padmanaban, and G. Wetzstein, “Neural holography with camera-in-the-loop training,” ACM Transactions on Graphics (TOG), vol. 39, no. 6, pp. 1–14, 2020.
[24] X. Chang, R. Zhao, S. Jiang, C. Shen, G. Zheng, C. Yang, and L. Bian, “Complex-domain-enhancing neural network for large-scale coherent imaging,” Advanced Photonics Nexus, vol. 2, no. 4, pp. 046006–046006, 2023.
[25] C. Zhong, X. Sang, B. Yan, H. Li, D. Chen, X. Qin, S. Chen, and X. Ye, “Real-time high-quality computer-generated hologram using complex-valued convolutional neural network,” IEEE Transactions on Visualization and Computer Graphics, 2023.
[26] Z. Dong, C. Xu, Y. Tang, Y. Ling, Y. Li, and Y. Su, “Vision transformer-based, high-fidelity, computer-generated holography,” in Advances in Display Technologies XIII, vol. 12443. SPIE, 2023, pp. 47–53.
[27] Z. Chen, S. Zheng, Z. Tong, and X. Yuan, “Physics-driven deep learning enables temporal compressive coherent diffraction imaging,” Optica, vol. 9, no. 6, pp. 677–680, 2022.
[28] G. Qu, P. Wang, and X. Yuan, “Dual-scale transformer for large-scale single-pixel imaging,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 25327–25337.
[29] Y. Gao and L. Cao, “Iterative projection meets sparsity regularization: towards practical single-shot quantitative phase imaging with in-line holography,” Light: Advanced Manufacturing, vol. 4, no. 1, pp. 1–17, 2023.
[30] C. A. Metzler, A. Maleki, and R. G. Baraniuk, “From denoising to compressed sensing,” IEEE Transactions on Information Theory, vol. 62, no. 9, pp. 5117–5144, 2016.
[31] Z. Xia, X. Pan, S. Song, L. E. Li, and G. Huang, “Vision transformer with deformable attention,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 4794–4803.
[32] E. Agustsson and R. Timofte, “Ntire 2017 challenge on single image super-resolution: Dataset and study,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2017, pp. 1122–1131.
[33] B. Lim, S. Son, H. Kim, S. Nah, and K. M. Lee, “Enhanced deep residual networks for single image super-resolution,” in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, July 2017.
[34] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.

Authorized licensed use limited to: National Research Nuclear University - MEPhi. Downloaded on November 06,2026 at 18:16:46 UTC from IEEE Xplore. Restrictions apply.