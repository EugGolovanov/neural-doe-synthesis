Check for updates

Research Article

Vol. 57, No. 14 / 10 May 2018 / Applied Optics

3859

# applied optics

# Deep-learning-generated holography

RYOICHI HORISAKI,1,2,* RYOSUKE TAKAGI,1 AND JUN TANIDA1

1Department of Information and Physical Sciences, Graduate School of Information Science and Technology, Osaka University, 1-5 Yamadaoka, Suita, Osaka 565-0871, Japan
2JST, PRESTO, 4-1-8 Honcho, Kawaguchi-shi, Saitama 332-0012, Japan
*Corresponding author: r.horisaki@ist.osaka-u.ac.jp

Received 12 February 2018; revised 12 April 2018; accepted 12 April 2018; posted 13 April 2018 (Doc. ID 323038); published 8 May 2018

We present a method for computer-generated holography based on deep learning. The inverse process of light propagation is regressed with a number of computationally generated speckle data sets. This method enables noniterative calculation of computer-generated holograms (CGHs). The proposed method was experimentally verified with a phase-only CGH. © 2018 Optical Society of America

OCIS codes: (090.1760) Computer holography; (090.1970) Diffractive optics; (110.1758) Computational imaging; (100.3190) Inverse problems; (100.4996) Pattern recognition, neural networks.

https://doi.org/10.1364/AO.57.003859

# 1. INTRODUCTION

Computer-generated holography is a technique for calculating an interference pattern that generates an arbitrary optical field [1–5]. Spatial light modulators (SLMs) have realized rewritable computer-generated holograms (CGHs). Promising applications of such CGHs include holographic 3D television and dynamic beam shaping, in both of which optical intensity distributions in an output space are controlled by CGHs [6–9]. A longstanding and challenging issue in computer-generated holography based on SLMs is the incomplete control of light waves in conventional SLMs; that is to say, the SLMs modulate only the amplitude or phase of the light waves. Iterative optimization algorithms have been used for calculating CGHs, such as the Gerchberg–Saxton (GS) method, the optimal-rotation-angle method, and so on [10–14]. Several noniterative methods have also been proposed [15–18]. However, they have some issues with image quality, spatial resolution, or field-of-view due to speckle noise, the down-sampling effect, and the zero-order and conjugate light.

Recently, machine learning has become a hot topic in the field of optics, and we have introduced machine learning techniques to optical sensing and control [19–23]. Several research projects based on deep learning (deep neural networks) have also been proposed for phase retrieval, superresolution, and ghost imaging [24–27]. A shallow neural network has been introduced for noniterative computer-generated holography [28]. However, in this method, a traditional iterative method is needed for generating the training data sets, and the computational cost is high.

In this paper, to solve the above issues, we present a non-iterative method for calculating CGHs based on deep learning. The deep network generates an interferometric pattern for

generating an arbitrary light field at a certain propagating distance. Our method uses a number of pairs of random input patterns and their propagating speckle-like intensity output patterns for training the network. We experimentally demonstrated our method with a phase-only CGH. The relationship between our proposed method and the learning-based phase retrieval methods [20,22–24] is analogous to the relationship between the GS method for computer-generated holography and the error-reduction method for phase retrieval [10,29]. Our method is a promising approach for high-speed parallel focusing, such as multispot stimulation in optogenetics and optical tweezers for biology [8,30–34].

# 2. METHOD

A diagram of the proposed method is shown in Fig. 1. In this paper, we assume a phase-only CGH implemented by the optical setup shown in Fig. 1. This forward process F[•] is expressed as

$$\mathbf{y} = \mathcal{F}[\mathbf{x}], \tag{1}$$

$$= |\mathcal{P}_z[\exp(i\mathbf{x})]|^2, \tag{2}$$

where x ∈ ℝN²*1 is a phase pattern displayed on the SLM, y ∈ ℝN²*1 is an intensity pattern that appears on the image sensor located at a distance z from the SLM, and Pz[•] is the Fresnel propagation at distance z [4]. Here, N is the number of pixels along one spatial dimension.

We use a deep convolutional residual network (ResNet) for calculating the inverse process of Eq. (1) [35]. Residual learning enables optimization of deep layers by preventing stagnation with skip connections. ResNet has been used for phase retrieval

1559-128X/18/143859-05 Journal © 2018 Optical Society of America