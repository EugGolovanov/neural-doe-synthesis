Check for updates

Research Article

Vol. 60, No. 4 / 1 February 2021 / Applied Optics

A323

applied optics

# Three-dimensional deeply generated holography [Invited]

RYOICHI HORISAKI,¹,²,* YOHEI NISHIZAKI,³ KATSUHISA KITAGUCHI,³ MAMORU SAITO,³ AND JUN TANIDA¹

¹Department of Information and Physical Sciences, Graduate School of Information Science and Technology, Osaka University, 1-5 Yamadaoka, Suita, Osaka 565-0871, Japan

²JST, PRESTO, 4-1-8 Honcho, Kawaguchi-shi, Saitama 332-0012, Japan

³Systems and Control Laboratory, Environmental Technology Research Division, Osaka Research Institute of Industrial Science and Technology, 1-6-50, Morinomiya, Joto-ku, Osaka 536-8553, Japan

*Corresponding author: r.horisaki@ist.osaka-u.ac.jp

Received 30 July 2020; revised 2 October 2020; accepted 27 October 2020; posted 28 October 2020 (Doc. ID 404151); published 28 January 2021

In this paper, we present a noniterative method for 3D computer-generated holography based on deep learning. A convolutional neural network is adapted for directly generating a hologram to reproduce a 3D intensity pattern in a given class. We experimentally demonstrated the proposed method with optical reproductions of multiple layers based on phase-only Fourier holography. Our method is noniterative, but it achieves a reproduction quality comparable with that of iterative methods for a given class. © 2021 Optical Society of America

https://doi.org/10.1364/AO.404151

## 1. INTRODUCTION

Computer-generated holography (CGH) is a technique for computationally synthesizing an interference pattern to produce an arbitrary optical field [1–4]. CGH has been used for various applications, such as laser processing for precision engineering, optical tweezers and stimulation in life science, and 3D displays for entertainment [5–8]. Spatial light modulators (SLMs) have realized rewritable holograms for dynamic light shaping [9]. However, SLMs cannot control both the amplitude and phase of light waves simultaneously but are capable of just amplitude control only or phase control only. Therefore, most of the algorithms for CGH iteratively calculate amplitude or phase holograms to reproduce the target fields; examples include the Gerchberg–Saxton (GS) method and the optimal-rotation-angle method [10,11]. One issue with those algorithms is the long calculation time due to the iterative process. A similar problem, which is a trade-off between the calculation time and reproduction quality, might be found in 3D CGH. For example, iterative methods for high-quality reproduction based on optimization procedures and noniterative fast methods based on superposition of holograms for each depth with certain assumptions, such as sparsity, have been reported [12–17].

Recently, machine learning, including deep learning, has had a major impact in various fields [18,19]. It has also become an attractive tool for imaging applications [20–24]. Our team has reported a noniterative method for CGH based on deep learning to solve the above-mentioned trade-off [25]. Here, we refer to this method as deeply generated holography (DGH), in

which a deep neural network is trained to solve the inverse problem to noniteratively calculate a hologram that reproduces a target optical pattern. DGH has recently been extended to adapt to a given target pattern, where a hologram is noniteratively but indirectly synthesized, and/or holograms for the target patterns are synthesized in advance for training [26–28]. In this paper, we further extend the concept of DGH to noniteratively and directly calculate a hologram without hologram synthesis in the training stage and furthermore to reproduce a 3D target pattern optically. Our study provides novel flexibility to assumptions in noniterative 3D-CGH methods with a priori knowledge of reproduced optical patterns based on deep learning. This is important for some applications of CGH, such as precision engineering and life science applications. In these applications, reproduced patterns might belong to a given or limited class, where the patterns have similar shapes, geometries, and other spatial characteristics.

## 2. METHOD

A schematic diagram of 3D-DGH is shown in Fig. 1. In this study, we assume an off-axis phase-only Fourier hologram θₖ(x, y) that reproduces a 3D optical intensity pattern iₖ(x, y, z). Here, x and y are the lateral coordinates, z is the axial coordinate along the optical axis, and k is the index of the 3D patterns. The numbers of elements along the x, y, z, and k dimensions are defined as nₓ, nᵧ, n₂, and nₖ, respectively. The network N directly generates a hologram θₖ from a 3D target pattern iₖ as

1559-128X/21/04A323-06 Journal © 2021 Optical Society of America