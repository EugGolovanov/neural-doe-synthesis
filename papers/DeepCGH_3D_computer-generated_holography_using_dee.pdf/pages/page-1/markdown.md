Check for updates

Research Article

Vol. 28, No. 18 / 31 August 2020 / Optics Express 26636

Optics EXPRESS

# DeepCGH: 3D computer-generated holography using deep learning

M. HOSSEIN EYBPOSH,¹,² NICHOLAS W. CAIRA,¹ MATHEW ATISA,³ PRANEETH CHAKRAVARTHULA,³ AND NICOLAS C. PÉGARD¹,²,⁴,*

¹Department of Applied Physical Sciences, University of North Carolina at Chapel Hill, NC 27599, USA

²Department of Biomedical Engineering, University of North Carolina at Chapel Hill, NC 27599, USA

³Department of Computer Science, University of North Carolina at Chapel Hill, NC 27599, USA

⁴Neuroscience Center, University of North Carolina at Chapel Hill, NC 27599, USA

*pegard@unc.edu

Abstract: The goal of computer-generated holography (CGH) is to synthesize custom illumination patterns by modulating a coherent light beam. CGH algorithms typically rely on iterative optimization with a built-in trade-off between computation speed and hologram accuracy that limits performance in advanced applications such as optogenetic photostimulation. We introduce a non-iterative algorithm, DeepCGH, that relies on a convolutional neural network with unsupervised learning to compute accurate holograms with fixed computational complexity. Simulations show that our method generates holograms orders of magnitude faster and with up to 41% greater accuracy than alternate CGH techniques. Experiments in a holographic multiphoton microscope show that DeepCGH substantially enhances two-photon absorption and improves performance in photostimulation tasks without requiring additional laser power.

© 2020 Optical Society of America under the terms of the OSA Open Access Publishing Agreement

## 1. Introduction

Computer generated holography (CGH) aims to synthesize custom light intensity distributions by modulating a coherent wavefront, typically by digitally encoding its phase with a spatial light modulator (SLM). CGH is the preferred method to create custom volumetric illumination in a broad range of applications including neural photostimulation [1–3], optical trapping [4,5], and 3D displays [6,7].

CGH algorithms identify the best possible wave modulation by solving a multidimensional, nonlinear, and non-convex inverse problem that is generally ill-posed. Holograms must be a free space solution of the wave propagation equation that conserve energy, and thus many target intensities are physically infeasible and cannot be rendered exactly. This issue is most notable in 3D CGH when the target intensities specified across several successive depth planes are mutually incompatible. Additional factors that prevent feasibility include the numerical aperture of the optical system, which sets the maximal allowable resolution, as well as hardware limitations such as finite SLM resolution. In practice, CGH solutions are always approximate and numerical methods are required to identify a feasible hologram that best matches the desired illumination pattern.

Aside from simple holographic superposition techniques [8], existing methods for CGH rely on iterative exploration. The most common approach is the Gercheberg-Saxton (GS) algorithm [9], which digitally propagates a complex wave back and forth between the image plane, where the intensity distribution is rendered, and the SLM plane, where the wavefront is modulated, while enforcing amplitude constraints at each step. This algorithm is simple and straightforward to implement, but yields sub-optimal solutions. More recently, advanced algorithms have been developed that compute holograms by solving an optimization problem with an explicit loss function. This includes non-convex optimization with a gradient descent [10], and methods based

#399624

Journal © 2020

https://doi.org/10.1364/OE.399624

Received 10 Jun 2020; revised 5 Aug 2020; accepted 17 Aug 2020; published 25 Aug 2020