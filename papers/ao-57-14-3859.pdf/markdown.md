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

3860

Vol. 57, No. 14 / 10 May 2019 / Applied Optics

Research Article

![img-0.jpeg](img-0.jpeg)

Fig. 1. Diagram illustrating the proposed method.

and has shown promising results [24]. The network architecture we use is shown in Fig. 2(a). The inverse process $\mathcal{F}^{-1}[\bullet]$ of Eq. (1) is written as

$$\hat{\mathbf{x}} = \mathcal{F}^{-1}[\hat{\mathbf{y}}]. \tag{3}$$

The output of this inverse process and the network in Fig. 2(a) is a phase pattern $\hat{\mathbf{x}} \in \mathbb{R}^{N^2 \times 1}$ displayed on the SLM, to reproduce a target intensity pattern $\hat{\mathbf{y}} \in \mathbb{R}^{N^2 \times 1}$ on the image sensor, as shown in Fig. 1.

![img-1.jpeg](img-1.jpeg)

![img-2.jpeg](img-2.jpeg)

![img-3.jpeg](img-3.jpeg)

![img-4.jpeg](img-4.jpeg)

![img-5.jpeg](img-5.jpeg)

Fig. 2. Structures of the (a) whole network, (b) D-block, (c) U-block, (d) R-block, and (e) S-block.

In the training process, the optical setup shown in Fig. 1 is simulated in a computer based on Eq. (2). A number of training pairs of random phase patterns $\tilde{\mathbf{x}} \in \mathbb{R}^{N^2 \times 1}$ on the SLM and their Fresnel propagating intensity patterns $\tilde{\mathbf{y}} \in \mathbb{R}^{N^2 \times 1}$ on the image sensor are calculated. The network is regressed to the inverse process of the optical propagation with the training pairs.

The network is composed of multiscale ResNets, as shown in Fig. 2(a), where $K$ is the number of filters at the convolutional layers. The pixel counts of both the holographic plane and the image sensor plane are assumed to be a power of 2 in this paper, but this is easily extendable to an arbitrary pixel count with zero padding. "D" is a block for downsampling, as shown in Fig. 2(b), "U" is a block for upsampling, as shown in Fig. 2(c), "R" is a block with residual convolutions, as shown in Fig. 2(d), and "S" is a convolutional block for a skip convolutional connection, as shown in Fig. 2(e). The definitions of the layers are as follows [36]: "BatchNorm" is a layer of the batch normalization [37]; "ReLU" is a layer of the rectified linear unit [38]; "Conv $(S, L)$" is a layer of the 2D convolution with a filter size $S$ and a stride $L$; "TConv $(S, L)$" is a layer of the transposed 2D convolution with the filter size $S$ and the stride $L$. The loss function of the final regression layer is the mean squared error.

### 3. EXPERIMENTAL DEMONSTRATION

The proposed method was demonstrated experimentally. The pixel counts of holograms and target patterns were $64^2 (= N^2)$, and the number of filters at the convolutional layers was $32 (= K)$. The training data sets were composed of uniform random phase patterns $\tilde{\mathbf{x}}$ and their Fresnel propagating patterns $\tilde{\mathbf{y}}$ at the distance $z$. The number of training speckle pairs was 100,000. A learning algorithm called "Adam" was used for optimizing the network with an initial learning rate of 0.001, a mini-batch size of 50, and a maximum number of epochs of 100 [39]. Those parameters were chosen experimentally. The code was implemented with MATLAB.

A collimated beam from a laser [1103P manufactured by Edmund Optics; wavelength ($\lambda$): 632.8 nm] was used for illuminating the transmissive SLM [LC 2012 manufactured by Holoeye; pixel pitch ($p$): 36 $\mu$m; pixel count: 768 $\times$ 1024] operating in the phase mode. The image sensor (PL-B953 manufactured by PixelLink; pixel pitch: 4.65 $\mu$m; pixel count: 768 $\times$ 1024) was located 13 cm (= $z$) from the SLM. These optical elements were arranged in an on-axis configuration, as the optical setup shown in Fig. 1.

In the first demonstration, single-spot focusing with the proposed method was experimentally performed. The results are shown in Fig. 3. The hologram $\hat{\mathbf{x}}$ calculated by providing a delta function to the trained network is shown in Fig. 3(a). The single-spot pattern $\hat{\mathbf{y}}$ in Fig. 3(b) was optically reproduced from the hologram. A line profile of the spot is shown in Fig. 3(c). The FWHM of the profile was 36.0 $\mu$m, and the theoretical diffraction limit was 35.7 $\mu$m (= $z\lambda/pN$) [4]. Therefore, the hologram based on our method approximately achieved the diffraction limit. The peak-to-background ratio was 17.9, which is calculated as $s/b$, where $s$ is the spot intensity value, and $b$ is the average intensity value in the non-spot area [40].

Research Article

Vol. 57, No. 14 / 10 May 2019 / Applied Optics

3861

![img-6.jpeg](img-6.jpeg)

![img-7.jpeg](img-7.jpeg)

![img-8.jpeg](img-8.jpeg)

Fig. 3. Experimental data of single-spot focusing. (a) Hologram. (b) Its reproduced pattern. (c) Line profile of the spot. Hologram (phase) pattern is normalized in the interval [-π, π].

In the second experiment, image reproduction was demonstrated based on our method, as shown in Fig. 4. The three target intensity patterns ŷ shown in Fig. 4(a) were composed of a two-by-two array of digits, which were randomly selected from the handwritten digit data set of MATLAB. The holograms x̂ in Fig. 4(b) were calculated by the trained network with the target patterns. Their optically reproduced results are shown in Fig. 4(c). The zero-order light was not observed. A dot-array-like artifact in Fig. 4(c) may have been caused by the Talbot effect of the wiring grid used for pixel-wise control of the liquid crystal on the SLM and is not speckle noise [4]. It might be possible to alleviate this with an SLM having a larger fill factor, such as a reflective SLM. The root mean squared error (RMSE) between the target intensity patterns in Fig. 4(a) and the reproduced results with the proposed method in Fig. 4(c) was 0.16. The holograms calculated based on the GS method are shown in Fig. 4(d). The GS method is a representative iterative algorithm for calculating CGHs, and the number of iterations in the experiment was set to 100 [10]. The optically reproduced results are shown in Fig. 4(e). The RMSE between the target patterns in Fig. 4(a) and the reproduced results with the GS method in Fig. 4(e) was 0.17. Therefore, the reproduced results of our method were comparable with those of the conventional method.

Computational times for calculating a single hologram based on our method and the GS method were 26 and 94 ms, respectively, when using a computer with an Intel Core i7 processor with a clock rate of 3.1 GHz and a memory size of 16 GB without any parallelization or a graphics processing unit. Therefore, our method calculates holograms with a

![img-9.jpeg](img-9.jpeg)

![img-10.jpeg](img-10.jpeg)

![img-11.jpeg](img-11.jpeg)

![img-12.jpeg](img-12.jpeg)

![img-13.jpeg](img-13.jpeg)

![img-14.jpeg](img-14.jpeg)

![img-15.jpeg](img-15.jpeg)

![img-16.jpeg](img-16.jpeg)

![img-17.jpeg](img-17.jpeg)

![img-18.jpeg](img-18.jpeg)

![img-19.jpeg](img-19.jpeg)

![img-20.jpeg](img-20.jpeg)

![img-21.jpeg](img-21.jpeg)

![img-22.jpeg](img-22.jpeg)

![img-23.jpeg](img-23.jpeg)

Fig. 4. Experimental data of image reproduction. (a) Target intensity patterns. (b) Holograms based on the proposed method and (c) their reproduced patterns. (d) Holograms based on the GS method and (e) their reproduced patterns. Hologram (phase) patterns are normalized in the interval [-π, π]. The reproduced patterns were normalized with the same light intensity value.

shorter computational time compared with the conventional iterative method. The training time in our method was 40 h.

The relationship between the number of training pairs and the focusing performance at different hologram sizes was numerically investigated by simulating the first experiment, as shown in Fig. 5. The number of training pairs was set to 1000, 10,000, and 100,000. The pixel counts (N²) of the holograms and target patterns were set to 16², 32², and 64². The other parameters and the procedures for training the networks and calculating the holograms were the same as those in the first experimental demonstration. The reproduced focusing patterns

3862

Vol. 57, No. 14 / 10 May 2019 / Applied Optics

Research Article

![img-24.jpeg](img-24.jpeg)

Fig. 5. Relationship between the number of training pairs and the RMSE in a focusing simulation.

were computationally calculated with Fresnel propagation. The focusing performance was evaluated with the RMSEs between the target and the numerically reproduced focusing patterns, as shown in Fig. 5. This result shows that a larger hologram and a larger number of training data sets provide better performance.

## 4. CONCLUSION

In this paper, we proposed a noniterative method for calculating CGHs based on a convolutional deep neural network. The network was regressed to the inverse process of the optical propagation with a number of speckle pairs. We experimentally demonstrated a phase-only CGH based on our method. The results were compared with those obtained with the GS method. The demonstrations showed reasonable image quality of the reproduced intensity pattern and a shorter computational time for the proposed method compared with the conventional one.

We used the network shown in Fig. 2(a) for calculating the CGH in this paper. Deep neural networks have many options and tuning parameters [36]. Further investigation of the network architecture is needed for improving the performance. In this paper, we demonstrated control of 2D intensity with a phase-only CGH using visible light. Our method is readily extendable to various wave fields, e.g., x-rays and acoustic waves, and higher-dimensional pattern shaping, such as 3D spatial focusing, multispectral (color) holography, and temporal pulse shaping. It is also applicable to simultaneous holographic control of the complex amplitude (both amplitude and phase) and amplitude-only/complex-amplitude CGHs. Therefore, the proposed technique is promising in various applications, including biomedical sensing/control, industrial engineering, and entertainment.

Funding. Japan Society for the Promotion of Science (JSPS) (JP17H02799, JP17K00233); Precursory Research for Embryonic Science and Technology (PRESTO) (JPMJPR17PB).

## REFERENCES

1. B. R. Brown and A. W. Lohmann, "Complex spatial filtering with binary masks," Appl. Opt. 5, 967–969 (1966).

2. W. H. Lee, "Sampled Fourier transform hologram generated by computer," Appl. Opt. 9, 639–643 (1970).

3. D. Leseberg and C. Frère, "Computer-generated holograms of 3-D objects composed of tilted planar segments," Appl. Opt. 27, 3020–3024 (1988).

4. J. W. Goodman, Introduction to Fourier Optics (McGraw-Hill, 1996).

5. G. Nehmetallah and P. P. Banerjee, "Applications of digital and analog holography in three-dimensional imaging," Adv. Opt. Photon. 4, 472–553 (2012).

6. C. Slinger, C. Cameron, and M. Stanley, "Computer-generated holography as a generic display technology," Computer 38, 46–53 (2005).

7. J. Geng, "Three-dimensional display technologies," Adv. Opt. Photon. 5, 456–535 (2013).

8. K. Dholakia and T. Čižmár, "Shaping the future of manipulation," Nat. Photonics 5, 335–342 (2011).

9. J. A. Rodrigo and T. Alieva, "Freestyle 3D laser traps: tools for studying light-driven particle dynamics and beyond," Optica 2, 812–815 (2015).

10. R. W. Gerchberg and W. O. Saxton, "A practical algorithm for the determination of the phase from image and diffraction plane pictures," Optik 35, 237–246 (1972).

11. J. Bengtsson, "Kinoform design with an optimal-rotation-angle method," Appl. Opt. 33, 6879–6884 (1994).

12. N. Yoshikawa, M. Itoh, and T. Yatagai, "Quantized phase optimization of two-dimensional Fourier kinoforms by a genetic algorithm," Opt. Lett. 20, 752–754 (1995).

13. T. Dresel, M. Beyerlein, and J. Schwider, "Design of computer-generated beam-shaping holograms by iterative finite-element mesh adaption," Appl. Opt. 35, 6865–6874 (1996).

14. T. G. Jabbour and S. M. Kuebler, "Vectorial beam shaping," Opt. Express 16, 7203–7213 (2008).

15. A. W. Lohmann and D. P. Paris, "Binary Fraunhofer holograms, generated by computer," Appl. Opt. 6, 1739–1748 (1967).

16. P. W. M. Tsang and T. C. Poon, "Novel method for converting digital Fresnel hologram to phase-only hologram based on bidirectional error diffusion," Opt. Express 21, 23680–23686 (2013).

17. P. W. M. Tsang, Y.-T. Chow, and T. C. Poon, "Generation of phase-only Fresnel hologram based on down-sampling," Opt. Express 22, 25208–25214 (2014).

18. T. Shimobaba and T. Ito, "Random phase-free computer-generated hologram," Opt. Express 23, 9549–9554 (2015).

19. T. Ando, R. Horisaki, and J. Tanida, "Speckle-learning-based object recognition through scattering media," Opt. Express 23, 33902–33910 (2015).

20. R. Horisaki, R. Takagi, and J. Tanida, "Learning-based imaging through scattering media," Opt. Express 24, 13738–13743 (2016).

21. R. Takagi, R. Horisaki, and J. Tanida, "Object recognition through a multi-mode fiber," Opt. Rev. 24, 117–120 (2017).

22. R. Horisaki, R. Takagi, and J. Tanida, "Learning-based focusing through scattering media," Appl. Opt. 56, 4358–4362 (2017).

23. R. Horisaki, R. Takagi, and J. Tanida, "Learning-based single-shot superresolution in diffractive imaging," Appl. Opt. 56, 8896–8901 (2017).

24. A. Sinha, J. Lee, S. Li, and G. Barbastathis, "Lensless computational imaging through deep learning," Optica 4, 1117–1125 (2017).

25. Y. Rivenson, Z. Göröcs, H. Günaydin, Y. Zhang, H. Wang, and A. Ozcan, "Deep learning microscopy," Optica 4, 1437–1443 (2017).

26. M. Lyu, W. Wang, H. Wang, H. Wang, G. Li, N. Chen, and G. Situ, "Deep-learning-based ghost imaging," Sci. Rep. 7, 17865 (2017).

27. Y. Jo, S. Park, J. Jung, J. Yoon, H. Joo, M.-H. Kim, S.-J. Kang, M. C. Choi, S. Y. Lee, and Y. Park, "Holographic deep learning for rapid optical screening of anthrax spores," Sci. Adv. 3, e1700606 (2017).

28. S. Yamauchi, Y.-W. Chen, and Z. Nakao, "Optimization of computer-generated holograms by an artificial neural network," in Proceedings of Second International Conference on Knowledge-Based Intelligent Electronic Systems (KES'98) (Cat. No. 98EX111) (IEEE, 1998), Vol. 3, pp. 220–223.

Research Article

Vol. 57, No. 14 / 10 May 2018 / Applied Optics

3863

29. J. R. Fienup, "Phase retrieval algorithms: a comparison," Appl. Opt. 21, 2758–2769 (1982).

30. A. M. Packer, L. E. Russell, H. W. P. Dalgleish, and M. Häusser, "Simultaneous all-optical manipulation and recording of neural circuit activity with cellular resolution in vivo," Nat. Methods 12, 140–146 (2014).

31. O. Hernandez, E. Papagiakoumou, D. Tanese, K. Fidelin, C. Wyart, and V. Emiliani, "Three-dimensional spatiotemporal focusing of holographic patterns," Nat. Commun. 7, 11928 (2016).

32. N. C. Pégard, A. R. Mardinly, I. A. Oldenburg, S. Sridharan, L. Waller, and H. Adesnik, "Three-dimensional scanless holographic optogenetics with temporal focusing (3D-SHOT)," Nat. Commun. 8, 1228 (2017).

33. D. G. Grier, "A revolution in optical manipulation," Nature 424, 810–816 (2003).

34. Y. Pang, H. Song, J. H. Kim, X. Hou, and W. Cheng, "Optical trapping of individual human immunodeficiency viruses in culture fluid reveals heterogeneity with single-molecule resolution," Nat. Nanotechnol. 9, 624–630 (2014).

35. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (IEEE, 2016), pp. 770–778.

36. I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning (MIT, 2016).

37. S. Ioffe and C. Szegedy, "Batch normalization: accelerating deep network training by reducing internal covariate shift," in Proceedings of the 32nd International Conference on Machine Learning (ICML'15) (JMLR, 2015), Vol. 37, pp. 448–456.

38. V. Nair and G. E. Hinton, "Rectified linear units improve restricted Boltzmann machines," in Proceedings of the 27th International Conference on Machine Learning (ICML'10) (Omnipress, 2010), pp. 807–814.

39. D. P. Kingma and J. Ba, "Adam: a method for stochastic optimization," in International Conference on Learning Representations (ICLR) (2015).

40. R. Horstmeyer, H. Ruan, and C. Yang, "Guidestar-assisted wavefront-shaping methods for focusing light into biological tissue," Nat. Photonics 9, 563–571 (2015).