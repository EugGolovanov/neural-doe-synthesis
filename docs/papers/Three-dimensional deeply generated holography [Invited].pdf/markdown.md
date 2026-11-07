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

A324

Vol. 60, No. 4 / 1 February 2021 / Applied Optics

Research Article

![img-0.jpeg](img-0.jpeg)

Fig. 1. Computational scheme of 3D-DGH.

$$\boldsymbol{\theta}_k = \mathcal{N}(\boldsymbol{i}_k, \boldsymbol{p}), \tag{1}$$

where $\boldsymbol{p}$ is a set of learnable parameters constituting the network.

The generated hologram $\boldsymbol{\theta}_k$ is computationally propagated onto each layer in the target space through the Fourier transform $\mathcal{F}$ and Fresnel propagation with the impulse response $\boldsymbol{r}(x, y, z)$ [29]. Assuming a collimated coherent beam, this forward process $\mathcal{P}$ is written as

$$\hat{\boldsymbol{i}}_k(x, y, z) = \mathcal{P}(\boldsymbol{\theta}_k) \tag{2}$$

$$= \mathcal{C}(|\mathcal{F}(\exp(j\mathcal{Z}(\boldsymbol{\theta}_k(x, y)))) \otimes \boldsymbol{r}(x, y, z)|^2), \tag{3}$$

where $\hat{\boldsymbol{i}}_k(x, y, z)$ is the simulated 3D intensity, $\otimes$ is the 2D convolution operator, and $\mathcal{Z}$ and $\mathcal{C}$ are operators for zero-padding to $n_x' \times n_y'$ pixels and clipping the top-left region of $n_x \times n_y$ pixels in the $x, y$ dimensions, respectively. These operators are introduced in the off-axis configuration.

The simulated intensity $\hat{\boldsymbol{i}}_k$ is evaluated with the following loss function $\mathcal{L}$:

$$= \sqrt{\frac{1}{n_k n_x n_y n_z} \left\| c_{k,z} \mathcal{P}(\mathcal{N}(\boldsymbol{i}_k, \boldsymbol{p})) - \boldsymbol{i}_k \right\|_2^2}, \tag{5}$$

where $\|\cdot\|_2$ is the norm. This loss function evaluates the root-mean-square error (RMSE) between the target and simulated intensities. A normalization coefficient $c_{k,z}$ is introduced here, which is written as

$$c_{k,z} = \frac{1}{n_x n_y} \sum_{x=1}^{n_x} \sum_{y=1}^{n_y} \boldsymbol{i}_k(x, y, z), \tag{6}$$

which assumes a perfect diffraction efficiency, where the whole of the light intensity illuminating the hologram area of $n_x \times n_y$ pixels contributes to shaping the target intensity pattern $\boldsymbol{i}_k$.

In this study, the set of learning parameters $\boldsymbol{p}$ is optimized by minimizing the loss function $\mathcal{L}$ with a state-of-the-art stochastic gradient decent algorithm called "Adam" [30]. U-Net was used for the network $\mathcal{N}$ [31]. U-Net and U-Net-like networks have been applied to various biomedical imaging applications and image retrieval from spatially mixed signals [32–36]. The networks consist of multiple encoder subnetworks and decoder subnetworks linked with skip connections. The encoders and decoders employ downsampling by max-pooling and upsampling by up-convolution, respectively. Each layer in the 3D intensity $\boldsymbol{i}_k$ is provided to each channel of U-Net as the input, and its output is the hologram $\boldsymbol{\theta}_k$.

The 3D-DGH method is compared with the 3D-GS method, which is an established algorithm for 3D-CGH [12]. In the 3D-GS method, the following steps are computationally iterated: 1) The optical field on the hologram is propagated onto each target layer; 2) the propagating amplitudes on the target layers are replaced by the 3D target pattern; 3) the resultant fields on each target layer are backpropagated separately onto the hologram plane; 4) the backpropagating fields from each target layer are averaged; 5) the hologram is updated using the phase of the resultant field. Here, the forward process in Step 2 is equivalent to Eqs. (2) and (3) in 3D-DGH.

### 3. EXPERIMENTAL DEMONSTRATION

In the demonstrations, the learning performance of the proposed method for a given data set was verified, where we supposed a class of reproduced patterns that were known and limited, such as laser processing and optical tweezers. Layers of the 3D target patterns were images of handwritten digits randomly selected from the EMNIST database, which is a set of handwritten characters [25,28,37]. Layers were divided into two-by-two subregions, and a single digit was located in a single subregion on each layer. The subregions with the digits were randomly and differently chosen in each target pattern without lateral overlapping between each layer. The pixel count of the hologram and the target areas on each layer was

$$\mathcal{L}(\boldsymbol{p}) = \sqrt{\frac{1}{n_k n_x n_y n_z} \sum_{k=1}^{n_k} \sum_{x=1}^{n_x} \sum_{y=1}^{n_y} \sum_{z=1}^{n_z} \left( c_{k,z} \hat{\boldsymbol{i}}_k(x, y, z) - \boldsymbol{i}_k(x, y, z) \right)^2} \tag{4}$$

Research Article

Vol. 60, No. 4 / 1 February 2021 / Applied Optics

A325

96 × 96 (= nₓ × nᵧ). This was more than 100 times smaller than the pixel count used in state-of-the-art studies of CGH due to the limitations of our computational resources, which restricted the performances of both 3D-DGH and 3D-GS to compare them under the same conditions [4,12–17]. In this situation, target patterns composed of sparse lines like handwritten digits in the EMNIST database were suitable for demonstrating both 3D-DGH and 3D-GS.

The pixel count of a zero-padded image obtained through the operator Z in Eq. (3) was 384 × 384 (= nₓ' × nᵧ'), which was determined to prevent crosstalk of the zeroth-order light to the target area. The number of target layers was set to 2, 3, and 4 (= n_z). The wavelength of the collimated beam was 532 nm. The pixel pitch along the x and axes was 36 μm. The focal length f of the Fourier transform lens was 80 mm. The interval between the layers was 2 mm, and the first layer was located 3 mm before the Fourier plane.

The numbers of images in 3D patterns of the training and test data sets were 279,600 and 400, respectively, without any overlapping between the data sets. In the training process with the Adam optimizer, the learning rate was 0.0001, the gradient decay factor was 0.9, and the squared gradient decay factor was 0.999. The number of epochs was 10, and the batch size was 50 (= n_k). The encoder depth of U-Net was five. These parameters were experimentally chosen. The training and testing of the network in 3D-DGH were performed on a computer with an Intel Xeon 6134 CPU running at 3.2 GHz, with 192 GB of RAM, and an NVIDIA Tesla V100 GPU with 16 GB of VRAM. The training process of 3D-DGH took about one day; further, 3D-GS was also performed on the same computer using CUDA-based GPU processing with the parallel fast Fourier transform. The code in the experimental demonstrations was implemented by using MATLAB and Python.

The 3D-DGH and 3D-GS methods were numerically analyzed with the above-mentioned conditions, as shown in Fig. 2. This figure shows the relationship between the calculation time of hologram synthesis, excluding the training process, and the loss L in Eqs. (4) and (5) for 100 patterns of the test data set. In the case of the 3D-GS method, the number of iterations was varied from 1 to 1000 with an interval of 1, and the calculation times were measured. Each point and line in Fig. 2 was calculated with the average of 100 trials in both the 3D-DGH and 3D-GS methods.

The plots in Fig. 2 trivially show that a larger number of target layers increased the reproduction error in both the 3D-DGH and 3D-GS methods. In the case of the 3D-GS method, a longer iteration process reduced the error; however, it was time-consuming. On the other hand, the 3D-DGH method achieved faster hologram synthesis with less errors than that of 3D-GS using 1000 iterations. Therefore, the 3D-DGH method alleviated the trade-off between the calculation time and the reproduction error in the 3D-GS method. With the 3D-GS method, the average calculation time for n_z = 2, 3, and 4 was 4.30 s at 1000 iterations. With the 3D-DGH method, the average calculation time was 29.3 ms.

The 3D-DGH and 3D-GS methods were experimentally compared, as shown in Fig. 3. The experimental conditions were the same as those in the above numerical analysis. The hologram was implemented with a transmissive SLM (LC2012

![img-1.jpeg](img-1.jpeg)

Fig. 2. Relationship between the calculation time and reproduction error.

manufactured by HOLOEYE, pixel count: 1024 × 768, pixel pitch: 36 μm) with the phase mode; it was illuminated with a collimated beam from a laser light source (DJ532-40 manufactured by Thorlabs, wavelength: 532 nm), as shown in the upper part in Fig. 1. The transmitting field on the SLM was Fourier-transformed by a lens (focal length f: 80 mm), and the resultant field propagated onto each target layer. The propagating fields on each layer were captured by a monochrome camera (DMK33UX273 manufactured by The Imaging Source, pixel count: 1440 × 1080, pixel pitch: 3.45 μm) with an exposure time of 25 μs. In the case of the 3D-GS method, the numbers of iterations for n_z = 2, 3, and 4 were set to 1000.

A 3D target pattern for four layers (n_z = 4) from the test data set is shown in Fig. 3(a). The first two and three layers also corresponded to the target patterns for two and three layers, respectively. A hologram for two layers (n_z = 2) with the 3D-GS method and its reproduction result is shown in Figs. 3(b) and 3(c), and those with the 3D-DGH method are shown in Figs. 3(d) and 3(e), respectively. A hologram for three layers (n_z = 3) with the 3D-GS method and its reproduction result is shown in Figs. 3(f) and 3(g), and those with the 3D-DGH method are shown in Figs. 3(h) and 3(i), respectively. Similarly, a hologram for four layers (n_z = 4) with the 3D-GS method and its reproduction result is shown in Figs. 3(j) and 3(k), and those with the 3D-DGH method are shown in Figs. 3(l) and 3(m), respectively. The intensities of reproduced patterns with the 3D-DGH method were higher than those with the 3D-GS method, although the latter's reproduced patterns might be sharper. A large number of layers caused artifacts due to interference from other layers in both methods. The reproduced image quality in both the 3D-DGH and 3D-GS methods might be improved by increasing the pixel count of the holograms, which is limited by the current computational resources.

The generalization performance of 3D-DGH with training digits in the previous demonstrations of Figs. 2 and 3 was verified with a test data set composed of uppercase and lowercase alphabetical letters randomly selected from the EMNIST database, as shown in Fig. 4(a), where n_z = 2. In the numerical analysis of this case, the average RMSE with 100 patterns was

A326

Vol. 60, No. 4 / 1 February 2021 / Applied Optics

Research Article

![img-2.jpeg](img-2.jpeg)

(a)

![img-3.jpeg](img-3.jpeg)

(b)

![img-4.jpeg](img-4.jpeg)

(c)

![img-5.jpeg](img-5.jpeg)

(d)

![img-6.jpeg](img-6.jpeg)

(e)

![img-7.jpeg](img-7.jpeg)

(f)

![img-8.jpeg](img-8.jpeg)

(g)

![img-9.jpeg](img-9.jpeg)

(h)

![img-10.jpeg](img-10.jpeg)

(i)

![img-11.jpeg](img-11.jpeg)

(j)

![img-12.jpeg](img-12.jpeg)

(k)

![img-13.jpeg](img-13.jpeg)

(l)

![img-14.jpeg](img-14.jpeg)

(m)

Fig. 3. Experimental results with different numbers of layers. (a) Target pattern for four layers. (b) Hologram with 3D-GS for two layers and (c) its reproduction result. (d) Hologram with 3D-DGH for two layers and (e) its reproduction result. (f) Hologram with 3D-GS for three layers and (g) its reproduction result. (h) Hologram with 3D-DGH for three layers and (i) its reproduction result. (j) Hologram with 3D-GS for four layers and (k) its reproduction result. (l) Hologram with 3D-DGH for four layers and (m) its reproduction result.

0.136, which was 7% worse than that of the previous simulation in Fig. 2. The hologram with 3D-DGH and its experimental result are shown in Figs. 4(b) and 4(c), respectively. The visual impact due to alphabetic characters, which were not trained, was not very strong in this case by comparison between Figs. 3(e) and 4(c).

The sensitivity against geometries of 3D-DGH with training patterns without lateral overlapping in the previous demonstrations of Figs. 2 and 3 was also evaluated with a test data set composed of patterns with lateral overlapping, as shown in Fig. 5(a), where $n_z = 2$. In the numerical analysis of this case,

![img-15.jpeg](img-15.jpeg)

(a)

![img-16.jpeg](img-16.jpeg)

(b)

(c)

Fig. 4. Experimental results by training digits and testing alphabetical letters. (a) Target pattern for two layers. (b) Hologram with 3D-DGH and (c) its reproduction result.

![img-17.jpeg](img-17.jpeg)

(a)

![img-18.jpeg](img-18.jpeg)

(b)

(c)

Fig. 5. Experimental results by training without lateral overlapping and testing with lateral overlapping. (a) Target pattern for two layers. (b) Hologram with 3D-DGH and (c) its reproduction result.

the average RMSE with 100 patterns was 0.153, which was 20% worse than that of the previous simulation in Fig. 2. The hologram with 3D-DGH and its experimental result are shown in Figs. 5(b) and 5(c), respectively. This degradation due to lateral overlapping might be caused by strong interference from other layers and the high sensitivity of 3D-DGH against untrained geometries.

Although the network adapted or overfitted the given training class, as shown by the results in the previous two paragraphs, optical patterns formed by CGH can be assumed in several applications, such as in laser processing and optical tweezers. This limited generalization performance might be improved by investigating the network architecture and increasing the computational power.

## 4. CONCLUSION

In summary, we proposed and demonstrated 3D-DGH, in which a hologram is noniteratively and directly calculated with a deep convolutional neural network to optically reproduce a 3D target pattern in a given class. The network was trained with a loss between the target pattern and multislice propagation from the hologram synthesized by the network. The 3D-DGH method was numerically and experimentally compared with the 3D-GS method. In the demonstrations, the EMNIST database was used for the training and test data sets. U-Net was employed for the network architecture of 3D-DGH as a first trial, although U-Net was not optimized for this application.

Research Article

Vol. 60, No. 4 / 1 February 2021 / Applied Optics

A327

The results showed that 3D-DGH realized 147 times faster 3D hologram synthesis with accuracy comparable with that of 3D-GS.

3D-DGH will increase the range of applications of CGH, especially in the fields of laser material processing and optical manipulation, where the class of generated optical patterns is assumed to be known. Future work should address the reproduction quality and generalization performance of 3D-DGH. These issues are due to the heavy training cost, which limits the pixel count of the hologram and the variety of the training data set. Optimization or a different choice of the network architecture for 3D-DGH and the use of more powerful processors, especially with large memory spaces, for training large data sets might help to overcome these issues. Also, an interesting direction is to apply our method to novel optical computing and manipulation [38,39]. The present study and the future work mentioned above will influence various fields, such as industry, biology, and entertainment.

Note added in proof. After submission of this paper, related articles were published [40–42].

Funding. Japan Society for the Promotion of Science (JP20H02657, JP20K05361): Precursory Research for Embryonic Science and Technology (JPMJPR17PB).

Disclosures. The authors declare no conflicts of interest.

# REFERENCES

1. G. Nehmetallah and P. P. Banerjee, “Applications of digital and analog holography in three-dimensional imaging,” Adv. Opt. Photon. 4, 472–553 (2012).

2. J. Geng, “Three-dimensional display technologies,” Adv. Opt. Photon. 5, 456–535 (2013).

3. J.-H. Park, “Recent progress in computer-generated holography for three-dimensional scenes,” J. Inf. Disp. 18, 1–12 (2017).

4. K. Matsushima, Introduction to Computer Holography, Series in Display Science and Technology (Springer, 2020).

5. M. Malinauskas, A. Žukauskas, S. Hasegawa, Y. Hayasaki, V. Mizeikis, R. Buividas, and S. Juodkazis, “Ultrafast laser processing of materials: from science to industry,” Light Sci. Appl. 5, e16133 (2016).

6. K. Dholakia and T. Čižmár, “Shaping the future of manipulation,” Nat. Photonics 5, 335–342 (2011).

7. A. M. Packer, L. E. Russell, H. W. P. Dalgleish, and M. Häusser, “Simultaneous all-optical manipulation and recording of neural circuit activity with cellular resolution in vivo,” Nat. Methods 12, 140–146 (2014).

8. K. Kumagai, S. Hasegawa, and Y. Hayasaki, “Volumetric bubble display,” Optica 4, 298–302 (2017).

9. N. Savage, “Digital spatial light modulators,” Nat. Photonics 3, 170–172 (2009).

10. R. W. Gerchberg and W. O. Saxton, “A practical algorithm for the determination of the phase from image and diffraction plane pictures,” Optik 35, 237–246 (1972).

11. J. Bengtsson, “Kinoform design with an optimal-rotation-angle method,” Appl. Opt. 33, 6879–6884 (1994).

12. R. Piestun, B. Spektor, and J. Shamir, “Wave fields in three dimensions: analysis and synthesis,” J. Opt. Soc. Am. A 13, 1837–1848 (1996).

13. H. Pang, J. Wang, A. Cao, M. Zhang, L. Shi, and Q. Deng, “Accurate hologram generation using layer-based method and iterative Fourier transform algorithm,” IEEE Photon. J. 9, 1–8 (2017).

14. P. Sun, S. Chang, S. Liu, X. Tao, C. Wang, and Z. Zheng, “Holographic near-eye display system based on double-convergence light Gerchberg-Saxton algorithm,” Opt. Express 26, 10140–10151 (2018).

15. H. G. Kim and Y. M. Ro, “Ultrafast layer based computer-generated hologram calculation with sparse template holographic fringe pattern for 3-D object,” Opt. Express 25, 30418–30427 (2017).

16. A. Gilles and P. Gioia, “Real-time layer-based computer-generated hologram calculation for the Fourier transform optical system,” Appl. Opt. 57, 8508–8517 (2018).

17. J. Jia, J. Si, and D. Chu, “Fast two-step layer-based method for computer generated hologram using sub-sparse 2D fast Fourier transform,” Opt. Express 26, 17487–17497 (2018).

18. Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” Nature 521, 436–444 (2015).

19. S. Webb, “Deep learning for biology,” Nature 554, 555–557 (2018).

20. D. Mayerich, M. J. Walsh, A. Kadjacsy-Balla, P. S. Ray, S. M. Hewitt, and R. Bhargava, “Stain-less staining for computed histopathology,” Technology 03, 27–31 (2015).

21. R. Horisaki, R. Takagi, and J. Tanida, “Learning-based imaging through scattering media,” Opt. Express 24, 13738–13743 (2016).

22. R. Horisaki, R. Takagi, and J. Tanida, “Learning-based focusing through scattering media,” Appl. Opt. 56, 4358–4362 (2017).

23. G. Barbastathis, A. Ozcan, and G. Situ, “On the use of deep learning for computational imaging,” Optica 6, 921–943 (2019).

24. Z. Luo, A. Yurt, R. Stahl, A. Lambrechts, V. Reumers, D. Braeken, and L. Lagae, “Pixel super-resolution for lens-free holographic microscopy using deep learning neural networks,” Opt. Express 27, 13581–13595 (2019).

25. R. Horisaki, R. Takagi, and J. Tanida, “Deep-learning-generated holography,” Appl. Opt. 57, 3859–3863 (2018).

26. M. H. Eybposh, N. W. Caira, P. Chakravarthula, M. Atisa, and N. C. Pégard, “High-speed computer-generated holography using convolutional neural networks,” in Biophotonics Congress: Biomedical Optics 2020 (Translational, Microscopy, OCT, OTS, BRAIN) (Optical Society of America, 2020), paper BTu2C.2.

27. T. Shimobaba, D. Blinder, P. Schelkens, Y. Yamamoto, I. Hoshi, T. Kakue, and T. Ito, “Deep-learning-assisted hologram calculation via low-sampling holograms,” in 8th International Congress on Advanced Applied Informatics (IIAI-AAI) (IEEE, 2019), pp. 936–941.

28. P. Cheremkhin, N. Evtikhiev, V. Krasnov, V. Rodin, D. Rymov, and R. Starikov, “Machine learning methods for digital holography and diffractive optics,” Procedia Comput. Sci. 169, 440–444 (2020).

29. J. W. Goodman, Introduction to Fourier Optics (McGraw-Hill, 1996).

30. D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in International Conference on Learning Representations (ICLR) (2015).

31. O. Ronneberger, P. Fischer, and T. Brox, “U-Net: convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention (MICCAI), N. Navab, J. Hornegger, W. M. Wells, and A. F. Frangi, eds. (Springer, 2015), pp. 234–241.

32. T. Falk, D. Mai, R. Bensch, Ö. Çiçek, A. Abdulkadir, Y. Marrakchi, A. Böhm, J. Deubner, Z. Jäckel, K. Seiwald, A. Dovzhenko, O. Tietz, C. Dal Bosco, S. Walsh, D. Saltukoglu, T. L. Tay, M. Prinz, K. Palme, M. Simons, I. Diester, T. Brox, and O. Ronneberger, “U-Net: deep learning for cell counting, detection, and morphometry,” Nat. Methods 16, 67–70 (2019).

33. W. Ouyang, A. Aristov, M. Lelek, X. Hao, and C. Zimmer, “Deep learning massively accelerates super-resolution localization microscopy,” Nat. Biotechnol. 36, 460–468 (2018).

34. E. Moen, D. Bannon, T. Kudo, W. Graf, M. Covert, and D. Van Valen, “Deep learning for cellular image analysis,” Nat. Methods 16, 1233–1246 (2019).

35. S. Li, M. Deng, J. Lee, A. Sinha, and G. Barbastathis, “Imaging through glass diffusers using densely connected convolutional networks,” Optica 5, 803–813 (2018).

36. E. Guo, S. Zhu, Y. Sun, L. Bai, C. Zuo, and J. Han, “Learning-based method to reconstruct complex targets through scattering medium beyond the memory effect,” Opt. Express 28, 2433–2446 (2020).

37. G. Cohen, S. Afshar, J. Tapson, and A. van Schaik, “EMNIST: extending MNIST to handwritten letters,” in International Joint Conference on Neural Networks (IEEE, 2017), pp. 2921–2926.

A328

Vol. 60, No. 4 / 1 February 2021 / Applied Optics

Research Article

38. Q. Ma, G. D. Bai, H. B. Jing, C. Yang, L. Li, and T. J. Cui, “Smart metasurface with self-adaptively reprogrammable functions,” Light Sci. Appl. 8, 98 (2019).

39. E. Goi, Q. Zhang, X. Chen, H. Luan, and M. Gu, “Perspective on photonic memristive neuromorphic computing,” PhotoniX 1, 3 (2020).

40. M. Hossein Eybposh, N. W. Caira, M. Atisa, P. Chakravarthula, and N. C. Pégard, “DeepCGH: 3D computer-generated holography using deep learning,” Opt. Express 28, 26636–26650 (2020).

41. J. Lee, J. Jeong, J. Cho, D. Yoo, B. Lee, and B. Lee, “Deep neural network for multi-depth hologram generation and its training strategy,” Opt. Express 28, 27137–27154 (2020).

42. Y. Peng, S. Choi, N. Padmanaban, and G. Wetzstein, “Neural holography with camera-in-the-loop training,” ACM Trans. Graph. 39, 185 (2020).