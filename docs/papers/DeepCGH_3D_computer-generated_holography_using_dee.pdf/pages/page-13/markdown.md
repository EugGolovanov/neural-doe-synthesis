Research Article

Vol. 28, No. 18/31 August 2020/Optics Express 26648

Optics EXPRESS

We further illustrate the generalizability of DeepCGH with a model that is trained on the lines dataset and tested on a natural image. Simulation results in Fig. 6(b) show intensity reconstructions by each method with patterns picked from the lines test dataset and a natural image. DeepCGH yields image quality that compare to holograms obtained with iterative algorithms. This example perfectly illustrates the generalizability of the mapping that DeepCGH learns from a specific data type to synthesize high-fidelity holograms of completely different nature. Although DeepCGH holograms seem to contain more speckle than holograms synthesized with iterative techniques, we note that DeepCGH yields this solution 33 and 3,678 times faster than GS and NOVO-CGH, respectively, at speeds that would be compatible with online hologram synthesis at video rate.

## 4. Conclusion

We have developed DeepCGH, a new algorithm based on CNNs for fast and accurate 3D computer-generated holography. DeepCGH operates with a fixed computation time that is predetermined by the hologram size and model complexity. We found that DeepCGH not only synthesizes extremely large holograms (up to 11 Megavoxels) at record speeds, but also reliably identifies solutions with greater accuracy than existing techniques. We validated DeepCGH with experiments in multiphoton holography where the enhanced accuracy of our method yields substantially more two-photon absorption without hardware modifications. Image plane holography simplifies the mapping that the CNN performs and best utilizes the capabilities of CNNs compared to other CNN-based CGH approaches. DeepCGH can be easily customized to accommodate various spatial discretizations of the volume of interest and SLM resolutions by adjusting the number of input channels, the interleaving factor, or number of kernels in the CNN model. Finally, DeepCGH enables unsupervised training of the CNN, allowing the model to be tailored to custom applications of holography by selecting training datasets that best match the desired real-world experimental conditions, as there is no need to explicitly provide ground truth phase masks. Further tailoring can be achieved by customizing the loss function to optimize the model to best execute a user-defined task and directly optimize for the desired outcome instead of hologram accuracy.

## Funding

Nvidia; Burroughs Wellcome Fund (5113244).

## Acknowledgments

The authors thank Professors S. Chowdhury, H. Fuchs, and A. Giovannucci for providing valuable feedback on our work.

## Disclosures

The authors declare no conflicts of interest.

## References

1. A. R. Mardinly, I. A. Oldenburg, N. C. Pégard, S. Sridharan, E. H. Lyall, K. Chesnov, S. G. Brohawn, L. Waller, and H. Adesnik, "Precise multimodal optical control of neural ensemble activity," Nat. Neurosci. 21(6), 881–893 (2018).

2. P. Pozzi, L. Maddalena, N. Ceffa, O. Soloviev, G. Vdovin, E. Carroll, and M. Verhaegen, "Fast calculation of computer generated holograms for 3d photostimulation through compressive-sensing gerchberg–saxton algorithm," Methods Protoc. 2(1), 2–15 (2019).

3. W. Yang and R. Yuste, "Holographic imaging and photostimulation of neural activity," Curr. Opin. Neurobiol. 50, 211–221 (2018).

4. K. C. Neuman and S. M. Block, "Optical trapping," Rev. Sci. Instrum. 75(9), 2787–2809 (2004).

5. S. Koller, "Optical trapping: Techniques and applications," in Student Research Celebration, (Montana State University, 2017), pp. 1–2.