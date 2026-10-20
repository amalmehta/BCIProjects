# Blindsight: literature notes

This project builds a simulator of cortical prosthetic vision and learned stimulus encoders for it. These notes cover the domain knowledge the code depends on, the papers it draws on, the metrics the field uses, and which ideas we adopt. I checked every reference below against Crossref, PubMed/PMC or arXiv (title, authors, venue, year, DOI). Numbers marked *(abstract)* come from the paper's abstract. Numbers marked *(full text)* come from the open-access full text.

This is an independent project. It is not affiliated with or endorsed by Neuralink.

## 1. Domain knowledge

### How a cortical visual prosthesis works
A camera image is converted into electrical microstimulation of primary visual cortex (V1). Each stimulated site evokes a *phosphene*, a spot of light at the visual-field location that the site represents (its receptive field). A cortical prosthesis bypasses the eye and optic nerve, so it can serve people whose blindness comes from retinal or optic-nerve loss. Neuralink's Blindsight device received FDA Breakthrough Device designation in September 2024. I only verified this through news coverage (e.g. https://mobihealthnews.com/node/154408), not through an FDA or Neuralink page.

### Retinotopy and cortical magnification
- The V1 map is roughly log-polar (Schwartz 1977). Polimeni, Balasubramanian & Schwartz (2006) proposed the *wedge-dipole* model, w = k log((z_α + a)/(z_α + b)) with z_α = r e^{iαθ}. It captures foveal compression, the vertical-meridian shear and the far-periphery behaviour.
- We use the human parameters used by van der Grinten et al. (2024): k = 17.3 mm, a = 0.75°, b = 120°, α = 0.95. Along the horizontal meridian, the linear magnification M(r) = k(1/(r+a) − 1/(r+b)) mm/deg reduces to Horton & Hoyt's (1991) estimate M = 17.3/(E + 0.75). Our implementation stays within 8.3 % of Horton & Hoyt over 0–10° (`results/validation.json`).
- **Consequence for implants:** a fixed cortical spacing maps to tiny visual spacing at the fovea and coarse spacing in the periphery. Phosphene size scales with 1/M, so it also grows with eccentricity. Winawer & Parvizi (2016) showed that phosphene size across sites is explained by roughly constant activated cortical area, divided by M.
- Much of human V1 lies inside the calcarine sulcus. Benson et al. (2012, 2014) give anatomy-based retinotopy templates that could replace our planar map for patient-specific planning.

### Current spread, phosphene size, brightness and thresholds
- **Current spread.** Tehovnik (1996) reviewed direct activation scaling as I = K r². Tehovnik & Slocum (2007) use r = √(I/K) with K = 675 µA/mm² to relate current to phosphene size; the K value is attributed to Tehovnik et al. (2006).
- **Size saturation.** Bosking et al. (2017) stimulated 93 surface electrodes in 13 people *(abstract)*. Phosphene size rose just above threshold, then quickly saturated, and was larger at larger eccentricities.
- **First human Utah array in V1.** Fernández et al. (2021) implanted a 96-channel Utah array in a blind volunteer for 6 months. The mean single-electrode threshold was 66.8 ± 36.5 µA, and simultaneous stimulation of several electrodes lowered thresholds *(abstract)*. They used 166 ms trains, 170 µs/phase pulses at 300 Hz *(full text)*. Also from the full text: 88 of 96 electrodes evoked phosphenes; thresholds ranged from 7 to 128 µA; groups of 4 contiguous electrodes had thresholds of 48.0 ± 22.6 µA; and Fig. 6A averages perceived brightness over 10 single electrodes, reaching its maximum near 90 µA. Round 2 recalibrates our threshold distribution to the mean and sd. Because Fig. 6A is an average over (probably low-threshold) electrodes, we compare it with three model readings, not one. Phosphenes were mostly described as pin-points, and the volunteer identified simple letters/shapes.
- **Earlier intracortical work.** Schmidt et al. (1996) implanted 38 intracortical microelectrodes. Thresholds were as low as 1.9 µA, and electrodes 500 µm apart gave separate phosphenes while 250 µm apart typically did not. Phosphenes faded during long trains, which is the habituation that dynaphos models.
- **High-channel-count macaque prosthesis.** Chen et al. (2020) implanted 1024 channels in V1/V4 of monkeys, which recognised shapes, motion and letters from multi-electrode stimulation.
- **Dynamic stimulation.** Beauchamp et al. (2020) "traced" letter shapes with dynamic current steering. Sighted and blind participants recognised forms better than with static multi-electrode patterns. This motivates temporal encoders. Round 2 adds one that manages habituation (below); dynamic tracing itself is still future work.
- **Habituation.** Schmidt et al. (1996) report phosphenes fading during long or repeated stimulation. The dynaphos trace models this as a charge-driven increase of the effective threshold, decaying over about 2000 s. Under that model, an electrode held at a constant current loses its phosphene within about a second, so an encoder must ration and rotate stimulation to sustain a percept.
- **Recruitment.** Histed, Bonin & Reid (2009) showed microstimulation activates sparse, distributed populations through axons. Phosphene "size" is therefore a perceptual, not anatomical, quantity.

### Safety constraints
- Stimulation must stay within charge-per-phase and charge-density limits that depend on electrode material and area (Shannon 1992; Cogan 2008).
- We impose two hard limits on every encoder: an amplitude ceiling of 100 µA (17 nC/phase at 170 µs), and at most 25 % of electrodes active per frame (a power/heat and crosstalk budget). The learned encoder also gets a soft charge penalty.
- These are engineering stand-ins: real limits must come from the device's electrode area and material. Fernández et al. (2021) stayed in a similar current range.

### The devices we model
- **Utah array** (Maynard, Nordhausen & Normann 1997): 10×10 rigid shanks, 400 µm pitch.
- **Thread-based implants** (Musk & Neuralink 2019): up to 3072 electrodes on 96 flexible threads, robot-inserted at 6 threads/min. That paper describes *recording*, and we found no public stimulation specification. Our `threads_N` layouts are therefore an assumption: N independently placed surface sites spread over both hemispheres.

## 2. Simulation and encoding papers

| Paper | What it contributes | What we use |
|---|---|---|
| van der Grinten et al. 2024, *eLife* 13:e85812 (dynaphos) | Differentiable PyTorch cortical simulator. Covers wedge-dipole mapping, Tehovnik current spread, a charge-based leaky-integrator activation with rheobase leak (23.9 µA), sigmoidal brightness (fit to Fernández Fig. 6A), per-electrode thresholds N(9.14e-8, 6.72e-8), and a habituation trace fit to Schmidt 1996 | All equations and constants, re-implemented from the paper text (the code is GPL; we wrote our own). We add a closed form for a single train and a separable/basis renderer for CPU training |
| de Ruyter van Steveninck et al. 2022, *J Vis* 22(2):20 | End-to-end encoder → simulator → decoder training of stimulation protocols | Our training objective (reconstruction + percept-similarity + sparsity) follows this design |
| Granley, Relic & Beyeler 2022, NeurIPS (arXiv:2205.13623) | "Hybrid neural autoencoder": a deep encoder inverts a fixed, patient-specific forward model, which beats conventional encoding for every simulated patient | Patient-specific forward models; our adapter fine-tuning is a cheap variant |
| Granley, Fauvel, Chalk & Beyeler 2023, NeurIPS (arXiv:2306.13104) | Human-in-the-loop Bayesian optimisation of encoder parameters from pairwise preferences; the deep encoder is *conditioned* on patient-specific parameters | Discussed as the route to adaptation without a calibrated forward model; our threshold-aware encoder conditions on measured per-electrode thresholds in the same spirit |
| Perez et al. 2018, AAAI (FiLM) | Feature-wise conditioning of a network on side information | Considered for conditioning; we inject per-electrode calibration features at the read-out instead, because the conditioning is per electrode, not global |
| Bengio, Léonard & Courville 2013 (arXiv:1308.3432) | Straight-through estimator for hard binary units | Gradient of the threshold-aware encoder's on/off gate |
| Werbos 1990, *Proc IEEE* | Backpropagation through time | Training the temporal encoder through the frame dynamics |
| Srivastava et al. 2014, *JMLR* (dropout) | Random unit dropping as training-time augmentation | Electrode-failure augmentation (dropping electrodes, not hidden units) |
| Barrese et al. 2013, *J Neural Eng* | Failure modes of chronically implanted Utah arrays in macaques (biological and material) | Motivates the electrode-failure robustness experiments |
| Küçükoğlu et al. 2022, *Int J Neural Syst* 32(11):2250052 | Deep RL with a phosphene simulator in the loop | Related work (task-based objectives) |
| Han et al. 2021, Augmented Humans '21 (arXiv:2102.00297) | Scene simplification for bionic vision; segmentation beat saliency and depth | Motivates our saliency-gated edge baseline (a light version without a segmentation net) |
| Beyeler et al. 2017 (SciPy) and Beyeler et al. 2019, *Sci Rep* 9:9199 | pulse2percept; retinal axon-pathway percept model | Contrast: retinal percepts are streaks along axon bundles, cortical percepts are compact blobs |
| Hou & Zhang 2007, CVPR | Spectral-residual saliency | Saliency map in the salient-edge baseline |
| Rassia & Pezaris 2018, *Sci Rep* 8:16310 | Simulated phosphene reading improves with training | Supports using a learned classifier as an "adapted observer" proxy |
| Wang et al. 2004, *IEEE TIP* | SSIM | Reconstruction metric |
| Cohen et al. 2017, IJCNN (arXiv:1702.05373) | EMNIST letters | Letter-recognition benchmark |

## 3. Metrics used in the field
- **Image similarity between percept and input:** SSIM and PSNR (de Ruyter van Steveninck 2022; Granley 2022), often alongside the decoder's reconstruction. We also report a *low-pass SSIM*, computed after blurring both images with σ = 0.375°. Phosphene vision cannot carry detail finer than the phosphene spacing, and raw SSIM on dotted percepts mostly measures dot texture.
- **Task performance:** letter or object recognition, normally by sighted observers viewing simulated prosthetic vision. As a proxy we train a CNN on each encoder's percepts, i.e. an observer that learned that encoding.
- **Stimulation cost:** active electrodes, total charge per phase, and maximum current.
- **Robustness:** performance under electrode failure.

## 4. What we adopt and what we add
1. **Simulator.** dynaphos physics re-implemented. We add:
   - an exact closed form for one 166 ms train that includes the within-train habituation trace;
   - a renderer that is separable (exact) or uses a two-anchor Gaussian basis (fast, for training);
   - patient sampling with dead electrodes (10 %), threshold spread and 0.4 mm cortical position jitter (the dynaphos defaults);
   - Utah-like and thread-like layouts;
   - validation figures against Horton & Hoyt, Fernández Fig. 6A, and Fernández's thresholds. Our thresholds are lower than Fernández's mean, and we report this mismatch.
2. **Encoders.**
   - Baselines: intensity sampling (uncalibrated and threshold-calibrated), Sobel edges, saliency-gated edges, and a non-causal per-image optimisation reference.
   - Proposed: a small fully convolutional encoder with a retinotopic read-out at each phosphene location, so it is position-aware and its parameter count is independent of N. A per-electrode gain/bias adapter is fine-tuned on a patient's calibrated forward model from a few unlabeled images.
3. **Evaluation.**
   - Five held-out simulated patients.
   - Metrics: SSIM, low-pass SSIM, PSNR, recognition accuracy, charge, and robustness to unannounced electrode failures.
   - Ablations: loss terms, safety limits, and electrode count/layout (Utah 96, 16×Utah 1024, threads 64/256/1024).

## 5. References (verified)
- Barrese JC, Rao N, Paroo K, et al. Failure mode analysis of silicon-based intracortical microelectrode arrays in non-human primates. *J Neural Eng* 10(6):066014, 2013. doi:10.1088/1741-2560/10/6/066014
- Beauchamp MS, Oswalt D, Sun P, et al. Dynamic stimulation of visual cortex produces form vision in sighted and blind humans. *Cell* 181(4):774–783.e5, 2020. doi:10.1016/j.cell.2020.04.033
- Bengio Y, Léonard N, Courville A. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv:1308.3432, 2013.
- Benson NC, Butt OH, Datta R, et al. The retinotopic organization of striate cortex is well predicted by surface topology. *Curr Biol* 22(21):2081–2085, 2012. doi:10.1016/j.cub.2012.09.014
- Benson NC, Butt OH, Brainard DH, Aguirre GK. Correction of distortion in flattened representations of the cortical surface allows prediction of V1–V3 functional organization from anatomy. *PLoS Comput Biol* 10(3):e1003538, 2014. doi:10.1371/journal.pcbi.1003538
- Beyeler M, Boynton GM, Fine I, Rokem A. pulse2percept: a Python-based simulation framework for bionic vision. *Proc. 16th Python in Science Conf.*, 81–88, 2017. doi:10.25080/shinma-7f4c6e7-00c
- Beyeler M, Nanduri D, Weiland JD, et al. A model of ganglion axon pathways accounts for percepts elicited by retinal implants. *Sci Rep* 9:9199, 2019. doi:10.1038/s41598-019-45416-4
- Bosking WH, Sun P, Ozker M, et al. Saturation in phosphene size with increasing current levels delivered to human visual cortex. *J Neurosci* 37(30):7188–7197, 2017. doi:10.1523/JNEUROSCI.2896-16.2017
- Chen X, Wang F, Fernández E, Roelfsema PR. Shape perception via a high-channel-count neuroprosthesis in monkey visual cortex. *Science* 370(6521):1191–1196, 2020. doi:10.1126/science.abd7435
- Cogan SF. Neural stimulation and recording electrodes. *Annu Rev Biomed Eng* 10:275–309, 2008. doi:10.1146/annurev.bioeng.10.061807.160518
- Cohen G, Afshar S, Tapson J, van Schaik A. EMNIST: extending MNIST to handwritten letters. *IJCNN* 2017, 2921–2926. doi:10.1109/IJCNN.2017.7966217
- de Ruyter van Steveninck J, Güçlü U, van Wezel R, van Gerven M. End-to-end optimization of prosthetic vision. *J Vis* 22(2):20, 2022. doi:10.1167/jov.22.2.20
- Fernández E, Alfaro A, Soto-Sánchez C, et al. Visual percepts evoked with an intracortical 96-channel microelectrode array inserted in human occipital cortex. *J Clin Invest* 131(23):e151331, 2021. doi:10.1172/JCI151331
- Granley J, Relic L, Beyeler M. Hybrid neural autoencoders for stimulus encoding in visual and other sensory neuroprostheses. *NeurIPS* 35, 2022. arXiv:2205.13623
- Granley J, Fauvel T, Chalk M, Beyeler M. Human-in-the-loop optimization for deep stimulus encoding in visual prostheses. *NeurIPS* 36, 2023. arXiv:2306.13104
- Han N, Srivastava S, Xu A, Klein D, Beyeler M. Deep learning–based scene simplification for bionic vision. *Augmented Humans '21*, 45–54, 2021. doi:10.1145/3458709.3458982
- Histed MH, Bonin V, Reid RC. Direct activation of sparse, distributed populations of cortical neurons by electrical microstimulation. *Neuron* 63(4):508–522, 2009. doi:10.1016/j.neuron.2009.07.016
- Horton JC, Hoyt WF. The representation of the visual field in human striate cortex. *Arch Ophthalmol* 109(6):816–824, 1991. doi:10.1001/archopht.1991.01080060080030
- Hou X, Zhang L. Saliency detection: a spectral residual approach. *CVPR* 2007, 1–8. doi:10.1109/CVPR.2007.383267
- Küçükoğlu B, Rueckauer B, Ahmad N, et al. Optimization of neuroprosthetic vision via end-to-end deep reinforcement learning. *Int J Neural Syst* 32(11):2250052, 2022. doi:10.1142/S0129065722500526
- Maynard EM, Nordhausen CT, Normann RA. The Utah intracortical electrode array: a recording structure for potential brain-computer interfaces. *Electroencephalogr Clin Neurophysiol* 102(3):228–239, 1997. doi:10.1016/S0013-4694(96)95176-0
- Mitchell DP. Spectrally optimal sampling for distribution ray tracing. *SIGGRAPH Comput Graph* 25(4):157–164, 1991. doi:10.1145/127719.122736 (best-candidate sampling used for thread layouts)
- Musk E, Neuralink. An integrated brain-machine interface platform with thousands of channels. *J Med Internet Res* 21(10):e16194, 2019. doi:10.2196/16194
- Perez E, Strub F, de Vries H, Dumoulin V, Courville A. FiLM: visual reasoning with a general conditioning layer. *AAAI* 32(1), 2018. doi:10.1609/aaai.v32i1.11671
- Polimeni JR, Balasubramanian M, Schwartz EL. Multi-area visuotopic map complexes in macaque striate and extra-striate cortex. *Vision Res* 46(20):3336–3359, 2006. doi:10.1016/j.visres.2006.03.006
- Rassia KEK, Pezaris JS. Improvement in reading performance through training with simulated thalamic visual prostheses. *Sci Rep* 8:16310, 2018. doi:10.1038/s41598-018-31435-0
- Schmidt EM, Bak MJ, Hambrecht FT, et al. Feasibility of a visual prosthesis for the blind based on intracortical microstimulation of the visual cortex. *Brain* 119(2):507–522, 1996. doi:10.1093/brain/119.2.507
- Schwartz EL. Spatial mapping in the primate sensory projection: analytic structure and relevance to perception. *Biol Cybern* 25(4):181–194, 1977. doi:10.1007/BF01885636
- Shannon RV. A model of safe levels for electrical stimulation. *IEEE Trans Biomed Eng* 39(4):424–426, 1992. doi:10.1109/10.126616
- Srivastava N, Hinton G, Krizhevsky A, Sutskever I, Salakhutdinov R. Dropout: a simple way to prevent neural networks from overfitting. *J Mach Learn Res* 15:1929–1958, 2014.
- Tehovnik EJ. Electrical stimulation of neural tissue to evoke behavioral responses. *J Neurosci Methods* 65(1):1–17, 1996. doi:10.1016/0165-0270(95)00131-X
- Tehovnik EJ, Tolias AS, Sultan F, Slocum WM, Logothetis NK. Direct and indirect activation of cortical neurons by electrical microstimulation. *J Neurophysiol* 96(2):512–521, 2006. doi:10.1152/jn.00126.2006
- Tehovnik EJ, Slocum WM. Phosphene induction by microstimulation of macaque V1. *Brain Res Rev* 53(2):337–343, 2007. doi:10.1016/j.brainresrev.2006.11.001
- van der Grinten M, de Ruyter van Steveninck J, Lozano A, et al. Towards biologically plausible phosphene simulation for the differentiable optimization of visual cortical prostheses. *eLife* 13:e85812, 2024. doi:10.7554/eLife.85812
- Wang Z, Bovik AC, Sheikh HR, Simoncelli EP. Image quality assessment: from error visibility to structural similarity. *IEEE Trans Image Process* 13(4):600–612, 2004. doi:10.1109/TIP.2003.819861
- Werbos PJ. Backpropagation through time: what it does and how to do it. *Proc IEEE* 78(10):1550–1560, 1990. doi:10.1109/5.58337
- Winawer J, Parvizi J. Linking electrical stimulation of human primary visual cortex, size of affected cortical area, neuronal responses, and subjective experience. *Neuron* 92(6):1213–1219, 2016. doi:10.1016/j.neuron.2016.11.008

Data: the brightness-vs-current points for Fernández et al. 2021 Fig. 6A were read off the figure by the dynaphos authors and published as `test/data/Fernandez_2021_fig6A.csv` in github.com/neuralcodinglab/dynaphos. We use them only as validation targets.
