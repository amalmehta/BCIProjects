# Telepathy: literature and domain notes

Scope: decoding 2D cursor velocity from intracortical spiking for a person with paralysis,
keeping the decoder accurate across days without daily recalibration, and measuring what
matters to the user: closed-loop throughput in bits per second on a grid-selection task.

All references below were checked against the publisher, arXiv/bioRxiv, PubMed Central or
the conference proceedings page. Neuralink is cited only through its own public pages; this
project is independent and not affiliated with or endorsed by Neuralink.

## 1. Domain knowledge

**Signal.** Utah arrays (96 electrodes) and Neuralink's threads both record extracellular
voltage. For decoding, the standard features are per-electrode *threshold crossings* (a spike
is counted whenever the high-pass filtered voltage drops below roughly -4.5 x RMS) or
*spike-band power*, binned at 10-50 ms. Spike sorting is usually skipped in clinical
systems: it is fragile across days and adds little decoding accuracy. We therefore merge all
sorted units and the "hash" unit of each electrode into one multi-unit count per channel.

**What is decoded.** Cursor BCIs decode *velocity* (not position): motor cortex is better
tuned to velocity/direction, and a velocity command lets the user correct errors through
visual feedback. Neural activity leads hand movement by ~100-150 ms, so offline decoders map
spikes at time t to kinematics at t + lag.

**Closed loop differs from open loop.** A decoder that reconstructs arm velocity best offline
is not necessarily the best online, because the user sees the cursor and corrects in real time
(Gilja 2012; Willett 2019). Two consequences matter for this project:
(i) *intention estimation*: ReFIT re-labels closed-loop data by rotating decoded velocity
toward the target, since that is what the user intended; (ii) offline R^2 must be backed up
by a closed-loop measure; we use a simulated user (Cunningham 2011; Willett 2019) and say
plainly that it is a simulator.

**Nonstationarity.** Recorded units appear, vanish and change waveform across days because of
micromotion, gliosis, electrode failure and (for Neuralink's first participant) thread
retraction. The population's low-dimensional dynamics stay stable even as individual
channels change (Gallego 2020). So you can stabilize a decoder by learning a
per-day mapping from channels into a fixed latent space and leaving the latent-to-behavior
readout fixed (Degenhart 2020; Farshchian 2019; Karpowicz NoMAD 2022/2025). The cheapest
practical fix is re-centering and re-scaling each channel's statistics (Jarosiewicz 2015).

**Throughput metric.** Grid tasks report the achieved bitrate
`B = log2(N - 1) * max(S_c - S_i, 0) / t`, with N selectable targets, S_c / S_i correct /
incorrect selections in time t (Nuyujukian 2015; Pandarinath 2017). The "N - 1" and the
subtraction of errors model a keyboard with a delete key: every error costs one extra
selection to undo. Neuralink's public "Webgrid" game uses this score (bits per second, BPS),
computed from net correct targets per minute and grid size.

**Field numbers to calibrate against.** BrainGate participants reached 1.4-3.7 bps on a grid
task (Pandarinath 2017). Neuralink reported its first PRIME participant at 8.0 BPS within months
(PRIME Study Progress Update, 2024) and, in October 2026, a 11.32 BPS record with participant
medians around 10 BPS after pretraining on large neural datasets (reported by secondary press
covering Neuralink's 1 Oct 2026 update; we did not have access to the primary post's text).
Able-bodied mouse users score about 10 BPS on Webgrid.

## 2. Key papers and what each contributes

| Paper | Contribution | Used here |
|---|---|---|
| Wu, Gao, Bienenstock, Donoghue, Black. *Bayesian population decoding of motor cortical activity using a Kalman filter.* Neural Computation 18(1), 2006. | Kalman filter (KF) with linear-Gaussian observation model of firing rates given kinematics. | KF baseline (velocity state). |
| Gilja, Nuyujukian, Chestek, Cunningham, Yu, Fan, Churchland, Kaufman, Kao, Ryu, Shenoy. *A high-performance neural prosthesis enabled by control algorithm design.* Nature Neuroscience 15(12):1752-1757, 2012. | ReFIT-KF: refit on closed-loop data after rotating velocities toward the target; position feedback treated as known. Doubled throughput. | ReFIT-style intention relabelling in the closed-loop simulator. |
| Nuyujukian, Fan, Kao, Ryu, Shenoy. *A high-performance keyboard neural prosthesis enabled by task optimization.* IEEE TBME 62(1):21-29, 2015. | Defines the achieved bitrate on grid keyboards; shows task design (grid size, dwell) changes throughput. | Bitrate definition; grid-size sweep. |
| Pandarinath, Nuyujukian, Blabe, Sorice, Saab, Willett, Hochberg, Shenoy, Henderson. *High performance communication by people with paralysis using an intracortical brain-computer interface.* eLife 6:e18554, 2017. | Human grid-task bitrates of 1.4-3.7 bps with ReFIT-style decoders. | Field reference numbers, task design. |
| Jarosiewicz et al. *Virtual typing by people with tetraplegia using a self-calibrating intracortical BCI.* Sci. Transl. Med. 7(313):313ra179, 2015. | Tracks neural feature means during use, corrects velocity bias, recalibrates on task data. | "Re-centering" (per-channel mean/std renormalization) baseline. |
| Sussillo, Stavisky, Kao, Ryu, Shenoy. *Making brain-machine interfaces robust to future neural variability.* Nature Communications 7:13749, 2016. | Multiplicative RNN trained on months of data plus synthetic perturbations (channel dropping, firing-rate changes) is robust to future recording changes. | Training-time augmentation (channel dropout, gain/offset jitter); multi-day training. |
| Cunningham, Nuyujukian, Gilja, Chestek, Ryu, Shenoy. *A closed-loop human simulator for investigating the role of feedback control in brain-machine interfaces.* J. Neurophysiology 105(4):1932-1949, 2011. | Online prosthesis simulator: closed-loop evaluation reveals decoder differences hidden offline. | Motivation for closed-loop simulator. |
| Willett et al. *Principled BCI decoder design and parameter selection using a feedback control model.* Scientific Reports 9:8881, 2019. | Simulated user with feedback-control model predicts online performance and tunes gain/smoothing. | User model with visual delay + feedback controller; per-decoder gain tuning in simulation. |
| Gallego, Perich, Chowdhury, Solla, Miller. *Long-term stability of cortical population dynamics underlying consistent behavior.* Nature Neuroscience 23:260-270, 2020. | Latent dynamics are stable over ~2 years despite unit turnover; aligned (CCA) latents allow a fixed decoder. | Rationale for latent alignment. |
| Degenhart, Bishop, Oby, Tyler-Kabara, Chase, Batista, Yu. *Stabilization of a brain-computer interface via the alignment of low-dimensional spaces of neural activity.* Nature Biomedical Engineering 4:672-685, 2020. | Unsupervised stabilizer: factor analysis per day, identify stable channels, Procrustes-align loadings; fixed latent decoder. | Implemented as the classical unsupervised alignment baseline. |
| Farshchian, Gallego, Cohen, Bengio, Miller, Solla. *Adversarial domain adaptation for stable brain-machine interfaces.* ICLR 2019 (arXiv:1810.00045). | Adversarial alignment of new-day inputs to a fixed autoencoder latent space. | Related work (distribution alignment). |
| Karpowicz, Ali, Wimalasena, Sedler, Keshtkaran, Bodkin, Ma, Rubin, Williams, Cash, Hochberg, Miller, Pandarinath. *Stabilizing brain-computer interfaces through alignment of latent dynamics* (NoMAD). bioRxiv 10.1101/2022.04.06.487388 (2022); Nature Communications 16 (2025), doi:10.1038/s41467-025-59652-y. | Freeze a dynamics model (LFADS); learn an alignment network for new days by maximizing spike likelihood under the frozen dynamics plus a KL term matching latent distributions. | Core of our proposed unsupervised alignment (likelihood + latent-moment KL with frozen backbone). |
| Karpowicz, Ye, Fan, Tostado-Marcos, Rizzoglio, Washington, Scodeler, de Lucena, Nason-Tomaszewski, Mender, Ma, Arneodo, Hochberg, Chestek, Henderson, Gentner, Gilja, Miller, Rouse, Gaunt, Collinger, Pandarinath. *Few-shot Algorithms for Consistent Neural Decoding (FALCON) Benchmark.* NeurIPS 2024 Datasets & Benchmarks (bioRxiv 10.1101/2024.09.15.613126). | Standard held-in/held-out multi-session protocol; zero-shot and few-shot recalibration evaluated by R^2 on later sessions. | Our evaluation protocol (held-in calibration days, held-out later days, few-shot recalibration curves). |
| Pei, Ye, Zoltowski, Wu, Chowdhury, Sohn, O'Doherty, Shenoy, Kaufman, Churchland, Jazayeri, Miller, Pillow, Park, Dyer, Pandarinath. *Neural Latents Benchmark '21.* NeurIPS 2021 Datasets & Benchmarks (arXiv:2109.04463). | MC_RTT: one session of the O'Doherty random-target data used as a benchmark. | Same task and monkey (Indy); we use 24 sessions instead of one. |
| O'Doherty, Cardoso, Makin, Sabes. *Nonhuman primate reaching with multichannel sensorimotor cortex electrophysiology.* Zenodo, 2017, doi:10.5281/zenodo.583331. | Continuous random-target cursor reaching, monkey Indy, 37 sessions over ~10 months, 96-ch M1 Utah array. | **Our dataset.** |
| Makin, O'Doherty, Cardoso, Sabes. *Superior arm-movement decoding from cortex with a new, unsupervised-learning algorithm.* J. Neural Engineering 15(2):026010, 2018. | Introduced the dataset; KF vs latent-state decoders on it. | Dataset context; KF baseline. |
| Pandarinath, O'Shea, Collins, Jozefowicz, Stavisky, Kao, Trautmann, Kaufman, Ryu, Hochberg, Henderson, Shenoy, Abbott, Sussillo. *Inferring single-trial neural population dynamics using sequential auto-encoders* (LFADS). Nature Methods 15:805-815, 2018. | Multi-session "stitching" with per-session read-in matrices into a shared dynamical system. | Per-day read-in layers in our model. |
| Ye, Pandarinath. *Representation learning for neural population activity with Neural Data Transformers.* Neurons, Behavior, Data analysis, and Theory, 2021 (arXiv:2108.01210). | Transformer with masked spike modelling; fast non-recurrent inference. | Masked-spike auxiliary objective; transformer backbone. |
| Ye, Collinger, Wehbe, Gaunt. *Neural Data Transformer 2: Multi-context pretraining for neural spiking activity.* NeurIPS 2023. | Pretraining across sessions/subjects, rapid adaptation to new contexts. | Multi-day pretraining + few-shot adaptation framing. |
| Ye et al. *A generalist intracortical motor decoder* (NDT3). NeurIPS 2025 (bioRxiv 10.1101/2025.02.02.634313). | 2000 h pretraining; gains when downstream data < 1.5 h; poor generalization outside fine-tuned inputs/outputs. | Motivation; limits of scale at CPU size. |
| Azabou, Arora, Ganesh, Mao, Nachimuthu, Mendelson, Richards, Perich, Lajoie, Dyer. *A unified, scalable framework for neural population decoding* (POYO). NeurIPS 2023 (arXiv:2310.16046). | Spike tokenization + PerceiverIO; few-shot adaptation to new sessions by learning only unit embeddings. | "Read-in only" few-shot fine-tuning (we learn only the per-day input layer). |
| Willsey, Nason-Tomaszewski, Ensel, Temmar, Mender, Costello, Patil, Chestek. *Real-time BMI in non-human primates achieves high-velocity prosthetic finger movements using a shallow feedforward neural network decoder.* Nature Communications 13:6899, 2022. | Neural-network decoders beat ReFIT-KF online (+36% throughput). | Evidence that nonlinear decoders matter online. |
| Fan, Hahn, Kamdar, Avansino, Wilson, Hochberg, Shenoy, Henderson, Willett. *Plug-and-play stability for intracortical BCIs: a one-year demonstration of seamless brain-to-text communication* (CORP). NeurIPS 2023 (arXiv:2311.03611). | Continual self-recalibration with pseudo-labels for >1 year. | Related work on label-free recalibration. |
| Neuralink. *PRIME Study Progress Update* (2024) and the *Webgrid* page (neuralink.com/webgrid). | Product context: Webgrid BPS, thread retraction, decoder changes that restored performance. | Product framing and metric. |

## 3. Metrics we use

* **Offline velocity R^2** (coefficient of determination averaged over x and y), the FALCON /
  NLB standard; computed on held-out trials, never on training data.
* **Cross-day R^2** on held-out later sessions as a function of days since calibration, and
  **few-shot recalibration curves** (R^2 vs. number of labelled reaches: 0, 1, 5, 20, 50).
* **Robustness**: R^2 vs. fraction of dropped (silenced) electrodes and under channel shuffles.
* **Closed-loop simulated bitrate** (bits/s) with the formula above, plus time-to-target and
  error rate. Always labelled as simulated.
* **Latency**: per-bin inference time on CPU.

## 4. Ideas adopted

1. Sorting-free per-electrode threshold-crossing counts at 20 ms (clinical practice).
2. Decode intended velocity at a fixed neural lead, chosen on calibration data only.
3. Strong classical baselines: ridge Wiener filter with 200 ms history, velocity Kalman filter,
   GRU; all trained on the same data and selected on validation splits.
4. Proposed decoder: per-day linear read-in -> shared causal transformer backbone (latent
   dynamics) -> velocity head, trained jointly on several calibration days with masked spike
   reconstruction (NDT) and Sussillo-style input perturbations (channel dropout, gain/offset
   jitter).
5. Cross-day stabilization without labels: NoMAD-style read-in alignment with a frozen
   backbone (spike likelihood + Gaussian KL between latent distributions), compared with
   Degenhart's FA-Procrustes stabilizer and Jarosiewicz-style per-channel re-centering.
6. Few-shot recalibration: read-in-only fine-tuning (POYO/NDT2 style) vs. full retraining.
7. Closed-loop WebGrid simulator: delayed feedback-control user model, data-fit neural
   encoding model with calibrated noise, dwell or click selection, ReFIT-style recalibration.
