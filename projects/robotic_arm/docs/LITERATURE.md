# Assisted Robotic Arm: literature and domain notes

These notes collect the domain knowledge behind this project: high-degree-of-freedom
intracortical decoding plus shared autonomy for an assistive robotic arm. Each reference
below was checked against Crossref, PubMed, arXiv or bioRxiv, the publisher's page, or
the dataset and registry records (DANDI, ClinicalTrials.gov) on 2026-10-06. Where only
an abstract was read, the entry says so. The BibTeX is in `paper/refs.bib`.

## 1. The product problem

People with cervical spinal cord injury or brainstem stroke lose arm and hand function.
An intracortical BCI (iBCI) records spiking activity from motor cortex and decodes the
movement the person is attempting. Those decoded velocities drive a robotic arm mounted
on a wheelchair or table, so the person can reach for, grasp and move everyday objects.

Neuralink's **CONVOY** study (ClinicalTrials.gov NCT06710626; started 2024-11-25,
enrolling by invitation, 3 participants estimated) has people already implanted in the
PRIME study use the N1 implant to control an "Assistive Robotic Arm". Its primary
outcome is the ability to control an assistive device through the N1 at 3 months. We
found no peer-reviewed robotic-arm results from Neuralink, and nothing public names the
arm model. This project is independent and not affiliated with Neuralink.

Three facts from the clinical literature shape the design:

1. **High-DoF continuous control is possible, but noisy.** BrainGate2 (Hochberg et al.
   2012, *Nature*) decoded 3-D endpoint velocity with a Kalman filter, plus a discrete
   grasp state from an LDA classifier. One participant touched the target on 48.8% (DLR
   arm) and 69.2% (DEKA arm) of trials; the other touched on 95.6%. The Pitt group
   (Collinger et al. 2013, *Lancet*) reached 7 DoF: 3-D translation, 3-D orientation
   and grasp, decoded with an optimal linear estimator and two-step calibration. Success
   on target tasks was 91.6%, but path efficiency was only about 0.38. Wodlinger et al.
   (2015, *J Neural Eng*) extended this to 10 DoF and found that most units were tuned
   to all dimensions and that calibration context matters.
2. **Grasping is the bottleneck.** Flesher et al. (2021, *Science*) halved ARAT trial
   times, from a median of 20.9 s to 10.2 s, by adding ICMS tactile feedback; the gain
   came mainly from time spent attempting grasps. Downey et al. (2016, *J NeuroEng
   Rehabil*) showed that computer-vision-guided **shared control** near the object
   raised ARAT success from 22% to 78% for subject 1 and from 0% to 46% for subject 2.
   On a multi-object selection task, success went from 46% to 92%, with median time
   falling from 26.3 s to 8.3 s.
3. **Neural recordings drift across days.** See Downey et al. 2018 (*J Neural Eng*),
   Jarosiewicz et al. 2015 (*Sci Transl Med*), Degenhart et al. 2020 (*Nat Biomed Eng*)
   and Karpowicz et al. 2025 (NoMAD, *Nat Commun*). A decoder fit on one day loses
   accuracy on later days, so daily recalibration time is a real product cost.

## 2. Data: FALCON H1

**FALCON** (Karpowicz et al., NeurIPS 2024 Datasets & Benchmarks; bioRxiv
10.1101/2024.09.15.613126) is a benchmark for few-shot, cross-session iBCI decoding. Its
**H1** task is on DANDI as dandiset 000954 (~102 MB):

- **Participant and recordings:** a participant in the Pitt sensorimotor iBCI trial
  (NCT01894802) with Utah arrays in motor cortex. Threshold crossings are binned at
  20 ms, and the NWB files contain 176 channels. The paper's figure says 172 channels;
  we use the shape of the data.
- **Task:** open-loop attempted reach-and-grasp. The participant follows a virtual arm,
  so the labels are *cued* (intended) velocities.
- **Outputs (7 DoF):** `tx, ty, tz` translation, `rx` wrist roll, and three grasp
  dimensions `g1..g3` (pinch, scoop, thumb abduction).
- **Sessions:** 6 held-in days with 5-9 min each, and 7 held-out days spanning 15 days,
  with 1.5-1.8 min of released calibration data each.
- **Metric:** per-session `r2_score(..., multioutput="variance_weighted")` over bins
  where `eval_mask` is true, averaged across sessions. Variance weighting means the
  high-variance grasp dimensions dominate the score. We therefore also report per-DoF R².
- **Reported baselines** (paper Table 1 and the EvalAI leaderboard; held-out / held-in):

  | Method | Held-out | Held-in |
  |---|---|---|
  | Wiener filter, static | 0.16 | 0.20 |
  | Wiener filter, oracle | 0.21 | 0.24 |
  | RNN, zero-shot | 0.09 | 0.31 |
  | RNN, oracle | 0.44 | 0.51 |
  | NDT2 Multi, few-shot supervised | 0.52 | 0.62 |
  | NDT2 Multi, oracle | 0.63 | 0.68 |

  The repo's sklearn example should report a held-in minival mean of 0.195.
- **Overlap in the public files, found in this project:** the first bins of every
  public held-in *calibration* file are exactly the matching *minival* file. Minival
  must therefore be removed from the calibration data before training. If it is not,
  minival scores are inflated: a GRU reached 0.80 here versus 0.63 after removal.
- **Splits not available to us:** FALCON keeps the evaluation split (the last 20% of
  each held-out file) private. We build our own cross-session protocol from the public
  files (see `DESIGN.md`), so our held-out numbers are **not** directly comparable to
  the leaderboard. Our held-in numbers use FALCON's public minival split.

## 3. Decoders

| Paper | Contribution | Use here |
|---|---|---|
| Wu et al. 2006, *Neural Comput*; Kim et al. 2008, *J Neural Eng* | Velocity Kalman filter. Velocity decoding beats position decoding in closed loop. | Kalman baseline |
| Gilja et al. 2012, *Nat Neurosci* (ReFIT) | Closed-loop recalibration with intention relabelling; halved acquisition time. | Discussed (needs closed-loop data) |
| Wiener filter (FALCON baseline) | Ridge regression on 30 bins (600 ms) of spike history. | Wiener baseline; lags chosen on validation |
| Sussillo et al. 2012, *J Neural Eng*; Sussillo et al. 2016, *Nat Commun* | RNN decoders. Training on many days plus synthetic perturbations (channel dropout, rate changes) makes them robust to future variability. | GRU baseline; augmentation in the proposed model |
| Pandarinath et al. 2018, *Nat Methods* (LFADS) | Session "stitching" through per-session read-in matrices onto shared latent dynamics. | Per-day read-ins in the proposed model |
| Ye & Pandarinath 2021, *NBDT* (NDT); Ye et al. 2023, NeurIPS (NDT2); Azabou et al. 2023, NeurIPS (POYO); Ye et al. 2025, NeurIPS (NDT3) | Transformer decoders pretrained across sessions, adapted few-shot through context tokens or unit embeddings. NDT2 is the strongest published FALCON H1 baseline. | Causal-Transformer ablation; "re-learn only the input layer" adaptation |
| Deo et al. 2024, *Sci Rep* | RNN decoders for bimanual control; time-warp augmentation. | Background |
| Degenhart et al. 2020; Karpowicz et al. 2025 (NoMAD); Wilson et al. 2025, *Nat Biomed Eng*; Fan et al. 2023, NeurIPS (CORP) | Unsupervised and pseudo-label recalibration. | Unsupervised per-day normalisation baseline; discussed |

## 4. Shared autonomy

The formalism we follow is **policy blending** (Dragan & Srinivasa 2013, *IJRR*):
`u = (1 − α) u_user + α u_robot`. The weight α is a monotone function of the robot's
confidence in its goal prediction, which comes from maximum-entropy inverse optimal
control: `P(ξ | g) ∝ exp(−C_g(ξ))`. Their user study found that aggressive assistance
helps when the prediction is right, and timid assistance is clearly better when it is
wrong (t(15) = 7.09, p < .001).

- **BCI-specific blending:**
  - Downey et al. 2016 use `C = (1 − α) R + α B`. The BMI weight is
    `α = (1 + e^{−aD + o})^{−1}` with a = 11.6 and o = 6.9, where D is the scaled
    distance to a vision-computed grasp pose. The robot takes over only inside the
    grasp envelope.
  - Muelling et al. 2017 (*Auton Robots*; RSS 2015, arXiv:1503.05451) is the full
    autonomy-infused teleoperation system: MaxEnt goal inference over detected objects
    plus sigmoid arbitration, with a safety threshold that lets the user break away.
- **Hindsight optimisation:** Javdani, Srinivasa & Bagnell 2015 (RSS; arXiv:1503.07619)
  and Javdani et al. 2018 (*IJRR*). Shared autonomy is cast as a POMDP over the unknown
  goal and solved with QMDP: `Q(b, a, u) = Σ_g b(g) Q_g(x, a, u)`. The robot assists
  toward what is common across likely goals before it is confident. Users were faster
  and gave less input, but did not clearly prefer it, which reflects a
  control-authority trade-off.
- **Learned copilots:**
  - Reddy, Dragan & Levine 2018 (RSS) learn a model-free deep-RL copilot and filter
    actions with `Q'(s, a) ≥ (1 − α) Q'(s, a*)`.
  - Lee et al. 2025 (*Nat Mach Intell*) use AI copilots with a non-invasive BCI. They
    made robotic-arm pick-and-place possible for a participant with paralysis.
- **Latent actions:** Losey et al. 2020 (ICRA), Jeon et al. 2020 (RSS) and Losey et al.
  2022 (*Auton Robots*). Relevant when the decoder has fewer DoF than the robot.
- **Acceptance and personalisation:** Gopinath, Jain & Argall 2017 (*RA-L*); Brooks &
  Szafir 2020 (IROS); Jain & Argall 2019 (*THRI*).
- **Data on human goal-directed manipulation:** HARMONIC (Newman et al. 2022, *IJRR*).

## 5. Simulating the closed loop

Offline R² does not equal closed-loop performance, because the user corrects errors
through visual feedback. Cunningham et al. 2011 (*J Neurophysiol* 105:1932-1949)
introduced closed-loop simulators for BMI design. Willett et al. 2017 (*J Neural Eng*
14:016001) modelled the feedback-control policies that iBCI users actually employ. Our
simulated user is a delayed feedback controller (200 ms visuomotor delay plus motor
noise). It drives an encoding model fitted to FALCON H1, and the resulting synthetic
spikes are decoded by a decoder calibrated exactly as in the clinic: an open-loop block
in which the cued kinematics are replayed.

## 6. The arm

- **Kinematics:** we use the classical DH table of the 7-DoF **Kinova Gen3** (Kinova
  Gen3 User Guide R07, Table 94):
  - all `a_i = 0` and `α_i = π/2`;
  - `d = [−0.2848, −0.0118, −0.4208, −0.0128, −0.3143, 0, −0.1674]` m;
  - θ offsets of π on joints 2-7;
  - a base transform with `α = π`.

  To this we add a 0.12 m gripper TCP offset, which is our assumption. Kinova Gen2 arms
  (JACO) are common on wheelchairs.
- **Usage restriction:** the Gen3 user guide itself states that the robot should not be
  used as an assistance robot in clinical trials. We use it only as a realistic
  kinematic model.

## 7. Metrics the field uses

- **Offline decoding:** R² per DoF and FALCON's variance-weighted R², reported within
  session and across sessions.
- **Closed-loop manipulation:**
  - task success rate, both overall and against chance (Collinger 2013);
  - completion time / time to grasp;
  - path efficiency, i.e. straight-line distance divided by path length (Collinger
    2013 reports about 0.30-0.38);
  - ARAT scores and times (Downey 2016; Flesher 2021).
- **Shared-autonomy studies** add:
  - user input effort and the fraction of control held by the user (Javdani 2015);
  - goal-prediction accuracy over time;
  - performance when the prediction is wrong (Dragan 2013).

## 8. Ideas adopted, and what is ours

- **Adopted:**
  - FALCON's protocol and metric;
  - Wiener and Kalman baselines;
  - GRU decoders with nonstationarity augmentation (Sussillo 2016);
  - per-day read-in "stitching" (LFADS, NDT2, POYO) with few-shot read-in-only
    recalibration;
  - MaxEnt goal inference;
  - Dragan-style confidence blending;
  - Downey-style distance blending (their exact constants);
  - Javdani-style hindsight assistance.
- **Proposed here: Decoder-Aware Shared Autonomy (DASA).** It has three parts:
  1. A goal likelihood based on the decoder's calibrated error model,
     `ẑ = G z*_g + e`, `e ~ N(0, Σ)`. G and Σ are estimated from the same open-loop
     calibration block used to fit the decoder. The log-likelihood is divided by the
     decoder error's integrated autocorrelation time, so that temporally correlated
     decoder errors are not counted as independent evidence.
  2. A goal-switching (HMM) prior, so the user can change their mind.
  3. An agreement gate that withdraws assistance when the user persistently pushes
     against it.

  The ingredients exist separately in the literature: HMM target inference (Wilson et
  al. 2025), Downey's and Muelling's safety threshold, and Dragan's "timid when wrong".
  What we contribute is combining them, and calibrating the inference against the
  decoder's actual error statistics.
