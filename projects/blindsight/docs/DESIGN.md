# Blindsight: system design

Blindsight is a research codebase for cortical visual prostheses. It has three parts:

- a differentiable simulator of the phosphenes produced by V1 microstimulation;
- stimulus encoders that turn camera images into per-electrode currents;
- an evaluation harness that compares those encoders on simulated patients.

No public human V1 stimulation dataset exists, so every result comes from simulation. The simulator is grounded in published psychophysics, and the ways it falls short are listed at the end of this document. This is an independent project, not affiliated with or endorsed by Neuralink.

## Architecture

```mermaid
flowchart LR
  subgraph Data
    D1[MNIST / EMNIST letters / CIFAR-10<br/>torchvision -> data/blindsight/*.pt]
  end
  subgraph Simulator["Simulator (blindsight/simulator.py)"]
    L[Layout<br/>utah_96, utah_1024, threads_N] --> M[Wedge-dipole map<br/>cortex mm -> visual field deg]
    P[Patient sampler<br/>dead electrodes, thresholds<br/>Fernandez 2021 fit, jitter] --> R
    M --> R[Renderer<br/>I_eff -> activation -> brightness, size -> Gaussians]
  end
  subgraph Encoders["Encoders (blindsight/encoders.py)"]
    B[Baselines<br/>intensity, edges, salient edges]
    E[Learned encoder<br/>U-Net + retinotopic read-out + per-electrode adapter]
    TA[Threshold-aware encoder<br/>calibration at the read-out,<br/>supra-threshold output]
    TE[Temporal encoder<br/>+ habituation trace, previous frame]
  end
  P -. calibration .-> TA & TE
  S[Safety layer<br/>clamp to 100 uA, top-k active]
  D1 --> B & E & TA & TE --> S --> R
  R --> Dec[Decoder<br/>training only] --> Loss[Loss: recon + percept + charge]
  R --> Eval[Evaluation<br/>SSIM, low-pass SSIM, PSNR,<br/>percept classifier, dropout]
  Eval --> Out[results/metrics.json, figures, site_export/]
```

## Components

| Module | Job |
|---|---|
| `cortex.py` | Wedge-dipole forward and inverse maps, cortical magnification, Horton & Hoyt reference |
| `layouts.py` | Electrode layouts on the flattened V1 surface: Utah-like grids (0.4 mm pitch) and thread-like best-candidate sites over both hemispheres |
| `simulator.py` | `SimParams` (every constant, two threshold models), `Patient` (per-electrode state), `sample_patient`, `PhospheneSimulator` (static, differentiable, exact and fast renderers; optional activation override and brightness gate), `frame_update` (closed-form state update over one 166 ms frame), `TemporalSimulator` (Euler reference for the dynamics) |
| `encoders.py` | Safety layer, baselines, learned encoder, threshold-aware/temporal encoder (`ThresholdAwareEncoder`, `Condition`), ONNX wrapper with a baked-in patient, known-dead masking, decoder, percept classifier, bilinear read-out |
| `train.py` | End-to-end training through the simulator with domain randomisation over training patients, optional electrode-failure augmentation (announced or unannounced) |
| `temporal.py` | Static and moving image sequences, rollout through the frame dynamics, BPTT fine-tuning of the temporal encoder, per-frame evaluation |
| `evaluate.py` | Rendering on held-out patients, metrics, percept classifier, failure robustness (unannounced and known-dead), sub-threshold budget statistics, patient adaptation, per-image optimisation reference |
| `validation.py` | Checks the simulator against published curves and writes the `fig_sim_*` figures |
| `summarize.py`, `figures.py`, `export.py` | Write `metrics.json`, the result figures, and the website artefacts |
| `run.py` | Command-line entry point (`--stage data|validate|train|temporal|evaluate|ablate|adapt|diagnostics|summarize|figures|export`). Finished runs and evaluations are skipped on re-runs (`--force` recomputes), so `run_cpu.sh` resumes after an interruption. |

## Main flows

1. **Training** (`--stage train`). For each `(dataset, variant, seed)`:
   - sample a batch of images and one training patient (the patient switches every 20 steps, from a pool of 16);
   - compute `a = Safety(E(x))`, then `P = Sim(a, patient; fast, straight-through gate)`, then `x_hat = D(P)`;
   - loss `L = MSE(x_hat, x) + [MSE(P, x) + 0.5 (1 - SSIM(P, x))] + 0.05 mean(a / I_max)`;
   - optimiser: Adam with a cosine learning-rate schedule.
   - Encoder variants (`variants` entries with `_datasets`/`_seeds`) are trained on every dataset: `threshold_aware` (calibration-conditioned, with announced failure augmentation, 3 seeds), `threshold_aware_noaug` and `dropout_aug` (generic encoder with unannounced failure augmentation), 1 seed each.
   - Failure augmentation: each training image, with probability 0.8, loses a fraction p ~ U(0, 0.5) of its electrodes. The threshold-aware encoder is told which (known-dead); the generic one is not.
2. **Temporal encoder** (`--stage temporal`). Starts from the seed-0 threshold-aware encoder, adds two per-electrode inputs (habituation trace, previous amplitude; their weights start at zero, so it starts as the single-frame encoder) and is fine-tuned by backpropagation through time over 8-frame sequences, half static and half moving, after a random 0–8-frame warm-up that sets a habituated starting state. Evaluated on 16-frame (2.7 s) sequences against frame-by-frame encoders and a no-training compensation rule (threshold-aware encoder told threshold + trace).
3. **Evaluation** (`--stage evaluate`). Five held-out patients (seeds 1000–1004).
   - Test images are assigned to patients round-robin and rendered with the exact renderer and the hard threshold.
   - Each encoder gets its own percept classifier, trained on 6k percepts (4 epochs) and tested on 2k.
   - Robustness: we add 10/25/50 % failures and re-evaluate without retraining, either unannounced (the encoder keeps the intact calibration) or known-dead (failed electrodes are zeroed before the safety layer so the top-k budget goes to working ones; conditioned encoders are also told).
   - Budget: the share of active currents at or below the patient's true threshold (they evoke nothing) and the share of charge they use.
   - Calibration error: the threshold-aware encoder is also evaluated when told thresholds with 20 % lognormal error.
   - Adaptation: a copy of the generic encoder is fine-tuned per patient, adapter only, on 256 unlabeled images for 150 steps.
4. **Ablations** (`--stage ablate`). Loss terms, safety limits, and electrode layouts, each trained on MNIST and compared with the intensity baseline on the same layout. Results are cached per variant in `results/eval/ablations/`.
5. **Adaptation curve and diagnostics** (`--stage adapt`, `--stage diagnostics`; the budget numbers now come from every evaluation row, so `diagnostics` is optional).
   - Adaptation curve: SSIM gain against the number of unlabeled images (16–1024), adapter-only against full fine-tuning, on two test patients.
   - Diagnostics: the share of delivered currents that are above each patient's threshold, and the median current.
6. **Export** (`--stage export`). Writes JSON, ONNX, PNG and figures for the website. `export.onnx_variant` picks which seed-0 encoder becomes `encoder.onnx` (CIFAR) and `encoder_letters.onnx`; a threshold-aware encoder is wrapped with the demo patient's (seed 1000) thresholds, alive mask and coverage map as graph constants, so the I/O stays `image` [batch, 1, 64, 64] -> `amplitude_ua` [batch, 1024].

## Where data lives

| What | Where |
|---|---|
| Raw datasets and the uint8 caches | `../../data/blindsight/` (gitignored, about 0.6 GB after removing unused EMNIST splits) |
| Checkpoints | `results/runs/<dataset>_<variant>_s<seed>/{encoder.pt, decoder.pt, train_log.json}` (about 150 kB each); `results/extra_runs/` keeps the collapsed 100 µA-initialised unconstrained run for reference |
| Raw evaluation | `results/eval/*.json`, plus example percepts in `*_examples.pt` |
| Aggregates | `results/metrics.json` (the only source of numbers for the paper) and `results/validation.json` |
| Figures | `results/figures/*.png` |
| Website artefacts | `results/site_export/` (schema in the project report and inside each JSON file) |

## Design decisions and trade-offs

- **Threshold recalibration (round 2).** Fernández et al. (2021) report thresholds of 66.8 ± 36.5 µA (88 of 96 electrodes, range 7–128 µA). The dynaphos fit gives 47 ± 15 µA.
  - New default `threshold_model: fernandez2021`. Threshold *currents* follow a normal truncated below at the rheobase (23.9 µA). Its parameters (µ = −59.6, σ = 82.2 µA) are solved so the truncated mean and sd equal 66.8 and 36.5; activation thresholds follow linearly.
  - We kept the rheobase, brightness sigmoid, size model and habituation constants. So the brightness *function* is unchanged, and so are magnification, size and the habituation curve.
  - Consequences: the minimum threshold is the rheobase (24 µA, not the 7 µA reported), and 17 % of electrodes need more than the 100 µA ceiling.
  - The old brightness check gated the sigmoid at the population-mean threshold. At 66.8 µA that gate removes the 10–60 µA points of Fig. 6A, which averages 10 electrodes. So `validation.json` reports three readings: mean-threshold electrode (the old key), supra-threshold function, and the mean over electrodes.
  - All v1 results are archived in `results/v1_dynaphos_thresholds/`, and `metrics.json` carries them under `v1_dynaphos_thresholds`.
- **Threshold-aware output instead of a learned per-patient adapter.**
  - The encoder reads each electrode's calibrated threshold and alive flag at the read-out, through a small MLP shared by all electrodes. It is conditioned the way Granley et al. (2023) condition a deep encoder on patient parameters, without per-electrode weights.
  - Its output is either 0 or a current between threshold + 1 µA and I_max, so no current is spent below the calibrated threshold, and electrodes that cannot reach threshold under the ceiling are never driven.
  - A blurred map of usable electrodes is a fifth input channel, so the U-Net can route content around holes.
- **Brightness gate.** The on/off decision is applied to phosphene brightness, not current. For a {0, 1} gate this is the same percept, but an "off" electrode still gets a gradient (the percept change if it were on). Gating the current left every off electrode in the simulator's zero-gradient dead zone, and training collapsed to all-off in a smoke test. The gate is a hard step with a straight-through sigmoid gradient (Bengio et al. 2013).
- **Frame dynamics in closed form.** `frame_update` integrates one constant-amplitude train exactly (activation decays with τ = 111 ms; the trace grows with delivered charge and decays with τ ≈ 2000 s). It matches Euler integration within 1 % over on/off frames (unit test), and from rest it equals the static mode, so sequence frame 1 is the single-image percept.

- **Re-implemented rather than reused dynaphos.** The dynaphos code is GPL-3.0, so we wrote our own implementation from the published equations. We changed two things:
  - Static mode uses an exact closed form for one 166 ms train from rest. It includes the habituation trace that builds up within the train, and agrees with Euler integration of the dynamics to within 1 % (unit test).
  - Gaussian sigma follows the paper's text: 2σ equals the phosphene diameter. The reference code uses half of that.
- **Separable and basis rendering.**
  - The exact renderer factorises each isotropic Gaussian into x and y profiles: memory O(B·N·(H+W)), one batched matmul.
  - Training uses a two-anchor basis: Gaussians precomputed at the mean-threshold current and at I_max, blended linearly by current. Its error against the exact renderer is below 0.03 per pixel (unit test). Evaluation always uses the exact renderer.
  - Negligible Gaussian tails are zeroed to avoid denormal floats, which slowed CPU matmuls about tenfold.
- **Display blur floor (σ_floor = 0.15°).** Intracortical phosphenes in this model are 0.05–0.3° across, often smaller than a 0.19° pixel. We add the floor in quadrature so sub-pixel phosphenes neither vanish nor alias. It is documented in the export so the website renders identically.
- **Hard threshold with a straight-through gradient.** The forward pass is always the hard detection threshold (train and test see the same percept). Gradients go through a sigmoid surrogate of width 0.1·A50.
- **Retinotopic read-out instead of a dense output layer.**
  - The encoder is fully convolutional with coordinate channels. Its map is sampled bilinearly at each phosphene's nominal location, using Gather only, so it exports to ONNX.
  - Only the 2N adapter parameters scale with N (26k parameters in total at width 24 with 1024 electrodes).
  - Per-electrode gain and bias are the only electrode-specific parameters, so they act as a patient adapter.
- **Safety enforced outside the encoder.** All encoders go through the same clamp-and-top-k layer, so comparisons are at equal safety. The learned encoder also pays a soft charge cost.
- **Low-pass SSIM.** Raw SSIM between a dotted percept and a smooth image mostly measures dot texture. We therefore also report SSIM after a 0.375° blur. Raw SSIM and PSNR are still reported, as the brief requires.
- **Recognition proxy.** A CNN trained on each encoder's percepts stands in for a user who has learned that encoding. It is an upper-bound-style proxy (no attention, memory or motor limits). It is not a psychophysics result.
- **CPU budget.** Width-24 networks, 64 px renders, 1500 steps at batch 32, and a 20k-image training subset. `configs/gpu.yaml` scales steps, data, width, patients and seeds; the code is unchanged.

## How it is tested

`pytest tests` runs 40 tests in about 1–3 minutes (machine load):

- **Maps:** wedge-dipole round trip, magnification against finite differences and Horton & Hoyt, sign conventions.
- **Layouts:** counts, extents, Utah pitch.
- **Simulator:** black for zero or sub-threshold input; the threshold-current formula checked against a hand computation; monotone, bounded brightness; size trends; closed form against Euler integration; fast against exact renderer; gradient flow; dead electrodes; peak location.
- **Metrics:** SSIM matches scikit-image, PSNR matches a hand calculation, bilinear taps hit pixel centres, blur, safety layer.
- **Encoders:** output shapes and ranges.
- **Training:** a tiny overfit test.
- **Round 2 (`test_improvements.py`):** Fernández threshold model moments and floor; dynaphos model unchanged; frame update against Euler and against the static mode; habituation silencing; brightness gate equals current gate; threshold-aware currents are supra-threshold or zero and respect known-dead masks; gate gradients exist; baked ONNX export keeps I/O names/shapes and matches torch; known-dead masking frees the top-k budget; failure-mask rate; sequence generation; temporal encoder starts equal to the threshold-aware one; frame 1 of a rollout equals the static percept.
- **Export:** the documented rendering formula and baseline algorithm, re-implemented from the JSON alone in NumPy and in plain JavaScript (`tests/js/render_check.mjs`, run through Node if it is installed), reproduce the exported test cases.

`scripts/run_cpu.sh` runs the whole pipeline end to end, followed by the tests.

## Known limits

- The simulator is a model, not data. Thresholds now match the mean and sd that Fernández et al. 2021 report, but not their shape or range: the minimum is the rheobase (24 µA, not 7 µA), and the tail is long (17 % above 100 µA). The population-mean brightness curve fits Fig. 6A less well than the brightness function itself. Several features remain uncalibrated for intracortical arrays:
  - phosphene shape and colour;
  - interactions between electrodes (current summation; threshold reduction with simultaneous stimulation);
  - eye movements and gaze-contingent shifts;
  - long-term threshold drift.
- The cortex is a plane: there are no sulci, so the coverage of the thread layouts is optimistic. The thread layout is our assumption, not a device specification.
- Temporal: habituation only. The trace decays with τ ≈ 2000 s (dynaphos constant), so within a 2.7 s sequence an electrode's lost sensitivity never recovers, and only rationing and rotation help. Dynamic current steering (Beauchamp et al. 2020) and eye movements are not modelled. The temporal encoder is fine-tuned at CPU scale (300 steps, one seed, MNIST and letters).
- The threshold-aware encoder needs a calibration session (thresholds, dead electrodes). Its "zero wasted current" holds only for the calibrated thresholds; we test 20 % calibration error separately.
- Recognition is measured by a CNN proxy, not by human observers.
- Datasets are small and low resolution (MNIST, EMNIST, 32-px CIFAR). Indoor navigation scenes are not covered.
- **End-to-end training can collapse into the sub-threshold dead zone.** Once currents fall below the rheobase and the detection threshold, gradients vanish. This happened without the percept loss, with a single 96-channel array, and in an unconstrained run initialised at 100 µA (log kept in `results/extra_runs/`). The percept-similarity term and the 50 µA initialisation prevent it in the main configuration.
- Adaptation assumes a calibrated patient-specific forward model (thresholds, dead electrodes) to backpropagate through. In a person this would come from mapping sessions, or be replaced by preference-based optimisation (Granley et al. 2023).
