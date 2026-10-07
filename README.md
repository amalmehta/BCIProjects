# BCI Portfolio

Four machine-learning projects for brain–computer interfaces — cursor control, speech, an assisted robotic arm and cortical vision — each with real public neural data where it exists, strong baselines, a closed-loop simulator, an interactive web demo and a full paper.

![The portfolio site: one card per project with its headline result](docs/images/portfolio.jpg)

| | |
|---|---|
| **Telepathy** — cursor decoding that holds up for 4 months, scored in simulated WebGrid bits/s<br>![WebGrid replay](docs/images/telepathy.jpg) | **Speech** — attempted speech → phonemes (CTC) → words (n-gram + GPT-2)<br>![Live decoding replay](docs/images/speech.jpg) |
| **Assisted Robotic Arm** — 7-DoF decoding + decoder-aware shared autonomy<br>![3D episode replay](docs/images/robotic-arm.jpg) | **Blindsight** — learned, safety-bounded stimulation encoder for 1024 V1 sites<br>![Live phosphene demo](docs/images/blindsight.jpg) |

```mermaid
flowchart LR
  M[Motor-cortex spikes] --> D[Decoders<br/>per-day alignment]
  D --> T[Telepathy<br/>cursor + WebGrid]
  D --> S[Speech<br/>phonemes → words]
  D --> A[Robotic Arm<br/>+ shared autonomy]
  C[Camera] --> E[Learned encoder] --> V[V1 stimulation<br/>Blindsight]
```

## Papers

- [Telepathy](papers/Telepathy.pdf) — Stable cursor decoding across months and a closed-loop WebGrid simulator
- [Speech](papers/Speech.pdf) — Decoding attempted speech from intracortical recordings on a CPU budget
- [Assisted Robotic Arm](papers/Assisted%20Robotic%20Arm.pdf) — Multi-day 7-DoF decoding and decoder-aware shared autonomy
- [Blindsight](papers/Blindsight.pdf) — Safe, patient-adaptive stimulus encoders in a differentiable phosphene simulator

## Docs

- [Instructions](docs/INSTRUCTIONS.md) — setup, run the site, reproduce each project on CPU or GPU
- [System design](docs/SYSTEM-DESIGN.md) — architecture, flows, data, decisions, testing, limits
- [File structure](docs/FILE-STRUCTURE.md) — what's where

Independent project by Amal Mehta; not affiliated with or endorsed by Neuralink. Closed-loop results come from simulated users.
