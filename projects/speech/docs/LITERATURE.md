# Speech: literature and domain notes

Background for decoding attempted speech from intracortical recordings into text.
Every reference below was checked against PubMed, the arXiv API or the publisher
page on 2026-10-06.

## 1. The problem in one paragraph

People with ALS, brainstem stroke or other conditions can lose the ability to
speak intelligibly (dysarthria or anarthria) while the cortical motor commands for
speech remain intact. A speech neuroprosthesis records neural activity from speech
motor cortex while the person *attempts* to speak, and decodes it into text (and
optionally a synthesized voice). The field's figures of merit are **word error rate
(WER)** on held-out sentences, the **vocabulary size** it holds for, **speaking rate**
(words per minute, wpm), **calibration burden** (how much data a new day or a new user
needs), and **latency**. Neuralink's speech-restoration device received FDA
Breakthrough Device designation in May 2025 (press coverage, e.g. MassDevice,
"Neuralink FDA breakthrough nod for speech impairment"), so this is a product
question for them as well as a research one.

## 2. Neurophysiology and signals

* **Recording site.** Ventral precentral gyrus (vPCG): area 6v (ventral premotor), area 4
  (primary motor) and area 55b, which hold the orofacial and laryngeal articulator
  representations. Utah-array recordings there carry phoneme information. In
  Willett et al. (2023), area 6v was highly informative and area 44 (Broca's area)
  carried little.
* **Features.** For every electrode, signals are band-passed (250–5000 Hz) and two
  features are computed in 20 ms bins: **threshold crossings** (TX, spikes crossing
  −4.5 × RMS, a proxy for multi-unit firing without spike sorting) and **spike-band
  power** (SBP, the mean squared high-frequency signal). With 256 electrodes this gives 512
  features per bin. Features are z-scored per block to remove slow drift.
* **Non-stationarity.** Electrode impedance changes, micromotion and units appearing
  or disappearing all shift the feature distribution from day to day.
  The published decoders handle this with **day-specific input layers** (an affine map
  per session) and **noise / constant-offset augmentation** (Sussillo et al., 2016;
  Willett et al., 2021/2023). Fan et al. (2023) go further and adapt continuously
  with no supervision ("plug-and-play" stability, CORP).
* **Attempted versus inner speech.** Kunz et al. (2025) show that inner speech is
  represented in the same motor cortex areas, but more weakly than attempted speech.
  That opens the possibility of decoding speech the person does not intend to say,
  which is a privacy issue. They demonstrate a "password" gate to address it.

## 3. Key papers

| Paper | Contribution | Numbers worth remembering |
|---|---|---|
| **Moses et al., 2021**, *NEJM* 385:217–227, doi:10.1056/NEJMoa2027540 | First demonstration of decoding attempted speech into sentences in a person with anarthria. Used ECoG over sensorimotor cortex, word classification plus an HMM language model. | 50-word vocabulary, about 15 wpm |
| **Willett et al., 2021**, *Nature* 593:249–254, doi:10.1038/s41586-021-03506-2 | Handwriting BCI: RNN decoding of attempted handwriting into characters. Introduced the RNN + day-specific-layer + augmentation recipe later reused for speech. | 90 characters/min |
| **Willett et al., 2023**, *Nature* 620:1031–1036, doi:10.1038/s41586-023-06377-x | Intracortical speech BCI (participant T12, ALS, arrays in 6v and 44). 5-layer GRU → 39 phonemes + silence with CTC, day-specific input layers, patching of 14 bins with stride 4, white-noise and constant-offset augmentation, Kaldi WFST decoding with a 3-gram/5-gram LM, then LLM rescoring. Released the data as the Brain-to-Text '24 benchmark. | 9.1% WER (50 words), 23.8% WER (125k words), 62 wpm |
| **Metzger et al., 2023**, *Nature* 620:1037–1046, doi:10.1038/s41586-023-06443-4 | High-density ECoG (253 electrodes) with multimodal output: text, synthesized speech and a talking avatar. Character/phone CTC decoding with an LM. | about 78 wpm, median 25% WER on a 1,024-word vocabulary |
| **Card et al., 2024**, *NEJM* 391:609–618, doi:10.1056/NEJMoa2314132 | Participant T15 (ALS, four 64-electrode arrays in left vPCG: 6v, area 4, 55b). Rapid calibration: usable on day 1 after 30 min of data. Sustained accuracy over 8.4 months and 248 h of conversational use. Released as the Brain-to-Text '25 dataset (the data this project uses). | 99.6% accuracy (50 words) on day 1; 90.2% (125k words) on day 2; 97.5% sustained; about 32 wpm |
| **Willett et al., 2024**, *Brain-to-Text Benchmark '24: Lessons Learned*, arXiv:2412.17227 | Write-up of the T12 competition. The largest gains came from **ensembling** several decoders and merging their outputs with a fine-tuned LLM. Better learning-rate schedules and a **diphone** objective also helped. **Transformers, Mamba and CNN front-ends did not beat the GRU** baseline on WER. | baseline 9.7% WER; winner 5.77% |
| **Li et al., 2024**, arXiv:2411.10657 (DCoND-LIFT) | Diphone (context-dependent) CTC targets plus LLM-based ensemble fusion. Won B2T'24. | PER 16.62 → 15.34% from diphones |
| **Benster et al., 2024**, arXiv:2403.05583 (LISA) | Cross-modal contrastive alignment of neural and audio representations, plus LLM-based selection among ensemble outputs. | — |
| **Fan et al., 2023**, NeurIPS, arXiv:2311.03611 | Plug-and-play stability: the decoder self-trains on its own LM-corrected outputs, so it needs no recalibration for a year. | — |
| **Feng et al., 2024**, arXiv:2406.11568 | End-to-end neural → LLM decoding (no phoneme stage), competitive with cascades. | — |
| **Wairagkar et al., 2025**, *Nature* 644:145–152, doi:10.1038/s41586-025-09127-3 | Same T15 participant: instantaneous voice synthesis, which needs causal decoding at low latency. | — |
| **Kunz et al., 2025**, *Cell* 188:4658–4673, doi:10.1016/j.cell.2025.06.015 | Inner speech in motor cortex. Privacy implications. | — |
| **Sussillo et al., 2016**, *Nat Commun* 7:13749, doi:10.1038/ncomms13749 | Data augmentation for robustness to future neural variability. | — |

### Sequence-modelling and decoding methods

* **CTC.** Graves, Fernández, Gomez & Schmidhuber, "Connectionist temporal classification",
  ICML 2006. Marginalizes over monotonic alignments with a blank token, so no frame-level
  phoneme labels are needed. That matters here because the participant cannot produce
  audio to align against.
* **CTC prefix beam search with an LM.** Hannun et al., 2014, arXiv:1408.2873.
* **Conformer.** Gulati et al., 2020, arXiv:2005.08100. Self-attention for global
  context plus depthwise convolution for local context, with macaron feed-forward layers.
* **Intermediate CTC.** Lee & Watanabe, 2021, arXiv:2102.03216. An auxiliary CTC loss on a
  middle layer, used as a regularizer.
  **Self-conditioned CTC.** Nozaki & Komatsu, 2021, arXiv:2104.02724. The intermediate
  posteriors are fed back into the residual stream, which relaxes CTC's
  conditional-independence assumption.
* **SpecAugment.** Park et al., 2019, arXiv:1904.08779. Time and frequency masking. We use
  the neural analogues: time masking, and *electrode* dropout in place of frequency masking.
* **Kneser–Ney smoothing.** Kneser & Ney, ICASSP 1995. Chen & Goodman, "An empirical study
  of smoothing techniques for language modeling", *Computer Speech & Language* 13(4), 1999.
* **LLM rescoring.** The published systems rescore n-best lists with OPT-6.7B. We use GPT-2
  small (Radford et al., 2019, "Language Models are Unsupervised Multitask Learners", OpenAI
  technical report) because it fits on a CPU.

### Text resources

* CMU Pronouncing Dictionary (cmudict, github.com/cmusphinx/cmudict), 39 ARPAbet phonemes.
* OpenWebText (Gokaslan & Cohen, 2019). We use the public 10k-document sample
  `stas/openwebtext-10k`.
* DailyDialog (Li et al., IJCNLP 2017). Conversational English, close in register to the
  Switchboard cue sentences.

## 4. Metrics the field uses

* **PER**: Levenshtein distance between the decoded and true phoneme sequences, summed over
  sentences and divided by the total number of reference phonemes. The official code counts
  the word-boundary/silence token, and so do we.
* **WER**: the same at the word level, after lowercasing and stripping punctuation. Pooled
  over the corpus rather than averaged per sentence.
* Also reported in the literature: words per minute, accuracy = 1 − WER, calibration
  time, and latency (time from the end of the attempt to the text, or per-step latency
  in streaming).

## 5. What this project adopts

| Idea | Source | Where |
|---|---|---|
| 512 TX+SBP features, 41-class phoneme CTC (39 phonemes, word boundary, blank) | Willett 2023; Card 2024 | `speech/phonemes.py`, `speech/data.py` |
| Day-specific affine + softsign input layers, identity init | Willett 2021/2023 | `models.DayInputLayers` |
| Gaussian smoothing, then patching (280 ms windows, 80 ms hop) | B2T'25 baseline code | `models.patch` |
| White noise (σ=1.0), constant offset (σ=0.2), random cut | B2T'24/'25 baselines | `augment.py` |
| **Time masking + electrode dropout** (new augmentations) | SpecAugment, adapted | `augment.py` |
| **Causal Conformer + intermediate / self-conditioned CTC** (proposed encoder) | Gulati 2020; Lee & Watanabe 2021; Nozaki & Komatsu 2021 | `models.ConformerDecoder` |
| **Posterior-averaging ensembles** | B2T'24 lessons (ensembling was the largest win) | `speech/evaluate.py` |
| Lexicon-constrained CTC beam search + Kneser–Ney trigram | Hannun 2014; WFST decoders in Willett 2023 | `beam.py`, `lm.py` |
| n-best rescoring with a small LLM (GPT-2) | Willett 2023; Card 2024 (OPT-6.7B) | `speech/rescore.py` |
| Calibration curve for a new day (day layer only vs full fine-tune) | Card 2024 (rapid calibration) | `speech/calibrate.py` |
| Causal versus bidirectional latency trade-off | Wairagkar 2025 (instantaneous synthesis) | `speech/latency.py` |

## 6. Ideas we read about but did not implement (and why)

* **Diphone targets** (DCoND): roughly 1,600 output classes need far more data and compute than
  a CPU budget allows. This is the first thing to add at GPU scale.
* **Fine-tuned LLM ensemble merging** (B2T'24 winners): needs a 7B-class LLM.
* **Unsupervised continual recalibration** (Fan et al., 2023): needs a long-term
  deployment loop. Our calibration study covers the supervised case only.
* **End-to-end neural → LLM decoding** (Feng et al., 2024): data-hungry, and needs GPU
  memory for the LLM.
