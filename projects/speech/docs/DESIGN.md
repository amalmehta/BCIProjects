# Speech: system design

Decoding attempted speech from intracortical recordings of speech motor cortex into
text: a neural-to-phoneme CTC decoder, then a phoneme-to-word language-model decoder.
This is an independent project, not affiliated with or endorsed by Neuralink.

## Architecture

```mermaid
flowchart LR
    subgraph Data["Data (ROOT/data/speech, gitignored)"]
        HF[(HF mirror of Dryad<br/>doi:10.5061/dryad.dncjsxm85)] -->|download.py<br/>stream, float16, delete| NPZ[sessions/*.npz<br/>512 feats x 20 ms]
        TXT[(CMUdict, OpenWebText-10k,<br/>DailyDialog)] --> LMB[build_lm.py]
        GPT[(GPT-2 small)]
    end
    NPZ --> PREP[train.prepare_data<br/>bin to 60 ms, train/dev/test,<br/>held-out new day]
    PREP --> AUG[augment.py<br/>noise, offset, cut,<br/>time mask, electrode dropout]
    AUG --> ENC{{models.py<br/>day-specific input layers →<br/>GRU or causal Conformer → 41-way CTC}}
    ENC -->|posteriors_{dev,test}.npz| EVAL[evaluate.py]
    LMB -->|KN trigram + vocab| BEAM[beam.py<br/>lexicon-trie CTC<br/>prefix beam search]
    EVAL --> BEAM --> NB[n-best] --> RES[rescore.py<br/>GPT-2] --> MET[metrics_systems.json]
    ENC --> CAL[calibrate.py<br/>new-day calibration]
    ENC --> ROB[robustness.py<br/>electrode loss]
    ENC --> LAT[latency.py]
    MET & CAL & ROB & LAT --> REP[report.py + figures.py]
    REP --> OUT[results/metrics.json<br/>results/figures/*.png<br/>results/site_export/*]
```

## Components

| Module | Job |
|---|---|
| `speech/phonemes.py` | 41-class alphabet (blank + 39 ARPAbet + word boundary `\|`), identical to the released labels |
| `speech/data.py` | Streams each session's HDF5 from the mirror, converts it to compressed float16 `.npz` (~48% of the HDF5 size) and deletes the HDF5. Loads `Trial`s. |
| `speech/download.py` | CLI: sessions + block metadata CSV + GPT-2 weights |
| `speech/augment.py` | Batch augmentations and Gaussian smoothing |
| `speech/models.py` | `GRUDecoder` (published baseline), `ConformerDecoder` (causal or bidirectional, intermediate/self-conditioned CTC), `DayInputLayers` |
| `speech/train.py` | Data splits, input re-standardisation, length-bucketed batching, short-utterance curriculum, CTC training with warm-up + cosine LR, gradient clipping and best-dev checkpointing; per-step trace (loss, pre-clip gradient norm, blank fraction); dumps posteriors |
| `speech/stability.py` | Summarises each run's trace: blank-plateau length, escape step, gradient-norm statistics |
| `speech/run_queue.py` | Resumable process pool over the config's `schedule` |
| `speech/lm.py` | CMUdict lexicon, phoneme trie, pure-NumPy interpolated Kneser–Ney trigram |
| `speech/build_lm.py` | Builds the LM corpus with evaluation sentences filtered out |
| `speech/beam.py` | Lexicon-constrained CTC prefix beam search with word LM, n-best output |
| `speech/rescore.py` | GPT-2 sentence log-likelihoods for n-best rescoring |
| `speech/evaluate.py` | Tunes every decoder setting on dev (lexicon-only bonus, α, β, beam, GPT-2 γ and bonus) with grids that extend past an edge optimum, then decodes test once for each system and ensemble; PER/WER with sentence-level bootstrap CIs; paired bootstrap tests between systems |
| `speech/calibrate.py` | New-day study: reuse, unsupervised adaptation, supervised day-layer calibration |
| `speech/adapt.py` | Unsupervised test-time adaptation of the new day's input layer: moment / CORAL statistics matching folded into the layer, and confidence-filtered pseudo-label self-training |
| `speech/robustness.py` | PER vs. fraction of electrodes silenced at test time; PER when decoding through the wrong day's input layer |
| `speech/latency.py` | Per-frame streaming cost and lookahead; LM-stage timings |
| `speech/report.py`, `speech/figures.py` | `metrics.json`, PNG figures, website export |
| `speech/paper_tables.py` | LaTeX tables and number macros for the paper, generated from `metrics.json` |

## Main flows

1. **Prepare.** `download` fetches the sessions, then `build_lm` builds the language model. Each
   trial is 512 z-scored features per 20 ms bin. On CPU these are averaged into 60 ms bins.
2. **Train.** Each `(run, seed)` in `schedule` trains under a fixed wall-clock budget on CPU
   (40 min, cosine schedule over time) or a fixed step budget on GPU. The checkpoint with the
   best dev greedy PER is kept, and its CTC log-posteriors on dev and test are saved.
3. **Decode.** For each system (a single model, or a seed ensemble that averages per-frame
   probabilities), `evaluate` picks the LM weight α and the word bonus β on 100 dev sentences.
   It then decodes the test set in four ways: greedy phonemes, lexicon only (α = 0),
   lexicon + 3-gram, and 3-gram + GPT-2 n-best rescoring with weight γ tuned on dev.
4. **Side studies.** Calibration to a new day held out from training, robustness to electrode
   loss and to day-layer mismatch, and latency.
5. **Report.** `metrics.json`, figures and the site export, then the paper's tables and number macros and the PDF.

## Where data lives

* Raw neural data and LM resources: `ROOT/data/speech/` (gitignored; ~2.2 GB on CPU).
  * `sessions/<session>_{train,val}.npz`: 11 sessions (10 training + 1 held-out day).
  * `lm/`: CMUdict, OpenWebText-10k, DailyDialog, and `built/` (KN trigram `kn3.npz`, vocabulary, statistics).
  * `gpt2/`: GPT-2 small weights (0.5 GB).
* Experiment outputs: `projects/speech/results/`.
  * `runs/<run>_s<seed>/`: `model.pt`, `log.jsonl`, `summary.json`, `posteriors_{dev,test}.npz`.
  * `decodes/<system>.json`: per-sentence decodes and metrics (cache).
  * `metrics.json`, `calibration.json`, `robustness.json`, `latency.json`, `figures/`, `site_export/`.

## Splits (no leakage)

* **train / dev**: the released `train` split of each of the 10 training sessions. 10% of
  each session's trials go to dev (fixed seed). Dev is used for checkpoint selection and
  LM-weight tuning.
* **test**: the released `val` split of the same 10 sessions (426 sentences). It is never used
  for any choice. (The competition `test` split has no labels.)
* **new day**: session `t15.2024.03.15`, 77 days after the last training session, never
  seen in base training. Its `train` split is the calibration pool and its `val` split is
  the calibration test set (48 sentences).
* **LM corpus**: any corpus sentence that matches a dev, test or new-day test sentence
  after normalization is removed (68 removed). Cue sentences of training trials are included,
  as a deployed system would include its own prompt corpus.

## Key design decisions and trade-offs

| Decision | Why | Cost |
|---|---|---|
| Brain-to-Text '25 (T15) data instead of the brief's Willett 2023 (T12) | Dryad's API now requires a bearer token (HTTP 401/403 for scripts). A public Hugging Face mirror of the T15 release has per-session files, which allows a subset that fits the disk budget. | Different participant than the brief named. T15 is the newer, larger and better-documented dataset. |
| 10 of 45 sessions, 60 ms bins, 2×256 GRU (CPU) | 8 shared cores. One 40-minute run gets about 500 steps of batch 32. | Absolute PER/WER are far from published numbers. `gpu.yaml` restores full scale. |
| Wall-clock cosine schedule on CPU | The machine is shared and its speed varies by 2–3×. A step schedule would be cut off mid-decay. | Step counts differ slightly between runs (reported in `summary.json`). |
| Pure-Python lexicon beam search + NumPy KN trigram instead of a Kaldi WFST + 5-gram | Runs anywhere with no compiled dependencies. The 3-gram fits in about 0.1 GB of RAM (published 5-gram: about 300 GB). | Slower (about 0.3 s per sentence), with a weaker LM than the published one. |
| GPT-2 small instead of OPT-6.7B for rescoring | Must fit in the CPU and disk budget. | Weaker rescoring. |
| No absolute positional encoding in the Conformer | The un-scaled sinusoid dominated the small patch embeddings. Causal masking and depthwise convolution supply order information. | — |
| Posterior averaging for ensembles | Cheap, and the B2T'24 write-up names ensembling as the largest gain. | Inference cost grows linearly with ensemble size. |

## How it is tested

`pytest tests` runs in under 2 minutes and covers:

* metrics against hand-computed edit distances and corpus-level pooling;
* CTC greedy collapse;
* augmentation invariants (electrode dropout zeroes both features of an electrode, time masks are contiguous, smoothing preserves constants);
* model shapes and log-prob normalization;
* **causality**: perturbing future inputs leaves earlier outputs of the causal GRU and Conformer unchanged, accounting for the smoothing half-width;
* day layers initialized to the identity;
* Kneser–Ney distributions summing to 1 for seen and unseen contexts;
* trie homophones;
* exact beam-search recovery from clean posteriors;
* the LM resolving an acoustically ambiguous word;
* an HDF5 → npz round trip with binning;
* a tiny GRU overfitting synthetic sequences.

`scripts/run_cpu.sh` with `CONFIG=configs/smoke.yaml` runs the whole pipeline end to end with tiny budgets.

## Known limits

* CPU scale. Ten sessions, 480 optimizer steps per model (one run stopped at 466 steps on the 75-minute
  safety limit), 2–3 seeds for the main models and 1 for the day-layer ablation. GRU training was
  seed-sensitive at this budget: seed 1 stalled for both GRU variants. Conclusions about small differences
  are tentative; see the bootstrap CIs.
* Runs we planned but cut for CPU time: a no-augmentation ablation, and a *trained* bidirectional GRU
  (its compute is still timed in the latency study, using untrained weights). Both are in `configs/gpu.yaml`.
* One participant (T15). Results say nothing about cross-participant transfer.
* The calibration test set has only 48 sentences, so the curve is noisy.
* Latency numbers are for a shared laptop CPU, not an embedded implant processor. The Conformer's
  per-frame cost assumes a KV cache that is not implemented; we also report the naive
  recomputation cost.
* The `val` split is the standard held-out set in the competition codebase. Its sentences come
  from the same corpora as training, but no sentence appears in both.
