# Instructions

How to set up, view the website, and reproduce each project on a CPU (small scale) or a CUDA GPU (full scale). Commands assume you start in the repo root.

## 1. Setup (macOS or Linux, CPU)

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "torch==2.2.2" "numpy<2" scipy pandas matplotlib scikit-learn h5py pynwb dandi tqdm pyyaml pillow scikit-image torchvision==0.17.2 pytest onnx onnxruntime transformers==4.46.3
```

`torch 2.2.2` is the last build for Intel macOS; on Apple Silicon or Linux a newer torch also works. You also need [Tectonic](https://tectonic-typesetting.github.io) to compile the papers and Node.js for the one JavaScript test.

## 2. View the website

```bash
.venv/bin/python scripts/build_site.py
```

```bash
.venv/bin/python -m http.server -d site 8765
```

Open http://localhost:8765. `build_site.py` copies every project's `results/site_export/` into `site/data/<project>/`, copies the paper PDFs into `site/papers/` and `papers/`, and writes the landing-page headlines from each project's metrics. Re-run it after any new results.

The demos need an internet connection for two libraries loaded from jsDelivr: three.js (robotic-arm 3D view) and onnxruntime-web (Blindsight learned encoder; without it the page falls back to the baseline encoder).

What to try:
- **Telepathy:** switch decoders in the WebGrid replay; pick reaches from different days, including the failure on day 83.
- **Speech:** replay a decode; switch the decoding stage (lexicon only → trigram → GPT-2) on the sentence list.
- **Assisted Robotic Arm:** pick an episode (try both change-of-mind episodes), drag to orbit, scrub the timeline.
- **Blindsight:** draw a letter, upload a photo or use the webcam; switch encoders; drag "Failed electrodes".

## 3. Run each project (CPU)

Each project lives in `projects/<name>/` and has the same layout. Run these from inside the project folder.

| Project | Tests | Quick end-to-end check | Full CPU pipeline (resumable) |
|---|---|---|---|
| Telepathy | `../../.venv/bin/python -m pytest -q` | `CFG=configs/smoke.yaml scripts/run_cpu.sh` | `scripts/run_cpu.sh` |
| Speech | `../../.venv/bin/python -m pytest -q tests` | `CONFIG=configs/smoke.yaml scripts/run_cpu.sh` (~15 min) | `scripts/run_cpu.sh` |
| Assisted Robotic Arm | `../../.venv/bin/python -m pytest -q tests` | `RA_RESULTS=/tmp/ra bash scripts/run_cpu.sh configs/smoke.yaml` | `bash scripts/run_cpu.sh` |
| Blindsight | `../../.venv/bin/python -m pytest -q tests` and `node tests/js/render_check.mjs results/site_export` | — | `bash scripts/run_cpu.sh` |

Each pipeline downloads its data into `data/<project>/` (gitignored):

| Project | Dataset | Size on disk |
|---|---|---|
| Telepathy | O'Doherty et al. 2017 random-target reaching (Zenodo 583331), 24 sessions | 17 MB after binning |
| Speech | Brain-to-Text '25 (Card et al. 2024), via a Hugging Face mirror; 11 sessions | 2.2 GB |
| Assisted Robotic Arm | FALCON H1 (DANDI 000954) | 113 MB |
| Blindsight | MNIST, EMNIST letters, CIFAR-10 | 572 MB |

To regenerate a paper after new results:

| Project | Command (from the project folder) |
|---|---|
| Telepathy | `../../.venv/bin/python -m telepathy.report && cd paper && tectonic main.tex` |
| Speech | `../../.venv/bin/python -m speech.paper_tables --config configs/cpu.yaml && cd paper && tectonic main.tex` |
| Assisted Robotic Arm | `../../.venv/bin/python -m robotic_arm.tables && cd paper && tectonic main.tex` |
| Blindsight | `cd paper && ../../../.venv/bin/python make_tables.py && tectonic main.tex` |

Every number in the papers is generated from the results JSON by these scripts, never typed by hand.

## 4. Full scale on a GPU

Each project has `scripts/run_gpu.sh`, `configs/gpu.yaml` and `requirements-gpu.txt`. On a fresh Linux CUDA machine, run this from the project folder:

```bash
bash scripts/run_gpu.sh
```

The script sets up its own environment, downloads the data and runs the same pipeline with the full-scale config, with the device auto-detected and AMP on CUDA. Then copy `projects/<name>/results/` back, regenerate the paper (above) and rebuild the site (section 2). None of the GPU scripts has been run yet.

Disk needed for the full-scale data: Speech about 11.6 GB (all 45 sessions plus GPT-2); the others under 1 GB each.
