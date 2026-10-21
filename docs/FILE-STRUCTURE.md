# File Structure

```
neuralink_product_dev/
├── README.md                      what it is, screenshots, links
├── papers/                        the four papers people open
│   ├── Telepathy.pdf
│   ├── Speech.pdf
│   ├── Assisted Robotic Arm.pdf
│   └── Blindsight.pdf
├── docs/
│   ├── INSTRUCTIONS.md            setup, run the site, reproduce on CPU/GPU
│   ├── SYSTEM-DESIGN.md           architecture, flows, data, decisions, tests, limits
│   ├── FILE-STRUCTURE.md          this file
│   └── images/                    README screenshots
├── scripts/
│   └── build_site.py              copies results + papers into site/, writes headlines
├── site/                          the static website (serve with http.server)
│   ├── index.html                 landing page: four project cards + overview diagram
│   ├── telepathy.html             WebGrid replay, reach replays, recalibration chart
│   ├── speech.html                live decoding replay, sentence browser, calibration chart
│   ├── robotic-arm.html           3D episode replay (three.js), shared-autonomy tables
│   ├── blindsight.html            live phosphene demo (draw/upload/webcam, ONNX encoder)
│   ├── assets/                    style.css, common.js, phosphene.js, icon.svg
│   ├── data/<project>/            generated: copies of each results/site_export/
│   └── papers/                    generated: copies of the PDFs
├── projects/
│   └── <project>/                 telepathy · speech · robotic_arm · blindsight
│       ├── <package>/             Python source (data, models, training, eval, simulator)
│       ├── configs/               cpu.yaml (verified here), gpu.yaml (full scale), smoke.yaml (most projects)
│       ├── scripts/               run_cpu.sh, run_gpu.sh
│       ├── tests/                 pytest suite (+ tests/js for Blindsight)
│       ├── results/               metrics.json, figures, logs, site_export/ (checkpoints gitignored)
│       ├── docs/                  LITERATURE.md (verified references), DESIGN.md
│       ├── paper/                 main.tex, refs.bib, generated tables, main.pdf
│       └── requirements-gpu.txt   pip requirements for the GPU box
├── data/                          downloaded datasets (gitignored)
└── .venv/                         Python 3.11 environment (gitignored)
```

## Where to look first in each project

| Project | Core model | Simulator | Entry point |
|---|---|---|---|
| Telepathy | `telepathy/models.py`, `telepathy/align.py` | `telepathy/simulator.py` (WebGrid) | `telepathy/experiments.py` |
| Speech | `speech/models.py`, `speech/ctc.py` | — (real data only) | `speech/train.py`, `speech/evaluate.py`, `speech/beam.py` |
| Assisted Robotic Arm | `robotic_arm/decoders/` | `robotic_arm/arm.py`, `robotic_arm/closed_loop.py`, `robotic_arm/autonomy.py` | `robotic_arm/run_decoding.py`, `robotic_arm/run_closed_loop.py` |
| Blindsight | `blindsight/encoders.py` | `blindsight/simulator.py`, `blindsight/cortex.py` | `blindsight/run.py` |
