"""Copy each project's exported results (results/site_export) and paper PDF into the website.

Run from the repo root:  .venv/bin/python scripts/build_site.py
Then serve:              .venv/bin/python -m http.server -d site 8765
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"

# slug -> paper file name people will see
PAPERS = {
    "telepathy": "Telepathy.pdf",
    "speech": "Speech.pdf",
    "robotic_arm": "Assisted Robotic Arm.pdf",
    "blindsight": "Blindsight.pdf",
}


def copy_tree(src: Path, dst: Path) -> int:
    """Mirror src into dst (replacing dst). Returns number of files copied."""
    if dst.exists():
        shutil.rmtree(dst)
    if not src.exists():
        return 0
    shutil.copytree(src, dst)
    return sum(1 for p in dst.rglob("*") if p.is_file())


def load(slug: str) -> dict:
    return json.loads((SITE / "data" / slug / "metrics.json").read_text())


def headlines() -> dict:
    """One headline number per project for the landing page, read from each project's metrics."""
    tp = load("telepathy")["decoders"]
    sp = {(r["model"], r["system"], r["setting"]): r for r in load("speech")["rows"]}
    sa = load("robotic_arm")["shared_autonomy"]["standard"]
    bs = load("blindsight")["main"]["mnist"]["encoders"]
    return {
        "telepathy": {
            "title": "Stable cursor decoding across months",
            "value": f"R² {tp['sld_proposed']['cross_day_r2_recentred']['mean_excluding_remapped_days']:.2f}",
            "label": "velocity decoding up to 138 days later, with no new labels",
            "thumb": "data/telepathy/fig_webgrid_traces.png",
        },
        "speech": {
            "title": "Attempted speech to text",
            "value": f"{100 * sp[('conformer_aug', 'single model', 'greedy (PER)')]['mean']:.1f}% PER",
            "label": "phoneme error, causal Conformer, 426 held-out sentences",
            "thumb": "data/speech/posteriors_example.png",
        },
        "robotic_arm": {
            "title": "Decoder-aware shared autonomy",
            "value": f"{100 * sa['direct']['success']:.0f}% → {100 * sa['dasa']['success']:.0f}%",
            "label": "simulated grasp success, decoder alone → with assistance",
            "thumb": "data/robotic_arm/arm_3d.png",
        },
        "blindsight": {
            "title": "Learned cortical stimulation encoder",
            "value": f"SSIM {bs['learned']['ssim']:.2f} vs {bs['intensity']['ssim']:.2f}",
            "label": "simulated percept fidelity, learned vs intensity sampling",
            "thumb": "data/blindsight/figures/fig_examples.png",
        },
    }


def main() -> None:
    (SITE / "papers").mkdir(parents=True, exist_ok=True)
    (ROOT / "papers").mkdir(exist_ok=True)
    for slug, paper_name in PAPERS.items():
        proj = ROOT / "projects" / slug
        n_data = copy_tree(proj / "results" / "site_export", SITE / "data" / slug)
        pdf = proj / "paper" / "main.pdf"
        if pdf.exists():
            shutil.copy2(pdf, SITE / "papers" / paper_name)
            shutil.copy2(pdf, ROOT / "papers" / paper_name)
        print(f"{slug:12s} data files: {n_data:3d}  paper: {'yes' if pdf.exists() else 'MISSING'}")
    (SITE / "data" / "headlines.json").write_text(json.dumps(headlines(), indent=2, ensure_ascii=False))
    print("wrote site/data/headlines.json")


if __name__ == "__main__":
    main()
