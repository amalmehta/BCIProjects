"""Shared matplotlib style: one categorical order, recessive axes, 200 dpi PNGs."""
from __future__ import annotations

from pathlib import Path

# Validated categorical order (blue, orange, aqua, yellow, magenta, green, violet, red).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
FIGSIZE = (7.0, 3.6)


def setup_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#b5b4ae", "axes.labelcolor": TEXT, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e3",
        "grid.linewidth": 0.6, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "legend.fontsize": 8, "lines.linewidth": 2, "font.family": "DejaVu Sans",
    })


def savefig(fig, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
