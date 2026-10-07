"""Assisted Robotic Arm: high-DoF intracortical decoding (FALCON H1) + shared autonomy
for a simulated 7-DoF assistive arm.

Paths can be overridden with the environment variables ``RA_DATA`` and ``RA_RESULTS``.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DATA_ROOT = Path(os.environ.get("RA_DATA", REPO_ROOT / "data" / "robotic_arm"))
RESULTS_ROOT = Path(os.environ.get("RA_RESULTS", PROJECT_ROOT / "results"))

DOF_NAMES = ["tx", "ty", "tz", "rx", "g1", "g2", "g3"]
