"""Visuotopic mapping between the visual field and the flattened V1 surface.

We use the wedge-dipole model of Polimeni, Balasubramanian & Schwartz (2006),
with the human parameters used by van der Grinten et al. (2024, eLife):

    w = k * log( b (z_w + a) / (a (z_w + b)) ),   z_w = r * exp(i * alpha * theta)

where z = r exp(i theta) is a visual-field location (eccentricity r in degrees,
polar angle theta in (-pi/2, pi/2) measured in the contralateral hemifield) and
w = x + i y is the cortical location in millimetres in one hemisphere. The
constant k*log(b/a) places the foveal representation at w = 0.

The derivative |dw/dz| along the horizontal meridian gives the linear cortical
magnification M(r) = k (1/(r + a) - 1/(r + b)) mm/deg, which for small r reduces
to the Horton & Hoyt (1991) estimate M = 17.3 / (r + 0.75).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WedgeDipole:
    """Parameters of the wedge-dipole visuotopic model (human V1 defaults)."""

    k: float = 17.3
    a: float = 0.75
    b: float = 120.0
    alpha: float = 0.95

    def to_cortex(self, ecc_deg: np.ndarray, theta_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map hemifield polar coordinates (deg, rad) to cortical (x, y) in mm."""
        zw = np.asarray(ecc_deg) * np.exp(1j * self.alpha * np.asarray(theta_rad))
        w = self.k * np.log(self.b * (zw + self.a) / (self.a * (zw + self.b)))
        return w.real, w.imag

    def to_visual_field(self, x_mm: np.ndarray, y_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inverse map: cortical (x, y) mm -> hemifield eccentricity (deg), polar angle (rad)."""
        w = np.asarray(x_mm) + 1j * np.asarray(y_mm)
        e = np.exp(w / self.k)
        zw = self.a * self.b * (e - 1.0) / (self.b - self.a * e)
        ecc = np.abs(zw)
        theta = np.angle(zw) / self.alpha
        return ecc, theta

    def magnification(self, ecc_deg: np.ndarray) -> np.ndarray:
        """Linear cortical magnification factor M(r) in mm of cortex per degree."""
        r = np.asarray(ecc_deg, dtype=float)
        return self.k * (1.0 / (r + self.a) - 1.0 / (r + self.b))


def horton_hoyt_magnification(ecc_deg: np.ndarray) -> np.ndarray:
    """Horton & Hoyt (1991) human V1 magnification estimate, M = 17.3 / (E + 0.75) mm/deg."""
    return 17.3 / (np.asarray(ecc_deg, dtype=float) + 0.75)


def hemifield_to_cartesian(ecc_deg: np.ndarray, theta_rad: np.ndarray, hemisphere: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert hemifield polar coordinates to full visual-field (x, y) in degrees.

    hemisphere: 0 = left V1 (represents the right visual field, x > 0),
                1 = right V1 (represents the left visual field, x < 0).
    """
    sign = np.where(np.asarray(hemisphere) == 0, 1.0, -1.0)
    x = sign * ecc_deg * np.cos(theta_rad)
    y = ecc_deg * np.sin(theta_rad)
    return x, y
