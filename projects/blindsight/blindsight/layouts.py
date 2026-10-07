"""Electrode layouts on the flattened V1 surface.

Two families are modelled:

* Utah-like arrays (Maynard et al. 1997; used by Fernandez et al. 2021 in a human
  and by Chen et al. 2020 in macaque): rigid square grids with 0.4 mm pitch.
  ``utah_96`` is a single 10x10 array with the four corners inactive (96 channels,
  one hemisphere), ``utah_1024`` is 16 arrays of 8x8 (Chen et al. 2020 scale),
  split over both hemispheres here.
* Thread-based layouts (``threads_<N>``): N independent insertion sites spread over
  the reachable V1 surface of both hemispheres, as a flexible-thread, robot-inserted
  system could place them. This is our *assumption* of what a high-channel
  thread-based implant could achieve, not a published Neuralink specification: we
  treat every site as a distinct surface location (contacts along one thread at
  different depths would share a retinotopic location and are not modelled
  separately). Sites are placed with uniform cortical density using Mitchell's
  best-candidate sampling, so their visual-field density follows cortical
  magnification automatically.

The cortical surface is idealised as the planar wedge-dipole domain (no sulci): in a
real brain much of V1 lies inside the calcarine sulcus and is not reachable from the
surface, so these layouts are optimistic about coverage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cortex import WedgeDipole, hemifield_to_cartesian


@dataclass
class Layout:
    """Nominal electrode geometry (before any patient-specific perturbation)."""

    name: str
    cortex_x_mm: np.ndarray
    cortex_y_mm: np.ndarray
    hemisphere: np.ndarray  # 0 = left V1 (right visual field), 1 = right V1
    implant: np.ndarray  # integer implant/array id per electrode

    @property
    def n(self) -> int:
        return int(self.cortex_x_mm.shape[0])

    def visual_field(self, model: WedgeDipole) -> dict[str, np.ndarray]:
        """Return eccentricity (deg), polar angle (deg, full field) and (x, y) in degrees."""
        ecc, theta = model.to_visual_field(self.cortex_x_mm, self.cortex_y_mm)
        x, y = hemifield_to_cartesian(ecc, theta, self.hemisphere)
        polar = np.degrees(np.arctan2(y, x))
        return {"ecc": ecc, "polar_deg": polar, "x": x, "y": y}


def _valid_mask(model: WedgeDipole, x: np.ndarray, y: np.ndarray, ecc_min: float, ecc_max: float) -> np.ndarray:
    ecc, theta = model.to_visual_field(x, y)
    return (ecc >= ecc_min) & (ecc <= ecc_max) & (np.abs(theta) < np.pi / 2 * 0.98)


def _cortical_bbox(model: WedgeDipole, ecc_max: float) -> tuple[float, float, float, float]:
    r = np.linspace(0.0, ecc_max, 200)
    t = np.linspace(-np.pi / 2, np.pi / 2, 200)
    rr, tt = np.meshgrid(r, t)
    x, y = model.to_cortex(rr, tt)
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def best_candidate_sites(model: WedgeDipole, n: int, ecc_min: float, ecc_max: float,
                         rng: np.random.Generator, n_candidates: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Mitchell's best-candidate sampling of ``n`` well-spread sites in one hemisphere's V1."""
    x0, x1, y0, y1 = _cortical_bbox(model, ecc_max)
    pool_x = rng.uniform(x0, x1, size=200_000)
    pool_y = rng.uniform(y0, y1, size=200_000)
    keep = _valid_mask(model, pool_x, pool_y, ecc_min, ecc_max)
    pool = np.stack([pool_x[keep], pool_y[keep]], 1)
    chosen = [pool[rng.integers(len(pool))]]
    for _ in range(n - 1):
        cand = pool[rng.integers(len(pool), size=n_candidates)]
        pts = np.asarray(chosen)
        d = np.min(((cand[:, None, :] - pts[None]) ** 2).sum(-1), axis=1)
        chosen.append(cand[np.argmax(d)])
    pts = np.asarray(chosen)
    return pts[:, 0], pts[:, 1]


def thread_layout(n_sites: int, ecc_min: float, ecc_max: float, seed: int = 0,
                  model: WedgeDipole | None = None, implants_per_hemisphere: int = 2) -> Layout:
    """Thread-based layout with ``n_sites`` sites split evenly over both hemispheres."""
    model = model or WedgeDipole()
    rng = np.random.default_rng(seed)
    xs, ys, hs, imps = [], [], [], []
    per_h = n_sites // 2
    for h in (0, 1):
        x, y = best_candidate_sites(model, per_h, ecc_min, ecc_max, rng)
        # Implants partition each hemisphere along the eccentricity axis (cortical x).
        order = np.argsort(np.argsort(x))
        imp = h * implants_per_hemisphere + (order * implants_per_hemisphere) // per_h
        xs.append(x); ys.append(y); hs.append(np.full(per_h, h)); imps.append(imp)
    return Layout(f"threads_{n_sites}", np.concatenate(xs), np.concatenate(ys),
                  np.concatenate(hs).astype(int), np.concatenate(imps).astype(int))


def _grid(n_side: int, pitch: float, drop_corners: bool) -> np.ndarray:
    c = (np.arange(n_side) - (n_side - 1) / 2) * pitch
    gx, gy = np.meshgrid(c, c)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    if drop_corners:
        lim = c.max()
        corner = (np.abs(pts[:, 0]) == lim) & (np.abs(pts[:, 1]) == lim)
        pts = pts[~corner]
    return pts


def utah_layout(n_arrays: int, n_side: int, ecc_min: float, ecc_max: float, seed: int = 0,
                pitch_mm: float = 0.4, drop_corners: bool = False,
                model: WedgeDipole | None = None, name: str | None = None) -> Layout:
    """Rigid Utah-like grids. One array goes in the left hemisphere; more are split over both."""
    model = model or WedgeDipole()
    rng = np.random.default_rng(seed)
    grid = _grid(n_side, pitch_mm, drop_corners)
    hemis = [0] if n_arrays == 1 else [i % 2 for i in range(n_arrays)]
    xs, ys, hs, imps = [], [], [], []
    centres: dict[int, list[np.ndarray]] = {0: [], 1: []}
    bx0, bx1, by0, by1 = _cortical_bbox(model, ecc_max)
    for i, h in enumerate(hemis):
        if n_arrays == 1:
            # Single array centred near 2 deg eccentricity on the horizontal meridian,
            # comparable to the parafoveal phosphene locations of Fernandez et al. (2021).
            cx, cy = model.to_cortex(np.array(2.0), np.array(0.0))
            centre, rot = np.array([float(cx), float(cy)]), 0.0
        else:
            best, best_d = None, -1.0
            for _ in range(400):
                cx, cy = rng.uniform(bx0, bx1), rng.uniform(by0, by1)
                rot = rng.uniform(0, np.pi / 2)
                R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
                pts = grid @ R.T + np.array([cx, cy])
                if not _valid_mask(model, pts[:, 0], pts[:, 1], ecc_min, ecc_max).all():
                    continue
                d = min([np.linalg.norm(np.array([cx, cy]) - c) for c in centres[h]] or [1e9])
                if d > best_d:
                    best, best_d = (np.array([cx, cy]), rot), d
            centre, rot = best
            centres[h].append(centre)
        R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        pts = grid @ R.T + centre
        xs.append(pts[:, 0]); ys.append(pts[:, 1])
        hs.append(np.full(len(pts), h)); imps.append(np.full(len(pts), i))
    n_total = sum(len(x) for x in xs)
    return Layout(name or f"utah_{n_total}", np.concatenate(xs), np.concatenate(ys),
                  np.concatenate(hs).astype(int), np.concatenate(imps).astype(int))


def make_layout(name: str, ecc_min: float, ecc_max: float, seed: int = 0) -> Layout:
    """Build a named layout: ``utah_96``, ``utah_1024`` or ``threads_<N>``."""
    if name == "utah_96":
        return utah_layout(1, 10, ecc_min, ecc_max, seed, drop_corners=True, name=name)
    if name == "utah_1024":
        return utah_layout(16, 8, ecc_min, ecc_max, seed, name=name)
    if name.startswith("threads_"):
        return thread_layout(int(name.split("_")[1]), ecc_min, ecc_max, seed)
    raise ValueError(f"unknown layout {name!r}")
