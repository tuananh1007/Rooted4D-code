r"""A fixed-topology encoding of a labelled mitral valve, for learned shape spaces.

``fit_valve`` resamples a valve into 32 angular sectors whose origin is the
first principal axis of the voxel cloud and whose leaflets occupy whichever
sectors their voxels happen to fall in. That is enough to score one fit against
its labels, which is what E40 and E42 do, but two fits do not share vertex
meaning: the principal axis has no preferred sign, and the leaflets claim
different numbers of sectors from case to case. A shape space needs every
vertex to mean the same thing in every case.

This encoding anchors everything to anatomy instead:

* **Frame.** The annulus plane and its normal come from ``fit_valve``'s frame
  (the normal points from annulus to coaptation). Angle zero points from the
  valve centre to the anterior leaflet's centroid, so the rotation about the
  normal is fixed by the anterior leaflet rather than by a principal axis.
* **Commissures.** Walking round the annulus from angle zero in both
  directions, each commissure is the first angle at which posterior voxels
  outnumber anterior ones near the annulus. The anterior arc runs between
  them through zero; the posterior arc is the rest of the ring.
* **Nodes.** A fixed number of annular nodes is placed at fixed fractions of
  each arc (12 anterior and 20 posterior by default, about the leaflets'
  shares of the annulus). Each node carries a depth profile of its own
  leaflet's voxels, sampled exactly as ``fit_valve`` samples a sector. The
  node is the profile's root, so attachment stays structural.

Every fit therefore has the same vertex count and the same vertex meaning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rooted4d.donors.valve import ANTERIOR_LABEL, POSTERIOR_LABEL, _valve_frame

MINIMUM_VOXELS = 20
COMMISSURE_BINS = 72  # 5-degree bins for the dominance walk


@dataclass(frozen=True)
class FixedValveFit:
    annulus: np.ndarray        # (n_anterior + n_posterior, 3): anterior nodes, then posterior
    anterior: np.ndarray       # (n_anterior, n_depth, 3); row 0 of each profile is its annulus node
    posterior: np.ndarray      # (n_posterior, n_depth, 3)
    commissure_angles: tuple[float, float]
    case: str = ""

    @property
    def n_anterior(self) -> int:
        return self.anterior.shape[0]

    @property
    def n_posterior(self) -> int:
        return self.posterior.shape[0]

    @property
    def n_depth(self) -> int:
        return self.anterior.shape[1]

    def vertices(self) -> np.ndarray:
        """Annulus first, then each leaflet's rows below the root, node by node."""

        return np.vstack([self.annulus, self.anterior[:, 1:].reshape(-1, 3),
                          self.posterior[:, 1:].reshape(-1, 3)])


def fixed_valve_topology(n_anterior: int = 12, n_posterior: int = 20, n_depth: int = 6) -> dict:
    """Vertex indices and faces of the encoding, in ``FixedValveFit.vertices`` order."""

    n_annulus = n_anterior + n_posterior
    below = n_depth - 1

    def sheet(first_node: int, count: int, first_interior: int) -> np.ndarray:
        # grid[k, d]: vertex of node k at depth d; depth 0 is the annulus node.
        grid = np.empty((count, n_depth), dtype=np.int64)
        grid[:, 0] = first_node + np.arange(count)
        grid[:, 1:] = first_interior + np.arange(count * below).reshape(count, below)
        return grid

    anterior = sheet(0, n_anterior, n_annulus)
    posterior = sheet(n_anterior, n_posterior, n_annulus + n_anterior * below)

    def strip(grid: np.ndarray) -> np.ndarray:
        faces = []
        for k in range(grid.shape[0] - 1):
            for d in range(n_depth - 1):
                a, b = grid[k, d], grid[k + 1, d]
                faces.append((a, b, grid[k + 1, d + 1]))
                faces.append((a, grid[k + 1, d + 1], grid[k, d + 1]))
        return np.asarray(faces, dtype=np.int64)

    return {"n_vertices": n_annulus + n_annulus * below, "annulus": np.arange(n_annulus),
            "anterior_grid": anterior, "posterior_grid": posterior,
            "faces": {"anterior": strip(anterior), "posterior": strip(posterior)},
            "components": {"anterior": np.sort(anterior.ravel()), "posterior": np.sort(posterior.ravel())}}


def _wrap(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _profile(local_points: np.ndarray, local_height: np.ndarray, n_depth: int) -> np.ndarray:
    """One sector's depth profile, sampled exactly as ``fit_valve`` samples it."""

    sheet = np.empty((n_depth, 3))
    sheet[0] = local_points[local_height <= np.quantile(local_height, 0.10)].mean(axis=0)
    levels = np.linspace(np.quantile(local_height, 0.10), np.quantile(local_height, 0.90), n_depth)
    for step in range(1, n_depth):
        band = (local_height >= levels[step - 1]) & (local_height <= levels[step])
        sheet[step] = local_points[band].mean(axis=0) if band.sum() >= 3 else sheet[step - 1]
    sheet[-1] = local_points[local_height >= np.quantile(local_height, 0.90)].mean(axis=0)
    return sheet


@dataclass(frozen=True)
class ValveCommissures:
    """A labelled valve's anatomical frame and its two commissures.

    Angle zero points from the valve centre to the anterior leaflet's centroid,
    about ``normal`` (annulus towards coaptation); ``angles`` are the two
    commissures in that frame, the anterior arc running from the first to the
    second through zero.
    """

    centre: np.ndarray
    normal: np.ndarray
    reference: np.ndarray
    binormal: np.ndarray
    angles: tuple[float, float]

    def direction(self, angle: float) -> np.ndarray:
        """The unit in-plane direction at ``angle`` in this frame."""

        return np.cos(angle) * self.reference + np.sin(angle) * self.binormal


def _labelled_points(labels: np.ndarray, zooms):
    volume = np.asarray(labels)
    spacing = np.asarray(zooms[:3], dtype=np.float64)
    occupied = np.argwhere(volume > 0)
    if len(occupied) < 500:
        return None
    points = occupied * spacing
    which = volume[tuple(occupied.T)]
    if (which == ANTERIOR_LABEL).sum() < 200 or (which == POSTERIOR_LABEL).sum() < 200:
        return None
    return points, which


def _commissure_frame(points: np.ndarray, which: np.ndarray):
    """Frame, per-point height and angle, and the commissure angles (None if the walk fails)."""

    centre, normal, _ = _valve_frame(points)
    centred = points - centre
    height = centred @ normal
    planar = centred - np.outer(height, normal)
    anterior_direction = planar[which == ANTERIOR_LABEL].mean(axis=0)
    reference = anterior_direction / np.linalg.norm(anterior_direction)
    binormal = np.cross(normal, reference)
    angle = np.arctan2(planar @ binormal, planar @ reference)
    frame = (centre, normal, reference, binormal)

    # Commissures: near the annulus, where each leaflet owns its arc, walk out
    # from angle zero until posterior voxels outnumber anterior ones.
    near = height <= np.quantile(height, 0.5)
    edges = np.linspace(-np.pi, np.pi, COMMISSURE_BINS + 1)
    counts = {label: np.histogram(angle[near & (which == label)], bins=edges)[0].astype(float)
              for label in (ANTERIOR_LABEL, POSTERIOR_LABEL)}
    kernel = np.array([0.25, 0.5, 0.25])

    def smooth(values):
        padded = np.concatenate([values[-1:], values, values[:1]])
        return np.convolve(padded, kernel, mode="valid")

    dominance = smooth(counts[ANTERIOR_LABEL]) - smooth(counts[POSTERIOR_LABEL])
    centres = 0.5 * (edges[:-1] + edges[1:])
    zero = int(np.argmin(np.abs(centres)))
    boundary = []
    for step in (-1, 1):
        index, walked = zero, 0
        while dominance[index % COMMISSURE_BINS] > 0.0 and walked < COMMISSURE_BINS:
            index += step
            walked += 1
        if walked == 0 or walked >= COMMISSURE_BINS:
            return frame, height, angle, None
        # The commissure lies between the last anterior bin and the first posterior one.
        boundary.append(float(centres[(index - step) % COMMISSURE_BINS] + step * np.pi / COMMISSURE_BINS))
    return frame, height, angle, (boundary[0], boundary[1])


def valve_commissures(labels: np.ndarray, zooms) -> ValveCommissures | None:
    """The two commissures of a labelled valve, from its leaflet labels (E101).

    The same walk ``fit_valve_fixed`` uses, without its resampling, so a valve
    whose leaflets are too sparse for twelve and twenty profiles still has its
    commissures. None where a leaflet is too sparse or the walk fails.
    """

    labelled = _labelled_points(labels, zooms)
    if labelled is None:
        return None
    frame, _, _, angles = _commissure_frame(*labelled)
    if angles is None:
        return None
    centre, normal, reference, binormal = frame
    return ValveCommissures(centre=centre, normal=normal, reference=reference, binormal=binormal,
                            angles=angles)


def fit_valve_fixed(labels: np.ndarray, zooms, *, n_anterior: int = 12, n_posterior: int = 20,
                    n_depth: int = 6, case: str = "") -> FixedValveFit | None:
    """Encode one labelled valve in the fixed topology, or None if a leaflet is too sparse."""

    labelled = _labelled_points(labels, zooms)
    if labelled is None:
        return None
    points, which = labelled
    _, height, angle, angles = _commissure_frame(points, which)
    if angles is None:
        return None
    low, high = angles
    anterior_span = _wrap(high - low) % (2.0 * np.pi)
    posterior_span = 2.0 * np.pi - anterior_span

    rows = {}
    for label, count, start, span in ((ANTERIOR_LABEL, n_anterior, low, anterior_span),
                                      (POSTERIOR_LABEL, n_posterior, high, posterior_span)):
        mine = which == label
        half = 0.5 * span / count
        profiles = []
        for k in range(count):
            target = _wrap(start + (k + 0.5) * span / count)
            for widen in (1.0, 2.0, 3.0):
                in_sector = mine & (np.abs(_wrap(angle - target)) <= widen * half)
                if in_sector.sum() >= MINIMUM_VOXELS:
                    break
            else:
                return None
            profiles.append(_profile(points[in_sector], height[in_sector], n_depth))
        rows[label] = np.stack(profiles)
    anterior, posterior = rows[ANTERIOR_LABEL], rows[POSTERIOR_LABEL]
    return FixedValveFit(annulus=np.vstack([anterior[:, 0], posterior[:, 0]]), anterior=anterior,
                         posterior=posterior, commissure_angles=(low, high), case=case)
