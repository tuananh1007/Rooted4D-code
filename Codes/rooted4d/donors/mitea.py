r"""Fit MITEA left-ventricle cavities into the shared topology.

MITEA is isotropic 3D echocardiography, not a short-axis stack, so the ACDC
fitter does not apply: it slices along the image z axis and assumes that axis is
the ventricular long axis. Here the long axis has to be found from the anatomy.

The reason this module exists is to estimate the session-discrepancy subspace
:math:`U_s`. Two scans of one subject at one phase differ only by between-session
effects, so a per-vertex difference field between them is a direct sample of the
quantity the model's session term is supposed to represent. Section 13 measured
that difference as about 1.19 mm after rigid alignment.

**One detail decides whether any of this is meaningful.** A ring built by
angular binning needs a reference direction, and the ventricle is near
axisymmetric, so an independently chosen reference for each scan leaves the two
fits rotated against each other by an arbitrary amount. The difference field
would then be dominated by a correspondence artefact that looks exactly like a
large session effect. Both scans are therefore rigidly aligned first and fitted
in a single shared frame derived from the first, so vertex ``k`` means the same
anatomical location in both.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "MiteaFit",
    "cavity_world_points",
    "fit_cavity_in_frame",
    "lv_frame",
    "rigid_align",
]

CAVITY_LABEL = 2


@dataclass(frozen=True)
class MiteaFit:
    """One MITEA cavity resampled onto a ring grid in a given frame."""

    vertices: np.ndarray
    n_rings: int
    n_angles: int
    subject: str
    scan: str
    phase: str


def cavity_world_points(path: str | Path, label: int = CAVITY_LABEL) -> np.ndarray:
    """World coordinates of the labelled cavity voxels."""

    import nibabel as nib

    image = nib.load(str(path))
    index = np.argwhere(np.asarray(image.dataobj) == label)
    if len(index) == 0:
        raise ValueError(f"no voxels with label {label} in {path}")
    return nib.affines.apply_affine(image.affine, index)


def lv_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre, long axis and a reference direction for a cavity point cloud.

    The long axis is the leading principal direction, which for a ventricle is
    base to apex. Its sign is fixed so it points from base toward apex, decided
    by which end carries the wider cross-section rather than by the sign the
    eigensolver happens to return, since that sign is arbitrary and would flip
    the fit end for end between two scans of the same heart.
    """

    cloud = np.asarray(points, dtype=np.float64)
    centre = cloud.mean(axis=0)
    centred = cloud - centre
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    axis = right[0]

    height = centred @ axis
    lower = centred[height < np.quantile(height, 0.25)]
    upper = centred[height > np.quantile(height, 0.75)]

    def spread(block: np.ndarray) -> float:
        radial = block - np.outer(block @ axis, axis)
        return float(np.linalg.norm(radial, axis=1).mean())

    if spread(upper) > spread(lower):
        axis = -axis
    # Reference direction: any unit vector orthogonal to the axis. It is derived
    # from the axis deterministically so the same cloud always yields the same
    # frame; consistency *between* scans comes from sharing one frame, not from
    # this choice.
    seed = np.array([1.0, 0.0, 0.0])
    if abs(seed @ axis) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    reference = seed - (seed @ axis) * axis
    reference /= np.linalg.norm(reference)
    return centre, axis, reference


def rigid_align(source: np.ndarray, target: np.ndarray, iterations: int = 15):
    """Rigid ICP of one cavity cloud onto another, with subsampling.

    Correspondences are inferred, because two independent segmentations of the
    same heart share no vertex identity. Returns the transformed source together
    with the rotation and translation.
    """

    rng = np.random.default_rng(0)

    def thin(cloud: np.ndarray, count: int = 2000) -> np.ndarray:
        if len(cloud) <= count:
            return cloud
        return cloud[rng.choice(len(cloud), count, replace=False)]

    fixed = thin(np.asarray(target, dtype=np.float64))
    moving = thin(np.asarray(source, dtype=np.float64))
    rotation = np.eye(3)
    translation = np.zeros(3)
    current = moving.copy()
    for _ in range(iterations):
        distance = np.sqrt(
            ((current[:, None, :] - fixed[None, :, :]) ** 2).sum(axis=2)
        )
        matched = fixed[distance.argmin(axis=1)]
        source_centre = current.mean(axis=0)
        target_centre = matched.mean(axis=0)
        matrix = (current - source_centre).T @ (matched - target_centre)
        u, _, vt = np.linalg.svd(matrix)
        sign = np.sign(np.linalg.det(vt.T @ u.T))
        step = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
        shift = target_centre - step @ source_centre
        current = current @ step.T + shift
        rotation = step @ rotation
        translation = step @ translation + shift
    aligned = np.asarray(source, dtype=np.float64) @ rotation.T + translation
    return aligned, rotation, translation


def fit_cavity_in_frame(
    points: np.ndarray,
    frame: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    n_rings: int = 8,
    n_angles: int = 32,
    subject: str = "",
    scan: str = "",
    phase: str = "",
) -> MiteaFit | None:
    """Resample a cavity onto rings in a *given* frame.

    The frame is an argument rather than being derived here, so that two scans
    of one subject can be fitted in the same one. Deriving it per scan would
    leave the two fits rotated against each other about a near-axisymmetric
    ventricle, and the resulting correspondence error would masquerade as a
    session effect.
    """

    centre, axis, reference = frame
    binormal = np.cross(axis, reference)
    cloud = np.asarray(points, dtype=np.float64)
    centred = cloud - centre
    height = centred @ axis
    planar = centred - np.outer(height, axis)
    angle = np.arctan2(planar @ binormal, planar @ reference)
    radius = np.linalg.norm(planar, axis=1)

    low, high = np.quantile(height, 0.02), np.quantile(height, 0.98)
    levels = np.linspace(low, high, n_rings)
    half = 0.5 * (levels[1] - levels[0]) if n_rings > 1 else 1.0
    edges = np.linspace(-np.pi, np.pi, n_angles + 1)
    sector = np.clip(np.digitize(angle, edges) - 1, 0, n_angles - 1)

    rings = np.empty((n_rings, n_angles, 3))
    for ring_index, level in enumerate(levels):
        slab = np.abs(height - level) <= half
        if slab.sum() < n_angles:
            return None
        slab_radius = radius[slab]
        slab_sector = sector[slab]
        # Outermost sample per sector, matching the ACDC contour rule: a mean
        # would bias the boundary inward wherever the segmentation is thick.
        profile = np.zeros(n_angles)
        np.maximum.at(profile, slab_sector, slab_radius)
        empty = profile <= 0.0
        if empty.all():
            return None
        if empty.any():
            present = np.flatnonzero(~empty)
            for index in np.flatnonzero(empty):
                nearest = present[np.argmin(np.abs(present - index))]
                profile[index] = profile[nearest]
        centres = 0.5 * (edges[:-1] + edges[1:])
        rings[ring_index] = (
            centre
            + level * axis
            + profile[:, None] * (np.cos(centres)[:, None] * reference
                                  + np.sin(centres)[:, None] * binormal)
        )

    vertices = rings.reshape(-1, 3)
    base = rings[0].mean(axis=0)
    apex = rings[-1].mean(axis=0) + half * axis
    return MiteaFit(
        vertices=np.vstack([vertices, base[None, :], apex[None, :]]),
        n_rings=n_rings,
        n_angles=n_angles,
        subject=subject,
        scan=scan,
        phase=phase,
    )
