r"""Fit a fixed-topology LV surface to ACDC segmentations.

Everything in the plane-selection experiment so far ran on procedural geometry.
The contour noise was measured on real ACDC contours, but the anatomy the
selector reasoned about was synthetic, so the chain from measured noise to
measured benefit still rested on an assumption. This closes that.

ACDC supplies short-axis stacks with one expert delineation per annotated frame,
label 3 being the left-ventricular cavity, at roughly 1.56 mm in plane and 10 mm
between slices. The fit resamples each cavity into a fixed cylindrical topology:
``n_rings`` levels along the long axis, ``n_angles`` samples around each ring,
plus a base and an apex cap. Every patient then has the same vertex count in the
same order, which is what a shape basis and a plane-selection Jacobian both
require.

The anisotropy is the honest limitation here. Ten millimetres between slices
against 1.56 mm in plane means the long-axis direction is interpolated from far
fewer measurements than the circumferential one, and a shape basis learned from
these fits will be correspondingly smoother along the axis than the truth. That
is a property of the acquisition, not of the fit, and it is the reason the
project wants sparse-slice reconstruction in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def slice_cavity_contour(
    mask: np.ndarray, n_angles: int, spacing_mm: tuple[float, float]
) -> np.ndarray | None:
    """Resample one slice's cavity boundary to fixed angular samples.

    Returns ``None`` when the slice holds too little cavity to define a
    boundary, which happens at the apex and beyond the valve plane. Callers must
    handle that rather than assume every slice contributes.
    """

    binary = np.asarray(mask).astype(bool)
    if binary.sum() < 20:
        return None
    rows, columns = np.nonzero(binary)
    points = np.stack([rows * spacing_mm[0], columns * spacing_mm[1]], axis=1)
    centre = points.mean(axis=0)

    padded = np.pad(binary, 1)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary = binary & ~interior
    rows, columns = np.nonzero(boundary)
    if rows.size < 8:
        return None
    edge = np.stack([rows * spacing_mm[0], columns * spacing_mm[1]], axis=1)
    relative = edge - centre
    angles = np.arctan2(relative[:, 0], relative[:, 1])
    radii = np.linalg.norm(relative, axis=1)

    # Radius as a function of angle, taking the outermost boundary pixel in each
    # angular bin. Using the maximum rather than a mean avoids biasing the
    # cavity inward where the boundary is two pixels thick.
    targets = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    resampled = np.empty(n_angles)
    order = np.argsort(angles)
    sorted_angles, sorted_radii = angles[order], radii[order]
    for index, target in enumerate(targets):
        difference = np.abs(np.angle(np.exp(1j * (sorted_angles - target))))
        window = difference < (np.pi / n_angles) * 2.0
        resampled[index] = (
            sorted_radii[window].max()
            if window.any()
            else sorted_radii[np.argmin(difference)]
        )
    return np.stack(
        [
            centre[0] + resampled * np.sin(targets),
            centre[1] + resampled * np.cos(targets),
        ],
        axis=1,
    )


@dataclass(frozen=True)
class LVFit:
    """One patient-frame fitted into the shared topology."""

    vertices: np.ndarray
    n_rings: int
    n_angles: int
    n_slices_used: int
    patient: str
    frame: int

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]


def lv_topology_faces(n_rings: int, n_angles: int) -> np.ndarray:
    """Triangles for the cylindrical topology, closed at base and apex.

    Closure matters: the enclosed-volume calculation the clinical quantities use
    is only defined on a closed surface, and an open base would make every
    volume meaningless rather than merely inaccurate.
    """

    faces: list[tuple[int, int, int]] = []
    for ring in range(n_rings - 1):
        for angle in range(n_angles):
            following = (angle + 1) % n_angles
            lower = ring * n_angles
            upper = (ring + 1) * n_angles
            faces.append((lower + angle, upper + angle, upper + following))
            faces.append((lower + angle, upper + following, lower + following))
    # Cap winding must oppose the lateral surface's traversal of the end rings,
    # or the surface is not consistently oriented and the divergence-theorem
    # volume is wrong. The lateral faces traverse ring 0 as (i+1, i) and the last
    # ring as (i, i+1), so the caps take the reverse of each. Getting this
    # backwards is silent: the surface still looks closed, every edge still has
    # two incident faces, and only the volume is corrupted.
    base_centre = n_rings * n_angles
    apex_centre = base_centre + 1
    last = (n_rings - 1) * n_angles
    for angle in range(n_angles):
        following = (angle + 1) % n_angles
        faces.append((base_centre, angle, following))
        faces.append((apex_centre, last + following, last + angle))
    return np.asarray(faces, dtype=np.int64)


def fit_lv_surface(
    labels: np.ndarray,
    zooms: tuple[float, float, float],
    *,
    n_rings: int = 8,
    n_angles: int = 32,
    cavity_label: int = 3,
    patient: str = "",
    frame: int = 0,
    base_first: bool | None = None,
) -> LVFit | None:
    """Resample an ACDC cavity segmentation into the shared topology.

    ``base_first`` says which end of the stack is the base. By default it is
    decided by the ring radii, as the published fits were. ACDC stacks all come
    in one order, slice 0 basal, and the radius test gets 4 of its 300 annotated
    frames wrong (end-systolic frames whose last slice is nearly empty, and a
    stack that stops short of the apex). A caller that needs vertex
    correspondence across patients passes ``True``.
    """

    volume = np.asarray(labels)
    if volume.ndim != 3:
        raise ValueError("labels must be a 3D short-axis stack")
    spacing = (float(zooms[0]), float(zooms[1]))
    slice_thickness = float(zooms[2])

    contours = []
    positions = []
    for index in range(volume.shape[2]):
        contour = slice_cavity_contour(volume[:, :, index] == cavity_label, n_angles, spacing)
        if contour is not None:
            contours.append(contour)
            positions.append(index * slice_thickness)
    if len(contours) < 3:
        return None

    # Resample along the long axis so every patient has the same ring count
    # regardless of how many slices happened to contain cavity.
    positions = np.asarray(positions)
    targets = np.linspace(positions[0], positions[-1], n_rings)
    stacked = np.stack(contours, axis=0)
    rings = np.empty((n_rings, n_angles, 3))
    for angle in range(n_angles):
        for axis in range(2):
            rings[:, angle, axis] = np.interp(targets, positions, stacked[:, angle, axis])
    rings[:, :, 2] = targets[:, None]

    vertices = rings.reshape(-1, 3)

    # Which end is the valve plane and which is the apex is not given by slice
    # order, which depends on the acquisition. The base has the wider cavity, so
    # decide by radius rather than assume. Getting it backwards would put a cone
    # where the valve plane belongs.
    first_radius = float(
        np.mean(np.linalg.norm(rings[0, :, :2] - rings[0, :, :2].mean(axis=0), axis=1))
    )
    last_radius = float(
        np.mean(np.linalg.norm(rings[-1, :, :2] - rings[-1, :, :2].mean(axis=0), axis=1))
    )
    base_is_first = first_radius >= last_radius if base_first is None else bool(base_first)

    # The two ends are not symmetric anatomically and must not be capped the
    # same way. The valve plane truncates the cavity, so the basal cap is flat at
    # the ring itself. The apex tapers to a point beyond the last slice, so its
    # cap is a cone extended half a slice. Both caps are fans to a single point,
    # so a cone fills only a third of the cylinder it spans; capping the base
    # with an extended cone was losing two thirds of a half-slice there.
    base = vertices[:n_angles].mean(axis=0) if base_is_first else vertices[-n_angles:].mean(axis=0)
    apex = vertices[-n_angles:].mean(axis=0) if base_is_first else vertices[:n_angles].mean(axis=0)
    apex[2] += (0.5 * slice_thickness) * (1.0 if base_is_first else -1.0)
    if not base_is_first:
        # Keep the cap ordering the topology expects: cap index n*R attaches to
        # ring 0 and index n*R+1 to the last ring.
        base, apex = apex, base
    return LVFit(
        vertices=np.vstack([vertices, base[None, :], apex[None, :]]),
        n_rings=n_rings,
        n_angles=n_angles,
        n_slices_used=len(contours),
        patient=patient,
        frame=frame,
    )


def load_acdc_fits(
    root: str | Path,
    *,
    n_rings: int = 8,
    n_angles: int = 32,
    max_patients: int | None = None,
) -> list[LVFit]:
    """Fit every annotated ACDC frame found under ``root``."""

    import nibabel as nib

    paths = sorted(Path(root).rglob("*_gt.nii.gz"))
    fits: list[LVFit] = []
    seen: set[str] = set()
    for path in paths:
        patient = path.parent.name
        if max_patients is not None and patient not in seen and len(seen) >= max_patients:
            continue
        seen.add(patient)
        image = nib.load(str(path))
        frame_token = path.name.split("_frame")[-1].split("_")[0]
        fit = fit_lv_surface(
            np.asarray(image.dataobj),
            image.header.get_zooms()[:3],
            n_rings=n_rings,
            n_angles=n_angles,
            patient=patient,
            frame=int(frame_token),
        )
        if fit is not None:
            fits.append(fit)
    return fits


@dataclass(frozen=True)
class ACDCShapeModel:
    """Identity and motion bases learned from ACDC fits.

    This is the work-package-3 object built from real geometry rather than from
    an analytic smooth basis. Identity modes come from principal components of
    the end-diastolic shapes; motion modes from principal components of the
    end-diastolic to end-systolic difference. Two phases is all ACDC annotates,
    which is enough for the volumes and the ejection fraction and not enough for
    a periodic basis, so no periodic structure is claimed here.
    """

    mean_shape: np.ndarray
    #: Mean end-diastolic to end-systolic deformation. Held separately from the
    #: motion basis because it is the physiologically dominant component:
    #: centring it away left the mean shape with an ejection fraction of exactly
    #: zero, a heart that does not beat.
    mean_motion: np.ndarray
    identity_basis: np.ndarray
    motion_basis: np.ndarray
    faces: np.ndarray
    n_rings: int
    n_angles: int
    identity_variance_explained: np.ndarray
    motion_variance_explained: np.ndarray
    n_patients: int

    @property
    def n_vertices(self) -> int:
        return self.mean_shape.shape[0] // 3

    @property
    def n_identity(self) -> int:
        return self.identity_basis.shape[1]

    @property
    def n_motion(self) -> int:
        return self.motion_basis.shape[1]

    def shape_at(self, identity: np.ndarray, motion: np.ndarray, phase: float) -> np.ndarray:
        """Vertices at ``phase`` 0 for end diastole and 1 for end systole."""

        flat = self.mean_shape + self.identity_basis @ identity
        flat = flat + float(phase) * (self.mean_motion + self.motion_basis @ motion)
        return flat.reshape(-1, 3)

    def project(self, end_diastolic: np.ndarray, end_systolic: np.ndarray):
        """Least-squares coefficients for one patient."""

        identity = self.identity_basis.T @ (end_diastolic.reshape(-1) - self.mean_shape)
        difference = end_systolic.reshape(-1) - end_diastolic.reshape(-1)
        return identity, self.motion_basis.T @ (difference - self.mean_motion)


def build_acdc_shape_model(
    fits: list[LVFit],
    *,
    n_identity: int = 10,
    n_motion: int = 3,
) -> ACDCShapeModel:
    """Learn identity and motion bases from paired end-diastolic/systolic fits.

    Shapes are centred on their own centroid before the decomposition, so the
    bases describe anatomy rather than where in the scanner the patient lay.
    Leaving translation in would spend the leading modes on position, and the
    plane selector would then be reasoning about a basis dominated by something
    a localizer already resolves.
    """

    by_patient: dict[str, list[LVFit]] = {}
    for fit in fits:
        by_patient.setdefault(fit.patient, []).append(fit)

    diastolic, systolic = [], []
    for patient, frames in sorted(by_patient.items()):
        if len(frames) < 2:
            continue
        ordered = sorted(frames, key=lambda f: f.frame)
        first, second = ordered[0].vertices, ordered[-1].vertices
        # The larger cavity is end diastole; frame order alone does not
        # guarantee it and mislabelling would invert every ejection fraction.
        if _cavity_extent(second) > _cavity_extent(first):
            first, second = second, first
        diastolic.append((first - first.mean(axis=0)).reshape(-1))
        systolic.append((second - first.mean(axis=0)).reshape(-1))
    if len(diastolic) < 4:
        raise ValueError("at least four paired patients are required")

    diastolic_matrix = np.stack(diastolic, axis=0)
    mean_shape = diastolic_matrix.mean(axis=0)
    centred = diastolic_matrix - mean_shape
    _, identity_values, identity_vectors = np.linalg.svd(centred, full_matrices=False)
    identity_basis = identity_vectors[:n_identity].T

    difference_matrix = np.stack(systolic, axis=0) - diastolic_matrix
    mean_motion = difference_matrix.mean(axis=0)
    _, motion_values, motion_vectors = np.linalg.svd(
        difference_matrix - mean_motion, full_matrices=False
    )
    motion_basis = motion_vectors[:n_motion].T

    reference = fits[0]
    return ACDCShapeModel(
        mean_shape=mean_shape,
        mean_motion=mean_motion,
        identity_basis=identity_basis,
        motion_basis=motion_basis,
        faces=lv_topology_faces(reference.n_rings, reference.n_angles),
        n_rings=reference.n_rings,
        n_angles=reference.n_angles,
        identity_variance_explained=(identity_values**2 / np.sum(identity_values**2))[
            :n_identity
        ],
        motion_variance_explained=(motion_values**2 / np.sum(motion_values**2))[:n_motion],
        n_patients=len(diastolic),
    )


def _cavity_extent(vertices: np.ndarray) -> float:
    """Scale proxy used only to decide which frame is end diastole."""

    centred = vertices - vertices.mean(axis=0)
    return float(np.mean(np.linalg.norm(centred, axis=1)))
