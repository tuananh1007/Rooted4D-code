r"""Fit MVSeg2023 mitral segmentations to a fixed valve topology.

The specification's Revision 0.5 route needs real valve donors in the same
topology as every other donor, so they can be paired with chamber donors inside
a split. This produces them.

MVSeg2023 labels the leaflets as thick tissue rather than as surfaces, around
140,000 voxels for an anterior leaflet, so a mid-surface has to be extracted
rather than a boundary taken. The structure that makes this tractable is the
valve's funnel: measured on the data, the combined leaflets run from a mean
radius near 25 mm at the annulus to near 13 mm at the coaptation edge, along the
normal of their own best-fit plane. Annulus and free edge are therefore
separable by position along that normal, without needing either to be labelled.

Fitted topology, matching section 3: one cyclic annular ring, and two open
leaflet sheets whose root rows *are* annular vertices rather than copies of
them. Attachment is structural in the donor as it is in the model.

What this does not do is claim the topology is frozen. Freezing needs the
annular-resolution study and clinical review that `production_template_frozen`
is still blocked on. This is a fixed topology, consistently applied.

**Known anatomical shortfall.** Over twenty training cases the fit reproduces the
annular-arc ordering correctly, the posterior leaflet occupying more of the
annulus than the anterior, but not the leaflet lengths. Both come out near 15 mm
where anatomy expects roughly 20 to 25 mm anterior against 10 to 15 mm
posterior, and measuring arc length along the sheet rather than straight-line
extent does not recover the ordering either. The likely cause is the depth
sampling: clipping each sector at its tenth and ninetieth height percentiles
truncates the longer anterior leaflet while the posterior's segmented tissue
runs up into the annular region. Donors are therefore geometrically consistent
and in correspondence, which is what pairing requires, but they should not be
used to support a claim about absolute leaflet length.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# MVSeg2023 label convention, taken from the dataset card rather than assumed:
# "0 - Background  1 - Posterior leaflet  2 - Anterior leaflet". These were the
# other way round here, which swapped the two leaflets everywhere downstream.
# The error was visible in the results and was misread: the fitted "posterior"
# leaflet came out longer than the "anterior" one, 33.5 mm against 27.3, which
# is the reverse of mitral anatomy, and that was written up as an unexplained
# anomaly in the leaflet-span metric. With the labels the right way round the
# measurement agrees with anatomy, and the anomaly was never in the data.
POSTERIOR_LABEL = 1
ANTERIOR_LABEL = 2


@dataclass(frozen=True)
class ValveFit:
    """One MVSeg2023 case in the shared valve topology."""

    annulus: np.ndarray
    anterior: np.ndarray
    posterior: np.ndarray
    anterior_root_indices: np.ndarray
    posterior_root_indices: np.ndarray
    case: str
    n_annulus: int
    n_depth: int

    @property
    def n_vertices(self) -> int:
        """Annulus plus the non-root rows of each leaflet.

        Root rows are annular vertices, not duplicates, so they are counted once.
        """

        return (
            len(self.annulus)
            + self.anterior.shape[0] * (self.n_depth - 1)
            + self.posterior.shape[0] * (self.n_depth - 1)
        )

    def annulus_diameter_mm(self) -> float:
        centre = self.annulus.mean(axis=0)
        return float(2.0 * np.linalg.norm(self.annulus - centre, axis=1).mean())

    def leaflet_span_mm(self, which: str) -> float:
        """Mean root-to-free-edge distance.

        This is *not* clinical leaflet length. Depth is binned along the
        valve-plane normal, so this measures vertical drop below the annulus,
        and two leaflets hanging at different angles can have equal drop and
        very different surface length. On MVSeg2023 it reports the posterior
        leaflet as the larger of the two, which is the ordering of the height
        extents in that frame and the opposite of the anatomical ordering of
        leaflet length. Use it as a geometric descriptor, not a measurement.
        """

        sheet = self.anterior if which == "anterior" else self.posterior
        return float(np.mean(np.linalg.norm(sheet[:, -1] - sheet[:, 0], axis=1)))


def _valve_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre, annulus-plane normal oriented towards the free edge, and a reference axis.

    The normal is oriented so that increasing height moves from annulus to
    coaptation, decided by which end is wider. Leaving the orientation to chance
    would flip root and free edge on roughly half the cases.
    """

    centre = points.mean(axis=0)
    centred = points - centre
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    normal = right[2]
    height = centred @ normal
    radius = np.linalg.norm(centred - np.outer(height, normal), axis=1)
    low, high = height < np.quantile(height, 0.25), height > np.quantile(height, 0.75)
    if radius[low].mean() < radius[high].mean():
        normal = -normal
    reference = right[0] - (right[0] @ normal) * normal
    return centre, normal, reference / np.linalg.norm(reference)


def fit_valve(
    labels: np.ndarray,
    zooms: tuple[float, float, float],
    *,
    n_annulus: int = 32,
    n_depth: int = 6,
    case: str = "",
    frame: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> ValveFit | None:
    """Resample one labelled valve into the shared topology.

    ``frame`` (centre, normal, in-plane reference axis) replaces the default
    frame, whose reference is the first principal axis of the voxel cloud and
    has no preferred sign. Sector 0 starts just past the direction opposite the
    reference, so the frame fixes where the ring's numbering begins (E101
    numbers a predicted valve from its commissures this way).
    """

    volume = np.asarray(labels)
    spacing = np.asarray(zooms[:3], dtype=np.float64)
    occupied = np.argwhere(volume > 0)
    if len(occupied) < 500:
        return None
    points = occupied * spacing
    centre, normal, reference = _valve_frame(points) if frame is None else (
        np.asarray(frame[0], dtype=np.float64), np.asarray(frame[1], dtype=np.float64),
        np.asarray(frame[2], dtype=np.float64))
    binormal = np.cross(normal, reference)

    centred = points - centre
    height = centred @ normal
    planar = centred - np.outer(height, normal)
    angle = np.arctan2(planar @ binormal, planar @ reference)
    which = volume[tuple(occupied.T)]

    # Each leaflet is resampled from its own voxels. Binning by angle alone
    # mixes them: near the commissures and throughout the coaptation zone both
    # leaflets occupy the same angular sector at different heights, so a shared
    # depth profile averaged across the two and reported a posterior leaflet
    # longer than the anterior, which is the wrong way round.
    edges = np.linspace(-np.pi, np.pi, n_annulus + 1)

    def resample(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sectors, sheets, occupied_bins = [], [], []
        for index in range(n_annulus):
            in_bin = mask & (angle >= edges[index]) & (angle < edges[index + 1])
            if in_bin.sum() < 20:
                continue
            local_height = height[in_bin]
            local_points = points[in_bin]
            root = local_points[
                local_height <= np.quantile(local_height, 0.10)
            ].mean(axis=0)
            levels = np.linspace(
                np.quantile(local_height, 0.10),
                np.quantile(local_height, 0.90),
                n_depth,
            )
            sheet = np.empty((n_depth, 3))
            sheet[0] = root
            for step in range(1, n_depth):
                band = (local_height >= levels[step - 1]) & (
                    local_height <= levels[step]
                )
                sheet[step] = (
                    local_points[band].mean(axis=0) if band.sum() >= 3 else sheet[step - 1]
                )
            # The free edge is the coaptation surface, and it is the only row
            # ``valve_donor_vertices`` takes from this fit, so it has to be the
            # actual edge rather than the top of the last band. The bands stop at
            # the 90th percentile of per-sector height, which leaves roughly half
            # the leaflet's extent above them. Averaging the deepest decile
            # reaches the edge while still averaging enough voxels to stay robust
            # to a ragged segmentation boundary.
            sheet[-1] = local_points[
                local_height >= np.quantile(local_height, 0.90)
            ].mean(axis=0)
            sectors.append(root)
            sheets.append(sheet)
            occupied_bins.append(index)
        if not sheets:
            return np.empty((0, 3)), np.empty((0, n_depth, 3)), np.empty(0, dtype=np.int64)
        return (
            np.stack(sectors, axis=0),
            np.stack(sheets, axis=0),
            np.asarray(occupied_bins, dtype=np.int64),
        )

    anterior_roots, anterior_sheets, anterior_bins = resample(which == ANTERIOR_LABEL)
    posterior_roots, posterior_sheets, posterior_bins = resample(which == POSTERIOR_LABEL)
    anterior_rows = list(anterior_sheets)
    posterior_rows = list(posterior_sheets)
    anterior_index = list(anterior_bins)
    posterior_index = list(posterior_bins)

    # The annular ring is the union of the two root arcs, one vertex per sector.
    annulus = np.zeros((n_annulus, 3))
    filled = np.zeros(n_annulus, dtype=bool)
    for root, index in zip(posterior_roots, posterior_bins):
        annulus[index] = root
        filled[index] = True
    for root, index in zip(anterior_roots, anterior_bins):
        annulus[index] = root if not filled[index] else 0.5 * (annulus[index] + root)
        filled[index] = True
    if not filled.all():
        present = np.flatnonzero(filled)
        if present.size < 4:
            return None
        for index in np.flatnonzero(~filled):
            nearest = present[np.argmin(np.abs(present - index))]
            annulus[index] = annulus[nearest]

    if len(anterior_rows) < 4 or len(posterior_rows) < 4:
        return None
    return ValveFit(
        annulus=annulus,
        anterior=np.stack(anterior_rows, axis=0),
        posterior=np.stack(posterior_rows, axis=0),
        anterior_root_indices=np.asarray(anterior_index, dtype=np.int64),
        posterior_root_indices=np.asarray(posterior_index, dtype=np.int64),
        case=case,
        n_annulus=n_annulus,
        n_depth=n_depth,
    )


def load_valve_fits(
    root: str | Path,
    *,
    n_annulus: int = 32,
    n_depth: int = 6,
    limit: int | None = None,
) -> list[ValveFit]:
    """Fit every labelled MVSeg2023 case under ``root``."""

    import nibabel as nib

    paths = sorted(Path(root).rglob("*-label.nii.gz"))
    if limit is not None:
        paths = paths[:limit]
    fits = []
    for path in paths:
        image = nib.load(str(path))
        fit = fit_valve(
            np.asarray(image.dataobj),
            image.header.get_zooms()[:3],
            n_annulus=n_annulus,
            n_depth=n_depth,
            case=path.name.replace("-label.nii.gz", ""),
        )
        if fit is not None:
            fits.append(fit)
    return fits
