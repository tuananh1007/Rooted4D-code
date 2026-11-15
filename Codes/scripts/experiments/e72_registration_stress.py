"""E72: the registration terms under an annular slide, re-implemented on v3.

Pre-registered in ``Docs/E65_E72_UNIFORMITY_PREREGISTRATION_2026-09-24.md``.
E20b's host was not preserved, so this re-implements it from its description
and is recorded as a re-implementation. No shape model is involved.

For every test sample the host registers the anatomy both modalities observe,
the vertices in both visible sets, CMR (reference frame) onto TEE, with the
staged robust ICP of ``register_point_sets`` at its default settings:

* points are each modality's time-averaged observed positions of those
  vertices, which removes most periodic motion without needing the phase;
* arc coordinates are the vertices' positions around the ring, the directed
  annular coordinate that a commissure-anchored annotation supplies;
* oriented normals are each modality's area-weighted vertex normals on its own
  observed surface (chamber for CMR, leaflets for TEE), accumulated over the
  faces whose three vertices it observed; a vertex with no such face takes the
  modality's mean surface normal, which is recorded per sample;
* the start is the true pose composed with a rotation of the CMR points about
  their own annular axis, by 0, 30, 60 or 90 degrees -- what an initializer
  that misplaced the commissure would hand over.

The arms zero the normal weight, the arc weight, or both.

    R4D_ALLOW_TEST=1 python e72_registration_stress.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import e59_unknown_alignment as e59  # noqa: E402
from uniformity_runner import frozen_s, run  # noqa: E402
from rooted4d.geometry import RigidTransform, toy_complex_for_vertices  # noqa: E402
from rooted4d.registration import RegistrationConfig, register_point_sets  # noqa: E402

BENCH = os.environ.get("R4D_BENCH", "v3")  # E100 passes a rebuilt instance
SLIDES_DEG = (0.0, 30.0, 60.0, 90.0)
DEFAULT = RegistrationConfig()
#: Arc weight at which a tangential slide costs the same in arc as in position:
#: (2 pi r / s)^2, with r = 27.26 mm the mean ring radius of the 90 v3 training
#: identities and s = 8 mm the default distance scale. At the default weight of 3
#: the arc cost of a 30-degree slide is about 0.02 against a distance cost near
#: 2.6, so that term cannot act; see the amendment in the pre-registration.
COMMENSURATE_ARC = 458.0
ARMS = {"full": DEFAULT,
        "position_only": replace(DEFAULT, normal_weight=0.0),
        "no_directed_arc": replace(DEFAULT, arc_weight=0.0),
        "neither": replace(DEFAULT, normal_weight=0.0, arc_weight=0.0),
        "full_commensurate": replace(DEFAULT, arc_weight=COMMENSURATE_ARC),
        "position_only_commensurate": replace(DEFAULT, normal_weight=0.0, arc_weight=COMMENSURATE_ARC)}
STATE: dict = {}


def initialise():
    pass


def observed_normals(vertex_ids, mean_points, faces):
    """Area-weighted normals over faces whose three vertices were all observed."""
    position = {int(v): i for i, v in enumerate(vertex_ids)}
    complete = [f for f in np.asarray(faces) if all(int(v) in position for v in f)]
    accumulated = np.zeros((len(vertex_ids), 3))
    for a, b, c in complete:
        ia, ib, ic = position[int(a)], position[int(b)], position[int(c)]
        cross = np.cross(mean_points[ib] - mean_points[ia], mean_points[ic] - mean_points[ia])
        for index in (ia, ib, ic):
            accumulated[index] += cross
    surface = accumulated.sum(axis=0)
    surface = surface / max(np.linalg.norm(surface), 1e-12)
    norms = np.linalg.norm(accumulated, axis=1)
    missing = norms <= 1e-12
    accumulated[~missing] /= norms[~missing, None]
    accumulated[missing] = surface
    return accumulated, int(missing.sum())


def rotation_about(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * skew + (1.0 - np.cos(angle_rad)) * skew @ skew


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a.T @ b
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))


def fit_sample(path_string: str) -> dict:
    path = Path(path_string)
    with np.load(path, allow_pickle=True) as blob:
        identity = str(blob["identity_id"])
        # Sorted by vertex identity, as benchmark_problem sorts them, so the
        # shared-vertex columns can be found by binary search.
        cmr_order = np.argsort(blob["cmr_indices"])
        tee_order = np.argsort(blob["tee_indices"])
        cmr_ids = np.asarray(blob["cmr_indices"], dtype=np.int64)[cmr_order]
        tee_ids = np.asarray(blob["tee_indices"], dtype=np.int64)[tee_order]
        cmr_points = np.asarray(blob["cmr_points"], dtype=float)[:, cmr_order, :]
        tee_points = np.asarray(blob["tee_points"], dtype=float)[:, tee_order, :]
        rotation = np.asarray(blob["pose_rotation"], dtype=float)
        translation = np.asarray(blob["pose_translation"], dtype=float)
        n_vertices = len(blob["template_vertices"])
    complex_ = toy_complex_for_vertices(n_vertices)
    n_ring = len(complex_.groups["annulus"])
    cmr_mean, tee_mean = cmr_points.mean(axis=0), tee_points.mean(axis=0)
    shared = np.intersect1d(cmr_ids, tee_ids)
    cmr_cols, tee_cols = np.searchsorted(cmr_ids, shared), np.searchsorted(tee_ids, shared)
    source, target = cmr_mean[cmr_cols], tee_mean[tee_cols]
    cmr_normals, cmr_fallback = observed_normals(cmr_ids, cmr_mean, complex_.faces["cmr_surface"])
    tee_normals, tee_fallback = observed_normals(tee_ids, tee_mean, complex_.faces["tee_surface"])
    arc = (shared % n_ring).astype(float) / n_ring
    centre = source.mean(axis=0)
    axis = np.linalg.svd(source - centre)[2][-1]
    row = {"sample": path.name, "identity": identity, "shared_vertices": int(len(shared)),
           "normal_fallbacks": [cmr_fallback, tee_fallback], "variants": {}}
    for slide in SLIDES_DEG:
        slide_rotation = rotation_about(axis, np.radians(slide))
        start_rotation = rotation @ slide_rotation
        start = RigidTransform(start_rotation, rotation @ (centre - slide_rotation @ centre) + translation)
        for arm, config in ARMS.items():
            try:
                result = register_point_sets(
                    source, target,
                    source_normals=cmr_normals[cmr_cols], target_normals=tee_normals[tee_cols],
                    source_arc_coordinates=arc, target_arc_coordinates=arc,
                    initial_transform=start, config=config)
                estimate = result.transform
                outcome = {"succeeded": True,
                           "rotation_deg": geodesic_deg(estimate.rotation, rotation),
                           "translation_mm": float(np.linalg.norm(estimate.translation - translation)),
                           "rmse_mm": float(result.rmse_mm)}
            except Exception as error:
                outcome = {"succeeded": False, "failure_reason": f"{type(error).__name__}: {error}"}
            row["variants"][f"slide{int(slide)}/{arm}"] = outcome
    return row


def main() -> int:
    _, _, _, root = frozen_s(BENCH)
    split = "val" if os.environ.get("R4D_SMOKE") == "1" else "test"  # smoke never reads test
    paths = e59.sample_paths(Path(root), split)
    return run(f"E72_{BENCH}_{split}", split, paths, fit_sample, initialise, (),
               extra={"experiment": "E72", "benchmark": root, "slides_deg": SLIDES_DEG,
                      "arms": {k: vars(v) for k, v in ARMS.items()}})


if __name__ == "__main__":
    sys.exit(main())
