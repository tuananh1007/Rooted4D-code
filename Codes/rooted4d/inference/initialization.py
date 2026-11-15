"""A starting alignment derived from the observations alone.

Every arm used to start from the identity pose and a phase offset of zero, with
a per-model grid search over phase offsets (Stage A) as the only data-driven
step. That is not an input-derived initialization, and it has a subtler defect
for comparing two shape models: Stage A's choice depends on the model's own
fitting cost, so two models fitted to the same observations need not even begin
their main optimization from the same alignment.

The start built here uses nothing but the observed point sequences, so it is
identical for any two models fitted to the same observations:

* **Phase** from circular dynamic time warping of shape-free features of the
  vertices both modalities observed, the same feature map and aligner late
  fusion uses for its post-hoc temporal registration.
* **Pose** from a Kabsch fit between the time-averaged positions of those same
  vertices, CMR (reference frame) to TEE (acquisition frame). Point observations
  on this benchmark carry vertex identity, so the correspondences are exact and
  no nearest-neighbour guessing is involved. Averaging over each modality's own
  frames removes most of the periodic motion without needing the phase at all.

The shape, motion and session coefficients start at zero, which places both
models at the same training mean. The start is a starting point, not an
estimate: the joint fit refines all of it, and any phase search the solver runs
still runs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from rooted4d.registration import align_periodic_features

from .joint import Observations, SolverConfig, initial_parameters
from .phase_pose import kabsch
from .scene import SceneModel, SceneParameters

#: Fewest shared observed vertices that give a determined rigid fit with some
#: redundancy. Three non-collinear points determine a rotation; four is the
#: minimum late fusion already requires for its temporal alignment.
MINIMUM_SHARED_VERTICES = 4


def curve_features(points: np.ndarray) -> np.ndarray:
    """Per-frame features of a point set that ignore which frame it sits in.

    Mean and spread of the radius about the centroid, mean height, and the
    centroid's distance from the origin. The first three are invariant to
    rotation about and translation of the set, so they can be compared across
    two acquisition frames before either pose is known. Identical to the map
    late fusion uses, so the two temporal alignments are the same method.
    """

    centered = points - points.mean(axis=1, keepdims=True)
    radii = np.linalg.norm(centered, axis=2)
    return np.stack(
        [
            radii.mean(axis=1),
            radii.std(axis=1),
            centered[:, :, 2].mean(axis=1),
            np.linalg.norm(points.mean(axis=1), axis=1),
        ],
        axis=1,
    )


def _columns(available: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    columns = np.searchsorted(available, wanted)
    if columns.size and (columns.max() >= available.size
                         or not np.array_equal(available[columns], wanted)):
        raise ValueError("requested vertices are absent from the observed sequence")
    return columns


def _matrix_to_axis_angle(rotation: np.ndarray) -> np.ndarray:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-12:
        return np.zeros(3)
    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.zeros(3)
    return axis / norm * angle


@dataclass(frozen=True)
class StartingAlignment:
    """Pose and phase estimated from the observations, with what they used."""

    pose_axis_angle: np.ndarray
    pose_translation: np.ndarray
    phase_offset: float
    shared_vertices: int
    temporal_cost: float


def input_derived_alignment(observations: Observations) -> StartingAlignment:
    """Estimate pose and phase from the observed points of shared vertices."""

    cmr_vertices = np.asarray(observations.cmr_vertices)
    tee_vertices = np.asarray(observations.tee_vertices)
    shared = np.intersect1d(cmr_vertices, tee_vertices)
    if len(shared) < MINIMUM_SHARED_VERTICES:
        raise ValueError(
            f"only {len(shared)} vertices are observed by both modalities; "
            f"{MINIMUM_SHARED_VERTICES} are needed for an input-derived start"
        )
    cmr = np.asarray(observations.cmr_points)[:, _columns(cmr_vertices, shared), :]
    tee = np.asarray(observations.tee_points)[:, _columns(tee_vertices, shared), :]

    temporal = align_periodic_features(curve_features(cmr), curve_features(tee))
    transform = kabsch(cmr.mean(axis=0), tee.mean(axis=0))
    return StartingAlignment(
        pose_axis_angle=_matrix_to_axis_angle(transform.rotation),
        pose_translation=np.asarray(transform.translation, dtype=np.float64),
        phase_offset=float(temporal.offset_cycles) % 1.0,
        shared_vertices=int(len(shared)),
        temporal_cost=float(temporal.normalized_cost),
    )


def start_from(
    alignment: StartingAlignment, model: SceneModel, config: SolverConfig
) -> SceneParameters:
    """Solver start for one model: the shared alignment, every coefficient at zero.

    The alignment is shared and the parameter vector is not, because arms fit
    different parameterizations of the same anatomy -- the detached model of
    the soft-attachment arm carries twice the identity coordinates -- so each
    arm builds its own zero-coefficient start around the one alignment.
    """

    return replace(
        initial_parameters(model, config),
        pose_axis_angle=np.array(alignment.pose_axis_angle, dtype=np.float64),
        pose_translation=np.array(alignment.pose_translation, dtype=np.float64),
        phase_offset=float(alignment.phase_offset),
    )


def input_derived_start(
    observations: Observations, model: SceneModel, config: SolverConfig
) -> SceneParameters:
    """Solver start: estimated pose and phase, every coefficient at zero."""

    return start_from(input_derived_alignment(observations), model, config)
