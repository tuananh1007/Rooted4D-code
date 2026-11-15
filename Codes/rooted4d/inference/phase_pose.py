"""Closed-form pose plus circular-offset search for the smoke benchmark.

This is an initialization/diagnostic, not the final joint Rooted4D optimizer. It
uses known point correspondences and a fixed nonlinear warp shape so a new
checkout can verify pose/phase conventions before image renderers are added.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rooted4d.geometry import RigidTransform
from rooted4d.motion import FourierMotion, MonotoneCircularPhaseWarp, circular_distance
from rooted4d.synthetic.benchmark import SyntheticCase, session_residual


@dataclass(frozen=True)
class RecoveryResult:
    phase_offset: float
    transform: RigidTransform
    rmse_mm: float
    phase_error_cycles: float
    rotation_error_degrees: float
    translation_error_mm: float

    def as_dict(self) -> dict[str, float]:
        return {
            "recovered_phase_offset": self.phase_offset,
            "rmse_mm": self.rmse_mm,
            "phase_error_cycles": self.phase_error_cycles,
            "rotation_error_degrees": self.rotation_error_degrees,
            "translation_error_mm": self.translation_error_mm,
        }


def kabsch(source: np.ndarray, target: np.ndarray) -> RigidTransform:
    """Rigidly align corresponding source points to target points."""

    left = np.asarray(source, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError("source and target must both have shape (points, 3)")
    if len(left) < 3:
        raise ValueError("At least three point correspondences are required")
    left_mean = left.mean(axis=0)
    right_mean = right.mean(axis=0)
    covariance = (left - left_mean).T @ (right - right_mean)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = right_mean - left_mean @ rotation.T
    return RigidTransform(rotation, translation)


def _rotation_error_degrees(estimated: np.ndarray, truth: np.ndarray) -> float:
    relative = estimated @ truth.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def recover_synthetic_phase_pose(
    case: SyntheticCase,
    n_offset_candidates: int = 256,
) -> RecoveryResult:
    """Recover circular offset and SE(3) pose under smoke-test oracle assumptions."""

    if n_offset_candidates < 8:
        raise ValueError("n_offset_candidates must be at least eight")
    motion = FourierMotion(case.template_vertices, case.motion_coefficients)
    observed = case.tee_points.reshape(-1, 3)
    best: tuple[float, float, RigidTransform] | None = None

    for offset in np.linspace(0.0, 1.0, n_offset_candidates, endpoint=False):
        warp = MonotoneCircularPhaseWarp(case.phase_logits, offset=float(offset))
        phases = warp.evaluate(case.tee_times)
        geometry = motion.evaluate(phases) + 0.5 * session_residual(
            phases,
            case.session_static,
            case.session_dynamic,
        )
        predicted = geometry[:, case.tee_indices, :].reshape(-1, 3)
        transform = kabsch(predicted, observed)
        residual = transform.apply(predicted) - observed
        rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=-1))))
        if best is None or rmse < best[0]:
            best = (rmse, float(offset), transform)

    assert best is not None
    rmse, offset, transform = best
    phase_error = float(circular_distance(np.array([offset]), np.array([case.phase_offset]))[0])
    return RecoveryResult(
        phase_offset=offset,
        transform=transform,
        rmse_mm=rmse,
        phase_error_cycles=phase_error,
        rotation_error_degrees=_rotation_error_degrees(transform.rotation, case.pose_rotation),
        translation_error_mm=float(np.linalg.norm(transform.translation - case.pose_translation)),
    )

