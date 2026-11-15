"""Anatomy-restricted robust SE(3) registration without oracle correspondences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rooted4d.geometry import RigidTransform


def _points(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape (items, 3)")
    if len(array) < 3:
        raise ValueError(f"{name} must contain at least three points")
    return array


def _unit_normals(values: np.ndarray, count: int, name: str) -> np.ndarray:
    array = _points(values, name)
    if len(array) != count:
        raise ValueError(f"{name} count must match its point set")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{name} contains a zero-length normal")
    return array / norms


def _weighted_rigid(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> RigidTransform:
    mass = np.asarray(weights, dtype=np.float64)
    if mass.shape != (len(source),) or np.any(mass < 0.0) or float(mass.sum()) <= 0.0:
        raise ValueError("weights must be non-negative and contain positive mass")
    mass = mass / mass.sum()
    source_center = np.sum(mass[:, None] * source, axis=0)
    target_center = np.sum(mass[:, None] * target, axis=0)
    left = source - source_center
    right = target - target_center
    covariance = left.T @ (mass[:, None] * right)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - source_center @ rotation.T
    return RigidTransform(rotation, translation)


def _compose(after: RigidTransform, before: RigidTransform) -> RigidTransform:
    rotation = after.rotation @ before.rotation
    translation = before.translation @ after.rotation.T + after.translation
    return RigidTransform(rotation, translation)


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


@dataclass(frozen=True)
class RegistrationConfig:
    max_iterations: int = 80
    distance_scale_mm: float = 8.0
    normal_weight: float = 0.75
    arc_weight: float = 3.0
    trim_fraction: float = 0.15
    huber_delta_mm: float = 3.0
    rotation_tolerance_rad: float = 1e-6
    translation_tolerance_mm: float = 1e-5
    relative_loss_tolerance: float = 1e-7

    def validate(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.distance_scale_mm <= 0.0 or self.huber_delta_mm <= 0.0:
            raise ValueError("distance_scale_mm and huber_delta_mm must be positive")
        if self.normal_weight < 0.0 or self.arc_weight < 0.0:
            raise ValueError("feature weights cannot be negative")
        if not 0.0 <= self.trim_fraction < 1.0:
            raise ValueError("trim_fraction must lie in [0, 1)")
        if min(
            self.rotation_tolerance_rad,
            self.translation_tolerance_mm,
            self.relative_loss_tolerance,
        ) < 0.0:
            raise ValueError("convergence tolerances cannot be negative")


@dataclass(frozen=True)
class RegistrationResult:
    transform: RigidTransform
    converged: bool
    iterations: int
    matched_fraction: float
    rmse_mm: float
    loss_history: np.ndarray
    target_indices: np.ndarray
    inlier_mask: np.ndarray


def _correspondence_cost(
    transformed: np.ndarray,
    target: np.ndarray,
    *,
    source_labels: np.ndarray | None,
    target_labels: np.ndarray | None,
    transformed_normals: np.ndarray | None,
    target_normals: np.ndarray | None,
    source_arc: np.ndarray | None,
    target_arc: np.ndarray | None,
    config: RegistrationConfig,
) -> np.ndarray:
    difference = transformed[:, None, :] - target[None, :, :]
    cost = np.sum(difference * difference, axis=-1) / config.distance_scale_mm**2
    if source_labels is not None and target_labels is not None:
        admissible = source_labels[:, None] == target_labels[None, :]
        if np.any(~np.any(admissible, axis=1)):
            missing = np.unique(source_labels[~np.any(admissible, axis=1)]).tolist()
            raise ValueError(f"Target has no admissible points for source labels {missing}")
        cost = np.where(admissible, cost, np.inf)
    if transformed_normals is not None and target_normals is not None:
        dots = np.clip(transformed_normals @ target_normals.T, -1.0, 1.0)
        cost += config.normal_weight * 0.5 * (1.0 - dots)
    if source_arc is not None and target_arc is not None:
        difference_arc = source_arc[:, None] - target_arc[None, :]
        circular = np.arctan2(
            np.sin(2.0 * np.pi * difference_arc),
            np.cos(2.0 * np.pi * difference_arc),
        ) / (2.0 * np.pi)
        valid = np.isfinite(source_arc)[:, None] & np.isfinite(target_arc)[None, :]
        cost += np.where(valid, config.arc_weight * circular**2, 0.0)
    return cost


def register_point_sets(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    source_labels: np.ndarray | None = None,
    target_labels: np.ndarray | None = None,
    source_normals: np.ndarray | None = None,
    target_normals: np.ndarray | None = None,
    source_arc_coordinates: np.ndarray | None = None,
    target_arc_coordinates: np.ndarray | None = None,
    source_landmarks: np.ndarray | None = None,
    target_landmarks: np.ndarray | None = None,
    initial_transform: RigidTransform | None = None,
    config: RegistrationConfig = RegistrationConfig(),
) -> RegistrationResult:
    """Estimate a proper rigid transform using staged landmark + robust ICP fitting.

    Correspondences are inferred from the observations. Patch labels prohibit
    cross-anatomy snapping; oriented normals and circular annular coordinates
    contribute additive costs. No ground-truth pose, phase, or vertex identity is
    consumed by this function.
    """

    config.validate()
    source = _points(source_points, "source_points")
    target = _points(target_points, "target_points")

    labels_source = None
    labels_target = None
    if (source_labels is None) != (target_labels is None):
        raise ValueError("source_labels and target_labels must be supplied together")
    if source_labels is not None:
        labels_source = np.asarray(source_labels)
        labels_target = np.asarray(target_labels)
        if labels_source.shape != (len(source),) or labels_target.shape != (len(target),):
            raise ValueError("label vectors must match their point counts")

    normals_source = None
    normals_target = None
    if (source_normals is None) != (target_normals is None):
        raise ValueError("source_normals and target_normals must be supplied together")
    if source_normals is not None:
        normals_source = _unit_normals(source_normals, len(source), "source_normals")
        normals_target = _unit_normals(target_normals, len(target), "target_normals")

    arc_source = None
    arc_target = None
    if (source_arc_coordinates is None) != (target_arc_coordinates is None):
        raise ValueError("source and target arc coordinates must be supplied together")
    if source_arc_coordinates is not None:
        arc_source = np.asarray(source_arc_coordinates, dtype=np.float64)
        arc_target = np.asarray(target_arc_coordinates, dtype=np.float64)
        if arc_source.shape != (len(source),) or arc_target.shape != (len(target),):
            raise ValueError("arc coordinate vectors must match their point counts")

    if (source_landmarks is None) != (target_landmarks is None):
        raise ValueError("source_landmarks and target_landmarks must be supplied together")
    if source_landmarks is not None:
        landmarks_source = _points(source_landmarks, "source_landmarks")
        landmarks_target = _points(target_landmarks, "target_landmarks")
        if landmarks_source.shape != landmarks_target.shape:
            raise ValueError("landmark arrays must have equal shape")
        transform = _weighted_rigid(
            landmarks_source, landmarks_target, np.ones(len(landmarks_source))
        )
    elif initial_transform is not None:
        transform = initial_transform
    else:
        transform = RigidTransform(np.eye(3), target.mean(axis=0) - source.mean(axis=0))

    history: list[float] = []
    converged = False
    target_indices = np.zeros(len(source), dtype=np.int64)
    inliers = np.ones(len(source), dtype=bool)
    for iteration in range(config.max_iterations):
        transformed = transform.apply(source)
        transformed_normals = (
            None if normals_source is None else normals_source @ transform.rotation.T
        )
        cost = _correspondence_cost(
            transformed,
            target,
            source_labels=labels_source,
            target_labels=labels_target,
            transformed_normals=transformed_normals,
            target_normals=normals_target,
            source_arc=arc_source,
            target_arc=arc_target,
            config=config,
        )
        target_indices = np.argmin(cost, axis=1)
        matched = target[target_indices]
        residual = np.linalg.norm(transformed - matched, axis=1)
        keep_count = max(3, int(np.ceil((1.0 - config.trim_fraction) * len(source))))
        order = np.argsort(residual, kind="stable")
        inliers = np.zeros(len(source), dtype=bool)
        inliers[order[:keep_count]] = True
        robust = np.minimum(1.0, config.huber_delta_mm / np.maximum(residual, 1e-12))
        weights = robust * inliers
        if np.count_nonzero(weights) < 3:
            raise RuntimeError("Registration has fewer than three usable correspondences")
        delta = _weighted_rigid(transformed, matched, weights)
        updated = _compose(delta, transform)
        loss = float(np.sum(weights * residual**2) / np.sum(weights))
        history.append(loss)
        rotation_step = _rotation_angle(delta.rotation)
        translation_step = float(np.linalg.norm(delta.translation))
        relative_change = (
            np.inf
            if len(history) < 2
            else abs(history[-2] - history[-1]) / max(history[-2], 1e-12)
        )
        transform = updated
        if (
            rotation_step <= config.rotation_tolerance_rad
            and translation_step <= config.translation_tolerance_mm
        ) or relative_change <= config.relative_loss_tolerance:
            converged = True
            break

    final = transform.apply(source)
    final_normals = None if normals_source is None else normals_source @ transform.rotation.T
    final_cost = _correspondence_cost(
        final,
        target,
        source_labels=labels_source,
        target_labels=labels_target,
        transformed_normals=final_normals,
        target_normals=normals_target,
        source_arc=arc_source,
        target_arc=arc_target,
        config=config,
    )
    target_indices = np.argmin(final_cost, axis=1)
    final_residual = np.linalg.norm(final - target[target_indices], axis=1)
    keep_count = max(3, int(np.ceil((1.0 - config.trim_fraction) * len(source))))
    inliers = np.zeros(len(source), dtype=bool)
    inliers[np.argsort(final_residual, kind="stable")[:keep_count]] = True
    rmse = float(np.sqrt(np.mean(final_residual[inliers] ** 2)))
    return RegistrationResult(
        transform=transform,
        converged=converged,
        iterations=len(history),
        matched_fraction=float(np.mean(inliers)),
        rmse_mm=rmse,
        loss_history=np.asarray(history, dtype=np.float64),
        target_indices=target_indices,
        inlier_mask=inliers,
    )
