"""Small SE(3) helpers for the synthetic benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    vector = np.asarray(axis_angle, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError("axis_angle must have shape (3,)")
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


@dataclass(frozen=True)
class RigidTransform:
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("rotation and translation must have shapes (3, 3) and (3,)")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
            raise ValueError("rotation must be orthonormal")
        if np.linalg.det(rotation) < 0.0:
            raise ValueError("rotation must be proper (determinant +1)")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @classmethod
    def from_axis_angle(cls, axis_angle: np.ndarray, translation: np.ndarray) -> "RigidTransform":
        return cls(axis_angle_to_matrix(axis_angle), np.asarray(translation, dtype=np.float64))

    def apply(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.shape[-1] != 3:
            raise ValueError("points must end with coordinate dimension 3")
        return values @ self.rotation.T + self.translation

    def inverse(self) -> "RigidTransform":
        inverse_rotation = self.rotation.T
        return RigidTransform(inverse_rotation, -self.translation @ inverse_rotation.T)

