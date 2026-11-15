"""Shared-degree-of-freedom assembly for heterogeneous anatomical parts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class AttachmentMap:
    """Map topologically separate local vertices to shared physical vertices.

    A chamber, annular curve, and open leaflets may retain separate local
    connectivity while duplicated interface vertices index the same global
    coordinate. Geometry assembled from a global coordinate array therefore
    satisfies attachment exactly, without an attachment loss.
    """

    local_to_global: Mapping[str, np.ndarray]
    n_global_vertices: int

    def __post_init__(self) -> None:
        if self.n_global_vertices <= 0:
            raise ValueError("n_global_vertices must be positive")
        normalized: dict[str, np.ndarray] = {}
        for name, indices in self.local_to_global.items():
            array = np.asarray(indices, dtype=np.int64)
            if array.ndim != 1 or array.size == 0:
                raise ValueError(f"Part '{name}' must have a non-empty one-dimensional map")
            if array.min() < 0 or array.max() >= self.n_global_vertices:
                raise ValueError(f"Part '{name}' contains an out-of-range physical vertex id")
            normalized[name] = array
        object.__setattr__(self, "local_to_global", normalized)

    @property
    def parts(self) -> tuple[str, ...]:
        return tuple(self.local_to_global)

    def assemble(self, global_coordinates: np.ndarray) -> dict[str, np.ndarray]:
        """Return local part coordinates while retaining exact shared interfaces."""

        coordinates = np.asarray(global_coordinates)
        if coordinates.ndim < 2 or coordinates.shape[-2:] != (self.n_global_vertices, 3):
            raise ValueError(
                "global_coordinates must have shape (..., n_global_vertices, 3); "
                f"received {coordinates.shape}"
            )
        return {
            name: coordinates[..., indices, :]
            for name, indices in self.local_to_global.items()
        }

    def occurrences(self) -> dict[int, list[tuple[str, int]]]:
        result: dict[int, list[tuple[str, int]]] = {}
        for name, indices in self.local_to_global.items():
            for local_index, global_index in enumerate(indices.tolist()):
                result.setdefault(global_index, []).append((name, local_index))
        return result

    def shared_vertex_ids(self) -> tuple[int, ...]:
        return tuple(
            global_index
            for global_index, occurrences in self.occurrences().items()
            if len(occurrences) > 1
        )

    def local_slices(self) -> dict[str, slice]:
        """Slices for parts in the concatenated local-vertex ordering."""

        result: dict[str, slice] = {}
        start = 0
        for name, indices in self.local_to_global.items():
            stop = start + len(indices)
            result[name] = slice(start, stop)
            start = stop
        return result

    def scalar_assembly_matrix(self) -> np.ndarray:
        """Dense reference form of the fixed binary local-to-physical map.

        Production training code should store this matrix sparsely.  Each row
        represents one local topological occurrence and contains exactly one
        nonzero pointing to its unique physical coordinate.
        """

        n_local = sum(len(indices) for indices in self.local_to_global.values())
        matrix = np.zeros((n_local, self.n_global_vertices), dtype=np.float64)
        row = 0
        for indices in self.local_to_global.values():
            count = len(indices)
            matrix[np.arange(row, row + count), indices] = 1.0
            row += count
        return matrix

    def coordinate_assembly_matrix(self) -> np.ndarray:
        """Return ``A = scalar_assembly_matrix() kron I_3``."""

        return np.kron(self.scalar_assembly_matrix(), np.eye(3, dtype=np.float64))

    def max_residual(self, local_coordinates: Mapping[str, np.ndarray]) -> float:
        """Maximum Euclidean disagreement among duplicated interface vertices."""

        maximum = 0.0
        for global_index, occurrences in self.occurrences().items():
            if len(occurrences) < 2:
                continue
            values = []
            for name, local_index in occurrences:
                if name not in local_coordinates:
                    raise KeyError(f"Missing coordinates for part '{name}'")
                array = np.asarray(local_coordinates[name])
                expected_vertices = len(self.local_to_global[name])
                if array.ndim < 2 or array.shape[-2:] != (expected_vertices, 3):
                    raise ValueError(
                        f"Part '{name}' must have shape (..., {expected_vertices}, 3); "
                        f"received {array.shape}"
                    )
                values.append(array[..., local_index, :])
            reference = values[0]
            for value in values[1:]:
                residual = np.linalg.norm(value - reference, axis=-1)
                maximum = max(maximum, float(np.max(residual)))
        return maximum

    def merge(self, local_coordinates: Mapping[str, np.ndarray], atol: float = 1e-6) -> np.ndarray:
        """Merge local coordinates into one physical array after consistency checking."""

        residual = self.max_residual(local_coordinates)
        if residual > atol:
            raise ValueError(
                f"Local interface coordinates disagree by {residual:.6g}, exceeding atol={atol}"
            )
        first_name = self.parts[0]
        lead_shape = np.asarray(local_coordinates[first_name]).shape[:-2]
        dtype = np.result_type(*(np.asarray(value).dtype for value in local_coordinates.values()))
        result = np.zeros((*lead_shape, self.n_global_vertices, 3), dtype=dtype)
        counts = np.zeros(self.n_global_vertices, dtype=np.int64)
        for name, indices in self.local_to_global.items():
            values = np.asarray(local_coordinates[name])
            if values.shape[:-2] != lead_shape:
                raise ValueError("All local parts must have the same leading dimensions")
            for local_index, global_index in enumerate(indices.tolist()):
                result[..., global_index, :] += values[..., local_index, :]
                counts[global_index] += 1
        if np.any(counts == 0):
            missing = np.flatnonzero(counts == 0).tolist()
            raise ValueError(f"No local occurrence exists for physical vertices {missing}")
        result /= counts.reshape((1,) * len(lead_shape) + (-1, 1))
        return result
