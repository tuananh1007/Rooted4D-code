r"""Run an actual mesh generator on a reconstruction and report whether it worked.

The simulation-readiness argument was previously made by arithmetic: measure the
interface gap, compare it against a tolerance a mesher is assumed to use, and
conclude. That is a proxy. It predicts a failure rather than observing one, and a
prediction about someone else's tool is worth less than running the tool.

This runs Gmsh. A surface is written, healed at a stated tolerance exactly as a
user would, and volume-meshed. What is reported is what happened: whether the
parts stayed joined, whether a volume mesh was produced at all, how many
tetrahedra, and the element quality. A soft-attached reconstruction whose
interface gap exceeds the healing tolerance arrives as separate shells, and the
consequences are then Gmsh's to demonstrate rather than ours to assert.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np


@dataclass(frozen=True)
class MeshingOutcome:
    """What a mesh generator did with one reconstruction."""

    healed_tolerance_mm: float
    surface_ok: bool
    n_surface_entities: int
    volume_ok: bool
    n_tetrahedra: int
    min_scaled_jacobian: float
    mean_scaled_jacobian: float
    error: str | None

    @property
    def usable(self) -> bool:
        """A valid volume mesh with no inverted elements.

        Surface-patch count is deliberately not part of this. Gmsh's surface
        classification splits a single closed shell into several patches by
        dihedral angle, which is a reparameterization detail and not a defect;
        requiring one patch marked a perfectly good 10,681-element mesh as
        unusable. What matters is that a volume was produced and that no element
        is inverted.
        """

        return (
            self.surface_ok
            and self.volume_ok
            and self.n_tetrahedra > 0
            and self.min_scaled_jacobian > 0.0
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "healed_tolerance_mm": self.healed_tolerance_mm,
            "surface_ok": self.surface_ok,
            "n_surface_entities": self.n_surface_entities,
            "volume_ok": self.volume_ok,
            "n_tetrahedra": self.n_tetrahedra,
            "min_scaled_jacobian": self.min_scaled_jacobian,
            "mean_scaled_jacobian": self.mean_scaled_jacobian,
            "usable": self.usable,
            "error": self.error,
        }


def _write_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    with path.open("w", encoding="ascii") as handle:
        handle.write("solid reconstruction\n")
        for triangle in triangles:
            a, b, c = points[triangle]
            normal = np.cross(b - a, c - a)
            length = np.linalg.norm(normal)
            normal = normal / length if length > 0 else np.zeros(3)
            handle.write(
                f"facet normal {normal[0]:.9e} {normal[1]:.9e} {normal[2]:.9e}\n"
            )
            handle.write("  outer loop\n")
            for point in (a, b, c):
                handle.write(
                    f"    vertex {point[0]:.9e} {point[1]:.9e} {point[2]:.9e}\n"
                )
            handle.write("  endloop\nendfacet\n")
        handle.write("endsolid reconstruction\n")


def mesh_with_gmsh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    tolerance_mm: float,
    characteristic_length_mm: float = 4.0,
    verbose: bool = False,
) -> MeshingOutcome:
    """Heal at ``tolerance_mm`` and attempt a tetrahedral mesh.

    ``Geometry.Tolerance`` is the knob a user actually turns when a surface will
    not close, so it is the one varied here. Failures are returned rather than
    raised: a reconstruction a mesher rejects is a result, not an exception.
    """

    import gmsh

    surface_ok = False
    volume_ok = False
    entities = 0
    n_tetrahedra = 0
    minimum = 0.0
    mean = 0.0
    error: str | None = None

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "surface.stl"
        _write_stl(path, vertices, faces)
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
            gmsh.option.setNumber("Geometry.Tolerance", tolerance_mm)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", characteristic_length_mm)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", characteristic_length_mm / 4.0)
            gmsh.merge(str(path))
            # Stitch triangles into surfaces and close the shell, which is where
            # a gap wider than the tolerance stops the parts from joining.
            gmsh.model.mesh.classifySurfaces(np.pi, True, True, np.pi)
            gmsh.model.mesh.createGeometry()
            surfaces = gmsh.model.getEntities(2)
            entities = len(surfaces)
            surface_ok = entities > 0
            loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
            gmsh.model.geo.addVolume([loop])
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(3)
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(3)
            for kind, tags, nodes in zip(element_types, element_tags, node_tags):
                if kind != 4:
                    continue
                n_tetrahedra = len(tags)
                quality = gmsh.model.mesh.getElementQualities(list(tags), "minSICN")
                quality = np.asarray(quality, dtype=np.float64)
                if quality.size:
                    minimum = float(quality.min())
                    mean = float(quality.mean())
            volume_ok = n_tetrahedra > 0
        except Exception as failure:  # noqa: BLE001 - the failure is the result
            error = f"{type(failure).__name__}: {failure}"
        finally:
            gmsh.finalize()

    return MeshingOutcome(
        healed_tolerance_mm=float(tolerance_mm),
        surface_ok=surface_ok,
        n_surface_entities=entities,
        volume_ok=volume_ok,
        n_tetrahedra=n_tetrahedra,
        min_scaled_jacobian=minimum,
        mean_scaled_jacobian=mean,
        error=error,
    )
