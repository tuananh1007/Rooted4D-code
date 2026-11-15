"""Strict loader and validator for a frozen heterogeneous production template."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rooted4d.constraints import AttachmentMap
from rooted4d.geometry.tetrahedra import tetrahedron_quality_report


TEMPLATE_SCHEMA = "rooted4d-frozen-template-manifest-v1"
REQUIRED_COMPONENTS = {
    "lv_endocardium",
    "lv_epicardium",
    "left_atrium",
    "anterior_leaflet",
    "posterior_leaflet",
}


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SurfacePatch:
    name: str
    local_to_global: np.ndarray
    faces: np.ndarray
    attachment_local_indices: np.ndarray

    def __post_init__(self) -> None:
        mapping = np.asarray(self.local_to_global, dtype=np.int64)
        faces = np.asarray(self.faces, dtype=np.int64)
        attachment = np.asarray(self.attachment_local_indices, dtype=np.int64)
        if mapping.ndim != 1 or len(mapping) < 3 or len(set(mapping.tolist())) != len(mapping):
            raise ValueError(f"{self.name}: local_to_global must contain >=3 unique indices")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError(f"{self.name}: faces must have non-empty shape (faces, 3)")
        if faces.min() < 0 or faces.max() >= len(mapping):
            raise ValueError(f"{self.name}: faces index outside the local vertex array")
        if attachment.ndim != 1 or len(set(attachment.tolist())) != len(attachment):
            raise ValueError(f"{self.name}: attachment_local_indices must be unique and 1D")
        if len(attachment) and (attachment.min() < 0 or attachment.max() >= len(mapping)):
            raise ValueError(f"{self.name}: attachment index is out of range")
        object.__setattr__(self, "local_to_global", mapping)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "attachment_local_indices", attachment)


@dataclass(frozen=True)
class TemplateValidationReport:
    passed: bool
    errors: tuple[str, ...]
    statistics: dict[str, Any]

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise ValueError("production template validation failed:\n- " + "\n- ".join(self.errors))


@dataclass(frozen=True)
class HeterogeneousTemplate:
    template_id: str
    vertices: np.ndarray
    tetrahedra: np.ndarray
    annulus_global_indices: np.ndarray
    components: Mapping[str, SurfacePatch]
    landmarks: Mapping[str, int]
    coordinate_system: str = "LPS"
    units: str = "mm"

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        tetrahedra = np.asarray(self.tetrahedra, dtype=np.int64)
        annulus = np.asarray(self.annulus_global_indices, dtype=np.int64)
        components = dict(self.components)
        landmarks = {str(key): int(value) for key, value in self.landmarks.items()}
        if not self.template_id:
            raise ValueError("template_id must be non-empty")
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
            raise ValueError("vertices must have finite shape (vertices, 3)")
        if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4 or len(tetrahedra) == 0:
            raise ValueError("tetrahedra must have non-empty shape (tetrahedra, 4)")
        if tetrahedra.min() < 0 or tetrahedra.max() >= len(vertices):
            raise ValueError("tetrahedra contain an out-of-range vertex index")
        if annulus.ndim != 1 or len(annulus) < 12 or len(set(annulus.tolist())) != len(annulus):
            raise ValueError("annulus must contain at least 12 unique global indices")
        if annulus.min() < 0 or annulus.max() >= len(vertices):
            raise ValueError("annulus contains an out-of-range global index")
        missing_components = sorted(REQUIRED_COMPONENTS - set(components))
        if missing_components:
            raise ValueError(f"missing production components: {', '.join(missing_components)}")
        if self.units != "mm" or self.coordinate_system not in {"LPS", "RAS"}:
            raise ValueError("template units must be mm and coordinates must be LPS or RAS")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "tetrahedra", tetrahedra)
        object.__setattr__(self, "annulus_global_indices", annulus)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "landmarks", landmarks)

    @property
    def attachment_map(self) -> AttachmentMap:
        return AttachmentMap(
            {name: patch.local_to_global for name, patch in self.components.items()},
            n_global_vertices=len(self.vertices),
        )

    def validate(self, minimum_tetrahedron_volume_mm3: float = 1e-6) -> TemplateValidationReport:
        errors: list[str] = []
        annulus = set(self.annulus_global_indices.tolist())
        required_attached = {
            "lv_endocardium",
            "left_atrium",
            "anterior_leaflet",
            "posterior_leaflet",
        }
        attachment_globals: dict[str, set[int]] = {}
        for name, patch in self.components.items():
            if patch.local_to_global.min() < 0 or patch.local_to_global.max() >= len(self.vertices):
                errors.append(f"{name}: local_to_global contains an invalid global index")
                continue
            root = set(patch.local_to_global[patch.attachment_local_indices].tolist())
            attachment_globals[name] = root
            if name in required_attached and not root:
                errors.append(f"{name}: no tagged annular attachment vertices")
            if not root.issubset(annulus):
                errors.append(f"{name}: attachment includes a vertex outside the annulus")
            xyz = self.vertices[patch.local_to_global][patch.faces]
            doubled_area = np.linalg.norm(
                np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
            )
            if np.any(doubled_area <= 1e-10):
                errors.append(f"{name}: contains degenerate faces")
        for chamber in ("lv_endocardium", "left_atrium"):
            if attachment_globals.get(chamber, set()) != annulus:
                errors.append(f"{chamber}: tagged annular loop must cover the complete ring")
        leaflet_union = attachment_globals.get("anterior_leaflet", set()) | attachment_globals.get(
            "posterior_leaflet", set()
        )
        if leaflet_union != annulus:
            errors.append("anterior/posterior leaflet root arcs must cover the complete annular ring")
        required_landmarks = {
            "anterolateral_commissure",
            "posteromedial_commissure",
            "anterior_saddle_horn",
            "posterior_annular_midpoint",
        }
        if required_landmarks - set(self.landmarks):
            errors.append("canonical annular landmarks are incomplete")
        for name, index in self.landmarks.items():
            if index not in annulus:
                errors.append(f"landmark {name} is not on the annulus")

        tetrahedra = self.vertices[self.tetrahedra]
        signed_six_volume = np.einsum(
            "ij,ij->i",
            tetrahedra[:, 1] - tetrahedra[:, 0],
            np.cross(
                tetrahedra[:, 2] - tetrahedra[:, 0],
                tetrahedra[:, 3] - tetrahedra[:, 0],
            ),
        )
        volume = signed_six_volume / 6.0
        if np.any(volume <= minimum_tetrahedron_volume_mm3):
            errors.append("tetrahedra are inverted, degenerate, or below the volume threshold")
        quality = tetrahedron_quality_report(self.vertices, self.tetrahedra)
        statistics = {
            "global_vertices": len(self.vertices),
            "tetrahedra": len(self.tetrahedra),
            "annulus_vertices": len(self.annulus_global_indices),
            "components": {name: len(patch.faces) for name, patch in self.components.items()},
            "minimum_signed_tetrahedron_volume_mm3": float(np.min(volume)),
            "minimum_scaled_jacobian": quality.minimum_scaled_jacobian,
            "mean_scaled_jacobian": quality.mean_scaled_jacobian,
            "fraction_scaled_jacobian_below_0.1": (
                quality.fraction_scaled_jacobian_below_threshold
            ),
            "assembly_rows": self.attachment_map.scalar_assembly_matrix().shape[0],
        }
        return TemplateValidationReport(not errors, tuple(errors), statistics)


def _required_array(archive: Any, key: str) -> np.ndarray:
    if key not in archive.files:
        raise ValueError(f"production template asset is missing array {key!r}")
    return np.asarray(archive[key]).copy()


def load_frozen_template(manifest_path: str | Path) -> HeterogeneousTemplate:
    """Load a digest-locked production template and validate all topology invariants."""

    manifest_file = Path(manifest_path).resolve()
    payload = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != TEMPLATE_SCHEMA:
        raise ValueError(f"manifest schema must be {TEMPLATE_SCHEMA}")
    if payload.get("status") != "frozen":
        raise ValueError("load_frozen_template accepts only status: frozen")
    asset_path = Path(str(payload.get("asset_npz", "")))
    if not asset_path.is_absolute():
        asset_path = manifest_file.parent / asset_path
    asset_path = asset_path.resolve()
    expected_digest = str(payload.get("asset_sha256", ""))
    actual_digest = _sha256_file(asset_path)
    if expected_digest != actual_digest:
        raise ValueError("production template asset SHA-256 does not match the manifest")
    component_specs = payload.get("components")
    if not isinstance(component_specs, dict):
        raise ValueError("manifest components must be an object")
    with np.load(asset_path, allow_pickle=False) as archive:
        components: dict[str, SurfacePatch] = {}
        for name, spec in component_specs.items():
            if not isinstance(spec, dict):
                raise ValueError(f"component {name!r} specification must be an object")
            components[str(name)] = SurfacePatch(
                name=str(name),
                local_to_global=_required_array(archive, str(spec.get("local_to_global_key"))),
                faces=_required_array(archive, str(spec.get("faces_key"))),
                attachment_local_indices=_required_array(
                    archive, str(spec.get("attachment_local_indices_key"))
                ),
            )
        template = HeterogeneousTemplate(
            template_id=str(payload.get("template_id", "")),
            vertices=_required_array(archive, str(payload.get("vertices_key"))),
            tetrahedra=_required_array(archive, str(payload.get("tetrahedra_key"))),
            annulus_global_indices=_required_array(
                archive, str(payload.get("annulus_global_indices_key"))
            ),
            components=components,
            landmarks=payload.get("landmarks", {}),
            coordinate_system=str(payload.get("coordinate_system", "")),
            units=str(payload.get("units", "")),
        )
    template.validate().raise_if_failed()
    return template
