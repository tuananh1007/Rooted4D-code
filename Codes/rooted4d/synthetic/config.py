"""Validated configuration for the semi-synthetic dataset builder."""

from __future__ import annotations

import dataclasses

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .reproducibility import validate_seed


SPLIT_NAMES = ("train", "val", "test")
CONFIG_SCHEMA = "rooted4d-semisynthetic-config-v1"


@dataclass(frozen=True)
class SplitFractions:
    train: float = 0.60
    val: float = 0.20
    test: float = 0.20

    def validate(self) -> None:
        values = (self.train, self.val, self.test)
        if any(value <= 0.0 for value in values):
            raise ValueError("train, val, and test fractions must all be positive")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("train, val, and test fractions must sum to 1")

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in SPLIT_NAMES}


@dataclass(frozen=True)
class DonorConfig:
    mode: str = "procedural_test"
    whole_heart_count: int = 15
    valve_count: int = 15
    manifest: str | None = None
    template_manifest: str | None = None
    split_lock: str | None = None
    #: Orient each valve to its chamber before carrying it onto the chamber's
    #: ring: re-index the valve's ring and free edge to the chamber's handedness,
    #: so the free edge lands on the ventricular side. MVSeg2023 and ACDC donor
    #: rings are ordered with opposite handedness, so without this every
    #: composite carries its valve upside down (see
    #: Docs/COMPOSITE_ORIENTATION_FINDING_2026-09-28.md). False, the default,
    #: reproduces every benchmark built before the option existed.
    orient_valves: bool = False

    def validate(self) -> None:
        if self.mode not in {"procedural_test", "test_manifest", "production_manifest"}:
            raise ValueError(
                "donors.mode must be procedural_test, test_manifest, or production_manifest"
            )
        if self.mode == "procedural_test":
            if self.whole_heart_count < 3 or self.valve_count < 3:
                raise ValueError("procedural donor counts must be at least 3")
        elif self.mode == "test_manifest" and not self.manifest:
            raise ValueError("donors.manifest is required for test_manifest")
        elif self.mode == "production_manifest" and (
            not self.manifest or not self.template_manifest or not self.split_lock
        ):
            raise ValueError(
                "donors.manifest, donors.template_manifest, and donors.split_lock are required "
                "for production_manifest"
            )


@dataclass(frozen=True)
class ObservationConfig:
    n_annulus: int = 24
    n_cmr_frames: int = 20
    n_tee_frames: int = 24
    fourier_order: int = 3
    cmr_noise_mm: float = 0.35
    tee_noise_mm: float = 0.45
    cmr_dropout_fraction: float = 0.10
    tee_dropout_fraction: float = 0.20
    output_domain: str = "surface_points"
    cmr_n_slices: int = 5
    cmr_image_shape: tuple[int, int] = (24, 24)
    cmr_pixel_spacing_mm: tuple[float, float] = (2.5, 2.5)
    cmr_slice_thickness_mm: float = 6.0
    cmr_surface_sigma_mm: float = 1.5
    tee_grid_shape: tuple[int, int, int] = (20, 12, 12)
    tee_surface_sigma_mm: float = 1.5
    image_noise_std: float = 0.015
    tee_speckle_std: float = 0.12
    #: The procedural instance's interiors (plan v2, W2.6): chamber rings between
    #: the annulus and the apex, and leaflet rows beyond the root. One and one,
    #: the defaults, are the reduced instance of every earlier benchmark.
    chamber_levels: int = 1
    leaflet_rows: int = 1
    #: The TEE image's acoustic shadow. Off only in E106's no-shadow twin, a
    #: diagnostic of what the unmodeled shadow does to image fitting.
    tee_shadow: bool = True

    def validate(self) -> None:
        if self.n_annulus < 12 or self.n_annulus % 2:
            raise ValueError("n_annulus must be an even integer of at least 12")
        if self.chamber_levels < 1 or self.leaflet_rows < 1:
            raise ValueError("chamber_levels and leaflet_rows must be at least one")
        if self.n_cmr_frames < 2 or self.n_tee_frames < 2:
            raise ValueError("each modality must have at least two frames")
        if self.fourier_order < 1:
            raise ValueError("fourier_order must be positive")
        if self.cmr_noise_mm < 0.0 or self.tee_noise_mm < 0.0:
            raise ValueError("observation noise cannot be negative")
        if not 0.0 <= self.cmr_dropout_fraction < 1.0:
            raise ValueError("cmr_dropout_fraction must lie in [0, 1)")
        if not 0.0 <= self.tee_dropout_fraction < 1.0:
            raise ValueError("tee_dropout_fraction must lie in [0, 1)")
        if self.output_domain not in {"surface_points", "image_likelihood"}:
            raise ValueError("output_domain must be surface_points or image_likelihood")
        if self.cmr_n_slices < 1:
            raise ValueError("cmr_n_slices must be positive")
        if len(self.cmr_image_shape) != 2 or any(int(value) < 2 for value in self.cmr_image_shape):
            raise ValueError("cmr_image_shape must contain two values >= 2")
        if len(self.cmr_pixel_spacing_mm) != 2 or any(
            float(value) <= 0.0 for value in self.cmr_pixel_spacing_mm
        ):
            raise ValueError("cmr_pixel_spacing_mm must contain two positive values")
        if self.cmr_slice_thickness_mm <= 0.0 or self.cmr_surface_sigma_mm <= 0.0:
            raise ValueError("CMR thickness and surface sigma must be positive")
        if len(self.tee_grid_shape) != 3 or any(int(value) < 2 for value in self.tee_grid_shape):
            raise ValueError("tee_grid_shape must contain three values >= 2")
        if self.tee_surface_sigma_mm <= 0.0:
            raise ValueError("tee_surface_sigma_mm must be positive")
        if self.image_noise_std < 0.0 or self.tee_speckle_std < 0.0:
            raise ValueError("image and speckle noise cannot be negative")


@dataclass(frozen=True)
class PerturbationConfig:
    pose_axis_angle_std_rad: float = 0.14
    pose_translation_std_mm: float = 9.0
    phase_offset_min_cycles: float = -0.20
    phase_offset_max_cycles: float = 0.20
    phase_logit_std: float = 0.55
    phase_knots: int = 8
    session_static_scale_min: float = 0.70
    session_static_scale_max: float = 1.30
    session_dynamic_scale_min: float = 0.70
    session_dynamic_scale_max: float = 1.30
    #: Path to a measured session subspace. When set, session residuals are drawn
    #: from it instead of the procedural fallback, which displaces vertices by
    #: about 0.2 mm where MITEA scan-rescan measures 1.19 mm correspondence-free
    #: and 3.08 mm per vertex. A benchmark built on the fallback cannot test the
    #: session term: its measured contribution was 0.003 mm against a 2.2 mm
    #: representation floor.
    session_subspace_path: str | None = None

    def validate(self) -> None:
        if self.pose_axis_angle_std_rad < 0.0 or self.pose_translation_std_mm < 0.0:
            raise ValueError("pose perturbation scales cannot be negative")
        if self.phase_offset_min_cycles >= self.phase_offset_max_cycles:
            raise ValueError("phase offset minimum must be below its maximum")
        if self.phase_logit_std < 0.0 or self.phase_knots < 2:
            raise ValueError("phase_logit_std must be non-negative and phase_knots >= 2")
        # Zero is allowed: E92 intervenes on the session field by scaling it,
        # down to no session change at all.
        if not 0.0 <= self.session_static_scale_min <= self.session_static_scale_max:
            raise ValueError("invalid static session-scale interval")
        if not 0.0 <= self.session_dynamic_scale_min <= self.session_dynamic_scale_max:
            raise ValueError("invalid dynamic session-scale interval")


@dataclass(frozen=True)
class SemisyntheticConfig:
    schema_version: str = CONFIG_SCHEMA
    root_seed: int = 20260823
    split_seed: int = 104729
    split: SplitFractions = field(default_factory=SplitFractions)
    donors: DonorConfig = field(default_factory=DonorConfig)
    variants_per_identity: int = 2
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)

    def validate(self) -> None:
        if self.schema_version != CONFIG_SCHEMA:
            raise ValueError(f"unsupported configuration schema: {self.schema_version}")
        validate_seed(self.root_seed, "root_seed")
        validate_seed(self.split_seed, "split_seed")
        self.split.validate()
        self.donors.validate()
        if self.variants_per_identity < 1:
            raise ValueError("variants_per_identity must be positive")
        self.observation.validate()
        self.perturbation.validate()

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["split"] = self.split.as_dict()
        if self.observation.chamber_levels == 1 and self.observation.leaflet_rows == 1:
            # The reduced instance's record stays as it was before W2.6.
            del value["observation"]["chamber_levels"], value["observation"]["leaflet_rows"]
        if self.observation.tee_shadow:
            # Every benchmark before E106's twin carries the shadow; its record stays as it was.
            del value["observation"]["tee_shadow"]
        return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def config_from_mapping(raw: dict[str, Any]) -> SemisyntheticConfig:
    reproducibility = _mapping(raw.get("reproducibility"), "reproducibility")
    split_raw = _mapping(raw.get("split"), "split")
    donors_raw = _mapping(raw.get("donors"), "donors")
    samples_raw = _mapping(raw.get("samples"), "samples")
    observation_raw = _mapping(raw.get("observation"), "observation")
    perturbation_raw = _mapping(raw.get("perturbation"), "perturbation")
    config = SemisyntheticConfig(
        schema_version=str(raw.get("schema_version", CONFIG_SCHEMA)),
        root_seed=reproducibility.get("root_seed", 20260823),
        split_seed=reproducibility.get("split_seed", 104729),
        split=SplitFractions(**split_raw),
        donors=DonorConfig(**donors_raw),
        variants_per_identity=samples_raw.get("variants_per_identity", 2),
        observation=ObservationConfig(**observation_raw),
        perturbation=PerturbationConfig(**perturbation_raw),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> SemisyntheticConfig:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("semi-synthetic configuration must contain a YAML mapping")
    config = config_from_mapping(raw)
    # Relative paths in a config are relative to the config, not to the working
    # directory. Every other path here is resolved explicitly and a new one that
    # is not would simply fail to load at build time.
    if config.perturbation.session_subspace_path is not None:
        subspace = Path(config.perturbation.session_subspace_path)
        if not subspace.is_absolute():
            subspace = (source.parent / subspace).resolve()
        config = dataclasses.replace(
            config,
            perturbation=dataclasses.replace(
                config.perturbation, session_subspace_path=str(subspace)
            ),
        )
    if config.donors.mode in {"test_manifest", "production_manifest"}:
        manifest = Path(str(config.donors.manifest))
        if not manifest.is_absolute():
            manifest = (source.parent / manifest).resolve()
        template_manifest_value = config.donors.template_manifest
        split_lock_value = config.donors.split_lock
        template_manifest = None
        split_lock = None
        if template_manifest_value is not None:
            template_manifest = Path(template_manifest_value)
            if not template_manifest.is_absolute():
                template_manifest = (source.parent / template_manifest).resolve()
        if split_lock_value is not None:
            split_lock = Path(split_lock_value)
            if not split_lock.is_absolute():
                split_lock = (source.parent / split_lock).resolve()
        config = SemisyntheticConfig(
            schema_version=config.schema_version,
            root_seed=config.root_seed,
            split_seed=config.split_seed,
            split=config.split,
            donors=DonorConfig(
                mode=config.donors.mode,
                whole_heart_count=config.donors.whole_heart_count,
                valve_count=config.donors.valve_count,
                manifest=str(manifest),
                template_manifest=(
                    str(template_manifest) if template_manifest is not None else None
                ),
                split_lock=str(split_lock) if split_lock is not None else None,
                orient_valves=config.donors.orient_valves,
            ),
            variants_per_identity=config.variants_per_identity,
            observation=config.observation,
            perturbation=config.perturbation,
        )
        config.validate()
    return config
