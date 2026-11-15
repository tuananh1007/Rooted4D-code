"""Build a leakage-safe semi-synthetic CMR/TEE geometry benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from rooted4d import __version__
from rooted4d.geometry import HeterogeneousTemplate, load_frozen_template

from .benchmark import SyntheticCase, generate_case
from .config import SPLIT_NAMES, SemisyntheticConfig
from .donors import Donor, combine_donors, load_donor_manifest, make_procedural_donors
from .reproducibility import (
    canonical_sha256,
    derive_seed,
    environment_record,
    generator_source_fingerprint,
    make_rng,
    sha256_file,
    write_json,
)


DATASET_SCHEMA = "rooted4d-semisynthetic-dataset-v2"
SAMPLE_SCHEMA = "rooted4d-semisynthetic-sample-v2"


def _split_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    """Allocate at least one donor to every split using largest remainders."""

    if total < len(SPLIT_NAMES):
        raise ValueError("at least three donors are needed for train/val/test")
    raw = {name: fractions[name] * total for name in SPLIT_NAMES}
    counts = {name: int(np.floor(raw[name])) for name in SPLIT_NAMES}
    unassigned = total - sum(counts.values())
    ranking = sorted(SPLIT_NAMES, key=lambda name: (-(raw[name] % 1.0), SPLIT_NAMES.index(name)))
    for name in ranking[:unassigned]:
        counts[name] += 1
    for empty in [name for name in SPLIT_NAMES if counts[name] == 0]:
        donor = max(
            (name for name in SPLIT_NAMES if counts[name] > 1),
            key=lambda name: (counts[name] - raw[name], counts[name], -SPLIT_NAMES.index(name)),
        )
        counts[donor] -= 1
        counts[empty] += 1
    if sum(counts.values()) != total or any(value < 1 for value in counts.values()):
        raise RuntimeError("internal split allocation error")
    return counts


def _partition_donors(
    donors: list[Donor],
    split_seed: int,
    kind: str,
    fractions: dict[str, float],
) -> dict[str, list[Donor]]:
    by_person: dict[str, list[Donor]] = {}
    for donor in sorted(donors, key=lambda item: item.donor_id):
        by_person.setdefault(donor.global_person_id, []).append(donor)
    people = sorted(by_person)
    order = make_rng(split_seed, "person-partition", kind).permutation(len(people))
    shuffled_people = [people[int(index)] for index in order]
    counts = _split_counts(len(shuffled_people), fractions)
    result: dict[str, list[Donor]] = {}
    offset = 0
    for split in SPLIT_NAMES:
        selected_people = shuffled_people[offset : offset + counts[split]]
        result[split] = [
            donor for person in selected_people for donor in by_person[person]
        ]
        offset += counts[split]
    return result


def _locked_partitions(
    donors: list[Donor], donor_splits: dict[str, str], kind: str
) -> dict[str, list[Donor]]:
    result = {split: [] for split in SPLIT_NAMES}
    for donor in sorted(donors, key=lambda item: item.donor_id):
        split = donor_splits.get(donor.donor_id)
        if split not in SPLIT_NAMES:
            raise ValueError(f"{kind} donor {donor.donor_id!r} is absent from the frozen split lock")
        result[split].append(donor)
    if any(not result[split] for split in SPLIT_NAMES):
        raise ValueError(f"Frozen split lock leaves an empty {kind} donor split")
    return result


def _load_frozen_donor_splits(path: str | Path, donors: list[Donor]) -> dict[str, str]:
    from rooted4d.data.split_lock import load_split_lock, validate_split_lock

    payload = load_split_lock(path)
    report = validate_split_lock(payload)
    if not report.passed:
        raise ValueError("frozen global split lock is invalid:\n- " + "\n- ".join(report.errors))
    if payload.get("status") != "frozen":
        raise ValueError("production_manifest requires a status: frozen global split lock")
    donor_to_split: dict[str, str] = {}
    donor_to_person: dict[str, str] = {}
    for record in payload["records"]:
        for donor_id in record["donor_ids"]:
            if donor_id in donor_to_split:
                raise ValueError(f"donor {donor_id!r} occurs more than once in the split lock")
            donor_to_split[donor_id] = record["split"]
            donor_to_person[donor_id] = record["global_person_id"]
    for donor in donors:
        if donor_to_person.get(donor.donor_id) != donor.global_person_id:
            raise ValueError(
                f"donor {donor.donor_id!r} person does not match the frozen split lock"
            )
    return donor_to_split


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _sample_arrays(
    case: SyntheticCase,
    sample_id: str,
    split: str,
    identity_id: str,
    heart_id: str,
    valve_id: str,
    heart_person_id: str,
    valve_person_id: str,
    variant_index: int,
    seeds: dict[str, int],
) -> dict[str, np.ndarray]:
    arrays = dict(case.arrays())
    arrays.update(
        {
            "sample_id": np.asarray(sample_id),
            "split": np.asarray(split),
            "identity_id": np.asarray(identity_id),
            "whole_heart_donor_id": np.asarray(heart_id),
            "valve_donor_id": np.asarray(valve_id),
            "whole_heart_global_person_id": np.asarray(heart_person_id),
            "valve_global_person_id": np.asarray(valve_person_id),
            "variant_index": np.asarray(variant_index, dtype=np.int64),
            "phase_offset": np.asarray(case.phase_offset, dtype=np.float64),
            "attachment_residual_mm": np.asarray(case.attachment_residual_mm, dtype=np.float64),
            "periodic_seam_mm": np.asarray(case.periodic_seam_mm, dtype=np.float64),
            "phase_minimum_increment": np.asarray(
                case.phase_minimum_increment, dtype=np.float64
            ),
        }
    )
    for name, indices in sorted(case.local_to_global.items()):
        arrays[f"local_to_global__{name}"] = indices
    for name, seed in sorted(seeds.items()):
        arrays[f"seed__{name}"] = np.asarray(seed, dtype=np.uint64)
    return arrays


def _factor_seeds(root_seed: int, identity_seed: int, variant_index: int) -> dict[str, int]:
    variant_seed = derive_seed(identity_seed, "variant", variant_index)
    return {
        "identity": identity_seed,
        "variant": variant_seed,
        "motion": derive_seed(identity_seed, "shared-motion"),
        "pose": derive_seed(variant_seed, "pose"),
        "phase": derive_seed(variant_seed, "phase"),
        "session": derive_seed(variant_seed, "session"),
        "cmr_dropout": derive_seed(variant_seed, "cmr-dropout"),
        "tee_dropout": derive_seed(variant_seed, "tee-dropout"),
        "cmr_noise": derive_seed(variant_seed, "cmr-noise"),
        "tee_noise": derive_seed(variant_seed, "tee-noise"),
        "cmr_image": derive_seed(variant_seed, "cmr-image"),
        "tee_image": derive_seed(variant_seed, "tee-image"),
        "tee_shadow": derive_seed(variant_seed, "tee-shadow"),
        "sample": derive_seed(root_seed, "sample", identity_seed, variant_index),
    }


def _load_session_subspace(path: str | Path | None):
    """Measured session subspace, or None to use the procedural fallback."""

    if path is None:
        return None
    blob = np.load(Path(path))
    return {
        "static_components": blob["static_components"],
        "static_coefficient_std": blob["static_coefficient_std"],
        "dynamic_components": blob["dynamic_components"],
        "dynamic_coefficient_std": blob["dynamic_coefficient_std"],
    }


def _draw_session_fields(subspace, rng, n_vertices):
    """Draw a session field pair from the measured subspace."""

    def draw(components, deviations):
        coefficients = rng.normal(0.0, 1.0, size=len(deviations)) * deviations
        return (coefficients @ components).reshape(n_vertices, 3)

    return (
        draw(subspace["static_components"], subspace["static_coefficient_std"]),
        draw(subspace["dynamic_components"], subspace["dynamic_coefficient_std"]),
    )


def _generate_sample(
    config: SemisyntheticConfig,
    template_vertices: np.ndarray,
    seeds: dict[str, int],
    topology: HeterogeneousTemplate | None,
    session_subspace=None,
) -> tuple[SyntheticCase, dict[str, Any]]:
    perturbation = config.perturbation
    pose_rng = np.random.default_rng(seeds["pose"])
    phase_rng = np.random.default_rng(seeds["phase"])
    session_rng = np.random.default_rng(seeds["session"])
    axis_angle = pose_rng.normal(0.0, perturbation.pose_axis_angle_std_rad, size=3)
    translation = pose_rng.normal(0.0, perturbation.pose_translation_std_mm, size=3)
    phase_offset = float(
        phase_rng.uniform(
            perturbation.phase_offset_min_cycles,
            perturbation.phase_offset_max_cycles,
        )
    )
    phase_logits = phase_rng.normal(
        0.0,
        perturbation.phase_logit_std,
        size=perturbation.phase_knots,
    )
    static_scale = float(
        session_rng.uniform(
            perturbation.session_static_scale_min,
            perturbation.session_static_scale_max,
        )
    )
    dynamic_scale = float(
        session_rng.uniform(
            perturbation.session_dynamic_scale_min,
            perturbation.session_dynamic_scale_max,
        )
    )
    session_fields = (
        _draw_session_fields(session_subspace, session_rng, len(template_vertices))
        if session_subspace is not None
        else None
    )
    observation = config.observation
    case = generate_case(
        seed=seeds["sample"],
        n_cmr_frames=observation.n_cmr_frames,
        n_tee_frames=observation.n_tee_frames,
        fourier_order=observation.fourier_order,
        cmr_noise_mm=observation.cmr_noise_mm,
        tee_noise_mm=observation.tee_noise_mm,
        cmr_dropout_fraction=observation.cmr_dropout_fraction,
        tee_dropout_fraction=observation.tee_dropout_fraction,
        tee_axis_angle_rad=tuple(float(value) for value in axis_angle),
        tee_translation_mm=tuple(float(value) for value in translation),
        phase_logits=tuple(float(value) for value in phase_logits),
        phase_offset=phase_offset,
        n_annulus=observation.n_annulus,
        template_vertices=template_vertices,
        component_seeds={
            "motion": seeds["motion"],
            "cmr_dropout": seeds["cmr_dropout"],
            "tee_dropout": seeds["tee_dropout"],
            "cmr_noise": seeds["cmr_noise"],
            "tee_noise": seeds["tee_noise"],
            **(
                {
                    "cmr_image": seeds["cmr_image"],
                    "tee_image": seeds["tee_image"],
                    "tee_shadow": seeds["tee_shadow"],
                }
                if observation.output_domain == "image_likelihood"
                else {}
            ),
        },
        session_static_scale=static_scale,
        session_dynamic_scale=dynamic_scale,
        session_static_field=session_fields[0] if session_fields else None,
        session_dynamic_field=session_fields[1] if session_fields else None,
        observation_domain=observation.output_domain,
        cmr_n_slices=observation.cmr_n_slices,
        cmr_image_shape=tuple(observation.cmr_image_shape),
        cmr_pixel_spacing_mm=tuple(observation.cmr_pixel_spacing_mm),
        cmr_slice_thickness_mm=observation.cmr_slice_thickness_mm,
        cmr_surface_sigma_mm=observation.cmr_surface_sigma_mm,
        tee_grid_shape=tuple(observation.tee_grid_shape),
        tee_surface_sigma_mm=observation.tee_surface_sigma_mm,
        image_noise_std=observation.image_noise_std,
        tee_speckle_std=observation.tee_speckle_std,
        topology=topology,
        chamber_levels=observation.chamber_levels,
        leaflet_rows=observation.leaflet_rows,
        tee_shadow=observation.tee_shadow,
    )
    parameters = {
        "tee_axis_angle_rad": axis_angle.tolist(),
        "tee_translation_mm": translation.tolist(),
        "phase_offset_cycles": phase_offset,
        "phase_logits": phase_logits.tolist(),
        "session_static_scale": static_scale,
        "session_dynamic_scale": dynamic_scale,
    }
    return case, parameters


def _load_donors(
    config: SemisyntheticConfig,
) -> tuple[
    list[Donor],
    list[Donor],
    HeterogeneousTemplate | None,
    dict[str, str] | None,
]:
    donor_config = config.donors
    rich = (config.observation.chamber_levels, config.observation.leaflet_rows) != (1, 1)
    if rich and donor_config.mode != "test_manifest":
        raise NotImplementedError("a richer procedural instance (W2.6) is built from a donor manifest")
    if donor_config.mode == "procedural_test":
        hearts, valves = make_procedural_donors(
            root_seed=config.root_seed,
            whole_heart_count=donor_config.whole_heart_count,
            valve_count=donor_config.valve_count,
            n_annulus=config.observation.n_annulus,
        )
        return hearts, valves, None, None
    if donor_config.mode == "test_manifest":
        if donor_config.manifest is None:
            raise ValueError("test manifest mode requires a donor manifest")
        hearts, valves = load_donor_manifest(
            donor_config.manifest,
            config.observation.n_annulus,
            chamber_levels=config.observation.chamber_levels,
            leaflet_rows=config.observation.leaflet_rows,
        )
        overlap = {item.global_person_id for item in hearts} & {
            item.global_person_id for item in valves
        }
        if overlap:
            raise ValueError(
                "test_manifest cannot safely split people shared across donor pools; "
                "use production_manifest with a frozen global split lock"
            )
        return hearts, valves, None, None
    if (
        donor_config.manifest is None
        or donor_config.template_manifest is None
        or donor_config.split_lock is None
    ):
        raise ValueError(
            "production manifest mode requires donor, template, and global split-lock manifests"
        )
    topology = load_frozen_template(donor_config.template_manifest)
    if len(topology.annulus_global_indices) != config.observation.n_annulus:
        raise ValueError("observation.n_annulus does not match the frozen template")
    hearts, valves = load_donor_manifest(
        donor_config.manifest,
        template=topology,
    )
    donor_splits = _load_frozen_donor_splits(
        donor_config.split_lock,
        hearts + valves,
    )
    return hearts, valves, topology, donor_splits


def _portable_config_record(config: SemisyntheticConfig) -> dict[str, Any]:
    record = config.as_dict()
    manifest = record["donors"].get("manifest")
    if manifest:
        record["donors"]["manifest"] = Path(manifest).name
    template_manifest = record["donors"].get("template_manifest")
    if template_manifest:
        record["donors"]["template_manifest"] = Path(template_manifest).name
    split_lock = record["donors"].get("split_lock")
    if split_lock:
        record["donors"]["split_lock"] = Path(split_lock).name
    # Recorded only when on, so benchmarks built before the option existed keep
    # an identical config record.
    if not record["donors"].get("orient_valves"):
        record["donors"].pop("orient_valves", None)
    return record


def _build_into(config: SemisyntheticConfig, root: Path) -> dict[str, Any]:
    session_subspace = _load_session_subspace(
        getattr(config.perturbation, "session_subspace_path", None)
    )
    hearts, valves, topology, donor_splits = _load_donors(config)
    fractions = config.split.as_dict()
    if donor_splits is None:
        heart_partitions = _partition_donors(
            hearts, config.split_seed, "whole_heart", fractions
        )
        valve_partitions = _partition_donors(
            valves, config.split_seed, "valve", fractions
        )
    else:
        heart_partitions = _locked_partitions(hearts, donor_splits, "whole_heart")
        valve_partitions = _locked_partitions(valves, donor_splits, "valve")

    resolved_config = _portable_config_record(config)
    config_path = write_json(root / "config.resolved.json", resolved_config)
    seed_records: list[dict[str, Any]] = []
    split_records: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    used_hearts: set[str] = set()
    used_valves: set[str] = set()

    for split in SPLIT_NAMES:
        split_hearts = heart_partitions[split]
        split_valves = valve_partitions[split]
        pair_count = min(len(split_hearts), len(split_valves))
        if pair_count < 1:
            raise ValueError(f"split {split!r} has no usable heart/valve donor pair")
        pairing_order = make_rng(config.split_seed, "pairing", split).permutation(len(split_valves))
        paired_valves = [split_valves[int(index)] for index in pairing_order[:pair_count]]
        for pair_index, (heart, valve) in enumerate(zip(split_hearts[:pair_count], paired_valves)):
            identity_token = {
                "whole_heart_donor_id": heart.donor_id,
                "valve_donor_id": valve.donor_id,
            }
            identity_id = f"identity-{canonical_sha256(identity_token)[:16]}"
            identity_seed = derive_seed(
                config.root_seed,
                "identity",
                heart.donor_id,
                valve.donor_id,
            )
            template = combine_donors(
                heart,
                valve,
                config.observation.n_annulus,
                template=topology,
                orient_valve=config.donors.orient_valves,
                chamber_levels=config.observation.chamber_levels,
                leaflet_rows=config.observation.leaflet_rows,
            )
            used_hearts.add(heart.donor_id)
            used_valves.add(valve.donor_id)
            for variant_index in range(config.variants_per_identity):
                seeds = _factor_seeds(config.root_seed, identity_seed, variant_index)
                sample_token = {
                    "identity_id": identity_id,
                    "variant_index": variant_index,
                }
                sample_id = f"sample-{canonical_sha256(sample_token)[:16]}"
                case, parameters = _generate_sample(
                    config, template, seeds, topology, session_subspace
                )
                relative_path = Path("samples") / split / f"{sample_id}.npz"
                sample_path = root / relative_path
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                arrays = _sample_arrays(
                    case,
                    sample_id,
                    split,
                    identity_id,
                    heart.donor_id,
                    valve.donor_id,
                    heart.global_person_id,
                    valve.global_person_id,
                    variant_index,
                    seeds,
                )
                np.savez_compressed(sample_path, **{key: arrays[key] for key in sorted(arrays)})
                sample_sha = sha256_file(sample_path)
                entry = {
                    "schema_version": SAMPLE_SCHEMA,
                    "sample_id": sample_id,
                    "split": split,
                    "identity_id": identity_id,
                    "identity_index_within_split": pair_index,
                    "variant_index": variant_index,
                    "whole_heart_donor_id": heart.donor_id,
                    "valve_donor_id": valve.donor_id,
                    "whole_heart_global_person_id": heart.global_person_id,
                    "valve_global_person_id": valve.global_person_id,
                    "source_datasets": {
                        "whole_heart": heart.source_dataset,
                        "valve": valve.source_dataset,
                    },
                    "relative_path": relative_path.as_posix(),
                    "sha256": sample_sha,
                    "seeds": seeds,
                    "ground_truth_parameters": parameters,
                    "observation_domain": case.observation_domain,
                    "invariants": {
                        "attachment_residual_mm": case.attachment_residual_mm,
                        "periodic_seam_mm": case.periodic_seam_mm,
                        "phase_minimum_increment": case.phase_minimum_increment,
                    },
                }
                split_records[split].append(entry)
                seed_records.append(
                    {
                        "sample_id": sample_id,
                        "split": split,
                        "identity_id": identity_id,
                        "seeds": seeds,
                    }
                )

    split_summary: dict[str, Any] = {}
    for split in SPLIT_NAMES:
        records = sorted(split_records[split], key=lambda value: value["sample_id"])
        manifest_path = _write_jsonl(root / "manifests" / f"{split}.jsonl", records)
        split_summary[split] = {
            "manifest": manifest_path.relative_to(root).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "sample_count": len(records),
            "identity_count": len({record["identity_id"] for record in records}),
        }

    seed_manifest = {
        "schema_version": "rooted4d-seed-manifest-v1",
        "root_seed": config.root_seed,
        "split_seed": config.split_seed,
        "derivation": "SHA-256 named path -> NumPy SeedSequence -> uint63",
        "procedural_donor_seeds": {
            donor.donor_id: derive_seed(config.root_seed, "donor", donor.kind, donor.donor_id)
            for donor in sorted(hearts + valves, key=lambda item: item.donor_id)
            if donor.source_dataset == "procedural"
        },
        "samples": sorted(seed_records, key=lambda value: value["sample_id"]),
    }
    seed_path = write_json(root / "seeds.json", seed_manifest)

    all_heart_ids = {donor.donor_id for donor in hearts}
    all_valve_ids = {donor.donor_id for donor in valves}
    if config.donors.split_lock is not None:
        split_lock_snapshot = root / "provenance" / "global_split_lock.json"
        split_lock_snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config.donors.split_lock, split_lock_snapshot)
        global_split_descriptor = {
            "status": "frozen",
            "path": split_lock_snapshot.relative_to(root).as_posix(),
            "sha256": sha256_file(split_lock_snapshot),
        }
    else:
        global_split_descriptor = {
            "status": "generated_from_split_seed",
            "path": None,
            "sha256": None,
        }
    source_manifests = {
        "donor_manifest_sha256": (
            sha256_file(config.donors.manifest) if config.donors.manifest is not None else None
        ),
        "template_manifest_sha256": (
            sha256_file(config.donors.template_manifest)
            if config.donors.template_manifest is not None
            else None
        ),
    }
    dataset_manifest = {
        "schema_version": DATASET_SCHEMA,
        "generator_version": __version__,
        "scope": (
            "Known-truth point- and image-likelihood-domain proof of concept. It does not establish "
            "patient-specific clinical CMR/TEE fusion validity."
        ),
        "observation_domain": config.observation.output_domain,
        "topology_kind": (
            "frozen_production_v1" if topology is not None else "procedural_test_v1"
        ),
        "template_id": topology.template_id if topology is not None else None,
        "global_split_lock": global_split_descriptor,
        "source_manifests": source_manifests,
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": sha256_file(config_path),
        },
        "reproducibility": {
            "root_seed": config.root_seed,
            "split_seed": config.split_seed,
            "seed_manifest": seed_path.relative_to(root).as_posix(),
            "seed_manifest_sha256": sha256_file(seed_path),
            "environment": environment_record(),
            "generator_source_sha256": generator_source_fingerprint(),
        },
        "split_order": list(SPLIT_NAMES),
        "splits": split_summary,
        "donor_partitions": {
            split: {
                "whole_heart": [donor.donor_id for donor in heart_partitions[split]],
                "valve": [donor.donor_id for donor in valve_partitions[split]],
                "whole_heart_people": sorted(
                    {donor.global_person_id for donor in heart_partitions[split]}
                ),
                "valve_people": sorted(
                    {donor.global_person_id for donor in valve_partitions[split]}
                ),
            }
            for split in SPLIT_NAMES
        },
        "unused_donors": {
            "whole_heart": sorted(all_heart_ids - used_hearts),
            "valve": sorted(all_valve_ids - used_valves),
        },
        "donor_records": [
            donor.public_record()
            for donor in sorted(hearts + valves, key=lambda item: item.donor_id)
        ],
    }
    write_json(root / "dataset.json", dataset_manifest)
    return dataset_manifest


def build_semisynthetic_dataset(
    config: SemisyntheticConfig,
    output_directory: str | Path,
) -> Path:
    """Build atomically and verify before exposing the final dataset directory."""

    config.validate()
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"output already exists: {output}; choose a new directory to preserve provenance"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        _build_into(config, temporary)
        from rooted4d.data.semisynthetic import audit_semisynthetic_dataset

        audit = audit_semisynthetic_dataset(temporary, verify_checksums=True, verify_arrays=True)
        write_json(temporary / "audit.json", audit.as_dict())
        audit.raise_if_failed()
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output
