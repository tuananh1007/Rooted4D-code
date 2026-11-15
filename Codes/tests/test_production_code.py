import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from rooted4d.data import audit_semisynthetic_dataset
from rooted4d.geometry import RigidTransform, load_frozen_template, make_toy_complex
from rooted4d.inference import infer_cross_modal_alignment
from rooted4d.observation import (
    default_cmr_stack,
    default_tee_grid,
    point_to_mesh_distance,
    render_cmr_surface_likelihood,
    render_tee_surface_likelihood,
)
from rooted4d.registration import (
    RegistrationConfig,
    align_periodic_features,
    register_point_sets,
)
from rooted4d.synthetic import build_semisynthetic_dataset, config_from_mapping
from rooted4d.synthetic import generate_case
from rooted4d.synthetic.reproducibility import sha256_file


def _fan_faces(count: int) -> np.ndarray:
    return np.asarray(
        [(index, (index + 1) % count, count) for index in range(count)],
        dtype=np.int64,
    )


def _write_valid_template(root: Path) -> tuple[Path, np.ndarray]:
    count = 12
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    ring = np.stack((10.0 * np.cos(theta), 8.0 * np.sin(theta), np.zeros(count)), axis=1)
    vertices = np.concatenate(
        (
            ring,
            np.asarray(
                [
                    [0.0, 0.0, -6.0],
                    [0.0, 0.0, -8.0],
                    [0.0, 0.0, 6.0],
                    [0.0, -2.0, -2.0],
                    [0.0, 2.0, -2.0],
                ]
            ),
        ),
        axis=0,
    )
    tetrahedra = np.asarray([[0, 1, 2, 12]], dtype=np.int64)
    tetra = vertices[tetrahedra[0]]
    signed = float(
        np.dot(
            tetra[1] - tetra[0],
            np.cross(tetra[2] - tetra[0], tetra[3] - tetra[0]),
        )
    )
    if signed <= 0.0:
        tetrahedra[0, [1, 2]] = tetrahedra[0, [2, 1]]

    arrays = {
        "vertices": vertices,
        "tetrahedra": tetrahedra,
        "annulus": np.arange(count, dtype=np.int64),
        "lv_map": np.concatenate((np.arange(count), [12])),
        "lv_faces": _fan_faces(count),
        "lv_attach": np.arange(count, dtype=np.int64),
        "epi_map": np.concatenate((np.arange(count), [13])),
        "epi_faces": _fan_faces(count),
        "epi_attach": np.asarray([], dtype=np.int64),
        "la_map": np.concatenate((np.arange(count), [14])),
        "la_faces": _fan_faces(count),
        "la_attach": np.arange(count, dtype=np.int64),
        "ant_map": np.concatenate((np.arange(0, 6), [15])),
        "ant_faces": np.asarray([(i, i + 1, 6) for i in range(5)], dtype=np.int64),
        "ant_attach": np.arange(6, dtype=np.int64),
        "post_map": np.concatenate((np.arange(6, 12), [16])),
        "post_faces": np.asarray([(i, i + 1, 6) for i in range(5)], dtype=np.int64),
        "post_attach": np.arange(6, dtype=np.int64),
    }
    asset = root / "template.npz"
    np.savez_compressed(asset, **arrays)
    component_keys = {
        "lv_endocardium": ("lv_map", "lv_faces", "lv_attach"),
        "lv_epicardium": ("epi_map", "epi_faces", "epi_attach"),
        "left_atrium": ("la_map", "la_faces", "la_attach"),
        "anterior_leaflet": ("ant_map", "ant_faces", "ant_attach"),
        "posterior_leaflet": ("post_map", "post_faces", "post_attach"),
    }
    manifest = {
        "schema_version": "rooted4d-frozen-template-manifest-v1",
        "status": "frozen",
        "template_id": "test-frozen-mitral-v1",
        "asset_npz": asset.name,
        "asset_sha256": sha256_file(asset),
        "vertices_key": "vertices",
        "tetrahedra_key": "tetrahedra",
        "annulus_global_indices_key": "annulus",
        "coordinate_system": "LPS",
        "units": "mm",
        "landmarks": {
            "anterolateral_commissure": 0,
            "posteromedial_commissure": 6,
            "anterior_saddle_horn": 3,
            "posterior_annular_midpoint": 9,
        },
        "components": {
            name: {
                "local_to_global_key": keys[0],
                "faces_key": keys[1],
                "attachment_local_indices_key": keys[2],
            }
            for name, keys in component_keys.items()
        },
    }
    manifest_path = root / "template.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path, vertices


class ObservationOperatorTests(unittest.TestCase):
    def test_exact_point_to_triangle_distance(self) -> None:
        vertices = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        faces = np.asarray([[0, 1, 2]])
        query = np.asarray([[0.25, 0.25, 0.0], [0.25, 0.25, 3.0], [3.0, 0.0, 0.0]])
        distance = point_to_mesh_distance(query, vertices, faces)
        np.testing.assert_allclose(distance, [0.0, 3.0, 1.0], atol=1e-10)

    def test_cmr_and_tee_renderers_are_bounded_and_deterministic(self) -> None:
        fixture = make_toy_complex(24)
        cmr_stack = default_cmr_stack(
            fixture.global_vertices,
            n_slices=3,
            image_shape=(10, 12),
            pixel_spacing_mm=(4.0, 4.0),
        )
        cmr = render_cmr_surface_likelihood(
            fixture.global_vertices,
            fixture.faces["cmr_surface"],
            cmr_stack,
            noise_std=0.01,
            seed=14,
        )
        self.assertEqual(cmr.shape, (1, 3, 10, 12))
        self.assertTrue(np.all((cmr >= 0.0) & (cmr <= 1.0)))
        tee_grid = default_tee_grid(fixture.global_vertices, grid_shape=(8, 6, 7))
        first = render_tee_surface_likelihood(
            fixture.global_vertices,
            fixture.faces["tee_surface"],
            tee_grid,
            speckle_std=0.1,
            noise_std=0.01,
            seed=22,
        )
        second = render_tee_surface_likelihood(
            fixture.global_vertices,
            fixture.faces["tee_surface"],
            tee_grid,
            speckle_std=0.1,
            noise_std=0.01,
            seed=22,
        )
        self.assertEqual(first.shape, (1, 8, 6, 7))
        np.testing.assert_array_equal(first, second)


class RegistrationTests(unittest.TestCase):
    def test_cross_modal_initializer_uses_only_segmented_annular_sequences(self) -> None:
        case = generate_case(
            seed=77,
            n_cmr_frames=24,
            n_tee_frames=24,
            cmr_noise_mm=0.01,
            tee_noise_mm=0.01,
            cmr_dropout_fraction=0.0,
            tee_dropout_fraction=0.0,
        )
        cmr_mask = case.cmr_indices < 24
        tee_mask = case.tee_indices < 24
        result = infer_cross_modal_alignment(
            case.cmr_points[:, cmr_mask],
            case.tee_points[:, tee_mask],
        )
        rotation_delta = result.cmr_to_tee.rotation @ case.pose_rotation.T
        angle_degrees = float(
            np.degrees(
                np.arccos(np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0))
            )
        )
        self.assertLess(angle_degrees, 1.0)
        self.assertLess(
            float(np.linalg.norm(result.cmr_to_tee.translation - case.pose_translation)),
            1.0,
        )
        phase_error = abs(
            float(
                np.arctan2(
                    np.sin(2.0 * np.pi * (result.temporal.offset_cycles - case.phase_offset)),
                    np.cos(2.0 * np.pi * (result.temporal.offset_cycles - case.phase_offset)),
                )
                / (2.0 * np.pi)
            )
        )
        self.assertLess(phase_error, 0.08)

    def test_non_oracle_registration_recovers_observed_landmark_pose(self) -> None:
        rng = np.random.default_rng(42)
        source = rng.normal(size=(80, 3)) * np.asarray([12.0, 7.0, 4.0])
        truth = RigidTransform.from_axis_angle(
            np.asarray([0.12, -0.08, 0.18]), np.asarray([7.0, -4.0, 3.0])
        )
        target = truth.apply(source) + rng.normal(0.0, 0.03, source.shape)
        order = rng.permutation(len(target))
        result = register_point_sets(
            source,
            target[order],
            source_landmarks=source[[0, 11, 37, 58]],
            target_landmarks=truth.apply(source[[0, 11, 37, 58]]),
            config=RegistrationConfig(max_iterations=30, trim_fraction=0.05),
        )
        rotation_delta = result.transform.rotation @ truth.rotation.T
        angle = np.arccos(np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0))
        self.assertLess(float(angle), 0.01)
        self.assertLess(float(np.linalg.norm(result.transform.translation - truth.translation)), 0.1)
        self.assertLess(result.rmse_mm, 0.1)

    def test_patch_labels_reject_missing_anatomy(self) -> None:
        source = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with self.assertRaises(ValueError):
            register_point_sets(
                source,
                source + 1.0,
                source_labels=np.asarray(["lv", "la", "lv"]),
                target_labels=np.asarray(["lv", "lv", "lv"]),
            )

    def test_circular_dtw_recovers_integer_phase_shift(self) -> None:
        count = 24
        shift = 4
        phase = np.arange(count) / count
        source = np.stack(
            (np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)), axis=1
        )
        target = np.roll(source, -shift, axis=0)
        alignment = align_periodic_features(source, target, band_fraction=0.3)
        expected = np.mod(phase + shift / count, 1.0)
        circular_error = np.abs(
            np.arctan2(
                np.sin(2.0 * np.pi * (alignment.target_to_source_phase - expected)),
                np.cos(2.0 * np.pi * (alignment.target_to_source_phase - expected)),
            )
            / (2.0 * np.pi)
        )
        self.assertLess(float(np.median(circular_error)), 1.0 / count + 1e-6)
        self.assertTrue(np.all(np.diff(alignment.target_to_source_unwrapped) > 0.0))


class ProductionManifestTests(unittest.TestCase):
    def test_frozen_template_and_production_donor_path_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template_manifest, base_vertices = _write_valid_template(root)
            template = load_frozen_template(template_manifest)
            self.assertTrue(template.validate().passed)

            donor_records = []
            for kind in ("whole_heart", "valve"):
                for index in range(3):
                    vertices = base_vertices.copy()
                    vertices[:, 0] *= 1.0 + 0.01 * index
                    if kind == "valve":
                        vertices[15:, 2] -= 0.2 * index
                    path = root / f"{kind}-{index}.npz"
                    np.savez_compressed(path, vertices=vertices)
                    donor_records.append(
                        {
                            "donor_id": f"{kind}-{index}",
                            "global_person_id": f"person-{kind}-{index}",
                            "kind": kind,
                            "path": path.name,
                            "sha256": sha256_file(path),
                            "source_dataset": "unit-test-fixed-topology",
                        }
                    )
            donor_manifest = root / "donors.json"
            donor_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "rooted4d-fixed-topology-donors-v2",
                        "topology_kind": "frozen_production_v1",
                        "template_id": template.template_id,
                        "donors": donor_records,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            split_records = []
            for index, split in enumerate(("train", "val", "test")):
                for kind, hex_character in (("whole_heart", "a"), ("valve", "b")):
                    split_records.append(
                        {
                            "global_person_id": f"person-{kind}-{index}",
                            "source_dataset": "unit-test-fixed-topology",
                            "source_subject_id_hash": (
                                "sha256:" + hex_character * 62 + f"{index:02x}"
                            ),
                            "split": split,
                            "session_ids": [f"session-{kind}-{index}"],
                            "donor_ids": [f"{kind}-{index}"],
                            "modalities": ["cmr" if kind == "whole_heart" else "tee"],
                        }
                    )
            split_lock = root / "split-lock.json"
            split_lock.write_text(
                json.dumps(
                    {
                        "schema_version": "rooted4d-global-split-lock-v1",
                        "lock_id": "unit-test-frozen-lock",
                        "status": "frozen",
                        "created_utc": "2026-08-23T00:00:00Z",
                        "identity_hash": {
                            "algorithm": "sha256",
                            "salt_id": "unit-test-salt-id",
                            "raw_identifiers_stored": False,
                            "linkage_table_location": "outside_repository",
                        },
                        "split_policy": {
                            "unit": "global_person",
                            "freeze_before_model_fitting": True,
                            "pair_donors_only_within_split": True,
                            "test_access": "one_final_locked_evaluation",
                        },
                        "records": split_records,
                        "artifact_provenance": [],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            config = config_from_mapping(
                {
                    "reproducibility": {"root_seed": 101, "split_seed": 103},
                    "donors": {
                        "mode": "production_manifest",
                        "manifest": str(donor_manifest),
                        "template_manifest": str(template_manifest),
                        "split_lock": str(split_lock),
                    },
                    "samples": {"variants_per_identity": 1},
                    "observation": {
                        "n_annulus": 12,
                        "n_cmr_frames": 4,
                        "n_tee_frames": 5,
                        "fourier_order": 2,
                        "output_domain": "surface_points",
                    },
                }
            )
            output = build_semisynthetic_dataset(config, root / "dataset")
            report = audit_semisynthetic_dataset(output)
            self.assertTrue(report.passed, report.errors)
            manifest = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["topology_kind"], "frozen_production_v1")
            self.assertEqual(manifest["template_id"], template.template_id)

    def test_template_digest_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _ = _write_valid_template(root)
            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            payload["asset_sha256"] = "0" * 64
            manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_frozen_template(manifest_path)


if __name__ == "__main__":
    unittest.main()
