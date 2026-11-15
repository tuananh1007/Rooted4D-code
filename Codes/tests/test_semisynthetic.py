import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from rooted4d.data import (
    SemisyntheticDataset,
    audit_semisynthetic_dataset,
    build_split_loaders,
    compare_semisynthetic_datasets,
    dataset_fingerprint,
)
from rooted4d.synthetic import build_semisynthetic_dataset, config_from_mapping
from rooted4d.synthetic.config import DonorConfig
from rooted4d.synthetic.dataset_cli import main as dataset_cli_main
from rooted4d.synthetic.donors import make_procedural_donors
from rooted4d.synthetic.reproducibility import sha256_file


def small_config(root_seed: int = 8675309):
    return config_from_mapping(
        {
            "schema_version": "rooted4d-semisynthetic-config-v1",
            "reproducibility": {"root_seed": root_seed, "split_seed": 65537},
            "split": {"train": 0.60, "val": 0.20, "test": 0.20},
            "donors": {
                "mode": "procedural_test",
                "whole_heart_count": 6,
                "valve_count": 6,
                "manifest": None,
                "template_manifest": None,
            },
            "samples": {"variants_per_identity": 2},
            "observation": {
                "n_annulus": 24,
                "n_cmr_frames": 8,
                "n_tee_frames": 10,
                "fourier_order": 2,
                "cmr_noise_mm": 0.20,
                "tee_noise_mm": 0.25,
                "cmr_dropout_fraction": 0.10,
                "tee_dropout_fraction": 0.20,
            },
            "perturbation": {
                "pose_axis_angle_std_rad": 0.10,
                "pose_translation_std_mm": 5.0,
                "phase_offset_min_cycles": -0.15,
                "phase_offset_max_cycles": 0.15,
                "phase_logit_std": 0.40,
                "phase_knots": 6,
                "session_static_scale_min": 0.80,
                "session_static_scale_max": 1.20,
                "session_dynamic_scale_min": 0.80,
                "session_dynamic_scale_max": 1.20,
            },
        }
    )


class SemisyntheticDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        cls.config = small_config()
        cls.dataset_root = build_semisynthetic_dataset(cls.config, cls.base / "dataset")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_full_audit_passes_with_nonempty_splits(self) -> None:
        report = audit_semisynthetic_dataset(
            self.dataset_root,
            verify_checksums=True,
            verify_arrays=True,
        )
        self.assertTrue(report.passed, report.errors)
        for split in ("train", "val", "test"):
            self.assertGreater(report.statistics["splits"][split]["samples"], 0)
            self.assertGreater(report.statistics["splits"][split]["identities"], 0)
        self.assertEqual(
            {
                split: report.statistics["splits"][split]["samples"]
                for split in ("train", "val", "test")
            },
            {"train": 8, "val": 2, "test": 2},
        )

    def test_identity_and_both_donor_pools_are_disjoint(self) -> None:
        datasets = {
            split: SemisyntheticDataset(
                self.dataset_root,
                split,
                verify_checksums=False,
            )
            for split in ("train", "val", "test")
        }
        for field in (
            "identity_id",
            "whole_heart_donor_id",
            "valve_donor_id",
            "whole_heart_global_person_id",
            "valve_global_person_id",
        ):
            values = {
                split: {entry[field] for entry in dataset.entries}
                for split, dataset in datasets.items()
            }
            self.assertTrue(values["train"].isdisjoint(values["val"]))
            self.assertTrue(values["train"].isdisjoint(values["test"]))
            self.assertTrue(values["val"].isdisjoint(values["test"]))

    def test_loader_returns_only_requested_split(self) -> None:
        loaders = build_split_loaders(
            self.dataset_root,
            batch_size=2,
            verify_checksums=True,
        )
        self.assertEqual(set(loaders), {"train", "val", "test"})
        seen = {}
        for split, loader in loaders.items():
            self.assertEqual(loader.split, split)
            sample_ids = []
            for batch in loader:
                self.assertEqual(batch["split"], split)
                self.assertTrue(all(item["split"] == split for item in batch["metadata"]))
                sample_ids.extend(item["sample_id"] for item in batch["metadata"])
            self.assertEqual(len(sample_ids), len(loader.dataset))
            seen[split] = set(sample_ids)
        self.assertTrue(seen["train"].isdisjoint(seen["val"] | seen["test"]))

    def test_loader_order_is_reproducible_by_epoch(self) -> None:
        loaders = build_split_loaders(
            self.dataset_root,
            batch_size=2,
            verify_checksums=False,
        )
        train = loaders["train"]
        train.set_epoch(4)
        first = [item["sample_id"] for batch in train for item in batch["metadata"]]
        train.set_epoch(4)
        second = [item["sample_id"] for batch in train for item in batch["metadata"]]
        self.assertEqual(first, second)

    def test_same_seed_produces_byte_identical_fingerprint(self) -> None:
        second = build_semisynthetic_dataset(self.config, self.base / "same-seed")
        self.assertTrue(compare_semisynthetic_datasets(self.dataset_root, second))
        self.assertEqual(dataset_fingerprint(self.dataset_root), dataset_fingerprint(second))

    def test_different_root_seed_changes_dataset(self) -> None:
        different_config = replace(self.config, root_seed=self.config.root_seed + 1)
        different = build_semisynthetic_dataset(different_config, self.base / "different-seed")
        self.assertFalse(compare_semisynthetic_datasets(self.dataset_root, different))

    def test_injected_cross_split_donor_leak_is_rejected(self) -> None:
        leaked = self.base / "leaked"
        shutil.copytree(self.dataset_root, leaked)
        train_path = leaked / "manifests" / "train.jsonl"
        val_path = leaked / "manifests" / "val.jsonl"
        train_entry = json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])
        val_entries = [
            json.loads(line) for line in val_path.read_text(encoding="utf-8").splitlines()
        ]
        val_entries[0]["whole_heart_donor_id"] = train_entry["whole_heart_donor_id"]
        val_path.write_text(
            "\n".join(
                json.dumps(item, sort_keys=True, separators=(",", ":"))
                for item in val_entries
            )
            + "\n",
            encoding="utf-8",
        )
        report = audit_semisynthetic_dataset(
            leaked,
            verify_checksums=False,
            verify_arrays=False,
        )
        self.assertFalse(report.passed)
        self.assertTrue(any("leakage" in error for error in report.errors))
        with self.assertRaises(ValueError):
            SemisyntheticDataset(leaked, "val", verify_checksums=False, strict=True)

    def test_injected_cross_split_source_person_leak_is_rejected(self) -> None:
        leaked = self.base / "person-leaked"
        shutil.copytree(self.dataset_root, leaked)
        train_path = leaked / "manifests" / "train.jsonl"
        val_path = leaked / "manifests" / "val.jsonl"
        train_entry = json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])
        val_entries = [
            json.loads(line) for line in val_path.read_text(encoding="utf-8").splitlines()
        ]
        val_entries[0]["whole_heart_global_person_id"] = train_entry[
            "whole_heart_global_person_id"
        ]
        val_path.write_text(
            "\n".join(
                json.dumps(item, sort_keys=True, separators=(",", ":"))
                for item in val_entries
            )
            + "\n",
            encoding="utf-8",
        )
        report = audit_semisynthetic_dataset(
            leaked,
            verify_checksums=False,
            verify_arrays=False,
        )
        self.assertFalse(report.passed)
        self.assertTrue(any("source_global_person_id leakage" in error for error in report.errors))

    def test_corrupt_sample_is_rejected(self) -> None:
        corrupt = self.base / "corrupt"
        shutil.copytree(self.dataset_root, corrupt)
        manifest = corrupt / "manifests" / "test.jsonl"
        entry = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
        sample = corrupt / entry["relative_path"]
        payload = bytearray(sample.read_bytes())
        payload[-1] ^= 0x01
        sample.write_bytes(payload)
        report = audit_semisynthetic_dataset(
            corrupt,
            verify_checksums=True,
            verify_arrays=False,
        )
        self.assertFalse(report.passed)
        self.assertTrue(any("sample checksum mismatch" in error for error in report.errors))

    def test_existing_output_is_never_overwritten(self) -> None:
        with self.assertRaises(FileExistsError):
            build_semisynthetic_dataset(self.config, self.dataset_root)


class SemisyntheticConfigurationTests(unittest.TestCase):
    def test_invalid_split_fractions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            config_from_mapping(
                {
                    "split": {"train": 0.7, "val": 0.2, "test": 0.2},
                }
            )

    def test_invalid_seed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            config_from_mapping({"reproducibility": {"root_seed": -1, "split_seed": 2}})

    def test_cli_generate_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cli-dataset"
            config_path = Path(__file__).resolve().parents[1] / "configs" / "semisynthetic_ci.yaml"
            with contextlib.redirect_stdout(io.StringIO()):
                generated = dataset_cli_main(
                    ["generate", "--config", str(config_path), "--output", str(output)]
                )
                audited = dataset_cli_main(["audit", "--root", str(output)])
            self.assertEqual(generated, 0)
            self.assertEqual(audited, 0)

    def test_fixed_topology_donor_manifest_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hearts, valves = make_procedural_donors(
                root_seed=31,
                whole_heart_count=3,
                valve_count=3,
                n_annulus=24,
            )
            records = []
            for donor in hearts + valves:
                path = root / f"{donor.donor_id}.npz"
                np.savez_compressed(path, vertices=donor.vertices)
                records.append(
                    {
                        "donor_id": donor.donor_id,
                        "global_person_id": f"test-person-{donor.donor_id}",
                        "kind": donor.kind,
                        "path": path.name,
                        "sha256": sha256_file(path),
                        "source_dataset": "test-real-derived",
                    }
                )
            manifest = root / "donors.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "rooted4d-fixed-topology-donors-v2",
                        "topology_kind": "procedural_test_v1",
                        "template_id": None,
                        "donors": records,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            config = replace(
                small_config(),
                donors=DonorConfig(mode="test_manifest", manifest=str(manifest)),
                variants_per_identity=1,
            )
            generated = build_semisynthetic_dataset(config, root / "generated")
            report = audit_semisynthetic_dataset(generated)
            self.assertTrue(report.passed, report.errors)
            self.assertEqual(report.statistics["total_samples"], 3)


if __name__ == "__main__":
    unittest.main()


class DataFingerprintTest(unittest.TestCase):
    """The data hash must ignore provenance metadata that does not affect arrays."""

    def test_data_fingerprint_ignores_the_manifest(self):
        import json
        import tempfile
        from pathlib import Path

        from rooted4d.data import data_fingerprint, dataset_fingerprint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ds"
            (root / "train").mkdir(parents=True)
            (root / "train" / "sample.npz").write_bytes(b"array-bytes")
            manifest = {"generator_version": "0.1.0"}
            (root / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")

            data_before = data_fingerprint(root)
            full_before = dataset_fingerprint(root)

            # A version bump changes provenance but not one generated byte.
            manifest["generator_version"] = "0.2.0"
            (root / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")

            self.assertEqual(data_fingerprint(root), data_before)
            self.assertNotEqual(dataset_fingerprint(root), full_before)

    def test_data_fingerprint_detects_a_changed_array(self):
        import tempfile
        from pathlib import Path

        from rooted4d.data import data_fingerprint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ds"
            (root / "train").mkdir(parents=True)
            target = root / "train" / "sample.npz"
            target.write_bytes(b"array-bytes")
            before = data_fingerprint(root)
            target.write_bytes(b"different-bytes")
            self.assertNotEqual(data_fingerprint(root), before)
