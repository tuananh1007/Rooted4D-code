import unittest

from rooted4d.data import find_split_leakage, get_dataset, load_catalog


class CatalogTests(unittest.TestCase):
    def test_key_resources_are_present(self) -> None:
        catalog = load_catalog()
        for key in ("acdc", "mnms2", "mitea", "mvseg2023", "local_paired_cmr_4dtee"):
            self.assertIn(key, catalog)

    def test_mvseg_is_explicitly_gated(self) -> None:
        entry = get_dataset("mvseg2023")
        self.assertEqual(entry.fetch_kind, "huggingface_dataset")
        self.assertIn("hf download", entry.command)
        self.assertIn("accept", entry.access)

    def test_mvaa_is_not_mislabeled_as_downloadable_data(self) -> None:
        entry = get_dataset("mvaa2026")
        self.assertIn("proposal_pdf_only", entry.status)
        self.assertEqual(entry.fetch_kind, "manual")

    def test_patient_split_leakage_is_detected(self) -> None:
        leakage = find_split_leakage(
            {"development": ["p001", "p002"], "locked_test": ["p003", "p001"]}
        )
        self.assertIn("p001", leakage)



if __name__ == "__main__":
    unittest.main()
