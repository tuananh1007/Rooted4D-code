import tempfile
import unittest
from pathlib import Path

import numpy as np

from rooted4d.data import get_dataset
from rooted4d.data.validation import validate_dataset
from rooted4d.inference import recover_synthetic_phase_pose
from rooted4d.synthetic import generate_case, save_case


class SyntheticTests(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        first = generate_case(seed=12)
        second = generate_case(seed=12)
        np.testing.assert_allclose(first.cmr_points, second.cmr_points)
        np.testing.assert_allclose(first.tee_points, second.tee_points)
        np.testing.assert_array_equal(first.tee_indices, second.tee_indices)

    def test_invariants_hold(self) -> None:
        case = generate_case(seed=5)
        self.assertLessEqual(case.attachment_residual_mm, 1e-12)
        self.assertLessEqual(case.periodic_seam_mm, 1e-12)
        self.assertGreater(case.phase_minimum_increment, 0.0)

    def test_saved_case_matches_catalog_layout(self) -> None:
        case = generate_case(seed=9)
        with tempfile.TemporaryDirectory() as directory:
            save_case(case, directory)
            report = validate_dataset(get_dataset("synthetic_rooted4d"), Path(directory))
            self.assertTrue(report.passed)
            self.assertEqual(report.matched_files, 2)

    def test_oracle_phase_pose_initializer(self) -> None:
        case = generate_case(seed=13, tee_noise_mm=0.25)
        recovered = recover_synthetic_phase_pose(case, n_offset_candidates=512)
        self.assertLess(recovered.phase_error_cycles, 0.005)
        self.assertLess(recovered.rotation_error_degrees, 0.5)
        self.assertLess(recovered.translation_error_mm, 0.5)


if __name__ == "__main__":
    unittest.main()
