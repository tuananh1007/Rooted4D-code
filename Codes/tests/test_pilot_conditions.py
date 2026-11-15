"""The pilot's evidence conditions must damage only what they claim to."""

import unittest

import numpy as np

from rooted4d.experiments.pilot_conditions import (
    CONDITIONS,
    apply_condition,
    heldout_is_untouched,
)
from rooted4d.inference import SceneModel, SolverConfig
from rooted4d.inference.problem import make_joint_problem


def _problem():
    model = SceneModel.procedural(n_modes=10, n_motion=3, n_session=2, n_phase_nodes=6)
    config = SolverConfig(fourier_order=2, session_order=1, max_iterations=5)
    return make_joint_problem(model, config, seed=0)


class PilotConditionTest(unittest.TestCase):
    def setUp(self):
        self.observations = _problem().observations

    def test_every_condition_leaves_the_scoring_target_intact(self):
        """A generator that corrupted the held-out partition would improve every
        arm at once and read as a method effect."""

        for condition in CONDITIONS:
            modalities = (None,) if condition == "nominal" else ("cmr", "tee")
            for modality in modalities:
                with self.subTest(condition=condition, modality=modality):
                    after, _ = apply_condition(self.observations, condition, modality)
                    self.assertTrue(heldout_is_untouched(self.observations, after))

    def test_conditions_are_deterministic_under_their_frozen_seed(self):
        for condition in ("localized", "missing", "session_shift"):
            with self.subTest(condition=condition):
                first, _ = apply_condition(self.observations, condition, "tee")
                second, _ = apply_condition(self.observations, condition, "tee")
                np.testing.assert_array_equal(first.tee_points, second.tee_points)
                np.testing.assert_array_equal(first.tee_vertices, second.tee_vertices)

    def test_corruption_actually_changes_the_evidence(self):
        """A condition that silently did nothing would make the pilot pass for
        the wrong reason."""

        for condition in ("localized", "session_shift"):
            with self.subTest(condition=condition):
                after, record = apply_condition(self.observations, condition, "tee")
                self.assertTrue(record["applied"])
                self.assertFalse(
                    np.array_equal(self.observations.tee_points, after.tee_points)
                )

    def test_missing_views_removes_columns_consistently(self):
        after, record = apply_condition(self.observations, "missing", "cmr")
        self.assertTrue(record["applied"])
        self.assertLess(after.cmr_vertices.size, self.observations.cmr_vertices.size)
        self.assertEqual(
            after.cmr_points.shape,
            (len(after.cmr_times), len(after.cmr_vertices), 3),
        )

    def test_only_the_named_modality_is_touched(self):
        after, _ = apply_condition(self.observations, "localized", "cmr")
        np.testing.assert_array_equal(after.tee_points, self.observations.tee_points)


if __name__ == "__main__":
    unittest.main()
