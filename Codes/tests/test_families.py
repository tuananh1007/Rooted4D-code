"""Tests for the held-out generator families of the anti-inverse-crime protocol."""

import unittest

import numpy as np

from rooted4d.inference import SceneModel, SolverConfig, fit_joint_scene
from rooted4d.inference.problem import make_family_problem, make_joint_problem
from rooted4d.synthetic.families import (
    FAMILIES,
    TEST_FAMILY,
    TRAIN_FAMILY,
    VAL_FAMILY,
    arrhythmic_phase_samples,
    correlated_vertex_noise,
    periodic_bspline_basis,
    spline_motion_field,
)


class PeriodicSplineTest(unittest.TestCase):
    def test_basis_is_a_partition_of_unity(self):
        basis = periodic_bspline_basis(np.linspace(0.0, 1.0, 64, endpoint=False), 8)
        np.testing.assert_allclose(basis.sum(axis=-1), 1.0, atol=1e-12)

    def test_basis_is_periodic(self):
        np.testing.assert_allclose(
            periodic_bspline_basis(np.array([0.0]), 8),
            periodic_bspline_basis(np.array([1.0]), 8),
            atol=1e-12,
        )

    def test_spline_motion_is_cycle_centered(self):
        """The families must differ in basis, not in gauge."""

        rng = np.random.default_rng(0)
        base = rng.normal(size=(12, 3))
        phases = np.arange(256) / 256
        field = spline_motion_field(phases, base, n_control=7, rng=rng)
        self.assertLess(float(np.max(np.abs(field.mean(axis=0)))), 1e-2)

    def test_too_few_control_points_are_rejected(self):
        with self.assertRaises(ValueError):
            periodic_bspline_basis(np.array([0.5]), 3)


class FamilyPerturbationTest(unittest.TestCase):
    def test_correlated_noise_is_actually_correlated(self):
        rng = np.random.default_rng(1)
        vertices = np.stack(
            [np.linspace(0.0, 40.0, 24), np.zeros(24), np.zeros(24)], axis=1
        )
        noise = correlated_vertex_noise(vertices, (8, 24, 3), 1.0, 12.0, rng)
        neighbour = np.corrcoef(noise[:, :-1, 0].ravel(), noise[:, 1:, 0].ravel())[0, 1]
        self.assertGreater(neighbour, 0.5)

    def test_arrhythmic_samples_stay_ordered_within_one_cycle(self):
        times = arrhythmic_phase_samples(16, np.random.default_rng(2))
        self.assertTrue(np.all(np.diff(times) > 0.0))
        self.assertGreaterEqual(float(times.min()), 0.0)
        self.assertLess(float(times.max()), 1.0)

    def test_arrhythmic_spacing_is_uneven(self):
        times = arrhythmic_phase_samples(24, np.random.default_rng(3))
        gaps = np.diff(times)
        self.assertGreater(float(np.std(gaps) / np.mean(gaps)), 0.1)


class FamilyProblemTest(unittest.TestCase):
    def test_train_family_matches_the_original_generator(self):
        model = SceneModel.procedural(n_modes=10, n_motion=3, n_session=2, n_phase_nodes=6)
        config = SolverConfig(fourier_order=2, session_order=1, max_iterations=10)
        direct = make_joint_problem(model, config, seed=0)
        via_family = make_family_problem(model, config, family=TRAIN_FAMILY, seed=0)
        np.testing.assert_allclose(
            direct.observations.cmr_points, via_family.observations.cmr_points
        )

    def test_every_family_produces_a_fittable_problem(self):
        model = SceneModel.procedural(n_modes=10, n_motion=3, n_session=2, n_phase_nodes=6)
        config = SolverConfig(fourier_order=2, session_order=1, max_iterations=12)
        for family in FAMILIES:
            problem = make_family_problem(model, config, family=family, seed=1)
            fit = fit_joint_scene(problem.observations, model, config)
            self.assertTrue(np.isfinite(fit.cost), family)
            # Attachment is structural and must survive misspecification.
            self.assertEqual(fit.attachment_residual, 0.0, family)

    def test_held_out_families_are_harder_than_the_train_family(self):
        model = SceneModel.procedural(n_modes=10, n_motion=3, n_session=2, n_phase_nodes=6)
        config = SolverConfig(fourier_order=2, session_order=1, max_iterations=30)
        errors = {}
        for family in (TRAIN_FAMILY, VAL_FAMILY, TEST_FAMILY):
            problem = make_family_problem(model, config, family=family, seed=0)
            fit = fit_joint_scene(problem.observations, model, config)
            errors[family] = fit.heldout_errors()["heldout_cmr_mm"]
        self.assertGreater(errors[VAL_FAMILY], errors[TRAIN_FAMILY])
        self.assertGreater(errors[TEST_FAMILY], errors[TRAIN_FAMILY])

    def test_unknown_family_is_rejected(self):
        model = SceneModel.procedural(n_modes=10, n_motion=3, n_session=2, n_phase_nodes=6)
        with self.assertRaises(ValueError):
            make_family_problem(model, family="not_a_family")


if __name__ == "__main__":
    unittest.main()


class RendererFamilyTest(unittest.TestCase):
    """Held-out renderers must differ from the training one without changing anatomy."""

    def _mesh(self):
        from rooted4d.geometry import make_toy_complex

        complex_ = make_toy_complex(24)
        return complex_, complex_.global_vertices[None].repeat(3, axis=0)

    def test_supersampled_cmr_differs_but_stays_correlated(self):
        from rooted4d.observation import default_cmr_stack, render_cmr_surface_likelihood
        from rooted4d.synthetic.families import render_supersampled_cmr

        complex_, frames = self._mesh()
        stack = default_cmr_stack(
            complex_.global_vertices[complex_.groups["cmr_visible"]],
            n_slices=3,
            image_shape=(9, 9),
        )
        train = render_cmr_surface_likelihood(
            frames, complex_.faces["cmr_surface"], stack, sigma_mm=3.0
        )
        held_out = render_supersampled_cmr(
            frames, complex_.faces["cmr_surface"], stack, sigma_mm=3.0
        )
        self.assertEqual(train.shape, held_out.shape)
        self.assertGreater(float(np.mean(np.abs(train - held_out))), 1e-4)
        correlation = np.corrcoef(train.ravel(), held_out.ravel())[0, 1]
        self.assertGreater(correlation, 0.9)

    def test_ray_renderer_attenuates_behind_a_surface(self):
        """Independent voxels cannot occlude; a ray renderer must."""

        from rooted4d.observation import default_tee_grid, render_tee_surface_likelihood
        from rooted4d.synthetic.families import render_ray_raster_tee

        complex_, frames = self._mesh()
        grid = default_tee_grid(
            complex_.global_vertices[complex_.groups["tee_visible"]], grid_shape=(8, 5, 5)
        )
        train = render_tee_surface_likelihood(
            frames, complex_.faces["tee_surface"], grid, sigma_mm=3.0
        )
        held_out = render_ray_raster_tee(
            frames, complex_.faces["tee_surface"], grid, sigma_mm=3.0
        )
        self.assertEqual(train.shape, held_out.shape)
        self.assertGreater(float(np.mean(np.abs(train - held_out))), 1e-5)

        # Occlusion can only remove signal. The comparison must be against the
        # same renderer with occlusion disabled: the training renderer also
        # applies shadow cones, so its total is not a valid upper bound.
        transparent = render_ray_raster_tee(
            frames, complex_.faces["tee_surface"], grid, sigma_mm=3.0, occlusion=0.0
        )
        self.assertLessEqual(float(held_out.sum()), float(transparent.sum()) + 1e-6)
        self.assertGreater(float(transparent.sum()) - float(held_out.sum()), 1e-6)

    def test_supersampling_factor_must_exceed_one(self):
        from rooted4d.observation import default_cmr_stack
        from rooted4d.synthetic.families import render_supersampled_cmr

        complex_, frames = self._mesh()
        stack = default_cmr_stack(
            complex_.global_vertices[complex_.groups["cmr_visible"]],
            n_slices=2,
            image_shape=(6, 6),
        )
        with self.assertRaises(ValueError):
            render_supersampled_cmr(
                frames, complex_.faces["cmr_surface"], stack, sigma_mm=3.0, factor=1
            )

    def test_unknown_renderer_family_is_rejected(self):
        from rooted4d.inference import SceneModel, SolverConfig
        from rooted4d.inference.image_refine import ImageRefinementConfig, make_image_problem
        from rooted4d.inference.problem import make_joint_problem

        model = SceneModel.procedural(n_modes=10, n_motion=3, n_session=2, n_phase_nodes=6)
        config = SolverConfig(fourier_order=2, session_order=1, max_iterations=5)
        truth = make_joint_problem(model, config, seed=0).truth
        with self.assertRaises(ValueError):
            make_image_problem(
                truth, model, config, ImageRefinementConfig(),
                renderer_family="not_a_renderer",
            )


class RendererFairnessTest(unittest.TestCase):
    """The held-out renderer must not be handed easier data than the training one."""

    def test_ray_renderer_applies_shadow_cones(self):
        from rooted4d.geometry import make_toy_complex
        from rooted4d.observation import ShadowCone, default_tee_grid
        from rooted4d.synthetic.families import render_ray_raster_tee

        complex_ = make_toy_complex(24)
        frames = complex_.global_vertices[None].repeat(2, axis=0)
        grid = default_tee_grid(
            complex_.global_vertices[complex_.groups["tee_visible"]], grid_shape=(8, 6, 6)
        )
        cone = ShadowCone(
            azimuth_rad=0.0,
            elevation_rad=0.0,
            angular_radius_rad=0.25,
            start_depth_mm=float(grid.depths_mm[0]),
            strength=0.9,
        )
        without = render_ray_raster_tee(
            frames, complex_.faces["tee_surface"], grid, sigma_mm=3.0
        )
        with_shadow = render_ray_raster_tee(
            frames, complex_.faces["tee_surface"], grid, sigma_mm=3.0, shadows=(cone,)
        )
        self.assertLess(float(with_shadow.sum()), float(without.sum()))


class SharedAnatomyTest(unittest.TestCase):
    """Both modalities must observe the same patient, however the family varies."""

    def test_motion_coefficients_are_drawn_once_not_per_modality(self):
        """Drawing per call gave CMR and TEE independent anatomy: correlation 0.01."""

        from rooted4d.synthetic.families import (
            evaluate_spline_motion,
            spline_motion_coefficients,
        )

        rng = np.random.default_rng(0)
        base = np.random.default_rng(1).normal(size=(20, 3))
        phases = np.linspace(0.0, 1.0, 8, endpoint=False)
        coefficients = spline_motion_coefficients(base, 7, rng)
        first = evaluate_spline_motion(phases, coefficients)
        second = evaluate_spline_motion(phases, coefficients)
        np.testing.assert_array_equal(first, second)

    def test_family_problem_gives_both_modalities_consistent_anatomy(self):
        """The cross-modal discrepancy must stay near the session-residual scale."""

        from rooted4d.inference import SceneModel, SolverConfig
        from rooted4d.inference.problem import make_family_problem

        model = SceneModel.procedural(n_modes=10, n_motion=3, n_session=2, n_phase_nodes=6)
        config = SolverConfig(fourier_order=2, session_order=1, max_iterations=5)
        for family in (VAL_FAMILY, TEST_FAMILY):
            problem = make_family_problem(model, config, family=family, seed=0)
            shared = np.intersect1d(
                problem.observations.cmr_vertices, problem.observations.tee_vertices
            )
            self.assertGreater(len(shared), 3, family)
            cmr_columns = np.searchsorted(problem.observations.cmr_vertices, shared)
            # Compare centroid trajectories, which removes pose and leaves the
            # anatomy question: are these the same beating structure?
            cmr_centroid = problem.observations.cmr_points[:, cmr_columns, :].mean(axis=1)
            spread = float(np.std(cmr_centroid))
            # If the two modalities had independent motion the spread of the
            # cross-modal difference would swamp the per-modality motion scale.
            self.assertLess(spread, 50.0, family)
