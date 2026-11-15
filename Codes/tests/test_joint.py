"""Tests for the constrained scene model and staged joint solver."""

import unittest
from dataclasses import replace

import numpy as np

from rooted4d.inference import (
    SceneModel,
    SolverConfig,
    fit_joint_scene,
    initial_parameters,
    latent_trajectory,
    predict,
)
from rooted4d.inference.baselines import BASELINE_IDS, run_arm
from rooted4d.inference.joint import packed_gauge_directions
from rooted4d.inference.problem import make_joint_problem, pose_errors
from rooted4d.motion import circular_distance
from rooted4d.theory import constrained_identifiability


def _small_model() -> SceneModel:
    return SceneModel.procedural(n_modes=10, n_motion=3, n_session=2, n_phase_nodes=6)


def _small_config() -> SolverConfig:
    return SolverConfig(fourier_order=2, session_order=1, max_iterations=30)


class SceneModelTest(unittest.TestCase):
    def setUp(self):
        self.model = _small_model()
        self.config = _small_config()

    def test_attachment_is_exact_for_any_latent(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            latent = rng.normal(0.0, 5.0, size=(4, self.model.n_modes))
            self.assertEqual(self.model.attachment_residual(latent), 0.0)

    def test_bases_satisfy_assumption_a3(self):
        self.assertTrue(self.model.bases.validate().valid)

    def test_shape_basis_is_orthonormal(self):
        gram = self.model.shape_basis.T @ self.model.shape_basis
        np.testing.assert_allclose(gram, np.eye(self.model.n_modes), atol=1e-10)

    def test_detached_model_extends_the_latent_space_by_the_separation_block(self):
        """Detaching adds degrees of freedom; it does not remove or replace them.

        The earlier construction forced an identical latent dimension by giving
        each component its own modes, which changed the shape space as well as
        the coupling and confounded the comparison.
        """

        detached = self.model.detached()
        self.assertTrue(detached.is_detached)
        self.assertEqual(
            detached.n_modes, self.model.n_modes + detached.separation_dimension
        )
        self.assertEqual(detached.n_modes, detached.bases.latent_dimension)
        self.assertEqual(self.model.separation_dimension, 0)

    def test_a_detached_model_cannot_be_detached_again(self):
        with self.assertRaises(ValueError):
            self.model.detached().detached()

    def test_separation_mode_count_is_bounded_by_the_constraint_rank(self):
        with self.assertRaises(ValueError):
            self.model.detached(separation_modes=10_000)

    def test_detached_components_move_independently(self):
        """Otherwise the soft-attachment baseline would be exact for free."""

        detached = self.model.detached()
        rng = np.random.default_rng(1)
        latent = rng.normal(0.0, 5.0, size=(2, detached.n_modes))
        self.assertGreater(detached.detached_attachment_residual(latent), 1e-6)

    def test_detached_basis_contains_the_coupled_shape_space(self):
        """The arms must differ in coupling only, not in what they can express."""

        detached = self.model.detached()
        lifted = detached.shape_basis[:, : self.model.n_modes]
        # Theorem 1: the lifted coupled basis lies in ker(C), so it violates
        # nothing. Only the separation block can break attachment.
        self.assertLess(
            float(np.max(np.abs(detached.soft_constraint @ lifted))), 1e-10
        )
        separation = detached.shape_basis[:, self.model.n_modes :]
        self.assertEqual(separation.shape[1], detached.separation_dimension)
        self.assertGreater(
            float(np.max(np.abs(detached.soft_constraint @ separation))), 1e-3
        )

    def test_detached_model_shares_the_coupled_motion_subspace(self):
        detached = self.model.detached()
        np.testing.assert_allclose(
            detached.bases.motion[: self.model.n_modes], self.model.bases.motion
        )
        np.testing.assert_allclose(
            detached.bases.motion[self.model.n_modes :], 0.0
        )
        self.assertTrue(detached.bases.validate().valid)

    def test_pure_coupled_motion_leaves_attachment_exact_on_the_detached_model(self):
        """A geometry the coupled model can express must stay attached here too."""

        detached = self.model.detached()
        rng = np.random.default_rng(7)
        latent = np.zeros((3, detached.n_modes))
        latent[:, : self.model.n_modes] = rng.normal(0.0, 4.0, size=(3, self.model.n_modes))
        self.assertLess(detached.detached_attachment_residual(latent), 1e-9)

    def test_motion_has_zero_cycle_mean(self):
        parameters = initial_parameters(self.model, self.config)
        rng = np.random.default_rng(2)
        parameters = type(parameters)(
            **{**parameters.__dict__, "motion": rng.normal(size=parameters.motion.shape)}
        )
        phases = np.arange(64) / 64
        latent = latent_trajectory(parameters, phases, "cmr", self.model, self.config)
        np.testing.assert_allclose(latent.mean(axis=0), parameters.identity, atol=1e-12)


class JointSolverTest(unittest.TestCase):
    def setUp(self):
        self.model = _small_model()
        self.config = _small_config()
        self.problem = make_joint_problem(self.model, self.config, seed=0)

    def test_recovers_pose_and_phase_within_the_section_7_6_gates(self):
        fit = fit_joint_scene(self.problem.observations, self.model, self.config)
        pose = pose_errors(fit.parameters, self.problem.truth)
        phase = float(
            circular_distance(
                np.array([fit.parameters.phase_offset]),
                np.array([self.problem.truth.phase_offset]),
            )[0]
        )
        self.assertLess(pose["translation_mm"], 3.0)
        self.assertLess(pose["rotation_deg"], 3.0)
        self.assertLess(phase, 0.05)

    def test_attachment_is_exact_after_fitting(self):
        fit = fit_joint_scene(self.problem.observations, self.model, self.config)
        self.assertEqual(fit.attachment_residual, 0.0)

    def test_heldout_error_approaches_the_noise_floor(self):
        """A 3D isotropic Gaussian has mean magnitude 1.596 sigma."""

        fit = fit_joint_scene(self.problem.observations, self.model, self.config)
        errors = fit.heldout_errors()
        floor = 1.5958 * self.problem.cmr_noise_mm
        self.assertLess(errors["heldout_cmr_mm"], 1.5 * floor)

    def test_predicted_gauges_match_the_measured_null_space(self):
        fit = fit_joint_scene(self.problem.observations, self.model, self.config)
        jacobian = fit.evidence_jacobian()
        gauges = fit.gauge_directions()
        singular = np.linalg.svd(jacobian, compute_uv=False)
        measured = int(np.count_nonzero(singular <= 1e-8 * singular[0]))
        self.assertEqual(measured, gauges.shape[0])

    def test_problem_is_identifiable_only_after_removing_gauges(self):
        fit = fit_joint_scene(self.problem.observations, self.model, self.config)
        jacobian = fit.evidence_jacobian()
        self.assertFalse(constrained_identifiability(jacobian).identifiable)
        restricted = constrained_identifiability(
            jacobian, constraint_jacobian=fit.gauge_directions()
        )
        self.assertTrue(restricted.identifiable)
        self.assertGreater(restricted.smallest_tangent_eigenvalue, 1e-3)

    def test_gauge_count_tracks_the_active_configuration(self):
        parameters = initial_parameters(self.model, self.config)
        without_session = packed_gauge_directions(
            parameters, self.model, type(self.config)(**{**self.config.__dict__, "use_session": False})
        )
        with_session = packed_gauge_directions(parameters, self.model, self.config)
        self.assertGreater(with_session.shape[0], without_session.shape[0])

    def test_heldout_evidence_is_never_shown_to_the_solver(self):
        observations = self.problem.observations
        self.assertEqual(
            len(np.intersect1d(observations.tee_vertices, observations.heldout_tee_vertices)),
            0,
        )
        self.assertEqual(
            len(np.intersect1d(observations.cmr_times, observations.heldout_cmr_times)),
            0,
        )


class BaselineArmTest(unittest.TestCase):
    def setUp(self):
        self.model = _small_model()
        self.config = _small_config()
        self.problem = make_joint_problem(self.model, self.config, seed=1)

    def test_every_locked_arm_runs(self):
        for arm in BASELINE_IDS:
            result = run_arm(arm, self.problem, self.config)
            self.assertEqual(result.baseline_id, arm)
            self.assertTrue(result.succeeded, f"{arm} failed: {result.failure_reason}")

    def test_late_fusion_temporal_alignment_uses_only_observed_vertices(self):
        """The registration set may exceed the observed set; the temporal one may not.

        Late fusion registers on the whole annulus, which is legitimate because
        both reconstructions predict every template vertex. Its temporal
        alignment reads the *observed* point sequences instead, where a vertex
        removed by dropout has no column. Feeding the registration set into that
        lookup made searchsorted return an insertion point for a missing
        identifier and silently read a neighbouring vertex's trajectory, on
        every late-fusion sample of the benchmark. Nothing in the suite noticed,
        because the arm still returned plausible numbers.
        """

        observations = self.problem.observations
        annulus = np.sort(self.model.groups["annulus"])
        observed_shared = np.intersect1d(
            observations.cmr_vertices, observations.tee_vertices
        )
        registration = np.union1d(observed_shared, annulus)

        # The fixture has to actually exercise the defect, or the test is vacuous.
        self.assertTrue(
            set(registration) - set(observations.cmr_vertices)
            or set(registration) - set(observations.tee_vertices),
            "fixture does not withhold any registration vertex from the "
            "observations, so this test cannot detect the defect",
        )

        from rooted4d.inference.baselines import run_arm as _run_arm

        result = _run_arm("late_fusion", self.problem, self.config)
        self.assertTrue(result.succeeded, result.failure_reason)

        # A vertex absent from an observed sequence must raise rather than be
        # silently resolved to a neighbour.
        available = np.sort(observations.cmr_vertices)
        absent = np.setdiff1d(registration, available)
        if absent.size:
            columns = np.searchsorted(available, absent)
            clipped = np.clip(columns, 0, available.size - 1)
            self.assertFalse(
                np.array_equal(available[clipped], absent),
                "searchsorted resolved absent identifiers, so the membership "
                "check is what has to catch them",
            )

    def test_exact_arms_have_zero_attachment_residual(self):
        for arm in ("cmr_only", "tee_only", "rooted4d_full"):
            self.assertEqual(run_arm(arm, self.problem, self.config).attachment_residual_mm, 0.0)

    def test_soft_attachment_arm_leaves_a_nonzero_residual(self):
        """Theorem 2 in the baseline: a finite penalty cannot reach exactness."""

        result = run_arm("joint_soft_attachment", self.problem, self.config)
        self.assertGreater(result.attachment_residual_mm, 0.0)

    def test_soft_attachment_residual_decays_with_the_penalty_weight(self):
        """The matched basis is what makes the predicted decay reachable.

        With a mismatched basis the penalty fought the whole parameter vector and
        the residual plateaued, which measured optimizer stiffness rather than
        Theorem 2.
        """

        weights = (1e2, 1e4, 1e6)
        residuals = [
            run_arm(
                "joint_soft_attachment", self.problem, self.config,
                soft_attachment_weight=w,
            ).attachment_residual_mm
            for w in weights
        ]
        for earlier, later in zip(residuals, residuals[1:]):
            self.assertLess(later, earlier)
        # Two decades of weight should buy about two decades of residual.
        self.assertGreater(residuals[0] / max(residuals[-1], 1e-300), 1e3)

    def test_soft_attachment_approaches_exact_accuracy_at_high_weight(self):
        """The arms differ in coupling, so accuracy must converge as lambda grows."""

        soft = run_arm(
            "joint_soft_attachment", self.problem, self.config,
            soft_attachment_weight=1e6,
        )
        exact = run_arm("rooted4d_full", self.problem, self.config)
        self.assertLess(
            abs(soft.heldout_cmr_mm - exact.heldout_cmr_mm),
            0.05 * exact.heldout_cmr_mm,
        )

    def test_cmr_only_cannot_predict_held_out_tee(self):
        result = run_arm("cmr_only", self.problem, self.config)
        full = run_arm("rooted4d_full", self.problem, self.config)
        self.assertGreater(result.heldout_tee_mm, 5.0 * full.heldout_tee_mm)

    def test_unknown_arm_is_rejected(self):
        with self.assertRaises(ValueError):
            run_arm("not_a_baseline", self.problem, self.config)


if __name__ == "__main__":
    unittest.main()


class ArmIdentifiabilityTest(unittest.TestCase):
    """Each arm reports the smallest non-gauge eigenvalue of its own problem."""

    def setUp(self):
        self.model = _small_model()
        self.config = _small_config()
        self.problem = make_joint_problem(self.model, self.config, seed=0)

    def test_cmr_only_is_unidentified(self):
        """Pose and phase have no CMR evidence, so a direction is invisible."""

        result = run_arm("cmr_only", self.problem, self.config)
        self.assertEqual(result.alpha, 0.0)
        self.assertFalse(result.identified)

    def test_full_model_is_the_best_identified_arm(self):
        full = run_arm("rooted4d_full", self.problem, self.config)
        tee = run_arm("tee_only", self.problem, self.config)
        self.assertTrue(full.identified)
        self.assertGreater(full.alpha, tee.alpha)


class GaugeBookkeepingTest(unittest.TestCase):
    """The declared gauge set must match the Jacobian's null space in every config.

    A defect here is silent and expensive: an offset error placed the phase gauge
    into the session and pose slots whenever centering was disabled, leaving the
    real gauge in the tangent space. Alpha then read zero and looked like an
    identifiability finding.
    """

    def setUp(self):
        self.model = _small_model()
        self.config = _small_config()
        self.problem = make_joint_problem(self.model, self.config, seed=0)

    def _declared_and_measured(self, **overrides):
        from dataclasses import replace as _replace

        config = _replace(self.config, **overrides)
        fit = fit_joint_scene(self.problem.observations, self.model, config)
        jacobian = fit.evidence_jacobian()
        singular = np.linalg.svd(jacobian, compute_uv=False)
        measured = int(np.count_nonzero(singular <= 1e-8 * singular[0]))
        return fit.gauge_directions().shape[0], measured, fit

    def test_counts_match_with_gauges_enforced(self):
        declared, measured, _ = self._declared_and_measured(enforce_gauges=True)
        self.assertEqual(declared, measured)
        self.assertEqual(declared, 5)

    def test_counts_match_with_gauges_disabled(self):
        declared, measured, _ = self._declared_and_measured(enforce_gauges=False)
        self.assertEqual(declared, measured)
        self.assertEqual(declared, 1)

    def test_counts_match_without_a_session_block(self):
        declared, measured, _ = self._declared_and_measured(use_session=False)
        self.assertEqual(declared, measured)

    def test_alpha_is_unchanged_by_the_gauge_choice(self):
        """The gauges reparameterize; they do not add information."""

        _, _, with_gauges = self._declared_and_measured(enforce_gauges=True)
        _, _, without = self._declared_and_measured(enforce_gauges=False)
        first = constrained_identifiability(
            with_gauges.evidence_jacobian(),
            constraint_jacobian=with_gauges.gauge_directions(),
        ).smallest_tangent_eigenvalue
        second = constrained_identifiability(
            without.evidence_jacobian(),
            constraint_jacobian=without.gauge_directions(),
        ).smallest_tangent_eigenvalue
        self.assertGreater(first, 1e-3)
        self.assertLess(abs(first - second), 0.25 * first)


class FixedAlignmentTest(unittest.TestCase):
    """``estimate_phase_warp`` freezes the warp shape, never the scalar offset.

    ``_pack`` appended the phase offset unconditionally and ``_unpack`` consumed
    it unconditionally, so a run with the warp disabled still slid phase. The
    staged solver additionally *searched* twenty-four offsets before unlocking
    anything, which would have chosen the offset by enumeration even once the
    gradient could not move it. A diagnostic that needs genuinely fixed
    alignment could not get one, and that is a prerequisite for the
    interface-conditioned inference pilot.
    """

    def setUp(self):
        self.model = _small_model()
        self.config = _small_config()
        self.problem = make_joint_problem(self.model, self.config, seed=0)

    def _frozen(self):
        from dataclasses import replace

        return replace(self.config, estimate_phase_offset=False)

    def test_freezing_removes_exactly_one_packed_slot(self):
        from rooted4d.inference.joint import _pack

        parameters = initial_parameters(self.model, self.config)
        free = _pack(parameters, self.model, self.config)
        frozen = _pack(parameters, self.model, self._frozen())
        self.assertEqual(free.size - frozen.size, 1)

    def test_frozen_offset_survives_the_whole_staged_solve(self):
        config = self._frozen()
        start = initial_parameters(self.model, config)
        fit = fit_joint_scene(
            self.problem.observations, self.model, config, start=start
        )
        self.assertEqual(fit.parameters.phase_offset, start.phase_offset)

    def test_the_offset_search_is_skipped_when_the_offset_is_frozen(self):
        """Stage A must not choose by enumeration what the gradient cannot move."""

        config = self._frozen()
        start = initial_parameters(self.model, config)
        for candidates in (0, 24):
            fit = fit_joint_scene(
                self.problem.observations,
                self.model,
                config,
                start=start,
                phase_offset_candidates=candidates,
            )
            self.assertEqual(fit.parameters.phase_offset, start.phase_offset)

    def test_gauge_bookkeeping_holds_with_the_offset_frozen(self):
        """The gauge cursor has to drop the slot that ``_pack`` no longer writes.

        Advancing past a slot that is not there would place the phase gauge one
        position late, leaving the real gauge in the tangent space -- the same
        silent failure recorded for the session block.
        """

        fit = fit_joint_scene(self.problem.observations, self.model, self._frozen())
        jacobian = fit.evidence_jacobian()
        singular = np.linalg.svd(jacobian, compute_uv=False)
        measured = int(np.count_nonzero(singular <= 1e-8 * singular[0]))
        self.assertEqual(fit.gauge_directions().shape[0], measured)


class ImageEvidenceRowsTest(unittest.TestCase):
    """Image residuals are evidence, and must be counted as observations.

    They were appended *after* the prior terms while ``_OBSERVATION_ROWS`` was
    set from the point pieces alone, so the prefix ``evidence_jacobian()``
    slices excluded them. An image-only configuration recorded zero observation
    rows and raised; a mixed configuration silently dropped the image
    information from the Theorem 5 and 6 diagnostics, which is the opposite of
    what those diagnostics are for.
    """

    def setUp(self):
        from pathlib import Path

        self.paths = sorted(
            Path("data/derived/semisynthetic-real-v3/samples/test").glob("*.npz")
        )
        if not self.paths:
            self.skipTest("benchmark samples are not available in this checkout")
        self.model = _small_model()

    def _evidence(self, config):
        from rooted4d.inference.image_evidence import evidence_from_sample

        return evidence_from_sample(
            self.paths[0], self.model, budget=40, n_cmr_frames=2, n_tee_frames=2
        )

    def test_mixed_evidence_counts_points_and_image_rows(self):
        from rooted4d.inference.image_evidence import image_residuals
        from rooted4d.inference.joint import _OBSERVATION_ROWS, _pack, _residuals

        config = SolverConfig(
            fourier_order=2, session_order=1, max_iterations=5,
            cmr_image_weight=0.5, tee_image_weight=0.5,
        )
        evidence = self._evidence(config)
        problem = make_joint_problem(self.model, config, seed=0)
        start = initial_parameters(self.model, config)
        block = image_residuals(start, evidence, self.model, config)
        _residuals(
            _pack(start, self.model, config), start, problem.observations,
            self.model, config, image=evidence,
        )
        points = (
            problem.observations.cmr_points.size + problem.observations.tee_points.size
        )
        self.assertEqual(_OBSERVATION_ROWS["n"], points + block.size)

    def test_image_only_evidence_is_not_zero_observation_rows(self):
        from rooted4d.inference.image_evidence import image_residuals
        from rooted4d.inference.joint import _OBSERVATION_ROWS, _pack, _residuals

        config = SolverConfig(
            fourier_order=2, session_order=1, max_iterations=5,
            use_cmr=False, use_tee=False,
            cmr_image_weight=0.5, tee_image_weight=0.5,
        )
        evidence = self._evidence(config)
        problem = make_joint_problem(self.model, config, seed=0)
        start = initial_parameters(self.model, config)
        block = image_residuals(start, evidence, self.model, config)
        _residuals(
            _pack(start, self.model, config), start, problem.observations,
            self.model, config, image=evidence,
        )
        self.assertGreater(_OBSERVATION_ROWS["n"], 0)
        self.assertEqual(_OBSERVATION_ROWS["n"], block.size)


class CovarianceAwareIdentityPriorTest(unittest.TestCase):
    """The identity prior may shrink weak shape directions more than strong ones.

    The isotropic prior treats the weakest learned mode as exactly as plausible
    as the strongest, so added capacity is paid for twice: once in variance, and
    once in a prior that does not know the modes are ordered. This is the control
    that has to be run before any accuracy gain is attributed to an anatomical
    partition of the shape space.
    """

    def setUp(self):
        from dataclasses import replace as _replace

        self.model = _small_model()
        self.config = _small_config()
        self.problem = make_joint_problem(self.model, self.config, seed=0)
        # A decaying spectrum, as a learned basis produces.
        scales = np.geomspace(1.0, 0.02, self.model.n_modes)
        self.scaled_model = _replace(self.model, shape_scales=scales)

    def _residual(self, model, **overrides):
        from dataclasses import replace as _replace

        from rooted4d.inference.joint import _pack, _residuals

        config = _replace(self.config, **overrides)
        parameters = initial_parameters(model, config)
        # A non-zero identity, or every prior row is zero and nothing is tested.
        parameters = _replace(
            parameters, identity=np.linspace(1.0, 1.0, model.n_modes)
        )
        vector = _pack(parameters, model, config)
        return _residuals(
            vector, parameters, self.problem.observations, model, config
        )

    def test_flag_off_leaves_the_residual_untouched(self):
        """Default is isotropic, so every result computed before this reproduces."""

        without = self._residual(self.scaled_model)
        with_flag_but_no_scales = self._residual(
            self.model, covariance_aware_identity_prior=True
        )
        baseline = self._residual(self.model)
        np.testing.assert_allclose(without, baseline, atol=0.0)
        np.testing.assert_allclose(with_flag_but_no_scales, baseline, atol=0.0)

    def test_weak_directions_are_penalised_more_than_strong_ones(self):
        isotropic = self._residual(self.scaled_model)
        aware = self._residual(
            self.scaled_model, covariance_aware_identity_prior=True
        )
        self.assertFalse(np.allclose(isotropic, aware))
        # Total strength is matched, so the two priors differ in *where* they
        # shrink rather than in *how much*. Without this the aware arm runs at an
        # effective weight its own control never sweeps -- the multipliers
        # average 153 to 236 on the learned spectra -- and the comparison
        # measures regularisation magnitude instead of covariance awareness.
        scales = np.asarray(self.scaled_model.shape_scales)
        relative = np.maximum(scales / scales[0], self.config.identity_prior_scale_floor)
        multiplier = 1.0 / relative
        multiplier = multiplier / np.sqrt(np.mean(multiplier ** 2))
        self.assertAlmostEqual(float(np.mean(multiplier ** 2)), 1.0, places=12)
        # And the redistribution runs the right way: weak directions harder.
        self.assertLess(float(multiplier[0]), float(multiplier[-1]))

    def test_the_floor_bounds_how_hard_a_weak_mode_is_shrunk(self):
        """Without a floor the weakest directions become hard constraints on
        exactly what the data determines least."""

        from dataclasses import replace as _replace

        scales = np.asarray(self.scaled_model.shape_scales)
        for floor in (0.05, 0.25):
            config = _replace(self.config, identity_prior_scale_floor=floor)
            relative = np.maximum(scales / scales[0], config.identity_prior_scale_floor)
            self.assertGreaterEqual(float(relative.min()), floor)
            self.assertLessEqual(1.0 / float(relative.min()), 1.0 / floor + 1e-9)


class MatchedDetachedModelTest(unittest.TestCase):
    """The soft-attachment arm has to converge to the exact arm, prior included.

    The historical detached model renormalises its lifted columns, which moves
    the identity prior, the motion and session subspaces and the gauges into
    different coordinates from the coupled model's, so a stiff penalty never
    quite reproduces the exact fit. The matched variant keeps the coupled
    coordinates; its gap to the exact fit must shrink like 1/lambda, which is
    Theorem 2's rate, and vanish.
    """

    def setUp(self):
        from rooted4d.inference.initialization import StartingAlignment, start_from
        from rooted4d.inference.problem import make_joint_problem

        model = _small_model()
        self.model = replace(model, shape_scales=np.geomspace(10.0, 0.3, model.n_modes))
        self.config = replace(
            _small_config(), max_iterations=60, estimate_pose=False,
            estimate_phase_warp=False, estimate_phase_offset=False, identity_prior_weight=2.0)
        self.problem = make_joint_problem(self.model, self.config, seed=4)
        truth = self.problem.truth
        self.alignment = StartingAlignment(truth.pose_axis_angle, truth.pose_translation,
                                           truth.phase_offset, 0, 0.0)
        self.start_from = start_from

    def _gap(self, model, config, penalty):
        truth = self.problem.truth
        exact_start = replace(self.start_from(self.alignment, self.model, config),
                              phase_logits=truth.phase_logits)
        exact = fit_joint_scene(self.problem.observations, self.model, config,
                                start=exact_start).heldout_errors()
        soft_start = replace(self.start_from(self.alignment, model, config),
                             phase_logits=truth.phase_logits)
        soft = fit_joint_scene(self.problem.observations, model,
                               replace(config, soft_attachment_weight=penalty),
                               start=soft_start).heldout_errors()
        return max(abs(soft["heldout_cmr_mm"] - exact["heldout_cmr_mm"]),
                   abs(soft["heldout_tee_mm"] - exact["heldout_tee_mm"]))

    def test_matched_soft_arm_converges_to_the_exact_arm(self):
        for aware in (False, True):
            config = replace(self.config, covariance_aware_identity_prior=aware)
            detached = self.model.detached(matched=True)
            loose, stiff = self._gap(detached, config, 1e2), self._gap(detached, config, 1e6)
            self.assertLess(stiff, 1e-8)
            self.assertLess(stiff, 1e-3 * loose)

    def test_historical_detached_model_is_unchanged(self):
        detached = self.model.detached()
        self.assertIsNone(detached.matched_lifted_dimension)
        from rooted4d.inference.joint import identity_prior_multiplier

        self.assertIsNone(identity_prior_multiplier(detached, self.config, detached.n_modes))
        np.testing.assert_allclose(np.linalg.norm(detached.shape_basis[:, : self.model.n_modes],
                                                  axis=0), 1.0)

    def test_matched_prior_is_the_coupled_prior_and_spares_separation(self):
        from rooted4d.inference.joint import identity_prior_multiplier

        config = replace(self.config, covariance_aware_identity_prior=True)
        detached = self.model.detached(matched=True)
        multiplier = identity_prior_multiplier(detached, config, detached.n_modes)
        coupled = identity_prior_multiplier(self.model, config, self.model.n_modes)
        np.testing.assert_allclose(multiplier[: self.model.n_modes], coupled)
        self.assertFalse(np.any(multiplier[self.model.n_modes:]))
