"""Tests for the section 4.2 gauge-fixed decomposition and Theorem 3."""

import unittest

import numpy as np

from rooted4d.priors import GaugedLatentBases, SessionDecomposition


def _bases(latent: int = 12, n_motion: int = 3, n_session: int = 2, seed: int = 11):
    rng = np.random.default_rng(seed)
    motion_samples = rng.normal(size=(40, latent))
    session_samples = rng.normal(size=(40, latent))
    return GaugedLatentBases.from_samples(
        motion_samples,
        session_samples,
        n_motion=n_motion,
        n_session=n_session,
    )


def _gauged_parameters(model, n_phases=16, seed=3):
    """Build a triple that already satisfies both section 4.2 centering gauges.

    Shared motion uses integer Fourier modes, whose cycle mean vanishes exactly on
    a uniform grid.  The session residuals are equal and opposite in the weighted
    sense, so ``sum_m w_m r_m(tau) = 0`` holds at every phase.
    """

    rng = np.random.default_rng(seed)
    identity = rng.normal(size=model.latent_dimension)
    phases = np.arange(n_phases) / n_phases
    motion = np.stack(
        [np.cos(2 * np.pi * (k + 1) * phases) for k in range(model.bases.n_motion)],
        axis=1,
    )
    # A static loading offset plus a dynamic discrepancy, as section 4.2 allows.
    discrepancy = rng.normal(size=model.bases.n_session) + (
        np.sin(2 * np.pi * phases)[:, None] * rng.normal(size=model.bases.n_session)
    )
    weights = model.weights
    session = {
        "cmr": discrepancy,
        "tee": -(weights["cmr"] / weights["tee"]) * discrepancy,
    }
    return identity, motion, session


class GaugedLatentBasesTest(unittest.TestCase):
    def test_learned_bases_satisfy_assumption_a3(self):
        validation = _bases().validate()
        self.assertTrue(validation.valid)
        self.assertLess(validation.block_orthogonality, 1e-12)

    def test_overlapping_blocks_are_rejected(self):
        column = np.eye(4)[:, :1]
        with self.assertRaises(ValueError):
            SessionDecomposition(GaugedLatentBases(motion=column, session=column.copy()))

    def test_rank_deficient_samples_are_reported(self):
        rng = np.random.default_rng(0)
        rank_one = np.outer(rng.normal(size=20), rng.normal(size=8))
        with self.assertRaises(ValueError):
            GaugedLatentBases.from_samples(
                rank_one,
                rng.normal(size=(20, 8)),
                n_motion=3,
                n_session=2,
            )

    def test_blocks_cannot_exceed_the_latent_dimension(self):
        with self.assertRaises(ValueError):
            GaugedLatentBases(motion=np.eye(3), session=np.eye(3)[:, :1])


class SessionDecompositionTest(unittest.TestCase):
    def setUp(self):
        self.model = SessionDecomposition(_bases())

    def test_decomposition_satisfies_both_gauges(self):
        identity, motion, session = _gauged_parameters(self.model)
        result = self.model.decompose(self.model.compose(identity, motion, session))
        residuals = result.gauge_residuals()
        self.assertLess(residuals["motion_cycle_mean"], 1e-12)
        self.assertLess(residuals["session_cross_session_mean"], 1e-12)

    def test_admissible_parameters_are_recovered_exactly(self):
        identity, motion, session = _gauged_parameters(self.model)
        result = self.model.decompose(self.model.compose(identity, motion, session))
        np.testing.assert_allclose(result.identity, identity, atol=1e-10)
        np.testing.assert_allclose(result.motion, motion, atol=1e-10)
        for name in ("cmr", "tee"):
            np.testing.assert_allclose(result.session[name], session[name], atol=1e-10)
        self.assertLess(result.max_unmodelled, 1e-10)

    def test_theorem_3_collapses_both_gauge_ambiguities(self):
        """Reparameterizations that leave every trajectory fixed must not change the result."""

        identity, motion, session = _gauged_parameters(self.model)
        rng = np.random.default_rng(21)
        shifted = self.model.reparameterize(
            identity,
            motion,
            session,
            motion_shift=rng.normal(size=self.model.bases.n_motion),
            session_shift=rng.normal(size=self.model.bases.n_session),
        )

        original = self.model.compose(identity, motion, session)
        moved = self.model.compose(*shifted[:2], shifted[2])
        for name in ("cmr", "tee"):
            np.testing.assert_allclose(original[name], moved[name], atol=1e-10)

        # The shifted parameters really are a different, gauge-violating triple.
        self.assertGreater(float(np.max(np.abs(shifted[1] - motion))), 1e-3)

        canonical = self.model.decompose(original)
        recanonical = self.model.canonicalize(*shifted[:2], shifted[2])
        np.testing.assert_allclose(recanonical.identity, canonical.identity, atol=1e-10)
        np.testing.assert_allclose(recanonical.motion, canonical.motion, atol=1e-10)
        for name in ("cmr", "tee"):
            np.testing.assert_allclose(
                recanonical.session[name], canonical.session[name], atol=1e-10
            )

    def test_session_weighting_moves_the_canonical_identity(self):
        """A static loading offset is split between identity and residual by w_m."""

        identity, motion, session = _gauged_parameters(self.model)
        trajectories = self.model.compose(identity, motion, session)
        weighted = SessionDecomposition(self.model.bases, weights={"cmr": 0.8, "tee": 0.2})
        result = weighted.decompose(trajectories)
        self.assertLess(result.gauge_residuals()["session_cross_session_mean"], 1e-12)
        self.assertGreater(
            float(np.max(np.abs(result.identity - identity))),
            1e-6,
            "a different session weighting must move the canonical identity",
        )

    def test_content_outside_the_subspaces_is_reported_not_absorbed(self):
        identity, motion, session = _gauged_parameters(self.model)
        trajectories = self.model.compose(identity, motion, session)
        rng = np.random.default_rng(5)
        stray = rng.normal(size=self.model.latent_dimension)
        stray -= self.model.bases.motion @ (self.model.bases.motion.T @ stray)
        stray -= self.model.bases.session @ (self.model.bases.session.T @ stray)
        phases = np.arange(trajectories["cmr"].shape[0]) / trajectories["cmr"].shape[0]
        trajectories["cmr"] = trajectories["cmr"] + np.sin(2 * np.pi * phases)[:, None] * stray
        result = self.model.decompose(trajectories)
        self.assertGreater(result.max_unmodelled, 1e-3)

    def test_mismatched_phase_grids_are_rejected(self):
        identity, motion, session = _gauged_parameters(self.model)
        trajectories = self.model.compose(identity, motion, session)
        trajectories["tee"] = trajectories["tee"][:-1]
        with self.assertRaises(ValueError):
            self.model.decompose(trajectories)

    def test_non_positive_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            SessionDecomposition(self.model.bases, weights={"cmr": 1.0, "tee": 0.0})


if __name__ == "__main__":
    unittest.main()
