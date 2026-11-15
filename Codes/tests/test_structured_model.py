"""Tests for the train-only interface-conditioned shape-space builder.

The properties checked here are the ones E53's comparison turned out not to
have: that the basis is learned from train alone, that it is orthonormal in the
same physical coordinates the global basis lives in, that the retained
dimension is a numerical rank rather than a column count, and that the scales
handed to the identity prior mean what the prior assumes they mean.
"""

import unittest
from pathlib import Path

import numpy as np

from rooted4d.experiments import structured_model
from rooted4d.experiments.structured_model import fit_structured_benchmark_model

BENCHMARK = Path("data/derived/semisynthetic-real-v3")


@unittest.skipUnless(
    sorted((BENCHMARK / "samples" / "train").glob("*.npz")),
    "benchmark samples are not available in this checkout",
)
class StructuredBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.provenance = fit_structured_benchmark_model(
            BENCHMARK, interface_dim=12, interior_dim=12
        )

    def test_reads_only_the_train_split(self):
        """The defect that made E53b a diagnostic rather than a result."""

        seen = []
        original = structured_model._identity_geometry

        def recording(root, split, *rest):
            seen.append(split)
            return original(root, split, *rest)

        structured_model._identity_geometry = recording
        try:
            fit_structured_benchmark_model(
                BENCHMARK, interface_dim=8, interior_dim=8
            )
        finally:
            structured_model._identity_geometry = original
        self.assertEqual(seen, ["train"])

    def test_retained_dimension_is_a_numerical_rank(self):
        self.assertEqual(self.model.n_modes, self.provenance["numerical_rank"])
        self.assertLessEqual(
            self.provenance["numerical_rank"], self.provenance["columns"]
        )
        self.assertEqual(
            self.provenance["rank_deficiency"],
            self.provenance["columns"] - self.provenance["numerical_rank"],
        )

    def test_basis_is_orthonormal_in_physical_displacement(self):
        """The global arm's basis is; the prior is only comparable if this is."""

        basis = self.model.shape_basis
        np.testing.assert_allclose(
            basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10
        )

    def test_attachment_is_exact_for_arbitrary_coefficients(self):
        rng = np.random.default_rng(20260921)
        for _ in range(8):
            latent = rng.normal(scale=5.0, size=self.model.n_modes)
            self.assertLess(self.model.attachment_residual(latent), 1e-9)

    def test_scales_descend(self):
        """``identity_prior_scale_floor`` is relative to ``shape_scales[0]``.

        A PCA basis supplies a descending spectrum for free and this one does
        not, so the builder imposes the order. If that were dropped, the floor
        would clamp against an arbitrary reference and bind the wrong modes.
        """

        scales = np.asarray(self.model.shape_scales)
        self.assertTrue(np.all(np.diff(scales) <= 1e-12))
        self.assertGreater(scales[0], 0.0)

    def test_scales_are_the_training_coefficient_deviation(self):
        """The same quantity the global arm records, measured not assumed."""

        templates, _, _ = structured_model._identity_geometry(BENCHMARK, "train")
        flat = (templates - templates.mean(axis=0)).reshape(len(templates), -1)
        coefficients = flat @ self.model.shape_basis
        expected = np.sqrt((coefficients**2).sum(axis=0) / (len(templates) - 1))
        np.testing.assert_allclose(self.model.shape_scales, expected, rtol=1e-10)

    def test_reported_variance_matches_a_direct_projection(self):
        templates, _, _ = structured_model._identity_geometry(BENCHMARK, "train")
        flat = (templates - templates.mean(axis=0)).reshape(len(templates), -1)
        basis = self.model.shape_basis
        residual = flat - (flat @ basis) @ basis.T
        expected = 1.0 - (residual**2).sum() / (flat**2).sum()
        self.assertAlmostEqual(
            self.provenance["variance_captured"], float(expected), places=10
        )

    def test_allocation_changes_the_span_not_the_rank_accounting(self):
        other, provenance = fit_structured_benchmark_model(
            BENCHMARK, interface_dim=18, interior_dim=10
        )
        self.assertEqual(other.n_modes, provenance["numerical_rank"])
        self.assertEqual(other.n_modes, self.model.n_modes)
        # Same rank, genuinely different subspace: if these coincided the
        # allocation would not be testing anything.
        overlap = self.model.shape_basis.T @ other.shape_basis
        self.assertLess(np.linalg.norm(overlap, "fro") ** 2, other.n_modes - 1.0)

    def test_rejects_an_empty_span(self):
        with self.assertRaises(ValueError):
            fit_structured_benchmark_model(
                BENCHMARK, interface_dim=0, interior_dim=0
            )

    def test_rejects_unknown_coordinates(self):
        with self.assertRaises(ValueError):
            fit_structured_benchmark_model(
                BENCHMARK, interface_dim=8, interior_dim=8, coordinates="blocks"
            )


def _physical_precision(model, config):
    """The quadratic form the solver's identity prior puts on displacement.

    Built from ``identity_prior_multiplier``, the function the residuals call,
    so this measures the prior the solver applies rather than a re-derivation of
    it. Returned normalised, because the overall strength is the weight's job
    and the question here is the shape of the penalty.
    """

    from rooted4d.inference.joint import identity_prior_multiplier

    basis = model.shape_basis
    multiplier = identity_prior_multiplier(model, config, basis.shape[1])
    weights = np.ones(basis.shape[1]) if multiplier is None else multiplier**2
    precision = (basis * weights) @ basis.T
    return precision / np.linalg.norm(precision)


@unittest.skipUnless(
    sorted((BENCHMARK / "samples" / "train").glob("*.npz")),
    "benchmark samples are not available in this checkout",
)
class CovarianceCoordinatesTest(unittest.TestCase):
    """The structured arm has to get the covariance treatment global PCA gets.

    PCA coordinates diagonalise their own training covariance, so a diagonal
    prior over them is the full empirical-covariance prior. The assembly axes
    do not, so the same diagonal prior over them silently discards the
    cross-covariance. These tests pin the difference down.
    """

    @classmethod
    def setUpClass(cls):
        from rooted4d.inference import SolverConfig

        cls.assembly, cls.assembly_provenance = fit_structured_benchmark_model(
            BENCHMARK, interface_dim=12, interior_dim=12, coordinates="assembly"
        )
        cls.covariance, cls.covariance_provenance = fit_structured_benchmark_model(
            BENCHMARK, interface_dim=12, interior_dim=12, coordinates="covariance"
        )
        templates, _, _ = structured_model._identity_geometry(BENCHMARK, "train")
        cls.train = (templates - templates.mean(axis=0)).reshape(len(templates), -1)
        cls.aware = SolverConfig(covariance_aware_identity_prior=True)
        cls.isotropic = SolverConfig(covariance_aware_identity_prior=False)

    def _covariance(self, basis):
        coefficients = self.train @ basis
        return coefficients.T @ coefficients / (len(self.train) - 1)

    def test_covariance_coordinates_diagonalise_training_covariance(self):
        covariance = self._covariance(self.covariance.shape_basis)
        offdiagonal = covariance - np.diag(np.diag(covariance))
        self.assertLess(np.abs(offdiagonal).max(), 1e-9 * np.abs(covariance).max())
        np.testing.assert_allclose(
            self.covariance.shape_scales, np.sqrt(np.diag(covariance)), rtol=1e-10
        )

    def test_assembly_axes_do_not(self):
        """Recorded so the approximation is never mistaken for the real thing."""

        axes = self.assembly_provenance["assembly_axes"]
        self.assertGreater(axes["max_abs_offdiagonal_correlation"], 0.5)
        self.assertGreater(axes["offdiagonal_covariance_fraction"], 0.1)

    def test_rotation_changes_coordinates_and_nothing_else(self):
        a, c = self.assembly, self.covariance
        self.assertEqual(a.n_modes, c.n_modes)
        np.testing.assert_allclose(a.template_physical, c.template_physical)
        np.testing.assert_allclose(
            a.shape_basis @ a.shape_basis.T, c.shape_basis @ c.shape_basis.T, atol=1e-12
        )
        for name in ("motion", "session"):
            left = a.shape_basis @ getattr(a.bases, name)
            right = c.shape_basis @ getattr(c.bases, name)
            np.testing.assert_allclose(left @ left.T, right @ right.T, atol=1e-12)

    def test_scalar_prior_is_the_same_penalty_in_either_coordinates(self):
        np.testing.assert_allclose(
            _physical_precision(self.assembly, self.isotropic),
            _physical_precision(self.covariance, self.isotropic),
            atol=1e-12,
        )

    def test_solver_prior_in_covariance_coordinates_is_the_full_covariance_prior(self):
        """The review's test: the physical penalty survives an equivalent rotation.

        The full empirical-covariance prior, written in the *assembly* axes with
        its covariance matrix carried along, is ``B Sigma^-1 B^T``. After the
        rotation the solver applies a diagonal prior, and that diagonal prior
        must be the same quadratic form on physical displacement.
        """

        scales = np.asarray(self.covariance.shape_scales)
        # Proportionality only holds where the numerical floor binds nothing.
        self.assertGreater(
            scales[-1] / scales[0], self.aware.identity_prior_scale_floor
        )
        covariance = self._covariance(self.assembly.shape_basis)
        basis = self.assembly.shape_basis
        full = basis @ np.linalg.inv(covariance) @ basis.T
        full = full / np.linalg.norm(full)
        np.testing.assert_allclose(
            _physical_precision(self.covariance, self.aware), full, atol=1e-9
        )

    def test_assembly_diagonal_prior_is_a_different_penalty(self):
        """The E57 estimator's prior is not the covariance prior, by a lot."""

        diagonal = _physical_precision(self.assembly, self.aware)
        full = _physical_precision(self.covariance, self.aware)
        self.assertGreater(np.linalg.norm(diagonal - full), 0.5)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(
    sorted((BENCHMARK / "samples" / "train").glob("*.npz")),
    "benchmark samples are not available in this checkout",
)
class FoldExclusionTest(unittest.TestCase):
    """A cross-validation fold must learn its bases without the scored identities."""

    def test_excluded_identities_are_not_learned_from(self):
        from rooted4d.experiments.benchmark_eval import fit_benchmark_model

        everyone = sorted({str(np.load(p, allow_pickle=True)["identity_id"])
                           for p in (BENCHMARK / "samples" / "train").glob("*.npz")})
        held_out = frozenset(everyone[:18])
        for build in (
            lambda ex: fit_benchmark_model(BENCHMARK, n_modes=24, exclude_identities=ex),
            lambda ex: fit_structured_benchmark_model(
                BENCHMARK, interface_dim=8, interior_dim=8, exclude_identities=ex),
        ):
            model, provenance = build(held_out)
            self.assertEqual(provenance["training_identities"], len(everyone) - 18)
            full, _ = build(None)
            self.assertFalse(np.allclose(model.template_physical, full.template_physical))


class ResolutionTest(unittest.TestCase):
    def test_complex_is_derived_from_the_vertex_count(self):
        from rooted4d.geometry import make_toy_complex, toy_complex_for_vertices

        for n in (24, 48, 96):
            derived = toy_complex_for_vertices(3 * n + 1)
            np.testing.assert_array_equal(derived.global_vertices,
                                          make_toy_complex(n_annulus=n).global_vertices)
        for bad in (72, 74, 36):
            with self.assertRaises(ValueError):
                toy_complex_for_vertices(bad)
