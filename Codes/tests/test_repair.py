"""Tests for independent fitting followed by interface repair.

The two properties the comparison depends on: exact repair really does leave
the two reconstructions sharing one ring over the whole cycle, and what it
leaves free is decided by the representation's ring map -- nothing for global
PCA at rank 48, the interiors for the structured basis.
"""

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from rooted4d.inference import SceneModel, SolverConfig
from rooted4d.inference.initialization import input_derived_alignment
from rooted4d.inference.problem import make_joint_problem
from rooted4d.inference.repair import (
    RING_PHASES,
    build_consensus,
    exact_repair,
    independent_fits,
    penalty_repair,
    register_rings,
    ring_map,
    ring_trajectory,
)

BENCHMARK = Path("data/derived/semisynthetic-real-v3")


class RepairTest(unittest.TestCase):
    def setUp(self):
        self.model = SceneModel.procedural(n_modes=10, n_motion=3, n_session=2, n_phase_nodes=6)
        self.config = SolverConfig(fourier_order=2, session_order=1, max_iterations=30)
        self.problem = make_joint_problem(self.model, self.config, seed=5)
        self.alignment = input_derived_alignment(self.problem.observations)

    def test_ring_map_is_exact(self):
        fit = independent_fits(self.problem.observations, self.model, self.config, self.alignment)[0]
        offset, matrix = ring_map(self.model, fit.config, fit.parameters)
        shape = np.concatenate([fit.parameters.identity, fit.parameters.motion.reshape(-1)])
        np.testing.assert_allclose(offset + matrix @ shape,
                                   ring_trajectory(fit.parameters, self.model, fit.config).reshape(-1),
                                   atol=1e-9)

    def test_a_ring_registers_to_itself_with_no_shift_and_no_motion(self):
        fit = independent_fits(self.problem.observations, self.model, self.config, self.alignment)[0]
        delta, transform, _, _, residual = register_rings(fit.parameters, fit.parameters,
                                                          self.model, fit.config)
        self.assertLess(min(delta, 1.0 - delta), 1e-6)
        np.testing.assert_allclose(transform.rotation, np.eye(3), atol=1e-6)
        self.assertLess(residual, 1e-10)

    def test_consensus_is_representable(self):
        cmr, tee, cmr_config, _ = independent_fits(
            self.problem.observations, self.model, self.config, self.alignment)
        consensus = build_consensus(cmr, tee, self.model, cmr_config)
        np.testing.assert_allclose(consensus.offset + consensus.matrix @ consensus.shape,
                                   consensus.target, atol=1e-9)
        if consensus.null_space.size:
            np.testing.assert_allclose(consensus.matrix @ consensus.null_space, 0.0, atol=1e-9)

    def test_exact_repair_shares_one_ring_over_the_whole_cycle(self):
        result = exact_repair(self.problem.observations, self.model, self.config,
                              alignment=self.alignment)
        self.assertLess(result["interface_gap_mm"], 1e-8)
        # Enforced at eight phases, it must hold between them as well.
        phases = np.linspace(0.0, 1.0, 101)
        gap = np.abs(ring_trajectory(result["cmr"].parameters, self.model, result["cmr"].config, phases)
                     - ring_trajectory(result["tee"].parameters, self.model, result["tee"].config, phases))
        self.assertLess(gap.max(), 1e-8)

    def test_penalty_repair_moves_monotonically_toward_the_exact_constraint(self):
        """A stiffer penalty pins the ring harder and costs the data more.

        It need not reach the consensus at any practical weight: directions that
        barely move the ring are barely restrained by a ring penalty, which is the
        conditioning limit of Eq. (soft-drift). So the test is monotonicity,
        bounded by the exactly constrained optimum, not convergence to it.
        """

        from rooted4d.inference.joint import _pack, _residuals

        def data_cost(result):
            fit = result["fit"]
            values = _residuals(_pack(fit.parameters, self.model, self.config),
                                fit.parameters, self.problem.observations, self.model, self.config)
            return float(values @ values)

        results = [penalty_repair(self.problem.observations, self.model, self.config,
                                  alignment=self.alignment, ring_weight=w) for w in (1e1, 1e4, 1e8)]
        deviations = [r["ring_deviation_mm"] for r in results]
        costs = [data_cost(r) for r in results]
        self.assertTrue(deviations[0] > deviations[1] > deviations[2], deviations)
        self.assertTrue(costs[0] <= costs[1] <= costs[2], costs)

    def test_the_repair_arms_carry_their_diagnostics_into_the_record(self):
        """E62's D5 reports each repair's registration, projection and freedom."""

        from rooted4d.inference.baselines import run_arm

        exact = run_arm("independent_exact_repair", self.problem, self.config,
                        alignment=self.alignment).as_dict()
        penalty = run_arm("independent_penalty_repair", self.problem, self.config,
                          alignment=self.alignment, options={"ring_weight": 1e4}).as_dict()
        for record in (exact, penalty):
            self.assertGreaterEqual(record["repair_registration_rms_mm"], 0.0)
            self.assertGreaterEqual(record["repair_projection_rms_mm"], 0.0)
        fit = independent_fits(self.problem.observations, self.model, self.config, self.alignment)[0]
        _, matrix = ring_map(self.model, fit.config, fit.parameters)
        tolerance = 1e-10 * np.linalg.svd(matrix, compute_uv=False)[0]
        rank = np.linalg.matrix_rank(matrix[:, : self.model.n_modes], tol=tolerance)
        self.assertEqual(exact["repair_free_identity_directions"], self.model.n_modes - rank)
        self.assertGreaterEqual(exact["repair_free_shape_directions"],
                                exact["repair_free_identity_directions"])
        self.assertIsNone(penalty["repair_free_identity_directions"])


@unittest.skipUnless(
    sorted((BENCHMARK / "samples" / "train").glob("*.npz")),
    "benchmark samples are not available in this checkout",
)
class RepairFreedomTest(unittest.TestCase):
    """What exact repair leaves free is the representation's ring map."""

    def test_global_pca_has_nothing_left_to_refit_and_structured_keeps_its_interiors(self):
        from rooted4d.experiments.benchmark_eval import fit_benchmark_model
        from rooted4d.experiments.structured_model import fit_structured_benchmark_model
        from rooted4d.inference import initial_parameters

        config = replace(SolverConfig(fourier_order=3, session_order=1), use_session=False)
        for build, free in (
            (lambda: fit_benchmark_model(BENCHMARK, n_modes=48)[0], 0),
            (lambda: fit_structured_benchmark_model(
                BENCHMARK, interface_dim=12, interior_dim=12, coordinates="covariance")[0], 36),
        ):
            model = build()
            _, matrix = ring_map(model, config, initial_parameters(model, config))
            singular = np.linalg.svd(matrix, compute_uv=False)
            rank = int(np.count_nonzero(singular > 1e-10 * singular[0]))
            identity_free = model.n_modes - np.linalg.matrix_rank(matrix[:, : model.n_modes], tol=1e-10 * singular[0])
            self.assertEqual(identity_free, free)
            self.assertLessEqual(rank, matrix.shape[1])


if __name__ == "__main__":
    unittest.main()
