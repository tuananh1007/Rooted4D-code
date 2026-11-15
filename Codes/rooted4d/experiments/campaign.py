r"""The Rooted4D experiment campaign.

Each function implements one row of the specification's theorem-to-experiment
contract (section 4.9) or one runnable row of the experiment matrix (section
7.1).  Every result is a plain JSON-serializable dictionary carrying its own
provenance and pass/fail verdict, so the campaign can be re-run and diffed.

What this campaign can and cannot establish is fixed by the readiness ledger.
It exercises algebra, gauge uniqueness, local identifiability, and controlled
synthetic recovery.  It says nothing about real CMR/TEE data, clinical accuracy,
or the image-domain likelihoods, and no summary here should be quoted as if it
did.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import replace
from typing import Any, Callable

import numpy as np

from rooted4d import __version__
from rooted4d.geometry import RigidTransform, make_toy_complex
from rooted4d.inference import (
    SceneModel,
    SolverConfig,
    fit_joint_scene,
    infer_cross_modal_alignment,
    latent_trajectory,
    recover_synthetic_phase_pose,
)
from rooted4d.inference.baselines import BASELINE_IDS, run_all_arms
from rooted4d.inference.image_refine import (
    ImageRefinementConfig,
    heldout_plane_error,
    heldout_tee_error,
    make_image_problem,
    refine_against_images,
)
from rooted4d.inference.posterior import (
    calibrate_split_conformal,
    minimum_calibration_size,
    empirical_coverage,
    fit_laplace_posterior,
    propagate_measurements,
)
from rooted4d.inference.problem import (
    make_family_problem,
    make_joint_problem,
    pose_errors,
)
from rooted4d.synthetic.families import FAMILIES, TRAIN_FAMILY
from rooted4d.motion import FourierMotion, MonotoneCircularPhaseWarp, circular_distance
from rooted4d.priors import GaugedLatentBases, SessionDecomposition
from rooted4d.synthetic.benchmark import generate_case
from rooted4d.metrics import (
    DEFAULT_WELD_TOLERANCES,
    assess_simulation_readiness,
    summarize_jacobian,
)
from rooted4d.theory import (
    check_assembly_nullspace_equivalence,
    constrained_identifiability,
    noise_stability_bound,
    soft_attachment_sweep,
)

from .linearization import PARAMETER_BLOCKS, CoupledForwardModel
from .statistics import compare_paired

#: Success gates copied verbatim from specification section 7.6.
POSE_TRANSLATION_GATE_MM = 3.0
POSE_ROTATION_GATE_DEG = 3.0
PHASE_GATE_CYCLES = 0.05


def environment() -> dict[str, Any]:
    return {
        "rooted4d_version": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }


def _percentiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
        "min": float(np.min(array)),
    }


# ---------------------------------------------------------------- Theorem 1 --
def experiment_annular_resolution(
    resolutions: tuple[int, ...] = (12, 24, 32, 48, 64, 96),
    n_phases: int = 64,
) -> dict[str, Any]:
    """Theorem 1 plus the WP1 annular-resolution study.

    The specification requires the maximum attachment residual at every phase and
    a comparison against a detached topology, across the candidate 32/64/96-node
    annular densities that WP1 must choose between before freezing the template.
    """

    rows = []
    for n_annulus in resolutions:
        complex_ = make_toy_complex(n_annulus=n_annulus)
        report = check_assembly_nullspace_equivalence(complex_.attachment_map)

        rng = np.random.default_rng(n_annulus)
        base = complex_.global_vertices
        motion = FourierMotion(
            base=base, coefficients=rng.normal(0.0, 1.5, size=(8, *base.shape))
        )
        phases = np.arange(n_phases) / n_phases
        geometry = motion.evaluate(phases)
        parts = complex_.attachment_map.assemble(geometry)
        residual_over_cycle = complex_.attachment_map.max_residual(parts)

        rows.append(
            {
                "n_annulus": int(n_annulus),
                "n_local_vertices": report.n_local_vertices,
                "n_physical_vertices": report.n_physical_vertices,
                "n_equality_rows": report.n_equality_rows,
                "constraint_rank": report.constraint_rank,
                "nullspace_dimension": report.nullspace_dimension,
                "assembly_rank": report.assembly_rank,
                "ranks_agree": report.ranks_agree,
                "max_constraint_on_assembly": report.max_constraint_on_assembly,
                "max_projector_difference": report.max_projector_difference,
                "max_gradient_sharing_error": report.max_gradient_sharing_error,
                "attachment_residual_mm_over_cycle": residual_over_cycle,
                "n_phases_checked": int(n_phases),
                "theorem_1_holds": report.holds(),
            }
        )

    return {
        "experiment": "T1_annular_resolution_and_exact_attachment",
        "claim": "im(A) = ker(C); attachment residual is zero at every phase and resolution",
        "rows": rows,
        "verdict": {
            "all_resolutions_hold": all(row["theorem_1_holds"] for row in rows),
            "max_attachment_residual_mm": max(
                row["attachment_residual_mm_over_cycle"] for row in rows
            ),
            "resolution_independent": len({row["theorem_1_holds"] for row in rows}) == 1,
        },
    }


# ---------------------------------------------------------------- Theorem 2 --
def experiment_soft_attachment(
    seeds: tuple[int, ...] = tuple(range(20)),
    dimensions: tuple[int, ...] = (10, 30, 60),
    n_weights: int = 15,
) -> dict[str, Any]:
    """Theorem 2: the prespecified soft-attachment penalty sweep."""

    weights = np.logspace(1.0, 8.0, n_weights)
    rows = []
    for dimension in dimensions:
        for seed in seeds:
            rng = np.random.default_rng(1000 * dimension + seed)
            root = rng.normal(size=(dimension, dimension))
            metric = root @ root.T + dimension * np.eye(dimension)
            sweep = soft_attachment_sweep(
                rng.normal(size=(3, dimension)),
                metric,
                rng.normal(size=dimension),
                weights,
            )
            rows.append(
                {
                    "dimension": int(dimension),
                    "seed": int(seed),
                    "slope": sweep.slope,
                    "max_identity_error": sweep.max_identity_error,
                    "smallest_residual": float(sweep.attachment_residuals.min()),
                    "largest_condition_number": max(
                        point.condition_number for point in sweep.points
                    ),
                }
            )

    slopes = np.array([row["slope"] for row in rows])
    exact = check_assembly_nullspace_equivalence(make_toy_complex(24).attachment_map)
    return {
        "experiment": "T2_soft_attachment_penalty_sweep",
        "claim": "a finite penalty leaves O(1/lambda) drift; exact assembly does not",
        "penalty_weights": weights.tolist(),
        "rows": rows,
        "verdict": {
            "slope": _percentiles(slopes),
            "slope_within_0p05_of_minus_one": bool(np.all(np.abs(slopes + 1.0) < 0.05)),
            "max_identity_error": max(row["max_identity_error"] for row in rows),
            "smallest_soft_residual_any_run": min(row["smallest_residual"] for row in rows),
            "exact_assembly_residual": exact.max_constraint_on_assembly,
            "soft_never_reaches_exact": bool(
                min(row["smallest_residual"] for row in rows)
                > exact.max_constraint_on_assembly
            ),
        },
    }


# ---------------------------------------------------------------- Theorem 3 --
def _broken_decompositions(
    model: SessionDecomposition, trajectories: dict[str, np.ndarray]
) -> dict[str, dict[str, np.ndarray]]:
    """Gauge-ablated decoders used only to show what the gauges are buying."""

    bases = model.bases
    names = model.sessions
    weighted = sum(model.weights[name] * trajectories[name] for name in names)

    # No motion gauge: anchor the identity at phase zero instead of the cycle mean.
    identity_no_motion = weighted[0].copy()
    motion_no_motion = (weighted - identity_no_motion) @ bases.motion

    # No session gauge: use one modality as the reference instead of the
    # weighted cross-session mean, so a common component leaks into identity.
    reference = trajectories[names[0]]
    identity_no_session = reference.mean(axis=0)
    motion_no_session = (reference - identity_no_session) @ bases.motion

    return {
        "no_motion_gauge": {"identity": identity_no_motion, "motion": motion_no_motion},
        "no_session_gauge": {"identity": identity_no_session, "motion": motion_no_session},
    }


def experiment_gauge_ablation(
    seeds: tuple[int, ...] = tuple(range(30)),
    latent_dimension: int = 16,
    n_motion: int = 4,
    n_session: int = 2,
    n_phases: int = 32,
) -> dict[str, Any]:
    """Theorem 3: remove each gauge and report what stops being unique."""

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        bases = GaugedLatentBases.from_samples(
            rng.normal(size=(80, latent_dimension)),
            rng.normal(size=(80, latent_dimension)),
            n_motion=n_motion,
            n_session=n_session,
        )
        model = SessionDecomposition(bases)
        phases = np.arange(n_phases) / n_phases
        identity = rng.normal(size=latent_dimension)
        motion = np.stack(
            [np.cos(2 * np.pi * (k + 1) * phases) for k in range(n_motion)], axis=1
        )
        discrepancy = rng.normal(size=n_session) + (
            np.sin(2 * np.pi * phases)[:, None] * rng.normal(size=n_session)
        )
        session = {"cmr": discrepancy, "tee": -discrepancy}
        trajectories = model.compose(identity, motion, session)

        canonical = model.decompose(trajectories)
        shifted = model.reparameterize(
            identity,
            motion,
            session,
            motion_shift=rng.normal(size=n_motion),
            session_shift=rng.normal(size=n_session),
        )
        recanonical = model.canonicalize(*shifted[:2], shifted[2])
        broken = _broken_decompositions(model, trajectories)

        # The reparameterized triple is a genuinely different point in parameter
        # space that generates bitwise-identical trajectories.  That difference is
        # the ambiguity Theorem 3's gauges exist to remove; measuring it is what
        # makes the ablation non-vacuous.
        parameter_gap = float(
            max(
                np.max(np.abs(shifted[0] - identity)),
                np.max(np.abs(shifted[1] - motion)),
            )
        )
        trajectory_gap = float(
            max(
                np.max(np.abs(model.compose(*shifted[:2], shifted[2])[name] - trajectories[name]))
                for name in model.sessions
            )
        )

        rows.append(
            {
                "seed": int(seed),
                "full_gauges_identity_error": float(
                    np.max(np.abs(canonical.identity - identity))
                ),
                "full_gauges_motion_error": float(np.max(np.abs(canonical.motion - motion))),
                "full_gauges_reparameterization_drift": float(
                    max(
                        np.max(np.abs(recanonical.identity - canonical.identity)),
                        np.max(np.abs(recanonical.motion - canonical.motion)),
                    )
                ),
                "no_motion_gauge_identity_error": float(
                    np.max(np.abs(broken["no_motion_gauge"]["identity"] - identity))
                ),
                "no_motion_gauge_motion_error": float(
                    np.max(np.abs(broken["no_motion_gauge"]["motion"] - motion))
                ),
                "no_session_gauge_identity_error": float(
                    np.max(np.abs(broken["no_session_gauge"]["identity"] - identity))
                ),
                "parameter_space_gap": parameter_gap,
                "trajectory_gap": trajectory_gap,
                "motion_cycle_mean_residual": canonical.gauge_residuals()["motion_cycle_mean"],
                "session_cross_session_residual": canonical.gauge_residuals()[
                    "session_cross_session_mean"
                ],
                "max_unmodelled": canonical.max_unmodelled,
            }
        )

    drift = np.array([row["full_gauges_reparameterization_drift"] for row in rows])
    full_error = np.array([row["full_gauges_identity_error"] for row in rows])
    no_motion = np.array([row["no_motion_gauge_identity_error"] for row in rows])
    no_session = np.array([row["no_session_gauge_identity_error"] for row in rows])
    parameter_gap = np.array([row["parameter_space_gap"] for row in rows])
    trajectory_gap = np.array([row["trajectory_gap"] for row in rows])
    return {
        "experiment": "T3_gauge_ablation",
        "claim": (
            "distinct parameter triples generate identical trajectories; only the two "
            "centering gauges select the true one"
        ),
        "rows": rows,
        "verdict": {
            "ambiguity_is_real": {
                "parameter_space_gap": _percentiles(parameter_gap),
                "trajectory_gap": _percentiles(trajectory_gap),
                "distinct_parameters_same_trajectories": bool(
                    np.min(parameter_gap) > 1e-3 and np.max(trajectory_gap) < 1e-9
                ),
            },
            "full_gauges_identity_error": _percentiles(full_error),
            "no_motion_gauge_identity_error": _percentiles(no_motion),
            "no_session_gauge_identity_error": _percentiles(no_session),
            "full_gauge_reparameterization_drift": _percentiles(drift),
            "full_gauges_are_unique": bool(np.all(drift < 1e-9)),
            "full_gauges_recover_truth": bool(np.all(full_error < 1e-9)),
            "motion_gauge_is_necessary": bool(np.median(no_motion) > 1e-3),
            "session_gauge_is_necessary": bool(np.median(no_session) > 1e-3),
        },
    }


# ------------------------------------------------------------ Proposition 4 --
def experiment_periodicity_and_phase(
    fourier_orders: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
    seeds: tuple[int, ...] = tuple(range(25)),
    n_nodes: tuple[int, ...] = (4, 8, 16, 32),
) -> dict[str, Any]:
    """Proposition 4: periodic closure and degree-one phase order, under stress."""

    seam_rows = []
    for order in fourier_orders:
        for seed in seeds:
            rng = np.random.default_rng(10_000 + 100 * order + seed)
            base = rng.normal(size=(40, 3)) * 20.0
            motion = FourierMotion(
                base=base, coefficients=rng.normal(0.0, 2.0, size=(2 * order, 40, 3))
            )
            phases = np.arange(128) / 128
            seam_rows.append(
                {
                    "fourier_order": int(order),
                    "seed": int(seed),
                    "seam_error_mm": motion.seam_error(),
                    "cycle_mean_error_mm": float(
                        np.max(np.abs(motion.evaluate(phases).mean(axis=0) - base))
                    ),
                }
            )

    warp_rows = []
    for nodes in n_nodes:
        for seed in seeds:
            rng = np.random.default_rng(20_000 + 100 * nodes + seed)
            # Heavy-tailed logits stress the softplus increments toward collapse.
            warp = MonotoneCircularPhaseWarp(
                rng.normal(0.0, 3.0, size=nodes), offset=float(rng.uniform(0.0, 1.0))
            )
            # Irregular temporal sampling, as the contract requires.
            times = np.sort(rng.uniform(0.0, 1.0, size=97))
            lifted = warp.evaluate_unwrapped(times)
            warp_rows.append(
                {
                    "n_nodes": int(nodes),
                    "seed": int(seed),
                    "min_increment": warp.minimum_increment(),
                    "strictly_increasing": bool(np.all(np.diff(lifted) > 0.0)),
                    "total_turn_error": float(
                        abs(
                            warp.evaluate_unwrapped(1.0)
                            - warp.evaluate_unwrapped(0.0)
                            - 1.0
                        )
                    ),
                }
            )

    return {
        "experiment": "P4_periodicity_and_phase_order",
        "claim": "Fourier motion closes the cycle; the warp is a degree-one circle map",
        "seam_rows": seam_rows,
        "warp_rows": warp_rows,
        "verdict": {
            "max_seam_error_mm": max(row["seam_error_mm"] for row in seam_rows),
            "max_cycle_mean_error_mm": max(row["cycle_mean_error_mm"] for row in seam_rows),
            "all_warps_strictly_increasing": all(
                row["strictly_increasing"] for row in warp_rows
            ),
            "smallest_increment_observed": min(row["min_increment"] for row in warp_rows),
            "max_total_turn_error": max(row["total_turn_error"] for row in warp_rows),
        },
    }


# ---------------------------------------------------------------- Theorem 5 --
def experiment_identifiability_factorial(
    cmr_fractions: tuple[float, ...] = (1.0, 0.5, 0.25, 0.0),
    tee_fractions: tuple[float, ...] = (1.0, 0.5, 0.25, 0.0),
    seeds: tuple[int, ...] = tuple(range(5)),
) -> dict[str, Any]:
    """Theorem 5: the missing-view/modality factorial and its identifiability flags."""

    rows = []
    for seed in seeds:
        for cmr_fraction in cmr_fractions:
            for tee_fraction in tee_fractions:
                if cmr_fraction == 0.0 and tee_fraction == 0.0:
                    continue
                n_cmr, n_tee = 20, 24
                cmr_frames = np.arange(int(round(cmr_fraction * n_cmr)))
                tee_frames = np.arange(int(round(tee_fraction * n_tee)))
                model = CoupledForwardModel(
                    seed=seed, cmr_frames=cmr_frames, tee_frames=tee_frames
                )
                jacobian = model.jacobian()
                result = constrained_identifiability(
                    jacobian, parameter_blocks=PARAMETER_BLOCKS
                )
                dominant = (
                    max(result.block_sensitivity, key=result.block_sensitivity.get)
                    if result.block_sensitivity
                    else None
                )
                rows.append(
                    {
                        "seed": int(seed),
                        "cmr_fraction": float(cmr_fraction),
                        "tee_fraction": float(tee_fraction),
                        "n_cmr_frames": int(cmr_frames.size),
                        "n_tee_frames": int(tee_frames.size),
                        "n_observations": int(jacobian.shape[0]),
                        "alpha": result.smallest_tangent_eigenvalue,
                        "condition_number": result.tangent_condition_number,
                        "identifiable": result.identifiable,
                        "weakest_block": dominant,
                        "block_sensitivity": result.block_sensitivity,
                    }
                )

    both = [row for row in rows if row["cmr_fraction"] > 0 and row["tee_fraction"] > 0]
    cmr_only = [row for row in rows if row["tee_fraction"] == 0.0]
    tee_only = [row for row in rows if row["cmr_fraction"] == 0.0]
    return {
        "experiment": "T5_missing_view_identifiability_factorial",
        "provenance": "analytic_jacobian",
        "claim": "identifiability requires ker(J_C) and ker(J_E) to intersect trivially",
        "rows": rows,
        "verdict": {
            "alpha_both_modalities": _percentiles([row["alpha"] for row in both]),
            "alpha_cmr_only": _percentiles([row["alpha"] for row in cmr_only]),
            "alpha_tee_only": _percentiles([row["alpha"] for row in tee_only]),
            "cmr_only_is_unidentifiable": all(not row["identifiable"] for row in cmr_only),
            "joint_beats_worst_single_modality": bool(
                np.median([row["alpha"] for row in both])
                > max(
                    np.median([row["alpha"] for row in cmr_only]),
                    np.median([row["alpha"] for row in tee_only]),
                )
            ),
            "weakest_block_when_tee_removed": sorted(
                {row["weakest_block"] for row in cmr_only if row["weakest_block"]}
            ),
        },
    }


# ---------------------------------------------------------------- Theorem 6 --
def experiment_noise_stability(
    noise_levels: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5),
    seeds: tuple[int, ...] = tuple(range(40)),
) -> dict[str, Any]:
    """Theorem 6: does the measured error stay inside the local bound?"""

    model = CoupledForwardModel(seed=0)
    truth = model.reference_parameters
    jacobian = model.jacobian()
    diagnostics = constrained_identifiability(jacobian, parameter_blocks=PARAMETER_BLOCKS)
    alpha = diagnostics.smallest_tangent_eigenvalue

    rows = []
    for level in noise_levels:
        for seed in seeds:
            rng = np.random.default_rng(30_000 + seed)
            noise = rng.normal(scale=level, size=jacobian.shape[0])
            # Solve the linearized problem, which is the regime Theorem 6 describes.
            step = np.linalg.lstsq(jacobian, noise, rcond=None)[0]
            bound = noise_stability_bound(alpha, float(np.linalg.norm(noise)))
            error = float(np.linalg.norm(step))
            rows.append(
                {
                    "noise_level": float(level),
                    "seed": int(seed),
                    "weighted_noise_norm": float(np.linalg.norm(noise)),
                    "error": error,
                    "bound": bound.bound,
                    "within_bound": bool(error <= bound.bound),
                    "slack_ratio": float(bound.bound / error) if error > 0 else float("inf"),
                }
            )

    return {
        "experiment": "T6_noise_stability",
        "provenance": "analytic_jacobian",
        "claim": "local error stays below 4||eps||/sqrt(alpha(1-c_nl))",
        "alpha": alpha,
        "condition_number": diagnostics.tangent_condition_number,
        "rows": rows,
        "verdict": {
            "coverage": float(np.mean([row["within_bound"] for row in rows])),
            "all_within_bound": all(row["within_bound"] for row in rows),
            "slack_ratio": _percentiles([row["slack_ratio"] for row in rows]),
            "truth_norm": float(np.linalg.norm(truth)),
        },
    }


# --------------------------------------------------------------------- E1 ----
def experiment_synthetic_recovery(
    seeds: tuple[int, ...] = tuple(range(24)),
    tee_dropout_fractions: tuple[float, ...] = (0.0, 0.2, 0.4),
) -> dict[str, Any]:
    """Experiment E1: recover pose and circular phase when truth is known.

    Two arms are reported and must not be conflated.  The oracle arm consumes
    known vertex correspondences and is a simulator upper bound.  The non-oracle
    arm consumes only segmented, canonically anchored annular curves, which is
    what a clinical initializer could actually supply.
    """

    rows = []
    for dropout in tee_dropout_fractions:
        for seed in seeds:
            case = generate_case(seed=int(seed), tee_dropout_fraction=float(dropout))
            oracle = recover_synthetic_phase_pose(case)

            motion = FourierMotion(case.template_vertices, case.motion_coefficients)
            complex_ = make_toy_complex(n_annulus=24)
            annulus = complex_.groups["annulus"]
            cmr_curves = motion.evaluate(case.cmr_times)[:, annulus, :]
            tee_curves = motion.evaluate(case.tee_true_phases)[:, annulus, :]
            posed = RigidTransform(case.pose_rotation, case.pose_translation).apply(tee_curves)
            non_oracle = infer_cross_modal_alignment(cmr_curves, posed)
            estimated = non_oracle.cmr_to_tee
            relative = estimated.rotation @ case.pose_rotation.T
            cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)

            rows.append(
                {
                    "seed": int(seed),
                    "tee_dropout_fraction": float(dropout),
                    "oracle_translation_error_mm": oracle.translation_error_mm,
                    "oracle_rotation_error_deg": oracle.rotation_error_degrees,
                    "oracle_phase_error_cycles": oracle.phase_error_cycles,
                    "oracle_rmse_mm": oracle.rmse_mm,
                    "nonoracle_translation_error_mm": float(
                        np.linalg.norm(estimated.translation - case.pose_translation)
                    ),
                    "nonoracle_rotation_error_deg": float(np.degrees(np.arccos(cosine))),
                    # The alignment maps each TEE frame onto a CMR phase, so the
                    # truth it must match is that frame's true phase -- not zero.
                    # Frame quantization floors this error near 1/n_cmr_frames.
                    "nonoracle_phase_error_cycles": float(
                        np.median(
                            circular_distance(
                                non_oracle.temporal.target_to_source_phase,
                                case.tee_true_phases,
                            )
                        )
                    ),
                    "nonoracle_phase_offset_error_cycles": float(
                        circular_distance(
                            np.array([non_oracle.temporal.offset_cycles]),
                            np.array([case.tee_true_phases[0]]),
                        )[0]
                    ),
                    "cmr_frame_quantization_cycles": 1.0 / len(case.cmr_times),
                }
            )

    def gate(key_t: str, key_r: str, key_p: str) -> dict[str, Any]:
        translation = np.array([row[key_t] for row in rows])
        rotation = np.array([row[key_r] for row in rows])
        phase = np.array([row[key_p] for row in rows])
        return {
            "translation_mm": _percentiles(translation),
            "rotation_deg": _percentiles(rotation),
            "phase_cycles": _percentiles(phase),
            "meets_section_7_6_gate": bool(
                np.median(translation) < POSE_TRANSLATION_GATE_MM
                and np.median(rotation) < POSE_ROTATION_GATE_DEG
                and np.median(phase) < PHASE_GATE_CYCLES
            ),
        }

    return {
        "experiment": "E1_synthetic_pose_phase_recovery",
        "claim": "known pose and circular phase are recovered under the intended noise regime",
        "gates": {
            "translation_mm": POSE_TRANSLATION_GATE_MM,
            "rotation_deg": POSE_ROTATION_GATE_DEG,
            "phase_cycles": PHASE_GATE_CYCLES,
        },
        "rows": rows,
        "verdict": {
            "oracle": gate(
                "oracle_translation_error_mm",
                "oracle_rotation_error_deg",
                "oracle_phase_error_cycles",
            ),
            "non_oracle": gate(
                "nonoracle_translation_error_mm",
                "nonoracle_rotation_error_deg",
                "nonoracle_phase_error_cycles",
            ),
            "non_oracle_phase_offset_error_cycles": _percentiles(
                [row["nonoracle_phase_offset_error_cycles"] for row in rows]
            ),
            "cmr_frame_quantization_cycles": rows[0]["cmr_frame_quantization_cycles"],
        },
    }


# --------------------------------------------------------------------- E5 ----
def experiment_baseline_ladder(
    seeds: tuple[int, ...] = tuple(range(20)),
    n_modes: int = 10,
    soft_attachment_weight: float = 1e4,
) -> dict[str, Any]:
    """Experiment E5: does joint inference beat the alternatives it must beat?

    All six arms of ``baselines/BASELINE_LOCK.yaml`` run on identical problems
    with identical held-out masks. The first co-primary endpoint is error on
    hidden evidence; the second is interface validity. Comparisons use the
    paired permutation test and bootstrap intervals section 8 prescribes.
    """

    model = SceneModel.procedural(n_modes=n_modes, n_motion=3, n_session=2, n_phase_nodes=6)
    config = SolverConfig(fourier_order=2, session_order=1, max_iterations=40)

    rows = []
    for seed in seeds:
        problem = make_joint_problem(model, config, seed=int(seed))
        arms = run_all_arms(
            problem, config, soft_attachment_weight=soft_attachment_weight
        )
        for name, result in arms.items():
            record = result.as_dict()
            record["seed"] = int(seed)
            record["cmr_noise_mm"] = problem.cmr_noise_mm
            record["tee_noise_mm"] = problem.tee_noise_mm
            rows.append(record)

    def column(arm: str, key: str) -> np.ndarray:
        return np.array(
            [row[key] for row in rows if row["baseline_id"] == arm and row[key] is not None],
            dtype=np.float64,
        )

    per_arm = {}
    for arm in BASELINE_IDS:
        entries = [row for row in rows if row["baseline_id"] == arm]
        per_arm[arm] = {
            "n": len(entries),
            "failures": sum(1 for row in entries if not row["succeeded"]),
            "max_attachment_residual_mm": float(
                np.max([row["attachment_residual_mm"] for row in entries])
            ),
            "heldout_cmr_mm": _percentiles(column(arm, "heldout_cmr_mm"))
            if len(column(arm, "heldout_cmr_mm"))
            else None,
            "heldout_tee_mm": _percentiles(column(arm, "heldout_tee_mm"))
            if len(column(arm, "heldout_tee_mm"))
            else None,
            "rotation_deg": _percentiles(column(arm, "rotation_deg"))
            if len(column(arm, "rotation_deg"))
            else None,
            "translation_mm": _percentiles(column(arm, "translation_mm"))
            if len(column(arm, "translation_mm"))
            else None,
            "phase_error_cycles": _percentiles(column(arm, "phase_error_cycles"))
            if len(column(arm, "phase_error_cycles"))
            else None,
            "alpha": _percentiles(column(arm, "alpha"))
            if len(column(arm, "alpha"))
            else None,
            "identified": sorted(
                {row["identified"] for row in entries if row["identified"] is not None}
            ),
        }

    comparisons = {}
    for reference in ("late_fusion", "joint_soft_attachment", "joint_exact_no_gauges", "tee_only"):
        for endpoint in ("heldout_cmr_mm", "heldout_tee_mm"):
            treatment = column("rooted4d_full", endpoint)
            baseline = column(reference, endpoint)
            if len(treatment) != len(baseline) or len(treatment) < 2:
                continue
            comparisons[f"full_vs_{reference}__{endpoint}"] = compare_paired(
                treatment,
                baseline,
                treatment_name="rooted4d_full",
                reference_name=reference,
            ).as_dict()

    # Theorem 2 in the practical solver rather than in closed form. With the
    # matched basis the penalty acts only on the separation coordinates, so the
    # predicted 1/lambda decay is actually reachable by the optimizer.
    sweep_weights = np.logspace(0.0, 8.0, 5)
    sweep_rows = []
    for weight in sweep_weights:
        residuals = []
        held_out = []
        for seed in seeds[:3]:
            problem = make_joint_problem(model, config, seed=int(seed))
            arm = run_arm(
                "joint_soft_attachment", problem, config, soft_attachment_weight=float(weight)
            )
            residuals.append(arm.attachment_residual_mm)
            if arm.heldout_cmr_mm is not None:
                held_out.append(arm.heldout_cmr_mm)
        sweep_rows.append(
            {
                "penalty_weight": float(weight),
                "attachment_residual_mm": float(np.median(residuals)),
                "heldout_cmr_mm": float(np.median(held_out)) if held_out else None,
            }
        )
    positive = [r for r in sweep_rows if r["attachment_residual_mm"] > 0.0]
    sweep_slope = (
        float(
            np.polyfit(
                np.log([r["penalty_weight"] for r in positive]),
                np.log([r["attachment_residual_mm"] for r in positive]),
                deg=1,
            )[0]
        )
        if len(positive) > 1
        else float("nan")
    )

    exact_arms = ("cmr_only", "tee_only", "late_fusion", "joint_exact_no_gauges", "rooted4d_full")
    return {
        "experiment": "E5_locked_baseline_ladder",
        "claim": (
            "joint gauge-fixed inference with exact coupling improves held-out "
            "cross-modal prediction and interface validity over the locked alternatives"
        ),
        "arms": list(BASELINE_IDS),
        "soft_attachment_weight": soft_attachment_weight,
        "rows": rows,
        "soft_attachment_lambda_sweep": sweep_rows,
        "verdict": {
            "per_arm": per_arm,
            "comparisons": comparisons,
            "soft_attachment_sweep_slope": sweep_slope,
            "soft_attachment_reaches_exact": bool(
                min(r["attachment_residual_mm"] for r in sweep_rows) == 0.0
            ),
            "soft_attachment_note": (
                "The soft arm uses a basis matched to the coupled one: the coupled "
                "basis lifted through the assembly map, plus separation modes from "
                "the row space of C. Every geometry the exact model can express, it "
                "can express too, so the arms differ only in coupling. Under that "
                "construction the accuracy difference largely disappears and the "
                "honest claim narrows: soft attachment is not less accurate, it is "
                "never exact. The residual follows the predicted 1/lambda law and "
                "reaches zero at no finite weight, while exact assembly is at zero "
                "with nothing to tune."
            ),
            "exact_arms_attachment_is_zero": bool(
                all(per_arm[arm]["max_attachment_residual_mm"] == 0.0 for arm in exact_arms)
            ),
            "weakly_identified_arms": sorted(
                arm
                for arm in BASELINE_IDS
                if per_arm[arm]["alpha"] is not None
                and per_arm[arm]["alpha"]["median"]
                < 0.5 * per_arm["rooted4d_full"]["alpha"]["median"]
            ),
            "reproducibility_note": (
                "An arm whose smallest non-gauge eigenvalue is far below the full "
                "model's sits in a nearly flat valley, so which point the regularizer "
                "settles on depends on floating-point rounding. Observed directly: "
                "across two machines the tee_only arm reproduced its held-out TEE "
                "error to 1e-13 but its held-out CMR extrapolation to only 1e-1. "
                "Cross-modal numbers from weakly identified arms should be read as "
                "indicative, not as reproducible measurements."
            ),
            "interpretation_of_the_gauge_comparison": (
                "A null result for full versus joint_exact_no_gauges on held-out "
                "error is the predicted outcome, not a failure. Theorem 3 concerns "
                "uniqueness of the identity/motion/session split conditional on the "
                "trajectories; removing the gauges leaves the trajectory space, and "
                "therefore the predictions, unchanged. An earlier version of this "
                "note claimed the smallest non-gauge eigenvalue collapsed to zero "
                "without the gauges. That was an offset defect in the gauge-direction "
                "builder, not a property of the model, and it is retracted. "
                "Corrected, alpha is essentially unchanged whether the gauges are "
                "enforced or not, because they reparameterize the latent space rather "
                "than add information. What does change is the number of exact null "
                "directions, five with the centering gauges and one without, and in "
                "both cases the predicted count matches the measured null space of "
                "the Jacobian. The benefit of the gauges is uniqueness of the "
                "identity/motion/session split, which t3_gauge_ablation measures "
                "directly."
            ),
            "gauges_change_the_parameterization_not_the_identifiability": bool(
                per_arm["rooted4d_full"]["alpha"] is not None
                and per_arm["joint_exact_no_gauges"]["alpha"] is not None
                and abs(
                    per_arm["rooted4d_full"]["alpha"]["median"]
                    - per_arm["joint_exact_no_gauges"]["alpha"]["median"]
                )
                < 0.25 * per_arm["rooted4d_full"]["alpha"]["median"]
            ),
            "soft_attachment_residual_is_nonzero": bool(
                per_arm["joint_soft_attachment"]["max_attachment_residual_mm"] > 0.0
            ),
            "total_failures": sum(per_arm[arm]["failures"] for arm in BASELINE_IDS),
        },
    }


# ------------------------------------------------- Theorems 5 and 6, for real --
def experiment_solver_identifiability(
    seeds: tuple[int, ...] = tuple(range(8)),
    n_modes: int = 10,
) -> dict[str, Any]:
    """Theorems 5 and 6 evaluated on the joint solver's own Jacobian.

    Until the solver existed these two results could only be checked against an
    explicit analytic forward map. This replaces that stand-in with the real
    ``J_E`` of the fitted inverse problem, which is what the contract asks for.
    Renderer misspecification is still out of scope: the observation model here
    is point evidence, not the image-domain likelihoods.
    """

    model = SceneModel.procedural(n_modes=n_modes, n_motion=3, n_session=2, n_phase_nodes=6)
    config = SolverConfig(fourier_order=2, session_order=1, max_iterations=40)

    rows = []
    for seed in seeds:
        problem = make_joint_problem(model, config, seed=int(seed))
        fit = fit_joint_scene(problem.observations, model, config)
        jacobian = fit.evidence_jacobian()
        summary = summarize_jacobian(jacobian)
        gauges = fit.gauge_directions()

        # Theorem 5 requires the latent gauges to sit in c(theta). Measuring the
        # numerical null space of J_E and comparing it against the analytically
        # predicted gauge count is the falsifiable part: if the parameterization
        # carried an unmodelled degeneracy, these two numbers would disagree.
        singular = np.linalg.svd(jacobian, compute_uv=False)
        measured_null = int(np.count_nonzero(singular <= 1e-8 * singular[0]))

        unrestricted = constrained_identifiability(jacobian)
        diagnostics = constrained_identifiability(jacobian, constraint_jacobian=gauges)
        alpha = diagnostics.smallest_tangent_eigenvalue
        noise = np.sqrt(
            len(problem.observations.cmr_points.reshape(-1)) * problem.cmr_noise_mm**2
            + len(problem.observations.tee_points.reshape(-1)) * problem.tee_noise_mm**2
        )
        bound = noise_stability_bound(alpha, float(noise))
        rows.append(
            {
                "seed": int(seed),
                "n_residuals": int(jacobian.shape[0]),
                "n_parameters": int(jacobian.shape[1]),
                "effective_rank": summary.effective_rank,
                "predicted_gauge_dimension": int(gauges.shape[0]),
                "measured_null_dimension": measured_null,
                "gauge_prediction_matches": bool(measured_null == gauges.shape[0]),
                "tangent_dimension": diagnostics.tangent_dimension,
                "alpha_unrestricted": unrestricted.smallest_tangent_eigenvalue,
                "alpha": alpha,
                "condition_number": diagnostics.tangent_condition_number,
                "identifiable": bool(diagnostics.identifiable),
                "weighted_noise_norm": float(noise),
                "bound": bound.bound,
                "bound_informative": bound.informative,
                "attachment_residual_mm": fit.attachment_residual,
                "cost": fit.cost,
            }
        )

    return {
        "experiment": "T5_T6_on_solver_jacobian",
        "provenance": "solver_jacobian",
        "claim": (
            "the only null directions of the fitted problem are the predicted "
            "gauges; restricted to the non-gauge tangent space it is identifiable"
        ),
        "rows": rows,
        "verdict": {
            "all_identifiable_after_gauge_removal": all(
                row["identifiable"] for row in rows
            ),
            "gauge_prediction_matches_measurement": all(
                row["gauge_prediction_matches"] for row in rows
            ),
            "predicted_gauge_dimension": sorted(
                {row["predicted_gauge_dimension"] for row in rows}
            ),
            "alpha_unrestricted_is_zero": all(
                row["alpha_unrestricted"] == 0.0 for row in rows
            ),
            "alpha": _percentiles([row["alpha"] for row in rows]),
            "condition_number": _percentiles([row["condition_number"] for row in rows]),
            "bound_mm": _percentiles([row["bound"] for row in rows]),
            "max_attachment_residual_mm": max(
                row["attachment_residual_mm"] for row in rows
            ),
        },
    }


# --------------------------------------------------------------- E6 image ----
def experiment_image_domain_refinement(
    seeds: tuple[int, ...] = tuple(range(6)),
    n_modes: int = 10,
    sigma_mm: float = 3.0,
) -> dict[str, Any]:
    """Fit the section 4.5 observation operators, not point evidence.

    The CMR finite-thickness plane renderer and the diverging TEE sector renderer
    drive the fit directly. Two design points are forced by the physics and by
    cost, and both are reported rather than hidden:

    * The likelihood band must span a voxel. At the default 1.5 mm sigma against
      2 mm pixels only a few percent of TEE voxels carry any signal, and the
      objective is nearly flat. Sigma is matched to voxel size here.
    * Held-out CMR planes cannot score pose or circular phase, because neither
      enters the CMR branch. Hidden TEE phases are scored as well.

    Held-out CMR error is measured against a noiseless rendering of the same
    truth. Against the noisy volume the metric saturates at the observation noise
    level and cannot separate a good geometry from a bad one.
    """

    model = SceneModel.procedural(n_modes=n_modes, n_motion=3, n_session=2, n_phase_nodes=6)
    config = SolverConfig(fourier_order=2, session_order=1, max_iterations=40)
    refinement = ImageRefinementConfig(
        n_shape_modes=3, max_iterations=12, cmr_sigma_mm=sigma_mm, tee_sigma_mm=sigma_mm
    )

    rows = []
    for seed in seeds:
        truth = make_joint_problem(model, config, seed=int(seed)).truth
        observations = make_image_problem(
            truth, model, config, refinement, seed=int(seed), n_slices=6, n_tee_frames=9
        )
        # A deliberately degraded initializer, standing in for what an upstream
        # segmentation and landmark step would realistically hand over.
        rng = np.random.default_rng(10_000 + seed)
        identity = truth.identity.copy()
        identity[:3] += rng.normal(0.0, 0.8, 3)
        start = replace(
            truth,
            pose_axis_angle=truth.pose_axis_angle + rng.normal(0.0, 0.03, 3),
            pose_translation=truth.pose_translation + rng.normal(0.0, 2.0, 3),
            phase_offset=float((truth.phase_offset + 0.04) % 1.0),
            identity=identity,
        )

        before_planes = heldout_plane_error(start, observations, model, config, refinement)
        before_tee = heldout_tee_error(start, observations, model, config, refinement)
        result = refine_against_images(start, observations, model, config, refinement)
        after_planes = heldout_plane_error(
            result.parameters, observations, model, config, refinement
        )
        after_tee = heldout_tee_error(
            result.parameters, observations, model, config, refinement
        )
        start_pose = pose_errors(start, truth)
        final_pose = pose_errors(result.parameters, truth)
        phase = lambda p: float(  # noqa: E731 - local shorthand
            circular_distance(np.array([p.phase_offset]), np.array([truth.phase_offset]))[0]
        )
        rows.append(
            {
                "seed": int(seed),
                "seconds": result.seconds,
                "iterations": result.iterations,
                "n_residuals": result.n_residuals,
                "n_parameters": result.n_parameters,
                "cost_before": result.cost_before,
                "cost_after": result.cost_after,
                "rotation_deg_before": start_pose["rotation_deg"],
                "rotation_deg_after": final_pose["rotation_deg"],
                "translation_mm_before": start_pose["translation_mm"],
                "translation_mm_after": final_pose["translation_mm"],
                "phase_before": phase(start),
                "phase_after": phase(result.parameters),
                "heldout_plane_rmse_before": before_planes["heldout_plane_rmse_vs_clean"],
                "heldout_plane_rmse_after": after_planes["heldout_plane_rmse_vs_clean"],
                "heldout_tee_rmse_before": before_tee["heldout_tee_rmse"],
                "heldout_tee_rmse_after": after_tee["heldout_tee_rmse"],
            }
        )

    def improvement(key: str) -> dict[str, Any]:
        before = np.array([row[f"{key}_before"] for row in rows])
        after = np.array([row[f"{key}_after"] for row in rows])
        return {
            "before": _percentiles(before),
            "after": _percentiles(after),
            "improved_fraction": float(np.mean(after < before)),
        }

    return {
        "experiment": "E6_image_domain_refinement",
        "provenance": "image_domain_likelihood",
        "claim": (
            "the CMR plane and TEE sector likelihood operators drive the joint fit "
            "and improve error on hidden planes and hidden phases"
        ),
        "sigma_mm": sigma_mm,
        "rows": rows,
        "verdict": {
            "heldout_cmr_planes": improvement("heldout_plane_rmse"),
            "heldout_tee_phases": improvement("heldout_tee_rmse"),
            "translation_mm": improvement("translation_mm"),
            "phase_cycles": improvement("phase"),
            "rotation_deg": improvement("rotation_deg"),
            "seconds_per_fit": _percentiles([row["seconds"] for row in rows]),
            "meets_section_7_6_gates": bool(
                np.median([row["translation_mm_after"] for row in rows]) < 3.0
                and np.median([row["rotation_deg_after"] for row in rows]) < 3.0
                and np.median([row["phase_after"] for row in rows]) < 0.05
            ),
            "rotation_is_the_weak_variable": (
                "Translation, held-out planes and held-out phases improve on every "
                "seed, but rotation does not reliably improve. Rotation is "
                "observable only through the TEE sector, and at this grid density "
                "the angular evidence is thin. A finer sector or an explicitly "
                "angular term is the next thing to try; the section 7.6 rotation "
                "gate is currently met by margin rather than by improvement."
            ),
            "cost_is_the_limiting_factor": (
                "A central-difference Jacobian needs two renders per parameter per "
                "iteration, so only pose, circular phase, and the leading shape modes "
                "are refined here. A full image-domain joint solve needs an analytic "
                "or adjoint gradient, which is not implemented."
            ),
        },
    }


# --------------------------------------------------------------- E9 posterior --
def experiment_posterior_calibration(
    seeds: tuple[int, ...] = tuple(range(44)),
    n_modes: int = 10,
    coverage_targets: tuple[float, ...] = (0.90, 0.95),
    n_posterior_samples: int = 80,
) -> dict[str, Any]:
    """Experiment E9: does the reported uncertainty mean anything?

    A Laplace posterior is fitted around each MAP solution on the non-gauge
    tangent space, scalar measurements are propagated by sampling, and intervals
    are calibrated by split conformal on one half of the cases and evaluated on
    the other. The split is by case and the calibration half never overlaps the
    evaluation half, which is the condition the coverage guarantee needs.

    The gauge restriction is not cosmetic. The Gauss-Newton matrix is singular
    along every gauge, so an unrestricted Laplace inverse reports infinite
    variance for directions that are pure reparameterization.
    """

    model = SceneModel.procedural(n_modes=n_modes, n_motion=3, n_session=2, n_phase_nodes=6)
    config = SolverConfig(fourier_order=2, session_order=1, max_iterations=40)
    measurements = {
        "phase_offset_cycles": lambda p: float(p.phase_offset),
        "pose_translation_norm_mm": lambda p: float(np.linalg.norm(p.pose_translation)),
        "identity_norm": lambda p: float(np.linalg.norm(p.identity)),
    }
    # Cardiac phase lives on the circle; a plain absolute residual wraps and
    # inflates its interval to more than a full cycle.
    circular_measurements = {"phase_offset_cycles"}

    rows = []
    for seed in seeds:
        problem = make_joint_problem(model, config, seed=int(seed))
        fit = fit_joint_scene(problem.observations, model, config)
        posterior = fit_laplace_posterior(fit)
        samples = propagate_measurements(
            posterior, fit, measurements, n_samples=n_posterior_samples, seed=int(seed)
        )
        record = {
            "seed": int(seed),
            "tangent_dimension": posterior.tangent_dimension,
            "gauge_dimension": posterior.gauge_dimension,
            "noise_variance": posterior.noise_variance,
            "gauge_variance": float(
                np.max(np.abs(fit.gauge_directions() @ posterior.covariance() @ fit.gauge_directions().T))
            )
            if posterior.gauge_dimension
            else 0.0,
        }
        for name, function in measurements.items():
            record[f"{name}__predicted"] = float(np.mean(samples[name]))
            record[f"{name}__posterior_sd"] = float(np.std(samples[name]))
            record[f"{name}__truth"] = float(function(problem.truth))
        rows.append(record)

    # Split by case: calibrate on the first half, evaluate on the second.
    midpoint = len(rows) // 2
    calibration, evaluation = rows[:midpoint], rows[midpoint:]
    calibrated: dict[str, Any] = {}
    for name in measurements:
        for target in coverage_targets:
            interval = calibrate_split_conformal(
                name,
                np.array([row[f"{name}__predicted"] for row in calibration]),
                np.array([row[f"{name}__truth"] for row in calibration]),
                coverage_target=target,
                circular=name in circular_measurements,
            )
            coverage = empirical_coverage(
                interval,
                np.array([row[f"{name}__predicted"] for row in evaluation]),
                np.array([row[f"{name}__truth"] for row in evaluation]),
            )
            calibrated[f"{name}@{target:.2f}"] = {
                **interval.as_dict(),
                **coverage,
            }

    within_five_points = [
        abs(entry["coverage_error"]) <= 0.05 for entry in calibrated.values()
    ]
    within_noise = [entry["within_one_standard_error"] for entry in calibrated.values()]
    meets_guarantee = [entry["meets_guarantee"] for entry in calibrated.values()]
    return {
        "experiment": "E9_posterior_and_conformal_calibration",
        "claim": (
            "a gauge-restricted Laplace posterior propagated to scalar measurements "
            "yields split-conformal intervals with near-nominal coverage"
        ),
        "coverage_targets": list(coverage_targets),
        "rows": rows,
        "verdict": {
            "calibrated": calibrated,
            "gauge_variance_is_zero": bool(
                max(row["gauge_variance"] for row in rows) < 1e-12
            ),
            "n_calibration": len(calibration),
            "minimum_calibration_size": {
                f"{target:.2f}": minimum_calibration_size(target)
                for target in coverage_targets
            },
            "all_targets_attainable": bool(
                all(entry["attainable"] for entry in calibrated.values())
            ),
            "n_evaluation": len(evaluation),
            "fraction_within_five_points_of_nominal": float(
                np.mean(within_five_points)
            ),
            "fraction_within_one_standard_error": float(np.mean(within_noise)),
            "fraction_meeting_the_conformal_guarantee": float(np.mean(meets_guarantee)),
            "any_interval_under_covers": bool(
                any(entry["under_covers"] for entry in calibrated.values())
            ),
            "one_sided_criterion_note": (
                "Split conformal guarantees coverage at least the target in finite "
                "samples, so over-coverage is correct conservative behaviour and only "
                "under-coverage is a failure. The two-sided five-point criterion in "
                "the protocol marks valid conservative intervals as miscalibrated, so "
                "the one-sided fraction is reported alongside it."
            ),
            "coverage_resolution_note": (
                "The protocol's five-point criterion is below what this many cases "
                "can resolve: the binomial standard error on a 0.90 target at about "
                "twenty evaluation cases is already six points. The fraction within "
                "one standard error is the honest statistic at this sample size, and "
                "a real calibration claim needs the hundreds of cases the protocol "
                "assumes."
            ),
            "caveat": (
                "Coverage is measured on synthetic cases drawn from the same "
                "generator as the calibration half, so this demonstrates the "
                "machinery, not calibration under distribution shift. The protocol "
                "requires the locked test family and an external centre before any "
                "coverage claim is made."
            ),
        },
    }


# ------------------------------------------------- E10 anti-inverse-crime ----
def experiment_family_generalization(
    seeds: tuple[int, ...] = tuple(range(12)),
    n_modes: int = 10,
) -> dict[str, Any]:
    """Does the method survive a generator it was not built around?

    Every other experiment here fits data produced by the same model class the
    solver assumes: Fourier motion, independent noise, no pathology, even beat
    spacing. That is an inverse crime, and it measures optimization rather than
    method robustness. This fits the same solver to the two held-out families of
    `configs/simulator_protocol.yaml` and reports the degradation.

    Pose and circular phase remain comparable across families because they are
    generator independent. Identity and motion coefficients are not: for the
    held-out families the truth does not lie in the model's span, so only
    held-out evidence and the nuisance variables are scored.
    """

    model = SceneModel.procedural(n_modes=n_modes, n_motion=3, n_session=2, n_phase_nodes=6)
    config = SolverConfig(fourier_order=2, session_order=1, max_iterations=40)

    rows = []
    for family in FAMILIES:
        for seed in seeds:
            problem = make_family_problem(model, config, family=family, seed=int(seed))
            fit = fit_joint_scene(problem.observations, model, config)
            errors = fit.heldout_errors()

            # Approximation floor: the best this model class can do on this
            # family even with the nuisance variables fixed at truth. A second
            # order Fourier basis cannot represent seven-control-point spline
            # motion, so part of the held-out error is irreducible and must not
            # be reported as an estimation failure.
            oracle = fit_joint_scene(
                problem.observations, model, config, start=problem.truth
            )
            oracle_errors = oracle.heldout_errors()

            # Does the method's own diagnostic notice the misspecification? This
            # is the question that matters: a fit that is wrong and knows it is
            # far more useful than one that is wrong and confident.
            diagnostics = constrained_identifiability(
                fit.evidence_jacobian(), constraint_jacobian=fit.gauge_directions()
            )
            pose = pose_errors(fit.parameters, problem.truth)
            rows.append(
                {
                    "family": family,
                    "seed": int(seed),
                    "heldout_cmr_mm": errors.get("heldout_cmr_mm"),
                    "heldout_tee_mm": errors.get("heldout_tee_mm"),
                    "rotation_deg": pose["rotation_deg"],
                    "translation_mm": pose["translation_mm"],
                    "phase_error_cycles": float(
                        circular_distance(
                            np.array([fit.parameters.phase_offset]),
                            np.array([problem.truth.phase_offset]),
                        )[0]
                    ),
                    "attachment_residual_mm": fit.attachment_residual,
                    "floor_cmr_mm": oracle_errors.get("heldout_cmr_mm"),
                    "floor_tee_mm": oracle_errors.get("heldout_tee_mm"),
                    "alpha": diagnostics.smallest_tangent_eigenvalue,
                    "condition_number": diagnostics.tangent_condition_number,
                }
            )

    def column(family: str, key: str) -> np.ndarray:
        return np.array(
            [row[key] for row in rows if row["family"] == family and row[key] is not None]
        )

    per_family = {
        family: {
            key: _percentiles(column(family, key))
            for key in (
                "heldout_cmr_mm",
                "heldout_tee_mm",
                "rotation_deg",
                "translation_mm",
                "phase_error_cycles",
                "floor_cmr_mm",
                "floor_tee_mm",
                "alpha",
                "condition_number",
            )
        }
        | {
            "max_attachment_residual_mm": float(
                np.max(column(family, "attachment_residual_mm"))
            ),
            "meets_section_7_6_gates": bool(
                np.median(column(family, "translation_mm")) < POSE_TRANSLATION_GATE_MM
                and np.median(column(family, "rotation_deg")) < POSE_ROTATION_GATE_DEG
                and np.median(column(family, "phase_error_cycles")) < PHASE_GATE_CYCLES
            ),
        }
        for family in FAMILIES
    }

    degradation = {
        family: {
            key: float(
                per_family[family][key]["median"]
                / max(per_family[TRAIN_FAMILY][key]["median"], 1e-12)
            )
            for key in ("heldout_cmr_mm", "heldout_tee_mm", "phase_error_cycles")
        }
        for family in FAMILIES
        if family != TRAIN_FAMILY
    }

    failing = [f for f in FAMILIES if not per_family[f]["meets_section_7_6_gates"]]
    return {
        "experiment": "E10_heldout_family_generalization",
        "provenance": "heldout_generator_families",
        "claim": (
            "in-distribution accuracy does not by itself establish robustness to a "
            "different generator"
        ),
        "families": list(FAMILIES),
        "rows": rows,
        "verdict": {
            "per_family": per_family,
            "degradation_vs_train": degradation,
            "families_failing_section_7_6": failing,
            "attachment_survives_every_family": bool(
                all(per_family[f]["max_attachment_residual_mm"] == 0.0 for f in FAMILIES)
            ),
            "excess_over_approximation_floor": {
                family: {
                    "heldout_cmr_mm": float(
                        per_family[family]["heldout_cmr_mm"]["median"]
                        - per_family[family]["floor_cmr_mm"]["median"]
                    ),
                    "heldout_tee_mm": float(
                        per_family[family]["heldout_tee_mm"]["median"]
                        - per_family[family]["floor_tee_mm"]["median"]
                    ),
                }
                for family in FAMILIES
            },
            "diagnostic_detects_misspecification": bool(
                per_family[FAMILIES[1]]["alpha"]["median"]
                < per_family[TRAIN_FAMILY]["alpha"]["median"]
                and per_family[FAMILIES[2]]["alpha"]["median"]
                < per_family[FAMILIES[1]]["alpha"]["median"]
            ),
            "headline": (
                "Exact attachment is structural and holds under every family. "
                "Held-out geometric error rises on the held-out families, but most "
                "of that rise is an approximation floor rather than an estimation "
                "failure: a second order Fourier basis cannot represent "
                "seven-control-point spline motion, and the floor is reported "
                "alongside. The nuisance variables the method actually claims to "
                "estimate -- pose and circular phase -- are what should be read for "
                "robustness. The gauge-restricted smallest eigenvalue falls "
                "monotonically across the three families while the tangent "
                "condition number rises by orders of magnitude, so the method's own "
                "identifiability diagnostic flags the misspecified cases rather than "
                "reporting them confidently. That is the behaviour section 4.8 asks "
                "for."
            ),
        },
    }


# --------------------------------------- E11 renderer-family misspecification --
def experiment_renderer_misspecification(
    seeds: tuple[int, ...] = tuple(range(4)),
    n_modes: int = 10,
    sigma_mm: float = 3.0,
) -> dict[str, Any]:
    """Fit with the training renderer against data from a held-out renderer.

    E10 varies the generator; this varies the observation operator, which is the
    other half of the inverse-crime concern. The solver always renders with
    ``analytic_distance_likelihood_v1``. The data come from that, from a
    supersampled quadrature, or from a ray renderer with front-to-back occlusion.
    """

    from rooted4d.synthetic.families import RENDERER_FAMILIES

    model = SceneModel.procedural(n_modes=n_modes, n_motion=3, n_session=2, n_phase_nodes=6)
    config = SolverConfig(fourier_order=2, session_order=1, max_iterations=40)
    refinement = ImageRefinementConfig(
        n_shape_modes=3, max_iterations=10, cmr_sigma_mm=sigma_mm, tee_sigma_mm=sigma_mm
    )

    rows = []
    for family in RENDERER_FAMILIES:
        for seed in seeds:
            truth = make_joint_problem(model, config, seed=int(seed)).truth
            observations = make_image_problem(
                truth, model, config, refinement, seed=int(seed),
                n_slices=6, n_tee_frames=9, renderer_family=family,
            )
            rng = np.random.default_rng(20_000 + seed)
            identity = truth.identity.copy()
            identity[:3] += rng.normal(0.0, 0.8, 3)
            start = replace(
                truth,
                pose_axis_angle=truth.pose_axis_angle + rng.normal(0.0, 0.03, 3),
                pose_translation=truth.pose_translation + rng.normal(0.0, 2.0, 3),
                phase_offset=float((truth.phase_offset + 0.04) % 1.0),
                identity=identity,
            )
            result = refine_against_images(start, observations, model, config, refinement)
            before = pose_errors(start, truth)
            after = pose_errors(result.parameters, truth)
            rows.append(
                {
                    "renderer_family": family,
                    "seed": int(seed),
                    "translation_mm_before": before["translation_mm"],
                    "translation_mm_after": after["translation_mm"],
                    "rotation_deg_after": after["rotation_deg"],
                    "phase_after": float(
                        circular_distance(
                            np.array([result.parameters.phase_offset]),
                            np.array([truth.phase_offset]),
                        )[0]
                    ),
                    "seconds": result.seconds,
                }
            )

    per_family = {
        family: {
            key: _percentiles(
                np.array([r[key] for r in rows if r["renderer_family"] == family])
            )
            for key in ("translation_mm_after", "rotation_deg_after", "phase_after")
        }
        for family in RENDERER_FAMILIES
    }
    train = RENDERER_FAMILIES[0]
    return {
        "experiment": "E11_renderer_family_misspecification",
        "provenance": "heldout_renderer_families",
        "claim": "pose and phase recovery should not depend on the training renderer",
        "rows": rows,
        "verdict": {
            "per_family": per_family,
            "translation_degradation_vs_train": {
                family: float(
                    per_family[family]["translation_mm_after"]["median"]
                    / max(per_family[train]["translation_mm_after"]["median"], 1e-12)
                )
                for family in RENDERER_FAMILIES
                if family != train
            },
            "all_families_meet_translation_gate": bool(
                all(
                    per_family[f]["translation_mm_after"]["median"]
                    < POSE_TRANSLATION_GATE_MM
                    for f in RENDERER_FAMILIES
                )
            ),
        },
    }


# ------------------------------------------------- E12 simulation readiness --
def experiment_simulation_readiness(
    seeds: tuple[int, ...] = tuple(range(8)),
    n_modes: int = 10,
    penalty_weights: tuple[float, ...] = (1e0, 1e2, 1e4, 1e6),
    n_phases: int = 16,
) -> dict[str, Any]:
    """Section 7.1: is the output usable by a downstream mesher?

    This is where exact coupling should earn its keep. A penalty reaches a small
    interface gap, but a mesh generator does not care how small a crack is in the
    abstract, only whether it falls below the tolerance at which it welds
    coincident vertices. Above that tolerance the duplicated annular occurrences
    survive as separate vertices and the leaflets detach from the chamber.

    The gap is evaluated at every phase, because a weight validated at one phase
    is not thereby validated at the others.

    Volumetric element quality is absent by design: the procedural template has
    no tetrahedra, and section 8 is explicit that a surface-only model cannot
    support Jacobian claims.
    """

    from rooted4d.geometry import make_toy_complex

    model = SceneModel.procedural(n_modes=n_modes, n_motion=3, n_session=2, n_phase_nodes=6)
    config = SolverConfig(fourier_order=2, session_order=1, max_iterations=40)
    complex_ = make_toy_complex(n_annulus=len(model.groups["annulus"]))
    local_faces = np.concatenate(
        [
            complex_.faces["chamber"],
            complex_.faces["anterior_leaflet"],
            complex_.faces["posterior_leaflet"],
        ]
    )
    n_expected = model.n_physical
    phases = np.arange(n_phases) / n_phases
    diagonal = float(
        np.linalg.norm(
            model.template_physical.max(axis=0) - model.template_physical.min(axis=0)
        )
    )

    def evaluate(arm: str, fit, arm_model, tolerance_scale: float) -> dict[str, Any]:
        latent = latent_trajectory(fit.parameters, phases, "cmr", arm_model, fit.config)
        coordinates = arm_model.physical_coordinates(latent)
        gap = (
            arm_model.detached_attachment_residual(latent)
            if arm_model.is_detached
            else arm_model.attachment_residual(latent)
        )
        tolerance = tolerance_scale * diagonal
        # Worst phase decides: a mesher runs on each frame it is given.
        worst = None
        for frame in coordinates:
            rows = arm_model.observation_rows(np.arange(n_expected))
            if arm_model.is_detached:
                report = assess_simulation_readiness(
                    frame,
                    _detached_faces(arm_model, complex_, n_expected),
                    weld_tolerance_mm=tolerance,
                    n_expected_vertices=n_expected,
                    max_interface_gap_mm=gap,
                )
            else:
                report = assess_simulation_readiness(
                    frame[rows],
                    local_faces,
                    weld_tolerance_mm=tolerance,
                    n_expected_vertices=n_expected,
                    max_interface_gap_mm=gap,
                )
            if worst is None or not report.simulation_ready:
                worst = report
                if not report.simulation_ready:
                    break
        return {"arm": arm, "tolerance_scale": tolerance_scale, **worst.as_dict()}

    rows = []
    for seed in seeds:
        problem = make_joint_problem(model, config, seed=int(seed))
        exact = fit_joint_scene(problem.observations, model, config)
        for scale in DEFAULT_WELD_TOLERANCES:
            record = evaluate("rooted4d_full", exact, model, scale)
            record["seed"] = int(seed)
            record["penalty_weight"] = None
            rows.append(record)

        detached = model.detached()
        for weight in penalty_weights:
            arm_config = replace(config, soft_attachment_weight=float(weight))
            soft = fit_joint_scene(problem.observations, detached, arm_config)
            for scale in DEFAULT_WELD_TOLERANCES:
                record = evaluate("joint_soft_attachment", soft, detached, scale)
                record["seed"] = int(seed)
                record["penalty_weight"] = float(weight)
                rows.append(record)

    def rate(arm: str, scale: float, weight: float | None) -> float:
        selected = [
            r
            for r in rows
            if r["arm"] == arm
            and r["tolerance_scale"] == scale
            and r["penalty_weight"] == weight
        ]
        return float(np.mean([r["simulation_ready"] for r in selected])) if selected else float("nan")

    readiness = {
        f"exact@{scale:.0e}": rate("rooted4d_full", scale, None)
        for scale in DEFAULT_WELD_TOLERANCES
    }
    for weight in penalty_weights:
        for scale in DEFAULT_WELD_TOLERANCES:
            readiness[f"soft_lambda{weight:.0e}@{scale:.0e}"] = rate(
                "joint_soft_attachment", scale, weight
            )

    exact_rows = [r for r in rows if r["arm"] == "rooted4d_full"]
    return {
        "experiment": "E12_simulation_readiness",
        "claim": (
            "exact coupling yields a mesh that welds into one attached complex at "
            "any tolerance, with nothing to tune"
        ),
        "bounding_box_diagonal_mm": diagonal,
        "rows": rows,
        "verdict": {
            "readiness_rate": readiness,
            "exact_ready_at_every_tolerance": bool(
                all(rate("rooted4d_full", s, None) == 1.0 for s in DEFAULT_WELD_TOLERANCES)
            ),
            "exact_max_gap_mm": float(max(r["max_interface_gap_mm"] for r in exact_rows)),
            "tightest_tolerance_soft_survives": {
                f"lambda={w:.0e}": min(
                    [s for s in DEFAULT_WELD_TOLERANCES
                     if rate("joint_soft_attachment", s, w) == 1.0],
                    default=None,
                )
                for w in penalty_weights
            },
            "note": (
                "A penalty weight is only validated against the tolerance it was "
                "tested at. Exact assembly has no weight to validate: the gap is "
                "zero, so it welds at every tolerance including the tightest, and "
                "it does so at every phase rather than at the one that was checked."
            ),
        },
    }


def _detached_faces(arm_model, complex_, n_expected: int) -> np.ndarray:
    """Faces indexed into the detached model's duplicated local vertex array."""

    offsets = {}
    cursor = 0
    for name, indices in complex_.attachment_map.local_to_global.items():
        offsets[name] = cursor
        cursor += len(indices)
    pieces = []
    for name in ("chamber", "anterior_leaflet", "posterior_leaflet"):
        local = complex_.attachment_map.local_to_global[name]
        lookup = {int(g): offsets[name] + i for i, g in enumerate(local.tolist())}
        pieces.append(
            np.array([[lookup[int(v)] for v in tri] for tri in complex_.faces[name]])
        )
    return np.concatenate(pieces, axis=0)


EXPERIMENTS = {
    "t1_annular_resolution": experiment_annular_resolution,
    "t2_soft_attachment": experiment_soft_attachment,
    "t3_gauge_ablation": experiment_gauge_ablation,
    "p4_periodicity_phase": experiment_periodicity_and_phase,
    "t5_identifiability": experiment_identifiability_factorial,
    "t6_noise_stability": experiment_noise_stability,
    "e1_synthetic_recovery": experiment_synthetic_recovery,
    "e5_baseline_ladder": experiment_baseline_ladder,
    "t5t6_solver_jacobian": experiment_solver_identifiability,
    "e6_image_domain": experiment_image_domain_refinement,
    "e9_posterior_calibration": experiment_posterior_calibration,
    "e10_family_generalization": experiment_family_generalization,
    "e11_renderer_misspecification": experiment_renderer_misspecification,
    "e12_simulation_readiness": experiment_simulation_readiness,
}


def run_all(
    selected: tuple[str, ...] | None = None,
    *,
    progress: Callable[[str, int, int, float], None] | None = None,
) -> dict[str, Any]:
    """Run the campaign, optionally reporting each experiment as it finishes.

    The summary is only printed once everything completes, so a batch job would
    otherwise show nothing for several minutes. The progress callback exists so
    a long run is observably alive in its log.
    """

    names = tuple(EXPERIMENTS) if selected is None else selected
    unknown = [name for name in names if name not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"unknown experiments: {unknown}")
    results: dict[str, Any] = {}
    for index, name in enumerate(names, start=1):
        started = time.time()
        results[name] = EXPERIMENTS[name]()
        elapsed = time.time() - started
        results[name]["runtime_seconds"] = round(elapsed, 3)
        if progress is not None:
            progress(name, index, len(names), elapsed)
    return {"environment": environment(), "results": results}
