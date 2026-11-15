r"""ActivePlaneTwin experiment: does uncertainty-driven plane selection pay?

Four arms, all scored on the same posterior variance of the same reported
quantities, so the only difference is which planes get acquired and in what
order.

``greedy_uncertainty``
    The proposed method. At each step, pick the plane maximizing expected
    variance reduction per unit cost, conditioned on what has already been
    acquired for this subject.

``protocol_order``
    The conventional short-axis stack in its usual order.

``random_order``
    Ordering ablation: shows how much of any advantage is simply from choosing
    good planes versus from choosing them in a good order.

``fixed_learned_pair``
    The comparison that matters. AcquisitionFocus learns a small number of view
    orientations offline and applies them to every subject. This arm picks the
    best fixed pair on a training split by average utility, then applies it
    unchanged to held-out subjects. If per-subject adaptation buys nothing, this
    arm matches the greedy one, and the contribution does not stand.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

import numpy as np

from rooted4d.geometry import make_toy_complex
from rooted4d.inference import SceneModel
from rooted4d.motion import periodic_basis

from .planes import (
    PlaneCandidate,
    long_axis_candidates,
    oblique_candidates,
    short_axis_stack,
)
from .quantities import clinical_quantities, quantity_gradients, relative_weights
from .selection import (
    fixed_order_plan,
    greedy_plan,
    plane_jacobian,
    posterior_precision,
    quantity_variance,
    score_candidates,
)

QUANTITIES = ("edv_mm3", "esv_mm3", "ejection_fraction")


@dataclass(frozen=True)
class Subject:
    """One simulated patient: latent state, gradients, and per-plane Jacobians."""

    seed: int
    parameters: np.ndarray
    gradients: np.ndarray
    weights: np.ndarray
    values: dict[str, float]
    jacobians: list[np.ndarray]


def build_scene(n_modes: int = 10, n_motion: int = 3, fourier_order: int = 2):
    """Scene, chamber faces, and the latent layout the selector operates on."""

    model = SceneModel.procedural(
        n_modes=n_modes, n_motion=n_motion, n_session=2, n_phase_nodes=6
    )
    chamber = make_toy_complex(n_annulus=len(model.groups["annulus"])).faces["chamber"]
    return model, chamber, fourier_order


def geometry_factory(model: SceneModel, chamber: np.ndarray, fourier_order: int):
    """Return ``geometry(theta, phase) -> (vertices, faces)`` for this scene."""

    n_identity = model.n_modes
    n_motion = model.bases.n_motion

    def geometry(theta: np.ndarray, phase: float):
        identity = theta[:n_identity]
        motion = theta[n_identity:].reshape(2 * fourier_order, n_motion)
        shared = periodic_basis(np.array([float(phase)]), fourier_order) @ motion
        latent = identity + shared @ model.bases.motion.T
        return model.physical_coordinates(latent)[0], chamber

    return geometry


def candidate_set(
    model: SceneModel,
    *,
    n_short_axis: int = 7,
    n_long_axis: int = 4,
    n_oblique: int = 4,
) -> list[PlaneCandidate]:
    """Short-axis stack, long-axis views, and oblique planes.

    The oblique planes matter: without them the comparison would only be about
    the order of a fixed protocol, and the selector could never demonstrate
    choosing something the protocol does not contain.
    """

    return (
        short_axis_stack(model.template_physical, n_planes=n_short_axis)
        + long_axis_candidates(model.template_physical, n_rotations=n_long_axis)
        + oblique_candidates(
            model.template_physical, n_rotations=n_oblique, tilts=(0.5,)
        )
    )


def make_subject(
    seed: int,
    model: SceneModel,
    chamber: np.ndarray,
    fourier_order: int,
    candidates: Sequence[PlaneCandidate],
    phases: np.ndarray,
    *,
    identity_scale: float = 0.8,
    motion_amplitude: float = 10.0,
) -> Subject:
    """Draw a subject whose ejection fraction is physiological.

    The motion amplitude is calibrated rather than arbitrary: at the default the
    scene produces roughly 52 mL end-diastolic, 25 mL end-systolic and an
    ejection fraction near 0.52. An amplitude that leaves ejection fraction at a
    few percent would make its gradient numerically negligible and the selector
    would silently optimize volumes alone.
    """

    rng = np.random.default_rng(seed)
    n_identity = model.n_modes
    n_motion = model.bases.n_motion
    parameters = np.concatenate(
        [
            rng.normal(0.0, identity_scale, n_identity),
            rng.normal(0.0, 1.0, 2 * fourier_order * n_motion) * motion_amplitude,
        ]
    )
    geometry = geometry_factory(model, chamber, fourier_order)
    gradients, values = quantity_gradients(geometry, parameters, phases, names=QUANTITIES)
    weights = relative_weights(values, QUANTITIES)

    jacobians = []
    for candidate in candidates:
        blocks = []
        for phase in phases:
            block = plane_jacobian(
                lambda theta, p=float(phase): geometry(theta, p),
                parameters,
                candidate,
                samples_per_segment=1,
            )
            if block.size:
                blocks.append(block)
        jacobians.append(
            np.concatenate(blocks, axis=0)
            if blocks
            else np.zeros((0, parameters.size))
        )
    return Subject(
        seed=seed,
        parameters=parameters,
        gradients=gradients,
        weights=weights,
        values=values,
        jacobians=jacobians,
    )


def ef_standard_deviation(precision: np.ndarray, gradients: np.ndarray) -> float:
    """Posterior standard deviation of ejection fraction alone.

    Selection uses a weighted objective over all reported quantities, but the
    stopping rule is stated on ejection fraction in its own units, because "stop
    when EF is known to within two percentage points" is a target a radiologist
    can actually set. This is the same contract CUTE-MRI applies to sampling
    density.
    """

    row = gradients[QUANTITIES.index("ejection_fraction") : QUANTITIES.index("ejection_fraction") + 1]
    return float(np.sqrt(max(quantity_variance(precision, row), 0.0)))


def _cost_to_ef_target(
    steps,
    candidates: Sequence[PlaneCandidate],
    jacobians: Sequence[np.ndarray],
    prior_precision: np.ndarray,
    gradients: np.ndarray,
    noise_variance: float,
    target_sd: float,
) -> tuple[float, float]:
    """Cost at which the ejection-fraction target is met, and the final SD."""

    by_name = {candidate.name: index for index, candidate in enumerate(candidates)}
    precision = np.array(prior_precision, dtype=np.float64, copy=True)
    reached = float("inf")
    standard_deviation = ef_standard_deviation(precision, gradients)
    for step in steps:
        precision = posterior_precision(
            precision, [jacobians[by_name[step.chosen]]], noise_variance
        )
        standard_deviation = ef_standard_deviation(precision, gradients)
        if standard_deviation <= target_sd and not np.isfinite(reached):
            reached = step.cumulative_cost
    return reached, standard_deviation


def learn_fixed_pair(
    subjects: Sequence[Subject],
    candidates: Sequence[PlaneCandidate],
    prior_precision: np.ndarray,
    noise_variance: float,
) -> tuple[int, int]:
    """Best fixed pair of planes averaged over training subjects.

    This stands in for an offline-optimized view configuration. It is chosen once
    and applied unchanged, which is the property being tested, not the particular
    optimizer used to find it.
    """

    best: tuple[float, tuple[int, int]] | None = None
    for first, second in combinations(range(len(candidates)), 2):
        total = 0.0
        for subject in subjects:
            precision = posterior_precision(
                prior_precision,
                [subject.jacobians[first], subject.jacobians[second]],
                noise_variance,
            )
            total += quantity_variance(precision, subject.gradients, subject.weights)
        if best is None or total < best[0]:
            best = (total, (first, second))
    assert best is not None
    return best[1]


def run_active_plane_experiment(
    seeds: tuple[int, ...] = tuple(range(10)),
    *,
    n_phases: int = 6,
    budget: int = 8,
    prior_precision_scale: float = 1e-2,
    noise_mm: float = 1.5,
    target_ef_sd: float = 0.02,
) -> dict[str, Any]:
    """Compare the four arms on cost to reach an ejection-fraction precision target.

    ``noise_mm`` is contour-level error, so it stands for segmentation error
    rather than image noise; 1.5 mm is a plausible figure and the results are
    reported against it rather than against an optimistic one.

    One modelling caveat applies to every arm equally. Samples along a contour
    are treated as independent measurements, which overstates how much a single
    plane informs the latent, so the absolute plane counts here are optimistic.
    The comparison between arms is unaffected, because all arms inherit the same
    assumption; a correlated contour-noise model would raise every count together.
    """

    model, chamber, fourier_order = build_scene()
    candidates = candidate_set(model)
    phases = np.linspace(0.0, 1.0, n_phases, endpoint=False)
    subjects = [
        make_subject(int(seed), model, chamber, fourier_order, candidates, phases)
        for seed in seeds
    ]

    dimension = subjects[0].parameters.size
    prior = np.eye(dimension) * prior_precision_scale
    noise_variance = noise_mm**2

    # The fixed configuration is learned on the first half and applied to the
    # second, so it is never evaluated on a subject it was tuned on.
    split = max(1, len(subjects) // 2)
    training, evaluation = subjects[:split], subjects[split:]
    fixed_pair = learn_fixed_pair(training, candidates, prior, noise_variance)

    rows = []
    for subject in evaluation:
        prior_variance = quantity_variance(prior, subject.gradients, subject.weights)
        prior_ef_sd = ef_standard_deviation(prior, subject.gradients)

        plans = {
            "greedy_uncertainty": greedy_plan(
                candidates,
                subject.jacobians,
                prior,
                subject.gradients,
                noise_variance,
                n_planes=budget,
                weights=subject.weights,
            ),
            "protocol_order": fixed_order_plan(
                candidates,
                subject.jacobians,
                prior,
                subject.gradients,
                noise_variance,
                order=list(range(budget)),
                weights=subject.weights,
            ),
            "random_order": fixed_order_plan(
                candidates,
                subject.jacobians,
                prior,
                subject.gradients,
                noise_variance,
                order=list(
                    np.random.default_rng(1000 + subject.seed).permutation(
                        len(candidates)
                    )[:budget]
                ),
                weights=subject.weights,
            ),
            "fixed_learned_pair": fixed_order_plan(
                candidates,
                subject.jacobians,
                prior,
                subject.gradients,
                noise_variance,
                order=list(fixed_pair),
                weights=subject.weights,
            ),
        }
        for arm, steps in plans.items():
            cost, final_sd = _cost_to_ef_target(
                steps,
                candidates,
                subject.jacobians,
                prior,
                subject.gradients,
                noise_variance,
                target_ef_sd,
            )
            rows.append(
                {
                    "arm": arm,
                    "seed": subject.seed,
                    "ejection_fraction": subject.values["ejection_fraction"],
                    "prior_variance": prior_variance,
                    "prior_ef_sd": prior_ef_sd,
                    "target_ef_sd": target_ef_sd,
                    "cost_to_target": cost,
                    "final_ef_sd": final_sd,
                    "final_variance": steps[-1].variance_after if steps else float("inf"),
                    "final_cost": steps[-1].cumulative_cost if steps else 0.0,
                    "chosen": [step.chosen for step in steps],
                }
            )

    def column(arm: str, key: str) -> np.ndarray:
        return np.array([row[key] for row in rows if row["arm"] == arm], dtype=np.float64)

    arms = ("greedy_uncertainty", "protocol_order", "random_order", "fixed_learned_pair")
    per_arm = {}
    for arm in arms:
        cost = column(arm, "cost_to_target")
        finite = cost[np.isfinite(cost)]
        per_arm[arm] = {
            "median_cost_to_target": float(np.median(finite)) if finite.size else None,
            "reached_target_fraction": float(np.mean(np.isfinite(cost))),
            "median_final_variance": float(np.median(column(arm, "final_variance"))),
            "median_final_ef_sd": float(np.median(column(arm, "final_ef_sd"))),
        }

    chosen = [row["chosen"] for row in rows if row["arm"] == "greedy_uncertainty"]
    oblique_used = float(
        np.mean([any(name.startswith("OB") for name in names) for names in chosen])
    ) if chosen else 0.0
    distinct_plans = len({tuple(names) for names in chosen})

    return {
        "experiment": "E13_active_plane_selection",
        "claim": (
            "sequential plane selection driven by uncertainty in reported "
            "quantities reaches a precision target at lower acquisition cost than "
            "a protocol order or an offline-fixed view configuration"
        ),
        "quantities": list(QUANTITIES),
        "n_candidates": len(candidates),
        "fixed_pair": [candidates[index].name for index in fixed_pair],
        "rows": rows,
        "verdict": {
            "per_arm": per_arm,
            "greedy_beats_protocol": bool(
                per_arm["greedy_uncertainty"]["median_cost_to_target"] is not None
                and per_arm["protocol_order"]["median_cost_to_target"] is not None
                and per_arm["greedy_uncertainty"]["median_cost_to_target"]
                < per_arm["protocol_order"]["median_cost_to_target"]
            ),
            "greedy_beats_fixed_configuration": bool(
                per_arm["greedy_uncertainty"]["median_cost_to_target"] is not None
                and per_arm["fixed_learned_pair"]["median_cost_to_target"] is not None
                and per_arm["greedy_uncertainty"]["median_cost_to_target"]
                < per_arm["fixed_learned_pair"]["median_cost_to_target"]
            ),
            "fraction_of_plans_using_an_oblique_plane": oblique_used,
            "distinct_plans_across_subjects": distinct_plans,
            "n_evaluation_subjects": len(evaluation),
            "note": (
                "Per-subject adaptation is the claim, so the number of distinct "
                "plans matters as much as the cost: if every subject received the "
                "same plan, an offline configuration would do just as well and the "
                "sequential machinery would be unnecessary."
            ),
        },
    }
