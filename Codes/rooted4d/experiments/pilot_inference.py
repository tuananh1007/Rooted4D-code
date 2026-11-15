r"""The Direction 1 pilot's estimators: a robust baseline and the budgeted rule.

Both are wrappers that compose the residual the joint solver already minimizes,
so neither needs a change to the core solver. That matters for a reason beyond
convenience: an accuracy claim has to come from a rule with a *different
optimum*, not from a different way of reaching the same one. With separable
private variables and identical losses, priors and constraints,

    min_a  sum_m  min_{v_m} L_m(a, v_m)  =  min_{a, v}  sum_m L_m(a, v_m),

so profiling and Schur elimination reproduce the joint fit exactly. The budgeted
rule below is constrained, and its solution differs from the joint fit precisely
when a constraint is active -- which is the regime the pilot is about, where one
modality's evidence is degraded and left unconstrained it drags shared anatomy.

Contents:

*   :func:`modality_loss` -- per-modality fit loss, from ``predict`` alone. Not
    ``SceneFit.cost``, which folds in the priors, and not a slice of the residual
    vector, which would re-couple this to the observation-row bookkeeping.
*   :func:`reference_losses` -- ``L_m*``, what each modality attains fitting on
    its own evidence over the same frozen span.
*   :func:`robust_joint_fit` -- baseline 3. IRLS with a Huber weight on the
    observation rows only; the priors are not robustified, because the intent is
    to down-weight suspect *evidence*, not suspect beliefs.
*   :func:`budgeted_joint_fit` -- the frozen rule.

Nothing here may see ``problem.truth``, the corruption label, or the corruption
magnitude. Reliability has to come from solver-available quantities or not be
claimed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from rooted4d.inference.joint import (
    _OBSERVATION_ROWS,
    _levenberg_marquardt,
    _pack,
    _residuals,
    _unpack,
    Observations,
    SceneFit,
    SolverConfig,
    fit_joint_scene,
    initial_parameters,
    predict,
)
from rooted4d.inference.scene import SceneModel, SceneParameters

MODALITIES = ("cmr", "tee")
#: Huber cut in units of the robust scale estimate. 1.345 is the usual choice
#: for 95% efficiency at the Gaussian; it is tuned on development identities in
#: the pilot, not chosen per case.
HUBER_DELTA_SCALE = 1.345


def modality_loss(
    parameters: SceneParameters,
    observations: Observations,
    model: SceneModel,
    config: SolverConfig,
    modality: str,
) -> float:
    """Mean squared distance between predicted and observed points, in mm^2."""

    if modality not in MODALITIES:
        raise ValueError("modality must be one of %r, got %r" % (MODALITIES, modality))
    cmr_prediction, tee_prediction = predict(parameters, observations, model, config)
    prediction = cmr_prediction if modality == "cmr" else tee_prediction
    observed = observations.cmr_points if modality == "cmr" else observations.tee_points
    if prediction is None or np.asarray(observed).size == 0:
        return 0.0
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(observed, dtype=np.float64)
    return float((difference ** 2).sum(axis=-1).mean())


def reference_losses(
    observations: Observations,
    model: SceneModel,
    config: SolverConfig,
    *,
    start: SceneParameters | None = None,
) -> dict[str, float]:
    """``L_m*``: what each modality attains fitting on its own evidence.

    Fitted over the *same* span as every other arm, because the pilot isolates
    the inference rule and the contract freezes the representation. Only the
    evidence differs between these reference fits.

    Two corrections, both found because the reference stopped being a floor.

    *It must start where the joint fit starts.* Without ``start`` these fits
    began at the prior, and under a frozen-alignment configuration that means
    pose stays at zero while the joint fit is aligned at the truth. The
    benchmark's TEE frame is offset by roughly 7 to 11 mm, so ``L_tee*`` was
    measuring an uncorrected rigid offset rather than TEE's attainable fit.

    *The prior must not be stronger here than in the joint problem.* The prior
    weights are fixed, so fitting one modality alone leaves the same prior
    competing with roughly half the evidence, and the reference comes out
    over-shrunk. Scaling the prior weights by that modality's share of the
    observation rows restores the balance. Before this, the joint fit beat the
    "floor" on TEE (ratios 0.92 to 1.24); after it, the ratios are 1.30 to 1.68
    and the reference is a floor again.
    """

    counts = {
        "cmr": int(np.asarray(observations.cmr_points).size),
        "tee": int(np.asarray(observations.tee_points).size),
    }
    total = max(sum(counts.values()), 1)

    out: dict[str, float] = {}
    for modality in MODALITIES:
        share = counts[modality] / total
        arm = replace(
            config,
            use_cmr=(modality == "cmr"), use_tee=(modality == "tee"),
            identity_prior_weight=config.identity_prior_weight * share,
            motion_prior_weight=config.motion_prior_weight * share,
            session_prior_weight=config.session_prior_weight * share,
        )
        fit = fit_joint_scene(observations, model, arm, start=start)
        out[modality] = modality_loss(fit.parameters, observations, model, arm, modality)
    return out


def reliability_budgets(
    reference: dict[str, float], base: float, *, cap: float = 50.0
) -> dict[str, float]:
    r"""Turn each modality's own self-fit residual into its fidelity budget.

    The method is not told which modality is degraded, and must not be. What it
    has is ``L_m*``: how well each modality fits *its own* evidence over the
    frozen span. A modality whose own evidence is internally consistent is, on
    that evidence alone, the more trustworthy description of the shared anatomy,
    so it receives the tight budget and is protected; one that cannot fit itself
    receives a proportionally looser budget and is allowed to move.

        e_m = base * (L_m* / min_k L_k*)

    The best-fitting modality gets exactly ``base``; the others scale up with
    their own misfit. ``base`` is the only free quantity and is calibrated on
    development identities.

    This is the point at which the pilot can fail for a reason worth reporting.
    Within-modality consistency is a proxy for reliability, not reliability. A
    coherent corruption that the modality fits *well* -- a plausible, smooth,
    wrong anatomy -- raises no flag here, so its bias is protected rather than
    suppressed. Disagreement alone cannot identify which side is wrong, and this
    rule does not pretend otherwise.
    """

    if not reference:
        return {}
    smallest = max(min(reference.values()), 1e-12)
    return {
        m: float(min(base * (max(reference[m], 1e-12) / smallest), cap))
        for m in reference
    }


def _huber_weights(residual: np.ndarray, delta: float) -> np.ndarray:
    magnitude = np.abs(residual)
    return np.where(magnitude <= delta, 1.0, delta / np.maximum(magnitude, 1e-12))


def robust_joint_fit(
    observations: Observations,
    model: SceneModel,
    config: SolverConfig,
    *,
    start: SceneParameters | None = None,
    rounds: int = 3,
    delta_scale: float = HUBER_DELTA_SCALE,
) -> dict[str, Any]:
    """Baseline 3: exact joint fitting with an IRLS Huber loss on the evidence.

    The weights are recomputed from the current residual each round, and applied
    only to the observation rows. A robust loss that also down-weighted the
    prior rows would change what the model believes as well as what it trusts,
    and the comparison would no longer be about evidence handling.
    """

    template = initial_parameters(model, config) if start is None else start
    vector = _pack(template, model, config)
    weights: np.ndarray | None = None
    history: list[float] = []

    for _ in range(max(1, rounds)):

        def residual_function(value: np.ndarray, base=template) -> np.ndarray:
            full = _residuals(value, base, observations, model, config)
            if weights is None:
                return full
            scaled = full.copy()
            rows = min(len(weights), full.shape[0])
            scaled[:rows] = full[:rows] * np.sqrt(weights[:rows])
            return scaled

        vector, cost, _ = _levenberg_marquardt(residual_function, vector, config)
        history.append(float(cost))

        full = _residuals(vector, template, observations, model, config)
        n_observation = int(_OBSERVATION_ROWS["n"])
        evidence = full[:n_observation]
        if evidence.size == 0:
            break
        # Median absolute deviation, so the scale is not set by the outliers the
        # weight is meant to suppress.
        scale = 1.4826 * float(np.median(np.abs(evidence - np.median(evidence))))
        scale = max(scale, 1e-9)
        weights = np.ones(full.shape[0])
        weights[:n_observation] = _huber_weights(evidence, delta_scale * scale)

    parameters = _unpack(vector, template, model, config)
    return {
        "parameters": parameters,
        "cost_history": history,
        "losses": {m: modality_loss(parameters, observations, model, config, m)
                   for m in MODALITIES},
    }


@dataclass(frozen=True)
class BudgetedResult:
    """Outcome of the budgeted rule, including why it did what it did."""

    parameters: SceneParameters
    losses: dict[str, float]
    reference: dict[str, float]
    budgets: dict[str, float]
    #: Constraints within tolerance of their bound at the solution. An empty set
    #: means the rule returned the joint fit and must be reported as such rather
    #: than as a distinct estimator.
    active: tuple[str, ...]
    #: Smallest common scaling of the budgets that the achieved solution
    #: satisfies. ``<= 1`` is feasible.
    slack_scaling: float
    incompatible: bool
    #: Did the penalty actually move the solution away from its warm start? A
    #: violated constraint with an unmoved solution is a numerical failure of
    #: the penalty, not an infeasible problem. Reporting those as the same thing
    #: would let a decorative constraint be written up as a hard case.
    penalty_moved_solution: bool
    #: The violation was still falling at the top of the multiplier ladder. That
    #: is a solver setting, not a property of the problem, and must never be
    #: reported as an incompatible case.
    unconverged: bool
    #: ``"constrained"`` or ``"joint_fallback"``. Without this the fallback's
    #: losses can be misread as the constrained solution's, which is exactly how
    #: an unconverged run first looked like an infeasible one.
    returned: str
    violation_history: tuple[float, ...]
    cost_history: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "losses": dict(self.losses),
            "reference": dict(self.reference),
            "budgets": dict(self.budgets),
            "active": list(self.active),
            "slack_scaling": self.slack_scaling,
            "incompatible": self.incompatible,
            "penalty_moved_solution": self.penalty_moved_solution,
            "unconverged": self.unconverged,
            "returned": self.returned,
            "violation_history": list(self.violation_history),
            "cost_history": list(self.cost_history),
        }


def budgeted_joint_fit(
    observations: Observations,
    model: SceneModel,
    config: SolverConfig,
    budgets: dict[str, float],
    *,
    reference: dict[str, float] | None = None,
    start: SceneParameters | None = None,
    multipliers: tuple[float, ...] = (1.0, 1e2, 1e4, 1e6),
    tolerance: float = 1e-3,
) -> BudgetedResult:
    r"""Minimise the joint objective subject to ``L_m <= (1 + e_m) L_m*``.

    Solved as a penalty sequence on the constraint violation, warm started from
    the unconstrained joint fit so that the comparison isolates the constraint
    rather than the basin. The multiplier is raised until the constraints hold
    to tolerance or the incompatibility policy fires.

    The incompatibility policy is fixed in advance: compute the smallest common
    scaling ``t`` of the budgets that the achieved solution satisfies. If
    ``t <= 1`` the constraints are compatible. If not, the case is reported as
    incompatible with its ``t`` recorded, the unconstrained joint fit is
    returned, and the case still counts in every endpoint's denominator. Silent
    relaxation is what would make a failure look like a success.
    """

    if reference is None:
        reference = reference_losses(observations, model, config, start=start)
    bounds = {m: (1.0 + float(budgets.get(m, 0.0))) * float(reference[m])
              for m in MODALITIES}

    joint = fit_joint_scene(observations, model, config, start=start)
    template = joint.parameters
    vector = _pack(template, model, config)
    warm_start = vector.copy()
    history: list[float] = [float(joint.cost)]

    # The penalty has to be commensurate with the objective rather than an
    # absolute constant. The evidence residual carries thousands of rows and a
    # cost of order 1e3, while a violation is of order 1e-2 mm^2, so a
    # multiplier of 10 contributes about 3e-2 to a cost of 1.7e3: the constraint
    # is decorative and the solver returns its warm start untouched, which is
    # exactly what the first version of this function did. Violations are
    # therefore expressed relative to their own bound, which is dimensionless,
    # and the ladder is scaled by the objective itself.
    scale = max(float(joint.cost), 1.0)

    def relative_violations(parameters: SceneParameters) -> np.ndarray:
        out = []
        for modality in MODALITIES:
            loss = modality_loss(parameters, observations, model, config, modality)
            out.append(max(0.0, loss / max(bounds[modality], 1e-12) - 1.0))
        return np.asarray(out, dtype=np.float64)

    violations: list[float] = [float(relative_violations(template).max())]

    for multiplier in multipliers:

        def residual_function(
            value: np.ndarray, base=template, mu=multiplier
        ) -> np.ndarray:
            full = _residuals(value, base, observations, model, config)
            parameters = _unpack(value, base, model, config)
            return np.concatenate(
                [full, np.sqrt(mu * scale) * relative_violations(parameters)]
            )

        vector, cost, _ = _levenberg_marquardt(residual_function, vector, config)
        history.append(float(cost))
        violations.append(
            float(relative_violations(_unpack(vector, template, model, config)).max())
        )

    moved = float(np.max(np.abs(vector - warm_start))) if vector.size else 0.0
    penalty_moved_solution = moved > 1e-9

    parameters = _unpack(vector, template, model, config)
    losses = {m: modality_loss(parameters, observations, model, config, m)
              for m in MODALITIES}

    # t: the smallest common budget scaling this solution satisfies.
    ratios = []
    for modality in MODALITIES:
        budget = float(budgets.get(modality, 0.0))
        excess = losses[modality] / max(reference[modality], 1e-12) - 1.0
        ratios.append(excess / budget if budget > 0.0 else (0.0 if excess <= 0 else np.inf))
    slack = float(max(ratios)) if ratios else 0.0
    satisfied = slack <= 1.0 + tolerance

    # A violation still decreasing at the top of the ladder means the sequence
    # was too short. Calling that "incompatible" reports a solver setting as a
    # property of the problem, which is how the first version of this function
    # mislabelled a run that converged perfectly well one rung further up.
    still_falling = len(violations) >= 2 and violations[-1] < violations[-2] - 1e-6
    unconverged = (not satisfied) and still_falling
    incompatible = (not satisfied) and not still_falling

    active = tuple(
        m for m in MODALITIES
        if losses[m] >= bounds[m] * (1.0 - tolerance)
    )

    if not satisfied:
        # Fall back to the unconstrained fit, flagged. The rule does not get to
        # quietly widen its own budget, and the caller can tell an infeasible
        # case from one that simply needed a longer ladder.
        return BudgetedResult(
            parameters=joint.parameters,
            losses={m: modality_loss(joint.parameters, observations, model, config, m)
                    for m in MODALITIES},
            reference=dict(reference), budgets=dict(budgets),
            active=(), slack_scaling=slack, incompatible=incompatible,
            penalty_moved_solution=penalty_moved_solution,
            unconverged=unconverged, returned="joint_fallback",
            violation_history=tuple(violations), cost_history=tuple(history),
        )

    return BudgetedResult(
        parameters=parameters, losses=losses, reference=dict(reference),
        budgets=dict(budgets), active=active, slack_scaling=slack,
        incompatible=False, penalty_moved_solution=penalty_moved_solution,
        unconverged=False, returned="constrained",
        violation_history=tuple(violations), cost_history=tuple(history),
    )
