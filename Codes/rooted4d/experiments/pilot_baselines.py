r"""Baseline 1 of the Direction 1 pilot: ring projection, then refitting.

The contract asks for "independent fitting followed by weighted ring projection
and refitting" as the arm the budgeted rule has to beat. That is a different
object from ``run_late_fusion``, which reconstructs each modality separately and
then recovers a rigid transform and a circular warp between them: registration
relates two anatomies, where this projects their disagreement onto the structure
they share and refits one anatomy through it.

The procedure:

1.  Fit each modality on its own evidence over the same frozen span. Two
    parameter vectors, two anatomies.
2.  Evaluate each fit's annulus trajectory and combine them into one consensus
    ring, weighted by each modality's own fit quality. The weight is
    ``1 / L_m``, the inverse of that modality's mean squared residual on its own
    evidence -- within-modality consistency, which the solver has. It is never
    the corruption label and never a truth-derived weight.
3.  Refit a single parameter vector against both modalities' evidence with the
    annulus pinned to that consensus, so the ring carries the fusion and each
    modality's interior follows.

Step 3 is what makes this a baseline rather than a second joint fit: the shared
structure is fixed by a projection computed *before* the refit, so evidence from
one modality can only reach the other through the ring.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from rooted4d.inference.joint import (
    _levenberg_marquardt,
    _pack,
    _residuals,
    _unpack,
    Observations,
    SolverConfig,
    fit_joint_scene,
    latent_trajectory,
)
from rooted4d.inference.scene import SceneModel, SceneParameters

from .pilot_inference import MODALITIES, modality_loss

#: Phases at which the consensus ring is formed and enforced. The ring is a
#: trajectory, not a static shape, so pinning it at one phase would leave the
#: modalities free to disagree everywhere else in the cycle.
RING_PHASES = np.linspace(0.0, 1.0, 8, endpoint=False)


def _ring_trajectory(
    parameters: SceneParameters,
    model: SceneModel,
    config: SolverConfig,
    rows: np.ndarray,
) -> np.ndarray:
    """Annulus coordinates over the cycle, as (phases, ring vertices, 3).

    **Known limitation, not fixed.** The session is hard-coded to ``"cmr"``, so
    when this reads the *independent TEE* fit it evaluates that fit under the
    CMR session. The consensus ring is therefore not quite the weighted average
    of the two reconstructions as each would render itself. Recorded rather than
    repaired because the pilot is stopped and changing it would invalidate the
    saved stage-1 results; anyone reusing this arm should fix it first, and its
    reported numbers do not establish how ring projection performs in general.
    """

    latent = latent_trajectory(parameters, RING_PHASES, "cmr", model, config)
    return model.physical_coordinates(latent)[:, rows, :]


def ring_projection_refit(
    observations: Observations,
    model: SceneModel,
    config: SolverConfig,
    *,
    start: SceneParameters | None = None,
    ring_weight: float = 1e3,
) -> dict[str, Any]:
    """Independent fits, a weighted consensus ring, then one refit through it."""

    annulus = np.sort(model.groups["annulus"])
    rows = model.observation_rows(annulus)

    counts = {
        "cmr": int(np.asarray(observations.cmr_points).size),
        "tee": int(np.asarray(observations.tee_points).size),
    }
    total = max(sum(counts.values()), 1)

    independent: dict[str, SceneParameters] = {}
    losses: dict[str, float] = {}
    for modality in MODALITIES:
        # Same prior-to-evidence balance as reference_losses, for the same
        # reason: a single-modality fit under the full prior is over-shrunk,
        # and these losses set the consensus weights.
        share = counts[modality] / total
        arm = replace(
            config, use_cmr=(modality == "cmr"), use_tee=(modality == "tee"),
            identity_prior_weight=config.identity_prior_weight * share,
            motion_prior_weight=config.motion_prior_weight * share,
            session_prior_weight=config.session_prior_weight * share,
        )
        fit = fit_joint_scene(observations, model, arm, start=start)
        independent[modality] = fit.parameters
        losses[modality] = modality_loss(
            fit.parameters, observations, model, arm, modality
        )

    # Inverse-loss weights: a modality that fits its own evidence badly is, by
    # its own evidence, the less reliable description of the shared ring. This
    # is the only reliability signal the arm is allowed.
    weights = {m: 1.0 / max(losses[m], 1e-9) for m in MODALITIES}
    total = sum(weights.values())
    consensus = sum(
        (weights[m] / total) * _ring_trajectory(independent[m], model, config, rows)
        for m in MODALITIES
    )

    template = independent["cmr"] if start is None else start
    vector = _pack(template, model, config)
    scale = max(float(np.mean(consensus ** 2)), 1e-9)

    def residual_function(value: np.ndarray, base=template) -> np.ndarray:
        full = _residuals(value, base, observations, model, config)
        parameters = _unpack(value, base, model, config)
        deviation = _ring_trajectory(parameters, model, config, rows) - consensus
        return np.concatenate(
            [full, np.sqrt(ring_weight / scale) * deviation.reshape(-1)]
        )

    vector, cost, iterations = _levenberg_marquardt(residual_function, vector, config)
    parameters = _unpack(vector, template, model, config)
    final_ring = _ring_trajectory(parameters, model, config, rows)

    return {
        "parameters": parameters,
        "cost": float(cost),
        "iterations": int(iterations),
        "independent_losses": losses,
        "weights": {m: float(weights[m] / total) for m in MODALITIES},
        "ring_deviation_mm": float(
            np.sqrt(((final_ring - consensus) ** 2).sum(-1)).max()
        ),
        "losses": {
            m: modality_loss(parameters, observations, model, config, m)
            for m in MODALITIES
        },
    }


def independent_reference(
    observations: Observations,
    model: SceneModel,
    config: SolverConfig,
    *,
    start: SceneParameters | None = None,
) -> dict[str, Any]:
    """Unattached single-modality fits, recorded as a component-fidelity floor.

    Not an arm that produces one anatomy: two parameter vectors cannot be scored
    as a joint reconstruction. It exists so the pilot can say how much fidelity
    each modality gives up by being attached to the other at all.

    **Known inconsistency, not fixed.** This keeps the full prior weights, while
    ``reference_losses`` and ``ring_projection_refit`` scale the single-modality
    priors by observation share. The two are therefore not on the same footing,
    and the difference between this floor and an attached fit is not a pure cost
    of attachment. Recorded rather than repaired for the same reason as above.
    """

    out: dict[str, Any] = {}
    for modality in MODALITIES:
        arm = replace(
            config, use_cmr=(modality == "cmr"), use_tee=(modality == "tee")
        )
        fit = fit_joint_scene(observations, model, arm, start=start)
        errors = fit.heldout_errors()
        out[modality] = {
            "loss": modality_loss(fit.parameters, observations, model, arm, modality),
            "heldout": {k: float(v) for k, v in errors.items()},
        }
    return out
