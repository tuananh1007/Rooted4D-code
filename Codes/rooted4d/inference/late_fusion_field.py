r"""Late fusion with E116's Gaussian-kernel residual field (E119).

Pre-registered in ``Docs/E119_LATE_FUSION_FIELD_PREREGISTRATION_2026-11-13.md``.
Late fusion's two reconstructions and their trajectory registration are kept as
they are (``late_fusion_trajectory``): its pose, timing and held-out masks. The
residual field of ``residual_field.py`` (Gaussian kernel on the physical
vertices, the solver's periodic basis in time, posterior mean in closed form,
scales in units of a 1 mm noise) is then added to them, in two ways:

``session``
    One field per reconstruction, each from its own modality's residuals: the
    CMR field on the CMR fit, in its frame and at the CMR phases; the TEE field
    on the TEE fit, in the TEE fit's model frame and at its own phases. Neither
    reconstruction sees the other's evidence, as in late fusion itself.
``shared``
    One field in the CMR fit's frame from both modalities' residuals, after the
    registration: the TEE fit's residuals are rotated into the CMR frame by the
    registration's rotation and placed at the CMR phases the registration maps
    the TEE fit's phases onto. The TEE reconstruction reads the field back
    through the same maps.

Held-out CMR error is the CMR fit plus its field, held-out TEE error the TEE fit
plus its field placed by the TEE fit's pose, on the masks late fusion is scored
on. The attachment gap is late fusion's registered gap with each side's field
added: a field changes each reconstruction's annulus, but it cannot make the two
reconstructions share one. With the shared field the two annuli move by the same
displacement, so the gap is late fusion's.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .baselines import ArmResult, late_fusion_trajectory_fit
from .initialization import StartingAlignment
from .joint import predict
from .problem import JointProblem
from .published_fusion import _failure, _mean_and_p95
from .recent_fusion import GAP_PHASES
from .residual_field import (
    FieldEvidence,
    config_name,
    evaluate_field,
    field_evidence,
    field_scales,
    gaussian_factor,
    normal_equations,
    solve_field,
)
from .trajectory_registration import inverse_phase, session_free_trajectory

ARM = "late_fusion_residual_field"


def late_fusion_evidence(cmr_fit, tee_fit, registration) -> dict[str, FieldEvidence]:
    """Each fit's own residuals (``cmr``, ``tee``) and the TEE residuals in the CMR fit's frame (``tee_in_cmr``)."""

    cmr = field_evidence(cmr_fit)["cmr"]
    tee = field_evidence(tee_fit)["tee"]
    # The registration maps the TEE fit's model frame into the CMR fit's: v -> R v, a row v as v @ R^T.
    rotation = np.asarray(registration.transform.rotation, dtype=np.float64)
    in_cmr = FieldEvidence(
        phases=np.asarray(registration.warp.evaluate(tee.phases), dtype=np.float64),
        rows=tee.rows,
        residuals=tee.residuals @ rotation.T)
    return {"cmr": cmr, "tee": tee, "tee_in_cmr": in_cmr}


def run_late_fusion_residual_field(
    problem: JointProblem,
    config,
    *,
    alignment: StartingAlignment | None,
    modality_overrides: dict | None,
    configs: list[tuple[str, float, float, float]],
) -> ArmResult:
    """``configs`` lists (sharing, static scale, dynamic scale, width); the first is the arm's."""

    if alignment is None:
        return _failure(ARM, "needs_the_input_derived_start")
    base, (cmr_fit, tee_fit, cmr_config, tee_config, registration) = late_fusion_trajectory_fit(
        problem, config, alignment=alignment, modality_overrides=modality_overrides)
    if registration is None or not base.succeeded:
        return replace(base, baseline_id=ARM)
    model, observations = problem.model, problem.observations
    order = config.fourier_order
    evidence = late_fusion_evidence(cmr_fit, tee_fit, registration)
    rotation = np.asarray(registration.transform.rotation, dtype=np.float64)
    tee_rotation = np.asarray(tee_fit.parameters.transform().rotation, dtype=np.float64)

    # Late fusion's own predictions on its held-out masks, by the path its errors are scored on.
    cmr_times = np.asarray(observations.heldout_cmr_times, dtype=np.float64)
    cmr_rows = model.observation_rows(np.asarray(observations.heldout_cmr_vertices))
    cmr_base, _ = predict(cmr_fit.parameters, observations, model, cmr_config,
                          cmr_times=cmr_times, cmr_vertices=observations.heldout_cmr_vertices,
                          tee_times=np.empty(0))
    tee_times = np.asarray(observations.heldout_tee_times, dtype=np.float64)
    tee_phases = np.asarray(tee_fit.parameters.warp().evaluate(tee_times), dtype=np.float64)
    tee_rows = model.observation_rows(np.asarray(observations.heldout_tee_vertices))
    _, tee_base = predict(tee_fit.parameters, observations, model, tee_config,
                          cmr_times=np.empty(0), tee_times=tee_times,
                          tee_vertices=observations.heldout_tee_vertices)
    # Both annuli over the cycle, as ``registered_interface_gap`` reads them.
    annulus = np.sort(model.groups["annulus"])
    annulus_rows = model.observation_rows(annulus)
    moving_phases = inverse_phase(registration.warp, GAP_PHASES)
    chamber = session_free_trajectory(cmr_fit.parameters, model, cmr_config, GAP_PHASES, annulus)
    valve = session_free_trajectory(tee_fit.parameters, model, cmr_config, moving_phases, annulus)

    def scored(cmr_field: np.ndarray, tee_field: np.ndarray, shared: bool) -> dict:
        """``tee_field`` is in the CMR frame at CMR phases when ``shared``, else in the TEE fit's frame at its phases."""

        out = {}
        if len(cmr_times):
            out["heldout_cmr_mm"], out["heldout_cmr_p95_mm"] = _mean_and_p95(
                cmr_base + evaluate_field(cmr_field, cmr_times, order)[:, cmr_rows],
                observations.heldout_cmr_points)
        if len(tee_times):
            if shared:
                at = np.asarray(registration.warp.evaluate(tee_phases), dtype=np.float64)
                displacement = evaluate_field(tee_field, at, order)[:, tee_rows] @ rotation
            else:
                displacement = evaluate_field(tee_field, tee_phases, order)[:, tee_rows]
            out["heldout_tee_mm"], out["heldout_tee_p95_mm"] = _mean_and_p95(
                tee_base + displacement @ tee_rotation.T, observations.heldout_tee_points)
        chamber_side = chamber + evaluate_field(cmr_field, GAP_PHASES, order)[:, annulus_rows]
        if shared:
            valve_moving = valve + evaluate_field(tee_field, GAP_PHASES, order)[:, annulus_rows] @ rotation
        else:
            valve_moving = valve + evaluate_field(tee_field, moving_phases, order)[:, annulus_rows]
        valve_side = registration.transform.apply(valve_moving)
        out["attachment_residual_mm"] = float(np.abs(valve_side - chamber_side).max())
        out["field_rms_mm"] = float(np.sqrt(np.mean((evaluate_field(cmr_field, GAP_PHASES, order) ** 2).sum(-1))))
        out["tee_field_rms_mm"] = float(np.sqrt(np.mean((evaluate_field(tee_field, GAP_PHASES, order) ** 2).sum(-1))))
        return out

    by_config, systems = {}, {}
    for sharing, static, dynamic, width in configs:
        name = config_name(sharing, static, dynamic, width)
        if width not in systems:
            factor = gaussian_factor(model.template_physical, width)
            systems[width] = (factor, {
                "shared": normal_equations([evidence["cmr"], evidence["tee_in_cmr"]], factor, order),
                "cmr": normal_equations([evidence["cmr"]], factor, order),
                "tee": normal_equations([evidence["tee"]], factor, order)})
        factor, equations = systems[width]
        scales = field_scales(static, dynamic, order)
        try:
            if sharing == "shared":
                field = solve_field(*equations["shared"], factor, scales)
                by_config[name] = scored(field, field, shared=True)
            elif sharing == "session":
                by_config[name] = scored(solve_field(*equations["cmr"], factor, scales),
                                         solve_field(*equations["tee"], factor, scales), shared=False)
            else:
                raise ValueError(f"unknown sharing: {sharing}")
        except np.linalg.LinAlgError:
            by_config[name] = {"heldout_cmr_mm": float("nan"), "heldout_tee_mm": float("nan")}

    primary = by_config[config_name(*configs[0])]
    finite = all(np.isfinite(primary.get(k, np.nan)) for k in ("heldout_cmr_mm", "heldout_tee_mm"))
    return replace(
        base,
        baseline_id=ARM,
        succeeded=finite,
        failure_reason=None if finite else "non_finite_score",
        attachment_residual_mm=primary.get("attachment_residual_mm", float("nan")),
        heldout_cmr_mm=primary.get("heldout_cmr_mm"),
        heldout_tee_mm=primary.get("heldout_tee_mm"),
        heldout_cmr_p95_mm=primary.get("heldout_cmr_p95_mm"),
        heldout_tee_p95_mm=primary.get("heldout_tee_p95_mm"),
        extras={"by_config": by_config, "configurations": [config_name(*c) for c in configs],
                "late_fusion": {"heldout_cmr_mm": base.heldout_cmr_mm, "heldout_tee_mm": base.heldout_tee_mm,
                                "attachment_residual_mm": base.attachment_residual_mm,
                                "rotation_deg": base.rotation_deg, "translation_mm": base.translation_mm,
                                "phase_map_cycles": base.phase_map_cycles},
                "modes": {f"{w:g}": int(f.shape[1]) for w, (f, _) in systems.items()}},
    )
