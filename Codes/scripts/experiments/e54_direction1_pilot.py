"""E54: the bounded Direction 1 inference pilot, stage 1 (known alignment).

Runs the protocol frozen in ``Docs/DIRECTION1_PILOT_RULE_FROZEN_2026-09-17.md``
against the identity split frozen in
``results/benchmark/DIRECTION1_PILOT_SPLIT.json``. Two stages:

``R4D_STAGE=calibration``
    Sweeps the one free quantity, the budget base, on the **18 development
    identities** under nominal evidence only, and reports the largest base whose
    nominal cost stays inside a tolerance. Evaluation identities are never
    touched here.

``R4D_STAGE=evaluation``
    Runs the 12 evaluation identities under nominal, localized corruption in
    each modality, and missing views in each modality, with the session stress
    kept separate. Scores every arm on the sample's own held-out partition,
    which no condition modifies.

Alignment is frozen at the truth for this stage, and that is a declared
diagnostic rather than the proposed capability: it removes the pose and phase
search so the comparison is about evidence handling alone. Stage 2, under
unknown alignment from input-derived initialization, is what would make it a
capability claim, and it is not run here. Freezing alignment is not the same as
supplying truth-derived reliability weights, which the rule never receives.

Arms, all over one frozen span at 32 coefficients:

* ``independent`` -- unattached single-modality fits, a component-fidelity floor
  rather than a joint reconstruction.
* ``ring_refit``  -- baseline 1, independent fits then weighted ring projection.
* ``joint``       -- baseline 2, exact joint fitting.
* ``robust``      -- baseline 3, the same with an IRLS Huber loss on evidence.
* ``budgeted``    -- the frozen rule, with budgets derived from within-modality
  consistency, never from the corruption label.
"""

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from rooted4d.experiments.benchmark_eval import (
    benchmark_problem,
    fit_benchmark_model,
    sample_paths,
)
from rooted4d.experiments.pilot_baselines import independent_reference, ring_projection_refit
from rooted4d.experiments.pilot_conditions import apply_condition, heldout_is_untouched
from rooted4d.experiments.pilot_inference import (
    budgeted_joint_fit,
    modality_loss,
    reference_losses,
    reliability_budgets,
    robust_joint_fit,
)
from rooted4d.inference import SolverConfig, fit_joint_scene, initial_parameters
from rooted4d.inference.joint import SceneFit, _pack

ROOT = Path(os.environ.get("R4D_BENCHMARK", "data/derived/semisynthetic-real-v3"))
SPLIT = Path(os.environ.get(
    "R4D_SPLIT", "results/benchmark/DIRECTION1_PILOT_SPLIT.json"))
OUT = Path(os.environ.get("R4D_OUT", "results/benchmark"))
STAGE = os.environ.get("R4D_STAGE", "calibration")
N_MODES = int(os.environ.get("R4D_MODES", "32"))
#: Swept on development identities only.
BUDGET_BASES = [float(v) for v in
                os.environ.get("R4D_BASES", "0.02,0.05,0.10,0.25,0.50").split(",")]
#: Frozen before evaluation: the most nominal degradation a budget may cost,
#: relative to the unconstrained joint fit, on development identities.
NOMINAL_TOLERANCE = float(os.environ.get("R4D_NOMINAL_TOLERANCE", "0.02"))

PRIMARY_CONDITIONS = (
    ("nominal", None),
    ("localized", "cmr"),
    ("localized", "tee"),
    ("missing", "cmr"),
    ("missing", "tee"),
)
#: Reported separately: it tests a different assumption and the contract forbids
#: pooling it into the primary corruption result.
STRESS_CONDITIONS = (("session_shift", "cmr"), ("session_shift", "tee"))

CONFIG = SolverConfig(
    fourier_order=3, session_order=1,
    estimate_pose=False, estimate_phase_warp=False, estimate_phase_offset=False,
    max_iterations=30,
)


def aligned_start(model, truth):
    """Stage-1 start: alignment frozen at truth, everything else at the prior."""

    base = initial_parameters(model, CONFIG)
    return replace(
        base,
        pose_axis_angle=truth.pose_axis_angle,
        pose_translation=truth.pose_translation,
        phase_offset=truth.phase_offset,
        phase_logits=truth.phase_logits,
    )


def heldout_of(parameters, observations, model):
    """Held-out error for a bare parameter vector, via the existing endpoint."""

    fit = SceneFit(
        parameters=parameters, cost=0.0, iterations=0, stage_costs=(),
        attachment_residual=0.0, packed=_pack(parameters, model, CONFIG),
        model=model, config=CONFIG, observations=observations,
    )
    return {k: float(v) for k, v in fit.heldout_errors().items()}


def load_identities():
    index = {}
    for path in sample_paths(ROOT, "val"):
        with np.load(path, allow_pickle=True) as blob:
            index.setdefault(str(blob["identity_id"]), []).append(path)
    return index


def run_arms(observations, model, truth, base, *, arms=None):
    """Every arm on one conditioned sample. Returns per-arm held-out error."""

    start = aligned_start(model, truth)
    wanted = arms or ("independent", "ring_refit", "joint", "robust", "budgeted")
    out = {}

    reference = reference_losses(observations, model, CONFIG, start=start)
    budgets = reliability_budgets(reference, base)

    if "independent" in wanted:
        out["independent"] = independent_reference(
            observations, model, CONFIG, start=start)

    if "ring_refit" in wanted:
        result = ring_projection_refit(observations, model, CONFIG, start=start)
        out["ring_refit"] = {
            "heldout": heldout_of(result["parameters"], observations, model),
            "weights": result["weights"],
        }

    joint = None
    if {"joint", "budgeted"} & set(wanted):
        joint = fit_joint_scene(observations, model, CONFIG, start=start)
    if "joint" in wanted:
        out["joint"] = {
            "heldout": heldout_of(joint.parameters, observations, model),
            "attachment_residual_mm": float(joint.attachment_residual),
        }

    if "robust" in wanted:
        result = robust_joint_fit(observations, model, CONFIG, start=start, rounds=2)
        out["robust"] = {"heldout": heldout_of(result["parameters"], observations, model)}

    if "budgeted" in wanted:
        result = budgeted_joint_fit(
            observations, model, CONFIG, budgets, reference=reference, start=start)
        out["budgeted"] = {
            "heldout": heldout_of(result.parameters, observations, model),
            **result.as_dict(),
        }

    out["reference_losses"] = reference
    out["budgets"] = budgets
    return out


def evaluate_sample(path, model, base, conditions, arms=None):
    problem = benchmark_problem(path, model, CONFIG)
    rows = []
    for condition, modality in conditions:
        observations, record = apply_condition(problem.observations, condition, modality)
        if not heldout_is_untouched(problem.observations, observations):
            raise RuntimeError(
                "condition %r on %r modified the held-out partition" % (condition, modality))
        started = time.time()
        arms_out = run_arms(observations, model, problem.truth, base, arms=arms)
        rows.append({
            "condition": condition, "modality": modality, "record": record,
            "seconds": round(time.time() - started, 2), "arms": arms_out,
        })
    return rows


def main() -> int:
    model, _ = fit_benchmark_model(
        ROOT, n_modes=N_MODES, n_motion=4, n_session=2, n_phase_nodes=8)
    split = json.loads(SPLIT.read_text())
    index = load_identities()
    OUT.mkdir(parents=True, exist_ok=True)

    if STAGE == "calibration":
        identities = split["development_identities"]
        print("calibrating the budget base on %d development identities" % len(identities),
              flush=True)
        rows = []
        for base in BUDGET_BASES:
            costs = []
            for identity in identities:
                for path in index[identity]:
                    out = evaluate_sample(
                        path, model, base, (("nominal", None),),
                        arms=("joint", "budgeted"))[0]["arms"]
                    joint = out["joint"]["heldout"]
                    budgeted = out["budgeted"]["heldout"]
                    for key in ("heldout_cmr_mm", "heldout_tee_mm"):
                        if joint.get(key):
                            costs.append(budgeted[key] / joint[key] - 1.0)
            worst = float(np.max(costs)) if costs else float("nan")
            median = float(np.median(costs)) if costs else float("nan")
            rows.append({"base": base, "nominal_cost_median": median,
                         "nominal_cost_worst": worst,
                         "within_tolerance": bool(worst <= NOMINAL_TOLERANCE)})
            print("  base %.3f  nominal cost median %+.4f worst %+.4f  %s"
                  % (base, median, worst,
                     "ok" if worst <= NOMINAL_TOLERANCE else "exceeds tolerance"),
                  flush=True)
        allowed = [r for r in rows if r["within_tolerance"]]
        # Smallest passing base, matching the corrected section 3 of the frozen
        # rule: protection comes from a *tight* budget, so selecting the largest
        # passing value would deliberately choose the least protective setting.
        # This code selected the largest until 18 September; both rules pick 3.0
        # on the grid actually swept, so no reported result depends on it.
        chosen = min(allowed, key=lambda r: r["base"])["base"] if allowed else None
        payload = {"experiment": "E54_direction1_pilot_calibration",
                   "stage": "calibration", "modes": N_MODES,
                   "development_identities": len(identities),
                   "nominal_tolerance": NOMINAL_TOLERANCE,
                   "rows": rows, "chosen_base": chosen}
        (OUT / "E54_calibration.json").write_text(json.dumps(payload, indent=2) + "\n")
        print("\nchosen budget base:", chosen)
        return 0

    base = float(os.environ["R4D_BASE"])
    identities = split["evaluation_identities"]
    print("evaluating %d identities at frozen base %.3f" % (len(identities), base), flush=True)
    results = []
    for identity in identities:
        for path in index[identity]:
            started = time.time()
            rows = evaluate_sample(
                path, model, base, PRIMARY_CONDITIONS + STRESS_CONDITIONS)
            results.append({"identity": identity, "sample": path.name, "rows": rows})
            print("  %s %s  %.1fs" % (identity, path.name, time.time() - started), flush=True)
    payload = {"experiment": "E54_direction1_pilot_stage1", "stage": "evaluation",
               "modes": N_MODES, "budget_base": base,
               "alignment": "frozen at truth (declared stage-1 diagnostic)",
               "identities": len(identities), "results": results}
    (OUT / "E54_stage1.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nwrote", OUT / "E54_stage1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
