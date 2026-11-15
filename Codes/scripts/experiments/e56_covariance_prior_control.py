"""E56: the mandatory control -- does covariance-aware shrinkage close the gap?

The campaign memo makes this the first experiment, and it is the cheapest thing
that can end the campaign. The v3 capacity sweep shows added global capacity
helping CMR and hurting TEE, which is at least as consistent with an isotropic
prior that cannot tell strong shape directions from weak ones as it is with any
argument about partitioning anatomy. If simply shrinking each direction by the
variance the training identities show along it recovers the benefit, then the
structured representation has nothing left to explain, and the improved
estimator is kept regardless.

Design:

* The shape space is learned on the **train** split only, from identities rather
  than samples, which ``fit_benchmark_model`` already does.
* Everything is measured on the **validation** identities. Test is untouched.
* Both prior families get the same weight grid and the same selection rule, so
  neither is beaten at a setting it was never given.
* Alignment is frozen at the truth. That isolates the prior from the pose and
  phase search; it is a development diagnostic, not an end-to-end claim.
* The two variants of an identity are averaged before anything is summarised,
  because they are not two independent observations.

Selection rule, fixed before the run: the weight minimising the sum of the
median held-out CMR and TEE errors, applied identically to both arms. Both
components are reported, so a gain funded by one modality is visible rather than
hidden inside the summary.
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
from rooted4d.inference import SolverConfig, fit_joint_scene, initial_parameters
from rooted4d.inference.joint import SceneFit, _pack

ROOT = Path(os.environ.get("R4D_BENCHMARK", "data/derived/semisynthetic-real-v3"))
OUT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / "E56_covariance_prior_control.json"
MODES = [int(v) for v in os.environ.get("R4D_MODES", "32,40,48").split(",")]
#: The first sweep put both arms' optimum at the smallest weight tested, which
#: means the grid did not bracket it and each arm was judged at an edge. The
#: two arms turn out to want opposite ends of the range -- the isotropic prior
#: is pure cost and wants to be switched off, the covariance-aware one wants to
#: be strong -- so the grid has to span both, and does: the isotropic optimum
#: plateaus below 0.01 and the aware optimum is interior at 4.0.
WEIGHTS = [float(v) for v in
           os.environ.get("R4D_WEIGHTS", "0.001,0.003,0.01,1.0,2.0,4.0,8.0").split(",")]
#: E56b: on this benchmark the floor is a tax, not a safeguard. Set below the
#: smallest relative scale the 48-mode spectrum shows (0.0148) so it binds
#: nothing.
FLOOR = float(os.environ.get("R4D_SCALE_FLOOR", "0.005"))


def base_config() -> SolverConfig:
    return SolverConfig(
        fourier_order=3, session_order=1,
        estimate_pose=False, estimate_phase_warp=False, estimate_phase_offset=False,
        max_iterations=30,
    )


def aligned_start(model, config, truth):
    parameters = initial_parameters(model, config)
    return replace(
        parameters,
        pose_axis_angle=truth.pose_axis_angle,
        pose_translation=truth.pose_translation,
        phase_offset=truth.phase_offset,
        phase_logits=truth.phase_logits,
    )


def heldout_of(parameters, observations, model, config):
    fit = SceneFit(
        parameters=parameters, cost=0.0, iterations=0, stage_costs=(),
        attachment_residual=0.0, packed=_pack(parameters, model, config),
        model=model, config=config, observations=observations,
    )
    return fit.heldout_errors()


def validation_samples():
    index = {}
    for path in sample_paths(ROOT, "val"):
        with np.load(path, allow_pickle=True) as blob:
            index.setdefault(str(blob["identity_id"]), []).append(path)
    return index


def evaluate(model, config, index):
    """Median held-out error over validation identities, variants averaged."""

    per_identity = {"cmr": [], "tee": []}
    for identity, paths in sorted(index.items()):
        cmr, tee = [], []
        for path in paths:
            problem = benchmark_problem(path, model, config)
            fit = fit_joint_scene(
                problem.observations, model, config,
                start=aligned_start(model, config, problem.truth))
            errors = heldout_of(fit.parameters, problem.observations, model, config)
            cmr.append(float(errors["heldout_cmr_mm"]))
            tee.append(float(errors["heldout_tee_mm"]))
        per_identity["cmr"].append(float(np.mean(cmr)))
        per_identity["tee"].append(float(np.mean(tee)))
    return {
        "median_cmr_mm": float(np.median(per_identity["cmr"])),
        "median_tee_mm": float(np.median(per_identity["tee"])),
        "n_identities": len(per_identity["cmr"]),
    }


def main() -> int:
    index = validation_samples()
    rows = []
    for n_modes in MODES:
        model, provenance = fit_benchmark_model(
            ROOT, n_modes=n_modes, n_motion=4, n_session=2, n_phase_nodes=8)
        scales = np.asarray(model.shape_scales)
        print("modes %d | training identities %d | spectrum %.3f -> %.3f (relative %.4f)"
              % (n_modes, provenance["training_identities"], scales[0], scales[-1],
                 scales[-1] / scales[0]), flush=True)
        for aware in (False, True):
            for weight in WEIGHTS:
                config = replace(
                    base_config(),
                    identity_prior_weight=weight,
                    covariance_aware_identity_prior=aware,
                    identity_prior_scale_floor=FLOOR,
                )
                started = time.time()
                summary = evaluate(model, config, index)
                summary.update({
                    "modes": n_modes, "covariance_aware": aware, "weight": weight,
                    "sum_mm": summary["median_cmr_mm"] + summary["median_tee_mm"],
                    "seconds": round(time.time() - started, 1),
                })
                rows.append(summary)
                print("  %-10s w=%-7.3g  CMR %.4f  TEE %.4f  sum %.4f  (%.0fs)"
                      % ("aware" if aware else "isotropic", weight,
                         summary["median_cmr_mm"], summary["median_tee_mm"],
                         summary["sum_mm"], summary["seconds"]), flush=True)

    best = {}
    for n_modes in MODES:
        for aware in (False, True):
            candidates = [r for r in rows if r["modes"] == n_modes
                          and r["covariance_aware"] == aware]
            if candidates:
                best[(n_modes, aware)] = min(candidates, key=lambda r: r["sum_mm"])

    print("\nbest weight per arm, selected on the pre-declared sum:")
    comparison = []
    for n_modes in MODES:
        iso = best.get((n_modes, False))
        awr = best.get((n_modes, True))
        if not (iso and awr):
            continue
        entry = {
            "modes": n_modes,
            "isotropic": iso, "covariance_aware": awr,
            "cmr_change_mm": awr["median_cmr_mm"] - iso["median_cmr_mm"],
            "tee_change_mm": awr["median_tee_mm"] - iso["median_tee_mm"],
            "sum_change_mm": awr["sum_mm"] - iso["sum_mm"],
        }
        comparison.append(entry)
        print("  modes %d | isotropic w=%.2f %.4f/%.4f | aware w=%.2f %.4f/%.4f "
              "| change %+.4f/%+.4f (sum %+.4f)"
              % (n_modes, iso["weight"], iso["median_cmr_mm"], iso["median_tee_mm"],
                 awr["weight"], awr["median_cmr_mm"], awr["median_tee_mm"],
                 entry["cmr_change_mm"], entry["tee_change_mm"], entry["sum_change_mm"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "E56_covariance_prior_control",
        "question": "does covariance-aware shrinkage of the global PCA prior "
                    "recover the benefit attributed to a structured representation",
        "split": "shape space learned on train identities; measured on validation "
                 "identities; test untouched",
        "alignment": "frozen at truth (development diagnostic, not end to end)",
        "selection_rule": "weight minimising median CMR + median TEE, identical "
                          "for both arms, fixed before the run",
        "scale_floor": FLOOR, "weights": WEIGHTS, "modes": MODES,
        "rows": rows, "comparison": comparison,
    }, indent=2) + "\n")
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
