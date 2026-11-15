r"""E57: does the structured span still beat global PCA once the prior is fair?

E53 reported the structured shape space ahead of global PCA. Two defects have
since been established: the structured arm was scored by a blockwise projection
while the global arm received an optimal one, and the allocations were selected
on test. E56 then removed a third asymmetry that had nothing to do with the
representation at all -- the identity prior was isotropic, and shrinking each
mode by its learned scale bought 0.55 mm of held-out error on the *global*
basis. That moves the bar the structured option has to clear.

This is the matched comparison:

* both arms are built from the train identities only, deduplicated;
* both are orthonormal in physical displacement coordinates, so the identity
  prior penalises the same physical quantity in each;
* both are at numerical rank 48, so neither is given more capacity;
* both get the covariance-aware prior at their own optimum over a shared weight
  grid, so neither is judged at a setting it was never offered;
* everything is measured on the validation identities. Test is untouched.

The structured arm is allowed three allocations of the same rank 48 and the
global arm has one configuration, so the structured arm gets three chances and
the global arm one. That asymmetry is deliberate and it only inflates the
structured arm: if the best of three still fails the gate, the failure is
conclusive, whereas a pass would need confirmation on a split that has not been
selected on.

The gate is the campaign memo's, made numeric here before the run: at least a
10% improvement in the balanced sum over the strongest global control, with
neither modality worsening by more than 5% of that control's value. It is a
development gate for deciding whether to keep building, not a clinical
threshold and not a publication significance criterion.
"""

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "Codes/scripts/experiments")

from e56_covariance_prior_control import (  # noqa: E402
    ROOT,
    base_config,
    evaluate,
    validation_samples,
)

from rooted4d.experiments.benchmark_eval import fit_benchmark_model  # noqa: E402
from rooted4d.experiments.structured_model import (  # noqa: E402
    fit_structured_benchmark_model,
)

RANK = int(os.environ.get("R4D_RANK", "48"))
#: Every allocation here assembles to numerical rank 48 on this benchmark, which
#: is checked rather than assumed below.
ALLOCATIONS = [(12, 12), (18, 10), (24, 8)]
#: The grid the canonical E57 artifact was produced with. The first pass used
#: only 0.5..16, left two structured arms at the bottom edge, and was rerun;
#: the default is the rerun's grid, so a plain invocation reproduces the
#: archived result rather than the superseded one.
WEIGHTS = [float(v) for v in
           os.environ.get("R4D_WEIGHTS",
                          "0.01,0.03,0.06,0.125,0.25,0.5,1.0,2.0,4.0,8.0,16.0").split(",")]
FLOOR = float(os.environ.get("R4D_SCALE_FLOOR", "0.005"))
OUT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / "E57_structured_vs_global.json"

#: Declared before the run.
SUM_IMPROVEMENT_TARGET = 0.10
PER_MODALITY_TOLERANCE = 0.05


def sweep(label, model, index, rows):
    best = None
    for weight in WEIGHTS:
        config = replace(
            base_config(),
            identity_prior_weight=weight,
            covariance_aware_identity_prior=True,
            identity_prior_scale_floor=FLOOR,
        )
        started = time.time()
        summary = evaluate(model, config, index)
        summary.update({
            "arm": label, "weight": weight, "modes": int(model.n_modes),
            "sum_mm": summary["median_cmr_mm"] + summary["median_tee_mm"],
            "seconds": round(time.time() - started, 1),
        })
        rows.append(summary)
        print("  %-22s w=%-6.3g CMR %.4f  TEE %.4f  sum %.4f  (%.0fs)"
              % (label, weight, summary["median_cmr_mm"], summary["median_tee_mm"],
                 summary["sum_mm"], summary["seconds"]), flush=True)
        if best is None or summary["sum_mm"] < best["sum_mm"]:
            best = summary
    edge = best["weight"] in (WEIGHTS[0], WEIGHTS[-1])
    best = dict(best, weight_at_grid_edge=bool(edge))
    print("  -> %-19s best w=%-6.3g sum %.4f%s"
          % (label, best["weight"], best["sum_mm"],
             "  EDGE: optimum not bracketed" if edge else ""), flush=True)
    return best


def main() -> int:
    index = validation_samples()
    rows, bests = [], {}

    global_model, provenance = fit_benchmark_model(
        ROOT, n_modes=RANK, n_motion=4, n_session=2, n_phase_nodes=8)
    print("global PCA | rank %d | %d training identities | variance %.5f"
          % (global_model.n_modes, provenance["training_identities"],
             provenance["variance_captured"]), flush=True)
    bests["global"] = sweep("global_pca", global_model, index, rows)

    structured = {}
    for interface_dim, interior_dim in ALLOCATIONS:
        model, prov = fit_structured_benchmark_model(
            ROOT, interface_dim=interface_dim, interior_dim=interior_dim,
            n_motion=4, n_session=2, n_phase_nodes=8)
        label = "structured_%d+3x%d" % (interface_dim, interior_dim)
        if model.n_modes != RANK:
            raise ValueError(
                "%s assembles to rank %d, not %d; the comparison would not be "
                "capacity matched" % (label, model.n_modes, RANK))
        print("%s | rank %d (%d columns) | variance %.5f"
              % (label, prov["numerical_rank"], prov["columns"],
                 prov["variance_captured"]), flush=True)
        structured[label] = sweep(label, model, index, rows)
        structured[label]["provenance"] = prov

    control = bests["global"]
    best_structured = min(structured.values(), key=lambda r: r["sum_mm"])
    sum_change = best_structured["sum_mm"] - control["sum_mm"]
    relative = -sum_change / control["sum_mm"]
    cmr_ok = (best_structured["median_cmr_mm"]
              <= control["median_cmr_mm"] * (1 + PER_MODALITY_TOLERANCE))
    tee_ok = (best_structured["median_tee_mm"]
              <= control["median_tee_mm"] * (1 + PER_MODALITY_TOLERANCE))
    passed = bool(relative >= SUM_IMPROVEMENT_TARGET and cmr_ok and tee_ok)

    print("\nstrongest global control : CMR %.4f  TEE %.4f  sum %.4f (w=%.3g)"
          % (control["median_cmr_mm"], control["median_tee_mm"],
             control["sum_mm"], control["weight"]))
    print("best structured (of %d)  : CMR %.4f  TEE %.4f  sum %.4f (%s, w=%.3g)"
          % (len(structured), best_structured["median_cmr_mm"],
             best_structured["median_tee_mm"], best_structured["sum_mm"],
             best_structured["arm"], best_structured["weight"]))
    print("change                   : %+.4f mm on the sum, %+.2f%% relative"
          % (sum_change, -relative * 100))
    print("gate                     : %s (needs >=%.0f%% and neither modality "
          "worse than +%.0f%%)"
          % ("PASS" if passed else "FAIL", SUM_IMPROVEMENT_TARGET * 100,
             PER_MODALITY_TOLERANCE * 100))
    if not passed:
        print("                           CMR within tolerance: %s | TEE: %s"
              % (cmr_ok, tee_ok))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "E57_structured_vs_global",
        "question": "at matched numerical rank and with the covariance-aware prior "
                    "on both arms, does the structured span beat global PCA",
        "split": "both bases learned on train identities; measured on validation "
                 "identities; test untouched",
        "alignment": "frozen at truth (development diagnostic, not end to end)",
        "multiplicity": "the structured arm was given %d allocations and the global "
                        "arm one; a failure is therefore conclusive and a pass "
                        "would need confirmation on an unselected split"
                        % len(ALLOCATIONS),
        "rank": RANK, "weights": WEIGHTS, "scale_floor": FLOOR,
        "gate": {
            "sum_improvement_target": SUM_IMPROVEMENT_TARGET,
            "per_modality_tolerance": PER_MODALITY_TOLERANCE,
            "passed": passed,
            "sum_change_mm": sum_change,
            "relative_improvement": relative,
            "cmr_within_tolerance": cmr_ok,
            "tee_within_tolerance": tee_ok,
        },
        "global_control": control,
        "structured": structured,
        "best_structured": best_structured,
        "rows": rows,
    }, indent=2) + "\n")
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
