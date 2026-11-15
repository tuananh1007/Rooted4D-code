"""E56b: is the covariance gain the covariance, or is it the floor?

E56 shrinks each shape direction by its training scale, with a floor so the
weakest directions do not divide by nearly zero. The floored-mode count grows
with capacity -- 5 of 32, 13 of 40, 21 of 48 -- and the largest gain appears at
48 modes, exactly where nearly half the spectrum sits on the floor. At that
point "covariance-aware" is partly "uniform shrinkage on the floored half", and
the two explanations predict different things:

* if the mechanism is the covariance, the gain survives a much *smaller* floor,
  where fewer modes are clamped and the learned shape does the work;
* if the mechanism is the floor, the gain peaks near one floor value and falls
  away on either side of it.

The weight is re-optimised inside each floor. Changing the floor changes the
effective strength of the prior, so comparing at a fixed weight would confound
strength with shape -- the error this control has already made twice.
"""

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, "Codes/scripts/experiments")

from e56_covariance_prior_control import (  # noqa: E402
    ROOT,
    base_config,
    evaluate,
    validation_samples,
)
from rooted4d.experiments.benchmark_eval import fit_benchmark_model  # noqa: E402

MODES = int(os.environ.get("R4D_MODES", "48"))
FLOORS = [float(v) for v in
          os.environ.get("R4D_FLOORS", "0.005,0.01,0.02,0.05,0.10,0.20").split(",")]
WEIGHTS = [float(v) for v in
           os.environ.get("R4D_WEIGHTS", "0.5,1.0,2.0,4.0").split(",")]
OUT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / "E56b_floor_attribution.json"


def main() -> int:
    index = validation_samples()
    model, _ = fit_benchmark_model(
        ROOT, n_modes=MODES, n_motion=4, n_session=2, n_phase_nodes=8)
    scales = np.asarray(model.shape_scales)

    rows = []
    for floor in FLOORS:
        clamped = int((scales / scales[0] < floor).sum())
        best = None
        for weight in WEIGHTS:
            config = replace(
                base_config(),
                identity_prior_weight=weight,
                covariance_aware_identity_prior=True,
                identity_prior_scale_floor=floor,
            )
            summary = evaluate(model, config, index)
            total = summary["median_cmr_mm"] + summary["median_tee_mm"]
            if best is None or total < best[0]:
                best = (total, weight, summary)
        total, weight, summary = best
        edge = weight in (WEIGHTS[0], WEIGHTS[-1])
        rows.append({
            "floor": floor, "clamped_modes": clamped, "of_modes": MODES,
            "best_weight": weight, "weight_at_grid_edge": bool(edge),
            "sum_mm": total,
            "median_cmr_mm": summary["median_cmr_mm"],
            "median_tee_mm": summary["median_tee_mm"],
        })
        print("  floor %.3f | clamped %2d/%d | best w=%-7.3g%s | CMR %.4f TEE %.4f sum %.4f"
              % (floor, clamped, MODES, weight, " EDGE" if edge else "     ",
                 summary["median_cmr_mm"], summary["median_tee_mm"], total), flush=True)

    totals = [r["sum_mm"] for r in rows]
    spread = float(max(totals) - min(totals))
    smallest_floor = rows[0]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "E56b_floor_attribution",
        "question": "does the covariance gain survive a small floor, or is the "
                    "floor itself the mechanism",
        "modes": MODES, "floors": FLOORS, "weights": WEIGHTS,
        "sum_spread_mm": spread,
        "sum_at_smallest_floor_mm": smallest_floor["sum_mm"],
        "clamped_at_smallest_floor": smallest_floor["clamped_modes"],
        "rows": rows,
    }, indent=2) + "\n")
    print("\nspread across floors: %.4f mm" % spread)
    print("at the smallest floor (%.3f, %d clamped) the sum is %.4f"
          % (smallest_floor["floor"], smallest_floor["clamped_modes"],
             smallest_floor["sum_mm"]))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
