r"""E57b: how wide is the structured arm's margin, really?

E57 puts the best structured allocation 8.53% below the strongest global
control on the balanced sum, against a pre-declared 10% development gate. A
point estimate on one side of a threshold is not a decision when the interval
around it is unknown and the two arms are evaluated on the same 30 identities.
This measures the margin as a paired quantity.

The comparison is paired at the identity, which is the resampling unit the rest
of this project uses: the same validation identity is reconstructed by both
arms, so the difference is within-identity and the bootstrap resamples
identities rather than samples. Variants of an identity are averaged first,
because they are not independent observations.

Two things this interval does NOT cover, both of which make it optimistic:

* the weights were selected on these same identities, one per arm, so the
  selection is inside the interval rather than outside it;
* the structured arm was chosen as the best of three allocations on these same
  identities, and the global arm had one configuration.

So this bounds the sampling noise in a margin that is already the winner's
side of two selections. It is a development interval, not a held-out one.
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
    aligned_start,
    base_config,
    heldout_of,
    validation_samples,
)

from rooted4d.experiments.benchmark_eval import (  # noqa: E402
    benchmark_problem,
    fit_benchmark_model,
)
from rooted4d.experiments.structured_model import (  # noqa: E402
    fit_structured_benchmark_model,
)
from rooted4d.inference import fit_joint_scene  # noqa: E402

RESAMPLES = int(os.environ.get("R4D_RESAMPLES", "20000"))
PERMUTATIONS = int(os.environ.get("R4D_PERMUTATIONS", "200000"))
SEED = int(os.environ.get("R4D_SEED", "20260910"))
FLOOR = 0.005
#: The settings E57 selected, each the interior optimum of its own arm.
GLOBAL_WEIGHT = float(os.environ.get("R4D_GLOBAL_WEIGHT", "4.0"))
STRUCTURED_WEIGHT = float(os.environ.get("R4D_STRUCTURED_WEIGHT", "0.25"))
ALLOCATION = (12, 12)
OUT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / "E57b_paired_interval.json"


def per_identity(model, config, index):
    """Held-out error per validation identity, variants averaged."""

    identities, cmr, tee = [], [], []
    for identity, paths in sorted(index.items()):
        rows_cmr, rows_tee = [], []
        for path in paths:
            problem = benchmark_problem(path, model, config)
            fit = fit_joint_scene(
                problem.observations, model, config,
                start=aligned_start(model, config, problem.truth))
            errors = heldout_of(fit.parameters, problem.observations, model, config)
            rows_cmr.append(float(errors["heldout_cmr_mm"]))
            rows_tee.append(float(errors["heldout_tee_mm"]))
        identities.append(identity)
        cmr.append(float(np.mean(rows_cmr)))
        tee.append(float(np.mean(rows_tee)))
    return identities, np.array(cmr), np.array(tee)


def main() -> int:
    index = validation_samples()

    global_model, _ = fit_benchmark_model(
        ROOT, n_modes=48, n_motion=4, n_session=2, n_phase_nodes=8)
    structured, prov = fit_structured_benchmark_model(
        ROOT, interface_dim=ALLOCATION[0], interior_dim=ALLOCATION[1],
        n_motion=4, n_session=2, n_phase_nodes=8)
    if structured.n_modes != global_model.n_modes:
        raise ValueError("the arms are not capacity matched")

    def config_for(weight):
        return replace(
            base_config(), identity_prior_weight=weight,
            covariance_aware_identity_prior=True, identity_prior_scale_floor=FLOOR)

    names_g, cmr_g, tee_g = per_identity(global_model, config_for(GLOBAL_WEIGHT), index)
    names_s, cmr_s, tee_s = per_identity(
        structured, config_for(STRUCTURED_WEIGHT), index)
    if names_g != names_s:
        raise ValueError("the two arms saw different identities; the pairing is broken")
    n = len(names_g)
    print("paired on %d validation identities" % n, flush=True)

    def summarise(cmr, tee):
        return float(np.median(cmr)), float(np.median(tee)), float(
            np.median(cmr) + np.median(tee))

    obs_g = summarise(cmr_g, tee_g)
    obs_s = summarise(cmr_s, tee_s)
    observed = {
        "cmr_change_mm": obs_s[0] - obs_g[0],
        "tee_change_mm": obs_s[1] - obs_g[1],
        "sum_change_mm": obs_s[2] - obs_g[2],
        "relative_improvement": (obs_g[2] - obs_s[2]) / obs_g[2],
    }

    rng = np.random.default_rng(SEED)
    draws = {k: np.empty(RESAMPLES) for k in observed}
    for r in range(RESAMPLES):
        take = rng.integers(0, n, size=n)
        bg = summarise(cmr_g[take], tee_g[take])
        bs = summarise(cmr_s[take], tee_s[take])
        draws["cmr_change_mm"][r] = bs[0] - bg[0]
        draws["tee_change_mm"][r] = bs[1] - bg[1]
        draws["sum_change_mm"][r] = bs[2] - bg[2]
        draws["relative_improvement"][r] = (bg[2] - bs[2]) / bg[2]

    intervals = {
        key: {
            "observed": observed[key],
            "lower": float(np.percentile(values, 2.5)),
            "upper": float(np.percentile(values, 97.5)),
        }
        for key, values in draws.items()
    }

    # Paired sign-flip permutation on the per-identity total, the project's
    # standard test for a within-identity difference.
    total_difference = (cmr_s + tee_s) - (cmr_g + tee_g)
    observed_mean = float(total_difference.mean())
    flips = rng.integers(0, 2, size=(PERMUTATIONS, n)) * 2 - 1
    null = (flips * total_difference).mean(axis=1)
    p_value = float((np.abs(null) >= abs(observed_mean) - 1e-15).mean())

    gate_target = 0.10
    clears = intervals["relative_improvement"]["lower"] >= gate_target
    excluded = intervals["relative_improvement"]["upper"] < gate_target

    print("\nglobal     CMR %.4f  TEE %.4f  sum %.4f" % obs_g)
    print("structured CMR %.4f  TEE %.4f  sum %.4f" % obs_s)
    for key in ("cmr_change_mm", "tee_change_mm", "sum_change_mm"):
        row = intervals[key]
        print("%-16s %+.4f mm  95%% CI [%+.4f, %+.4f]"
              % (key.replace("_change_mm", ""), row["observed"],
                 row["lower"], row["upper"]))
    row = intervals["relative_improvement"]
    print("relative         %+.2f%%    95%% CI [%+.2f%%, %+.2f%%]"
          % (row["observed"] * 100, row["lower"] * 100, row["upper"] * 100))
    print("per-identity total difference %+.4f mm, sign-flip p = %.5f"
          % (observed_mean, p_value))
    print("\n10%% gate: %s"
          % ("cleared -- the interval is entirely above it" if clears
             else "ruled out -- the interval is entirely below it" if excluded
             else "undecided -- the interval spans the threshold"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "E57b_paired_interval",
        "question": "is the structured arm's margin over the strongest global "
                    "control separable from sampling noise, and does it reach "
                    "the declared 10% development gate",
        "pairing": "same validation identity through both arms; variants averaged; "
                   "bootstrap resamples identities",
        "optimism": "weights were selected on these identities and the structured "
                    "allocation was the best of three on them; the interval covers "
                    "sampling noise only, not selection",
        "n_identities": n, "resamples": RESAMPLES, "permutations": PERMUTATIONS,
        "seed": SEED,
        "global": {"weight": GLOBAL_WEIGHT, "median_cmr_mm": obs_g[0],
                   "median_tee_mm": obs_g[1], "sum_mm": obs_g[2]},
        "structured": {"allocation": list(ALLOCATION), "weight": STRUCTURED_WEIGHT,
                       "median_cmr_mm": obs_s[0], "median_tee_mm": obs_s[1],
                       "sum_mm": obs_s[2], "provenance": prov},
        "intervals": intervals,
        "paired_total": {"mean_difference_mm": observed_mean, "sign_flip_p": p_value},
        "gate": {"target": gate_target, "cleared": bool(clears),
                 "ruled_out": bool(excluded)},
        "per_identity": {"identities": names_g,
                         "global_cmr_mm": cmr_g.tolist(), "global_tee_mm": tee_g.tolist(),
                         "structured_cmr_mm": cmr_s.tolist(),
                         "structured_tee_mm": tee_s.tolist()},
    }, indent=2) + "\n")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
