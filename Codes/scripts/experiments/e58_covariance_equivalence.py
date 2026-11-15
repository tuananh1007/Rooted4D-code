r"""E58: does the structured gain survive equivalent covariance treatment?

E57 compared a global PCA basis under its full empirical-covariance prior
against a structured basis under a *diagonal approximation* of one: the
structured coordinates are singular vectors of the assembly operator, they do
not diagonalise the training covariance (largest coefficient correlation 0.94,
48% of the covariance's Frobenius norm off the diagonal), and the solver's
diagonal prior over them discards that structure. E57's errors are the errors
of the estimator it ran; what they cannot do is isolate the representation.

The rules are pre-registered in ``Docs/E58_PREREGISTRATION_2026-09-22.md``,
committed before the validation run and audited before that. Six arms at
numerical rank 48, all built from train identities, measured on the 30
validation identities with alignment frozen at truth:

  global_scalar                 global PCA, scalar prior on displacement
  global_covariance             global PCA, full empirical-covariance prior
  structured_scalar_assembly    structured, scalar prior, assembly axes
  structured_scalar_covariance  structured, scalar prior, covariance axes
  structured_covariance         structured, full empirical-covariance prior
  structured_diagonal           structured, E57's diagonal approximation

Preconditions, each computed rather than judged:

  P0  the harness reproduces E57b's per-identity arrays for the two arms E57b
      ran (structured_diagonal at 0.25, global_covariance at 4.0);
  P1  the two scalar structured arms -- one estimator in two sets of axes --
      agree on every identity at every eligible weight;
  P2  w = 0 is eligible for a span only if every validation sample leaves the
      identity determined by the visible evidence, which is decided from the
      masks alone before any fit.

``R4D_PHASE=smoke`` runs P1 on training identities only, including the
training samples whose masks make the structured span singular, so the check
can be exercised without looking at any validation result.
"""

import dataclasses
import json
import os
import subprocess
import sys
import time
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
    sample_paths,
)
from rooted4d.experiments.structured_model import (  # noqa: E402
    fit_structured_benchmark_model,
)
from rooted4d.inference import fit_joint_scene  # noqa: E402
from rooted4d.inference.joint import identity_prior_multiplier  # noqa: E402

PHASE = os.environ.get("R4D_PHASE", "full")
RANK = 48
ALLOCATION = (12, 12)
#: Declared in the pre-registration. Zero is the review's missing endpoint;
#: 0.25 and 0.06 are the saved E57 structured optima; 32 and 64 are added
#: because equal weight is not equal strength across arms (see kappa below)
#: and an optimum near the old upper edge should not force a post-hoc
#: extension.
WEIGHTS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.06, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0,
           8.0, 16.0, 32.0, 64.0]
FLOOR = 0.005
P0_TOLERANCE_MM = 1e-6
#: LM stops on relative improvement, so on masks that leave a leaflet nearly
#: unobserved it leaves up to ~1.4e-3 mm of coordinate-dependent remainder at
#: w = 0.001 (audit emulation on training masks). The tolerance is several
#: times that and an order of magnitude below the effects under test.
P1_TOLERANCE_MM = 1e-2
#: sigma_min / sigma_max of the shape basis restricted to the visible vertices.
#: Below RANK_RATIO the identity is not determined by the evidence at w = 0;
#: below THIN_RATIO a sample is flagged for the descriptive tail breakdown.
RANK_RATIO = 1e-8
THIN_RATIO = 1e-2
EXTENSION_LIMITS = (1e-5, 1024.0)
RESAMPLES = 20000
PERMUTATIONS = 200000
SEED = 20260910
OUT = Path(os.environ.get("R4D_OUT", "results/benchmark"))
PREREGISTRATION = "Docs/E58_PREREGISTRATION_2026-09-22.md"
E57B = Path("results/benchmark/E57b_paired_interval.json")

ARMS = {
    # name: (model key, span, covariance-aware prior)
    "global_scalar": ("global", "global", False),
    "global_covariance": ("global", "global", True),
    "structured_scalar_assembly": ("structured_assembly", "structured", False),
    "structured_scalar_covariance": ("structured_covariance", "structured", False),
    "structured_covariance": ("structured_covariance", "structured", True),
    "structured_diagonal": ("structured_assembly", "structured", True),
}
PRIMARY = ("structured_covariance", "global_covariance")
#: Arms whose edges block the outcome: the primary pair, and structured_diagonal
#: because the diagonal-only clause reads it. Edges elsewhere are flagged only.
GATED = ("structured_covariance", "global_covariance", "structured_diagonal")
SECONDARY = [
    ("structured_diagonal", "global_covariance"),
    ("structured_diagonal", "structured_covariance"),
    ("structured_scalar_assembly", "global_scalar"),
]


def build_models():
    models, provenance = {}, {}
    models["global"], provenance["global"] = fit_benchmark_model(
        ROOT, n_modes=RANK, n_motion=4, n_session=2, n_phase_nodes=8)
    for axes in ("assembly", "covariance"):
        key = "structured_" + axes
        models[key], provenance[key] = fit_structured_benchmark_model(
            ROOT, interface_dim=ALLOCATION[0], interior_dim=ALLOCATION[1],
            n_motion=4, n_session=2, n_phase_nodes=8, coordinates=axes)
    for key, model in models.items():
        if model.n_modes != RANK:
            raise ValueError("%s has rank %d, not %d" % (key, model.n_modes, RANK))
        # Prior strength per unit weight. With the multiplier normalised to unit
        # RMS the covariance-aware penalty is w * kappa * sum(c_i^2 / s_i^2), so
        # equal w is NOT equal strength across bases; compare arms in w * kappa.
        multiplier = identity_prior_multiplier(
            model, config_for(1.0, True), model.n_modes)
        scales = np.asarray(model.shape_scales)
        per_mode = multiplier ** 2 * scales ** 2
        provenance[key]["kappa_per_unit_weight_mm2"] = float(np.mean(per_mode))
        provenance[key]["kappa_spread"] = float(per_mode.max() / per_mode.min())
    return models, provenance


def config_for(weight, aware):
    return replace(base_config(), identity_prior_weight=weight,
                   covariance_aware_identity_prior=aware,
                   identity_prior_scale_floor=FLOOR)


def observability(model, observations):
    """sigma_min / sigma_max of the basis at the visible vertices, from masks only."""

    visible = np.union1d(observations.cmr_vertices, observations.tee_vertices)
    rows = model.shape_basis.reshape(-1, 3, model.n_modes)[visible].reshape(-1, model.n_modes)
    singular = np.linalg.svd(rows, compute_uv=False)
    return float(singular[-1] / singular[0])


def mask_census(models, index):
    """Per-sample observability for each span. Reads masks, never errors."""

    groups = models["global"].groups
    free = np.sort(groups["leaflet_free"])
    leaflets = {"anterior": set(free[: free.size // 2].tolist()),
                "posterior": set(free[free.size // 2:].tolist())}
    census = []
    for identity, paths in sorted(index.items()):
        for path in paths:
            observations = benchmark_problem(path, models["global"], base_config()).observations
            tee = set(np.asarray(observations.tee_vertices).tolist())
            census.append({
                "identity": identity, "sample": path.name,
                "visible_free": {k: len(v & tee) for k, v in leaflets.items()},
                "global": observability(models["global"], observations),
                "structured": observability(models["structured_covariance"], observations),
            })
    return census


def per_identity(model, config, index):
    """Held-out error per identity, variants averaged, plus solver iterations."""

    names, cmr, tee, iterations = [], [], [], []
    for identity, paths in sorted(index.items()):
        rows_cmr, rows_tee = [], []
        for path in paths:
            problem = benchmark_problem(path, model, config)
            fit = fit_joint_scene(problem.observations, model, config,
                                  start=aligned_start(model, config, problem.truth))
            errors = heldout_of(fit.parameters, problem.observations, model, config)
            rows_cmr.append(float(errors["heldout_cmr_mm"]))
            rows_tee.append(float(errors["heldout_tee_mm"]))
            iterations.append(int(fit.iterations))
        names.append(identity)
        cmr.append(float(np.mean(rows_cmr)))
        tee.append(float(np.mean(rows_tee)))
    return names, np.array(cmr), np.array(tee), iterations


def sum_of_medians(cmr, tee):
    return float(np.median(cmr) + np.median(tee))


def evaluate_row(models, arm, weight, index, extension=False):
    key, _, aware = ARMS[arm]
    config = config_for(weight, aware)
    started = time.time()
    names, cmr, tee, iterations = per_identity(models[key], config, index)
    row = {"weight": weight, "extension": extension, "identities": names,
           "cmr_mm": cmr.tolist(), "tee_mm": tee.tolist(),
           "median_cmr_mm": float(np.median(cmr)), "median_tee_mm": float(np.median(tee)),
           "sum_of_medians_mm": sum_of_medians(cmr, tee),
           "max_total_iterations": int(max(iterations)),
           # Iterations are summed over the solver's stages, each capped at
           # max_iterations, so a total below the cap proves no stage hit it.
           "cap_possibly_binding": bool(max(iterations) >= config.max_iterations),
           "seconds": round(time.time() - started, 1)}
    print("  %-29s w=%-8g CMR %.4f TEE %.4f sum %.4f  (%.0fs, iters<=%d)%s"
          % (arm, weight, row["median_cmr_mm"], row["median_tee_mm"],
             row["sum_of_medians_mm"], row["seconds"], row["max_total_iterations"],
             "  [extension]" if extension else ""), flush=True)
    return row


def select(rows, eligible_zero):
    pool = [r for r in rows if r["weight"] > 0.0 or eligible_zero]
    best = min(pool, key=lambda r: r["sum_of_medians_mm"])
    weights = sorted(r["weight"] for r in pool)
    return best, best["weight"] == weights[-1], (best["weight"] == weights[0]
                                                 and best["weight"] > 0.0)


def extend(models, arm, rows, index, eligible_zero):
    """Mechanical extension for a gated arm whose optimum sits at an edge.

    Upper edge: double until two consecutive weights exceed the running
    minimum or the limit is passed. Lower edge (only possible when w = 0 is
    ineligible): halve under the same rule. Declared in the pre-registration,
    so no choice about the extension is made after seeing results.
    """

    best, upper, lower = select(rows, eligible_zero)
    if not (upper or lower):
        return rows, "bracketed"
    factor = 2.0 if upper else 0.5
    weight, worse = best["weight"], 0
    while worse < 2:
        weight *= factor
        if not (EXTENSION_LIMITS[0] <= weight <= EXTENSION_LIMITS[1]):
            return rows, "limit reached"
        row = evaluate_row(models, arm, weight, index, extension=True)
        rows.append(row)
        if row["sum_of_medians_mm"] < best["sum_of_medians_mm"]:
            best, worse = row, 0
        else:
            worse += 1
    return rows, "extended"


def paired_statistics(cmr_a, tee_a, cmr_b, tee_b):
    """Arm a minus arm b, paired at the identity, resampling identities.

    A fresh generator per comparison, so every comparison's interval is the
    same whatever order the comparisons are made in, and the one E57b ran can
    be reproduced draw for draw.
    """

    rng = np.random.default_rng(SEED)
    n = len(cmr_a)
    base_b = sum_of_medians(cmr_b, tee_b)
    observed = {
        "cmr_difference_of_medians_mm": float(np.median(cmr_a) - np.median(cmr_b)),
        "tee_difference_of_medians_mm": float(np.median(tee_a) - np.median(tee_b)),
        "sum_of_medians_change_mm": sum_of_medians(cmr_a, tee_a) - base_b,
    }
    observed["relative_improvement"] = -observed["sum_of_medians_change_mm"] / base_b
    draws = {k: np.empty(RESAMPLES) for k in observed}
    for r in range(RESAMPLES):
        t = rng.integers(0, n, size=n)
        a, b = sum_of_medians(cmr_a[t], tee_a[t]), sum_of_medians(cmr_b[t], tee_b[t])
        draws["cmr_difference_of_medians_mm"][r] = np.median(cmr_a[t]) - np.median(cmr_b[t])
        draws["tee_difference_of_medians_mm"][r] = np.median(tee_a[t]) - np.median(tee_b[t])
        draws["sum_of_medians_change_mm"][r] = a - b
        draws["relative_improvement"][r] = (b - a) / b
    intervals = {k: {"observed": observed[k],
                     "lower": float(np.percentile(v, 2.5)),
                     "upper": float(np.percentile(v, 97.5))}
                 for k, v in draws.items()}
    total = (cmr_a + tee_a) - (cmr_b + tee_b)
    flips = rng.integers(0, 2, size=(PERMUTATIONS, n)) * 2 - 1
    null = (flips * total).mean(axis=1)
    descriptive = {
        "cmr_identities_better": int((cmr_a < cmr_b).sum()),
        "cmr_identities_worse": int((cmr_a > cmr_b).sum()),
        "tee_identities_better": int((tee_a < tee_b).sum()),
        "tee_identities_worse": int((tee_a > tee_b).sum()),
        "cmr_p90_mm": [float(np.percentile(cmr_a, 90)), float(np.percentile(cmr_b, 90))],
        "tee_p90_mm": [float(np.percentile(tee_a, 90)), float(np.percentile(tee_b, 90))],
        "cmr_max_mm": [float(cmr_a.max()), float(cmr_b.max())],
        "tee_max_mm": [float(tee_a.max()), float(tee_b.max())],
        "median_per_identity_sum_mm": [float(np.median(cmr_a + tee_a)),
                                       float(np.median(cmr_b + tee_b))],
        # Named for what it tests: the MEAN paired per-identity sum, not the
        # sum-of-medians headline.
        "mean_per_identity_sum_change_mm": float(total.mean()),
        "mean_per_identity_sum_sign_flip_p": float(
            (np.abs(null) >= abs(total.mean()) - 1e-15).mean()),
    }
    return intervals, descriptive


def cross_fitted(rows, eligible_zero):
    """Leave-one-identity-out weight selection, from the rows already computed.

    Each identity is scored at the weight the arm's own rule picks on the other
    29, so the resulting arrays carry weight-selection noise that the in-sample
    interval does not. Descriptive; the decision uses the registered rule.
    """

    pool = [r for r in rows if r["weight"] > 0.0 or eligible_zero]
    n = len(pool[0]["identities"])
    cmr, tee, chosen = np.empty(n), np.empty(n), []
    for i in range(n):
        keep = np.arange(n) != i
        best = min(pool, key=lambda r: sum_of_medians(
            np.array(r["cmr_mm"])[keep], np.array(r["tee_mm"])[keep]))
        cmr[i], tee[i] = best["cmr_mm"][i], best["tee_mm"][i]
        chosen.append(best["weight"])
    return cmr, tee, chosen


def git_state():
    watched = [PREREGISTRATION, "Codes/scripts/experiments/e58_covariance_equivalence.py",
               "Codes/scripts/experiments/e56_covariance_prior_control.py", "Codes/rooted4d"]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--"] + watched,
                           capture_output=True, text=True, check=True).stdout.strip()
    return revision, dirty


def smoke() -> int:
    """P1 on training identities, including the masks that make the span singular."""

    models, _ = build_models()
    index = {}
    for path in sample_paths(ROOT, "train"):
        with np.load(path, allow_pickle=True) as blob:
            index.setdefault(str(blob["identity_id"]), []).append(path)
    census = mask_census(models, index)
    thin = sorted(census, key=lambda c: c["structured"])[:5]
    thin_ids = {c["identity"] for c in thin}
    chosen = dict(list(sorted(index.items()))[:3])
    chosen.update({k: v for k, v in index.items() if k in thin_ids})
    print("smoke on %d training identities (%d chosen for singular masks: %s)"
          % (len(chosen), len(thin_ids),
             ", ".join("%.1e" % c["structured"] for c in thin)), flush=True)
    worst = 0.0
    for weight in (0.0, 0.001, 0.01, 1.0):
        config = config_for(weight, False)
        _, ca, ta, ia = per_identity(models["structured_assembly"], config, chosen)
        _, cc, tc, ic = per_identity(models["structured_covariance"], config, chosen)
        gap = float(max(np.abs(ca - cc).max(), np.abs(ta - tc).max()))
        if weight > 0.0:
            worst = max(worst, gap)
        print("  w=%-6g max |assembly - covariance| %.2e mm%s | max iterations %d / %d"
              % (weight, gap, "  (w=0: ill posed on singular masks; excluded by P2)"
                 if weight == 0.0 else "", max(ia), max(ic)), flush=True)
    ok = worst <= P1_TOLERANCE_MM
    print("P1 smoke over w>0: %s (worst %.2e mm, tolerance %.0e)"
          % ("PASS" if ok else "FAIL", worst, P1_TOLERANCE_MM))
    return 0 if ok else 1


def main() -> int:
    if PHASE == "smoke":
        return smoke()
    revision, dirty = git_state()
    if dirty and not os.environ.get("R4D_ALLOW_DIRTY"):
        print("refusing to run: the pre-registration or the code it covers is not "
              "committed\n" + dirty)
        return 2

    models, provenance = build_models()
    index = validation_samples()

    # P2 first, from the masks alone, before any fit.
    census = mask_census(models, index)
    eligible_zero = {span: all(c[span] >= RANK_RATIO for c in census)
                     for span in ("global", "structured")}
    print("P2 w=0 eligibility from masks: %s | smallest ratio global %.2e, structured %.2e"
          % (eligible_zero, min(c["global"] for c in census),
             min(c["structured"] for c in census)), flush=True)

    results = {}
    for arm in ARMS:
        results[arm] = [evaluate_row(models, arm, w, index) for w in WEIGHTS]

    extensions = {}
    for arm in GATED:
        span = ARMS[arm][1]
        results[arm], extensions[arm] = extend(models, arm, results[arm], index,
                                               eligible_zero[span])

    def gap(a, b):
        return float(max(np.abs(np.array(a["cmr_mm"]) - np.array(b["cmr_mm"])).max(),
                         np.abs(np.array(a["tee_mm"]) - np.array(b["tee_mm"])).max()))

    # P0: the harness reproduces E57b's per-identity arrays exactly.
    e57b = json.loads(E57B.read_text())["per_identity"]
    row_at = {arm: {r["weight"]: r for r in rows} for arm, rows in results.items()}
    p0_rows = {"structured_diagonal": (0.25, "structured"), "global_covariance": (4.0, "global")}
    p0 = {}
    for arm, (weight, prefix) in p0_rows.items():
        row = row_at[arm][weight]
        if row["identities"] != e57b["identities"]:
            raise ValueError("P0: identity order differs from E57b")
        p0[arm] = float(max(
            np.abs(np.array(row["cmr_mm"]) - np.array(e57b[prefix + "_cmr_mm"])).max(),
            np.abs(np.array(row["tee_mm"]) - np.array(e57b[prefix + "_tee_mm"])).max()))
    p0_pass = max(p0.values()) <= P0_TOLERANCE_MM

    # P1 over every weight eligible for the structured span.
    p1 = {str(a["weight"]): gap(a, b)
          for a, b in zip(results["structured_scalar_assembly"],
                          results["structured_scalar_covariance"])
          if a["weight"] > 0.0 or eligible_zero["structured"]}
    p1_pass = max(p1.values()) <= P1_TOLERANCE_MM
    cap_rows = [(arm, r["weight"]) for arm, rows in results.items()
                for r in rows if r["cap_possibly_binding"]]
    print("\nP0 harness reproduces E57b: %s (max %s)" % ("PASS" if p0_pass else "FAIL", p0))
    print("P1 rotation invariance: %s (worst %.2e mm over %d weights)"
          % ("PASS" if p1_pass else "FAIL", max(p1.values()), len(p1)))
    print("iteration cap possibly binding in: %s" % (cap_rows or "no row"))

    selected = {}
    for arm, rows in results.items():
        span = ARMS[arm][1]
        best, upper, lower = select(rows, eligible_zero[span])
        selected[arm] = dict(best, upper_edge=bool(upper), lower_edge=bool(lower),
                             gated=arm in GATED)
        kappa = provenance[ARMS[arm][0]]["kappa_per_unit_weight_mm2"] if ARMS[arm][2] else None
        selected[arm]["strength_w_times_kappa"] = (None if kappa is None
                                                   else best["weight"] * kappa)
        flag = ("  UPPER EDGE" if upper else "  LOWER EDGE (limit w->0+)" if lower
                else "  (zero-prior endpoint)" if best["weight"] == 0.0 else "")
        print("  %-29s selected w=%-8g sum %.4f%s%s"
              % (arm, best["weight"], best["sum_of_medians_mm"],
                 "" if kappa is None else "  w*kappa=%.2f" % (best["weight"] * kappa),
                 flag + ("" if flag == "" or arm in GATED else "  [descriptive, not gating]")))

    comparisons = {}
    for a, b in [PRIMARY] + SECONDARY:
        ra, rb = selected[a], selected[b]
        if ra["identities"] != rb["identities"]:
            raise ValueError("pairing broken between %s and %s" % (a, b))
        intervals, descriptive = paired_statistics(
            np.array(ra["cmr_mm"]), np.array(ra["tee_mm"]),
            np.array(rb["cmr_mm"]), np.array(rb["tee_mm"]))
        comparisons["%s_vs_%s" % (a, b)] = {"intervals": intervals, "descriptive": descriptive}
        s = intervals["sum_of_medians_change_mm"]
        print("\n%s vs %s\n  sum of medians %+.4f mm [%+.4f, %+.4f] | relative improvement "
              "(positive = first arm better) %+.2f%%"
              % (a, b, s["observed"], s["lower"], s["upper"],
                 intervals["relative_improvement"]["observed"] * 100))
        for key in ("cmr_difference_of_medians_mm", "tee_difference_of_medians_mm"):
            v = intervals[key]
            print("  %-30s %+.4f mm [%+.4f, %+.4f]" % (key, v["observed"], v["lower"], v["upper"]))
        d = descriptive
        print("  CMR better/worse %d/%d | TEE better/worse %d/%d | TEE P90 %.3f vs %.3f"
              " | TEE max %.3f vs %.3f"
              % (d["cmr_identities_better"], d["cmr_identities_worse"],
                 d["tee_identities_better"], d["tee_identities_worse"],
                 d["tee_p90_mm"][0], d["tee_p90_mm"][1], d["tee_max_mm"][0], d["tee_max_mm"][1]))

    # Cross-fitted weight selection for the primary pair (descriptive).
    cf = {}
    for arm in PRIMARY:
        cf[arm] = cross_fitted(results[arm], eligible_zero[ARMS[arm][1]])
    cf_intervals, cf_descriptive = paired_statistics(cf[PRIMARY[0]][0], cf[PRIMARY[0]][1],
                                                     cf[PRIMARY[1]][0], cf[PRIMARY[1]][1])
    s = cf_intervals["sum_of_medians_change_mm"]
    print("\ncross-fitted weight selection (descriptive): sum of medians %+.4f mm [%+.4f, %+.4f]"
          % (s["observed"], s["lower"], s["upper"]))

    # Descriptive tail breakdown by mask observability of the structured span.
    thin_identities = sorted({c["identity"] for c in census if c["structured"] < THIN_RATIO})
    tails = {}
    for arm in PRIMARY:
        row = selected[arm]
        flagged = [row["tee_mm"][i] for i, n in enumerate(row["identities"]) if n in thin_identities]
        other = [row["tee_mm"][i] for i, n in enumerate(row["identities"]) if n not in thin_identities]
        tails[arm] = {"thin_identities_tee_mm": flagged,
                      "median_tee_other_mm": float(np.median(other)) if other else None}
    print("identities with a thin structured mask (ratio < %.0e): %d -> %s"
          % (THIN_RATIO, len(thin_identities),
             {arm: ["%.2f" % v for v in t["thin_identities_tee_mm"]] for arm, t in tails.items()}))

    # Outcome, per the pre-registration.
    primary = comparisons["%s_vs_%s" % PRIMARY]
    s = primary["intervals"]["sum_of_medians_change_mm"]
    gated_edges = [arm for arm in GATED if selected[arm]["upper_edge"] or selected[arm]["lower_edge"]]
    readable = p0_pass and p1_pass and not gated_edges
    if not readable:
        outcome = "UNREADABLE"
    elif s["upper"] < 0.0:
        outcome = "R1"
    elif s["lower"] > 0.0:
        outcome = "R3"
    else:
        outcome = "R2"
    claimable = [m for m in ("cmr", "tee")
                 if primary["intervals"]["%s_difference_of_medians_mm" % m]["upper"] < 0.0]
    tee_tail_bound = primary["descriptive"]["tee_p90_mm"][0] > primary["descriptive"]["tee_p90_mm"][1]
    carried = [m for m in claimable if not (m == "tee" and tee_tail_bound)]
    detail = outcome
    if outcome == "R1" and not carried:
        detail = "R1-empty"
    diagonal = comparisons["structured_diagonal_vs_global_covariance"]["intervals"]
    diagonal_only = outcome in ("R2", "R3") and diagonal["sum_of_medians_change_mm"]["upper"] < 0.0
    print("\noutcome: %s | claimable modalities %s | TEE tail bound %s | carried claim %s%s"
          % (detail, claimable, tee_tail_bound, carried or "none",
             " | diagonal-only clause" if diagonal_only else ""))
    if not readable:
        print("  unreadable because: P0 %s, P1 %s, gated edges %s" % (p0_pass, p1_pass, gated_edges))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "E58_covariance_equivalence.json").write_text(json.dumps({
        "experiment": "E58_covariance_equivalence",
        "preregistration": PREREGISTRATION,
        "git_revision": revision, "working_tree_clean": not dirty,
        "solver_config": dataclasses.asdict(base_config()),
        "question": "does the structured gain over global PCA survive giving both "
                    "arms the full empirical-covariance prior",
        "split": "bases from train identities; measured on validation identities; "
                 "test unused by this campaign but examined by earlier E53 selection",
        "alignment": "frozen at truth (development diagnostic, not end to end)",
        "selection_conditional": "the 12+3x12 allocation was chosen in E57 on these "
                                 "same identities and each arm's weight is chosen "
                                 "here on them; the decision interval covers sampling "
                                 "noise only; the cross-fitted interval adds weight "
                                 "selection; nothing here covers allocation selection",
        "rank": RANK, "allocation": list(ALLOCATION), "weights": WEIGHTS,
        "scale_floor": FLOOR, "resamples": RESAMPLES, "permutations": PERMUTATIONS,
        "seed": SEED, "provenance": provenance,
        "mask_census": census,
        "preconditions": {
            "p0_harness_reproduces_e57b_max_gap_mm": p0, "p0_passed": bool(p0_pass),
            "p0_tolerance_mm": P0_TOLERANCE_MM,
            "p1_rotation_invariance_gap_mm_by_weight": p1, "p1_passed": bool(p1_pass),
            "p1_tolerance_mm": P1_TOLERANCE_MM,
            "p2_zero_weight_eligible": eligible_zero, "p2_rank_ratio": RANK_RATIO,
            "iteration_cap_possibly_binding": cap_rows,
        },
        "extensions": extensions,
        "selected": {k: {kk: vv for kk, vv in v.items()
                         if kk not in ("identities", "cmr_mm", "tee_mm")}
                     for k, v in selected.items()},
        "comparisons": comparisons,
        "cross_fitted_primary": {
            "intervals": cf_intervals, "descriptive": cf_descriptive,
            "weights_chosen": {arm: cf[arm][2] for arm in PRIMARY}},
        "thin_mask_tails": {"ratio_threshold": THIN_RATIO,
                            "identities": thin_identities, "by_arm": tails},
        "outcome": outcome, "outcome_detail": detail,
        "claimable_modalities": claimable, "tee_tail_bound": bool(tee_tail_bound),
        "carried_claim": carried, "diagonal_only_clause": bool(diagonal_only),
        "results": results,
    }, indent=2) + "\n")
    print("wrote", OUT / "E58_covariance_equivalence.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
