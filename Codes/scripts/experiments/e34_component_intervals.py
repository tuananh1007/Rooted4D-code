"""Bootstrap confidence intervals for the component-fitting scores.

Panels A and B of the paper reported point estimates with intervals marked
pending. The reason they were pending is that the naive bootstrap is wrong
here: the chamber evaluation has 300 frames from 150 patients, two frames per
patient, and resampling frames would treat 300 correlated observations as 300
independent ones and produce intervals that are too narrow. The resampling unit
has to be the patient, with both of that patient's frames carried together.

The valve evaluation is one case per subject, so its unit is the case. The
ejection-fraction rows are already one per patient.

Percentile intervals on the median are reported, matching how the point
estimates are summarised. 20000 resamples at a fixed seed.
"""

import json
from pathlib import Path

import numpy as np

RESAMPLES = 20_000
SEED = 20260910
CHAMBER = Path("results/benchmark/component_chamber_acdc_full.json")
VALVE = Path("results/21928873/component_valve_corrected.json")
OUTPUT = Path("results/benchmark/E34_component_intervals.json")


def cluster_bootstrap_median(values_by_unit, rng):
    """Percentile interval on the median, resampling whole units with replacement."""
    units = list(values_by_unit)
    pooled = np.concatenate([values_by_unit[u] for u in units])
    point = float(np.median(pooled))

    lengths = np.array([values_by_unit[u].size for u in units])
    # Pre-flatten so each resample is one fancy-index rather than a Python loop.
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    flat = pooled
    draws = rng.integers(0, len(units), size=(RESAMPLES, len(units)))

    medians = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        picked = draws[i]
        idx = np.concatenate([np.arange(offsets[u], offsets[u + 1]) for u in picked])
        medians[i] = np.median(flat[idx])
    low, high = (float(v) for v in np.percentile(medians, [2.5, 97.5]))
    return {"point": point, "ci_low": low, "ci_high": high,
            "units": len(units), "observations": int(lengths.sum())}


def group(rows, unit_key, value_fn):
    grouped: dict = {}
    for row in rows:
        value = value_fn(row)
        if value is None or not np.isfinite(value):
            continue
        grouped.setdefault(row[unit_key], []).append(float(value))
    return {k: np.asarray(v, dtype=float) for k, v in grouped.items()}


def main() -> int:
    rng = np.random.default_rng(SEED)
    result: dict = {
        "experiment": "E34_component_intervals",
        "resamples": RESAMPLES,
        "seed": SEED,
        "statistic": "median, percentile bootstrap",
        "resampling_unit": {
            "chamber": "patient, both frames carried together",
            "valve": "case",
            "ejection_fraction": "patient",
        },
        "chamber": {},
        "valve": {},
    }

    chamber = json.loads(CHAMBER.read_text())
    for name, key in (
        ("symmetric_mm", "symmetric_mean_mm"),
        ("hausdorff_mm", "hausdorff_mm"),
    ):
        by_patient = group(chamber["rows"], "patient",
                           lambda r, k=key: r["fit_vs_segmentation"][k])
        result["chamber"][name] = cluster_bootstrap_median(by_patient, rng)

    ef_rows = chamber["ejection_fraction"]["rows"]
    ef_key = next(k for k in ef_rows[0] if "error" in k.lower())
    ef_by_patient = {i: np.array([abs(float(r[ef_key]))]) for i, r in enumerate(ef_rows)}
    result["chamber"]["ef_absolute_points"] = cluster_bootstrap_median(ef_by_patient, rng)

    valve = json.loads(VALVE.read_text())
    for leaflet in ("anterior", "posterior"):
        for name, key in (("symmetric_mm", "symmetric_mean_mm"),
                          ("hausdorff_mm", "hausdorff_mm")):
            by_case = group(valve["rows"], "case",
                            lambda r, l=leaflet, k=key: r[l][k])
            result["valve"]["%s_%s" % (leaflet, name)] = cluster_bootstrap_median(by_case, rng)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")

    for panel in ("chamber", "valve"):
        print("\n%s" % panel)
        for name, stat in result[panel].items():
            print("  %-28s %7.3f  [%7.3f, %7.3f]  (%d units, %d obs)"
                  % (name, stat["point"], stat["ci_low"], stat["ci_high"],
                     stat["units"], stat["observations"]))
    print("\nwrote %s" % OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
