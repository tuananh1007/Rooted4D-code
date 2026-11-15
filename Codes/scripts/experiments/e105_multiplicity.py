"""E105: equivalence bounds and multiplicity on E100's test sets (weak point 6).

Pre-registered in ``Docs/E103_E107_REVIEWER_ANALYSES_PREREGISTRATION_2026-10-03.md``.
The bootstrap draws of each reported comparison are regenerated with the same
code path, seed and draw order as ``e62_stage_b.paired`` and
``e88_replication.identity_difference``; their 2.5 and 97.5 percentiles must
reproduce the reported intervals exactly (the code check). From the same draws:

* the one-sided 95% upper bound of Rooted4D minus late fusion (sum, CMR, TEE);
* Bonferroni-adjusted intervals for E100's eight key differences, at
  1 - 0.05/8 (one test set) and 1 - 0.05/48 (the six benchmarks together).

    python e105_multiplicity.py <rows root> <E100_main.json> <E100_replication.json>
        -> results/benchmark/E105/E105_multiplicity.json

``<rows root>`` holds one directory per test set, named as below, each with the
chain's test shards.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E103_E107_REVIEWER_ANALYSES_PREREGISTRATION_2026-10-03.md"
OUT = Path("results/benchmark/E105")
TRAJECTORY = "S/late_fusion_trajectory"
#: test set -> (report file, key in that report)
TEST_SETS = {
    "v4n96o": ("main", "v4n96o"), "v3n96o": ("main", "v3n96o"), "v4n48o": ("main", "v4n48o"),
    "v3n48o": ("main", "v3n48o"), "v3o": ("main", "v3o"), "v4oriented": ("main", "v4oriented"),
    **{f"v4n96os{k}": ("replication", f"v4s{k}") for k in range(1, 6)},
    **{f"v3n96os{k}": ("replication", f"v3s{k}") for k in range(1, 6)},
}
MAIN_SIX = ("v4n96o", "v3n96o", "v4n48o", "v3n48o", "v3o", "v4oriented")
KEYS = ("late_fusion_sum", "late_fusion_cmr", "late_fusion_tee", "trajectory_rotation_deg",
        "trajectory_translation_mm", "trajectory_phase_map_cycles", "trajectory_attachment_residual_mm",
        "global_basis_sum")
LEVELS = {"per_test_set": 0.05 / 8, "six_benchmarks": 0.05 / 48}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paired_draws(stage, rows, first, second) -> dict:
    """``stage.paired``'s draws, in its order: sum, CMR, TEE from one generator."""

    _, cmr_a = stage.per_identity(rows, first, "heldout_cmr_mm")
    _, tee_a = stage.per_identity(rows, first, "heldout_tee_mm")
    _, cmr_b = stage.per_identity(rows, second, "heldout_cmr_mm")
    _, tee_b = stage.per_identity(rows, second, "heldout_tee_mm")
    rng = np.random.default_rng(stage.SEED)
    n = len(cmr_a)
    draws = {k: np.empty(stage.RESAMPLES) for k in ("sum", "cmr", "tee")}
    for r in range(stage.RESAMPLES):
        t = rng.integers(0, n, size=n)
        draws["sum"][r] = stage.sum_of_medians(cmr_a[t], tee_a[t]) - stage.sum_of_medians(cmr_b[t], tee_b[t])
        draws["cmr"][r] = np.nanmedian(cmr_a[t]) - np.nanmedian(cmr_b[t])
        draws["tee"][r] = np.nanmedian(tee_a[t]) - np.nanmedian(tee_b[t])
    observed = {"sum": stage.sum_of_medians(cmr_a, tee_a) - stage.sum_of_medians(cmr_b, tee_b),
                "cmr": float(np.nanmedian(cmr_a) - np.nanmedian(cmr_b)),
                "tee": float(np.nanmedian(tee_a) - np.nanmedian(tee_b))}
    return {k: (observed[k], draws[k]) for k in draws}


def endpoint_draws(stage, rows, first, second, key):
    """``identity_difference``'s draws for one endpoint."""

    _, a = stage.per_identity(rows, first, key)
    _, b = stage.per_identity(rows, second, key)
    rng = np.random.default_rng(stage.SEED)
    draws = np.empty(stage.RESAMPLES)
    for r in range(stage.RESAMPLES):
        t = rng.integers(0, len(a), size=len(a))
        draws[r] = np.nanmedian(a[t]) - np.nanmedian(b[t])
    return float(np.nanmedian(a) - np.nanmedian(b)), draws


def key_draws(stage, rows) -> dict:
    fusion = paired_draws(stage, rows, "S/joint", "S/independent")
    representation = paired_draws(stage, rows, "S/joint", "G/joint")
    out = {f"late_fusion_{m}": fusion[m] for m in ("sum", "cmr", "tee")}
    for key in ("rotation_deg", "translation_mm", "phase_map_cycles", "attachment_residual_mm"):
        out[f"trajectory_{key}"] = endpoint_draws(stage, rows, "S/joint", TRAJECTORY, key)
    out["global_basis_sum"] = representation["sum"]
    return out


def reported(entry: dict, key: str) -> dict:
    if key.startswith("late_fusion_"):
        return entry["T3_fusion_cost"]["comparison"][key.split("_")[-1]]
    if key.startswith("trajectory_"):
        return entry["rooted4d_minus_trajectory"][key[len("trajectory_"):]]
    return entry["T1_representation"]["comparison"]["sum"]


def main(rows_root: str, main_report: str, replication_report: str) -> None:
    stage = _load("e62_stage_b")
    reports = {"main": json.loads(Path(main_report).read_text())["instances"],
               "replication": json.loads(Path(replication_report).read_text())["instances"]}
    result = {"experiment": "E105", "preregistration": PREREGISTRATION,
              "statistic": "E62's draws, regenerated; Bonferroni levels " + json.dumps(LEVELS),
              "code_check": {}, "test_sets": {}}
    worst_check = 0.0
    for name, (source, key) in TEST_SETS.items():
        rows = stage.load_rows([str(p) for p in sorted((Path(rows_root) / name).glob("*")) if p.is_dir()]
                               or [str(Path(rows_root) / name)])
        entry = reports[source][key]
        draws = key_draws(stage, rows)
        record = {}
        for k in KEYS:
            observed, values = draws[k]
            reference = reported(entry, k)
            lower, upper = float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
            deviation = max(abs(lower - reference["lower"]), abs(upper - reference["upper"]),
                            abs(observed - reference["observed"]))
            worst_check = max(worst_check, deviation)
            item = {"observed": observed, "lower": lower, "upper": upper,
                    "excludes_zero": bool(upper < 0 or lower > 0)}
            for level_name, alpha in LEVELS.items():
                lo = float(np.percentile(values, 100 * alpha / 2))
                hi = float(np.percentile(values, 100 * (1 - alpha / 2)))
                item[level_name] = {"lower": lo, "upper": hi, "excludes_zero": bool(hi < 0 or lo > 0)}
            if k.startswith("late_fusion_"):
                item["one_sided_95_upper"] = float(np.percentile(values, 95))
            record[k] = item
        result["test_sets"][name] = record
        print(name, {k: (round(v["observed"], 3), v["excludes_zero"], v["per_test_set"]["excludes_zero"],
                         v["six_benchmarks"]["excludes_zero"]) for k, v in record.items()}, flush=True)
    result["code_check"] = {"max_deviation": worst_check, "reproduces": worst_check < 1e-12}
    # Summaries the reading asks for.
    bounds = {m: {name: result["test_sets"][name][f"late_fusion_{m}"]["one_sided_95_upper"]
                  for name in TEST_SETS} for m in ("sum", "cmr", "tee")}
    result["equivalence"] = {m: {"largest": max(v.values()), "test_set": max(v, key=v.get),
                                 "largest_main_six": max(v[n] for n in MAIN_SIX)} for m, v in bounds.items()}
    survival = {}
    for name, record in result["test_sets"].items():
        for k, item in record.items():
            if item["excludes_zero"]:
                survival.setdefault(k, {"unadjusted": 0, "per_test_set": 0, "six_benchmarks": 0, "lost": []})
                survival[k]["unadjusted"] += 1
                for level_name in LEVELS:
                    if item[level_name]["excludes_zero"]:
                        survival[k][level_name] += 1
                if not item["per_test_set"]["excludes_zero"]:
                    survival[k]["lost"].append(name)
    result["survival"] = survival
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "E105_multiplicity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"code_check": result["code_check"], "equivalence": result["equivalence"],
                      "survival": survival}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
