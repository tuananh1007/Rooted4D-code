"""E87: late fusion with trajectory registration, the strong late-fusion baseline.

Pre-registered in ``Docs/E87_TRAJECTORY_LATE_FUSION_PREREGISTRATION_2026-09-28.md``.
The arm (``late_fusion_trajectory``) reuses late fusion's two independent fits
at their frozen settings and replaces its registration by a trajectory
registration (``rooted4d/inference/trajectory_registration.py``). Nothing is
tuned: each spec is derived mechanically from the frozen test spec of its
benchmark, and the E59 driver runs it.

    python e87_trajectory_late_fusion.py spec  <bench> [test|val]  -> Codes/configs/e87/<bench>_<split>.json
    python e87_trajectory_late_fusion.py analyze v4=<dir> v3=<dir> v3n48=<dir> v3n96=<dir> [pilot=<dir>]

Benchmarks: v4 (realistic session), v3 (small session) at 24 nodes, and v3 at 48
and 96 nodes. On v4 the test spec also re-runs Rooted4D unchanged, as a
same-run reference whose fitted parameters are kept for the downstream
measurements; the new arm keeps its parameters on every benchmark.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E87_TRAJECTORY_LATE_FUSION_PREREGISTRATION_2026-09-28.md"
OUT = Path("Codes/configs/e87")
REPORT = Path("results/benchmark/E87/E87_trajectory_late_fusion.json")
ARM = "S/late_fusion_trajectory"
FROZEN = {"v4": "Codes/configs/e75/v4_e75_test.json",
          "v3": "Codes/configs/e62/v3_test.json",
          "v3n48": "Codes/configs/e75/v3n48_e75_test.json",
          "v3n96": "Codes/configs/e85/v3n96_e85_test.json"}
#: The test runs whose rows the tables report, and which E87 is compared with.
ORIGINAL = {"v4": "results/benchmark/E75/v4_test",
            "v3": "results/benchmark/E64/v3_test",
            "v3n48": "results/benchmark/E75/v3n48_test",
            "v3n96": "results/benchmark/E85/v3n96_test"}
ENDPOINTS = ("rotation_deg", "translation_mm", "phase_error_cycles", "phase_map_cycles",
             "attachment_residual_mm")
EXPECTATION = ("Strong late fusion: translation ≈ 3 mm (may beat Rooted4D), rotation ≈ 2°, "
               "phase 0.01–0.05 cycles, gap several mm. Claims kept regardless: interface, "
               "phase, one anatomy.")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE = _load("e62_stage_b")


def _copy(value):
    return json.loads(json.dumps(value))


def cmd_spec(bench: str, split: str = "test") -> None:
    if split not in ("test", "val"):
        raise SystemExit("split must be test or val")
    frozen = json.loads(Path(FROZEN[bench]).read_text())
    by_name = {v["name"]: v for v in frozen["variants"]}
    # Every model of the frozen spec is kept, so the shared start is computed
    # exactly as it was in the run being compared with.
    spec = {key: frozen[key] for key in ("benchmark", "start", "models", "base_config")}
    spec.update({"experiment": "E87", "preregistration": PREREGISTRATION, "split": split,
                 "frozen_from": FROZEN[bench]})
    options = dict(_copy(by_name["S/independent"]["options"]), store_parameters=split == "test")
    variants = [{"name": ARM, "model": "S", "arm": "late_fusion_trajectory", "options": options}]
    if split == "val":
        # The pilot checks the code path and timing beside the arm it replaces.
        variants.append(_copy(by_name["S/independent"]))
    elif bench == "v4":
        rerun = _copy(by_name["S/joint"])
        rerun["name"] = "S/joint_rerun"
        rerun["options"] = dict(rerun.get("options") or {}, store_parameters=True)
        variants.append(rerun)
    spec["variants"] = variants
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{bench}_{split}.json"
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print("wrote", path, "with", len(variants), "variants")


def _merged(new_dir: str, original_dir: str) -> list:
    """Original test rows with E87's variants added; names never collide."""

    original = {r["sample"]: r for r in _rows(original_dir)}
    merged = []
    for row in _rows(new_dir):
        base = original[row["sample"]]
        assert base["identity"] == row["identity"]
        clash = set(base["variants"]) & set(row["variants"])
        assert not clash, clash
        merged.append({"sample": row["sample"], "identity": row["identity"],
                       "variants": {**base["variants"], **row["variants"]}})
    # Every original sample exactly once: a count alone would pass a duplicated
    # row standing in for a missing one (a restart under another shard layout).
    names = [r["sample"] for r in merged]
    assert len(names) == len(set(names)) and set(names) == set(original), (
        len(names), len(set(names)), len(original))
    return merged


def _check_spec(bench: str, new_dir: str) -> None:
    """The rows must come from this benchmark's E87 test spec: all four share sample names."""

    spec = json.loads((Path(new_dir) / "spec.json").read_text())
    frozen = json.loads(Path(FROZEN[bench]).read_text())
    assert (spec["experiment"], spec["split"], spec["benchmark"], spec["frozen_from"]) == (
        "E87", "test", frozen["benchmark"], FROZEN[bench]), (bench, spec["benchmark"])


def _rows(directory: str) -> list:
    rows = []
    for shard in sorted(Path(directory).glob("shard_*.jsonl")):
        rows += [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]
    return rows


def identity_difference(rows, first, second, key):
    """E62's statistic for any endpoint (same seed and draw order as ``paired``)."""

    ids_a, a = STAGE.per_identity(rows, first, key)
    ids_b, b = STAGE.per_identity(rows, second, key)
    assert ids_a == ids_b
    rng = np.random.default_rng(STAGE.SEED)
    draws = np.empty(STAGE.RESAMPLES)
    for r in range(STAGE.RESAMPLES):
        t = rng.integers(0, len(a), size=len(a))
        draws[r] = np.nanmedian(a[t]) - np.nanmedian(b[t])
    lower, upper = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return {"observed": float(np.nanmedian(a) - np.nanmedian(b)), "lower": lower, "upper": upper,
            "reading": ("first lower" if upper < 0 else "first higher" if lower > 0
                        else "no detectable difference")}


def _values(rows, name, key):
    """Per-sample values in row order, NaN where the variant failed or has none."""

    out = []
    for row in rows:
        outcome = row["variants"].get(name) or {}
        value = outcome.get(key) if outcome.get("succeeded") else None
        out.append(np.nan if value is None else float(value))
    return np.array(out)


def reproduction_of_fits(rows) -> dict:
    """The new arm's fits are late fusion's: held-out errors must agree per sample."""

    out = {}
    for key in ("heldout_cmr_mm", "heldout_tee_mm"):
        difference = _values(rows, ARM, key) - _values(rows, "S/independent", key)
        out[key] = {"max_abs_difference": float(np.nanmax(np.abs(difference))),
                    "median_difference": float(np.nanmedian(_values(rows, ARM, key))
                                               - np.nanmedian(_values(rows, "S/independent", key)))}
    out["passed"] = all(abs(v["median_difference"]) <= 0.01 for v in out.values())
    return out


def reproduction_of_joint(rows) -> dict:
    out = {}
    for key in ("heldout_cmr_mm", "heldout_tee_mm", "rotation_deg", "translation_mm",
                "phase_map_cycles"):
        rerun = STAGE.sample_medians(rows, "S/joint_rerun")[key]
        original = STAGE.sample_medians(rows, "S/joint")[key]
        out[key] = {"rerun": rerun, "original": original, "difference": rerun - original}
    out["passed"] = all(abs(out[k]["difference"]) <= 0.01 for k in ("heldout_cmr_mm", "heldout_tee_mm"))
    return out


def distribution(rows, key) -> dict:
    values = _values(rows, ARM, key)
    values = values[np.isfinite(values)]
    return {"median": float(np.median(values)), "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)), "max": float(values.max()),
            "samples": int(values.size)}


def bench_report(bench: str, new_dir: str) -> dict:
    _check_spec(bench, new_dir)
    rows = _merged(new_dir, ORIGINAL[bench])
    arms = [ARM, "S/independent", "S/exact_repair", "S/joint"]
    report = {
        "identities": len({r["identity"] for r in rows}), "samples": len(rows),
        "original_rows": ORIGINAL[bench], "e87_rows": new_dir,
        "medians": {name: STAGE.sample_medians(rows, name) for name in arms},
        "reproduction_of_fits": reproduction_of_fits(rows),
        "registration_rms_mm": distribution(rows, "repair_registration_rms_mm"),
        # The interface statement reports the gap as measured, with its distribution.
        "gap_mm": distribution(rows, "attachment_residual_mm"),
        "rooted4d_minus_trajectory": {k: identity_difference(rows, "S/joint", ARM, k)
                                      for k in ENDPOINTS},
        "trajectory_minus_feature": {k: identity_difference(rows, ARM, "S/independent", k)
                                     for k in ENDPOINTS},
        "trajectory_minus_exact_repair": {k: identity_difference(rows, ARM, "S/exact_repair", k)
                                          for k in ENDPOINTS if k != "attachment_residual_mm"},
    }
    translation = _values(rows, ARM, "translation_mm") < _values(rows, "S/joint", "translation_mm")
    report["samples_where_trajectory_translation_beats_rooted4d"] = int(np.sum(translation))
    if any("S/joint_rerun" in r["variants"] for r in rows):
        report["reproduction_of_joint"] = reproduction_of_joint(rows)
    return report


def cmd_analyze(*arguments: str) -> None:
    given = dict(argument.split("=", 1) for argument in arguments)
    report = {"experiment": "E87", "preregistration": PREREGISTRATION,
              "statistics": "E62's: identity-averaged, difference of medians, %d identity "
                            "resamples, seed %d; descriptive, a difference is read only where "
                            "the 95%% interval excludes zero" % (STAGE.RESAMPLES, STAGE.SEED),
              "expectation_as_written": EXPECTATION, "benchmarks": {}}
    if "pilot" in given:
        rows = _rows(given["pilot"])
        report["validation_pilot"] = {
            "samples": len(rows),
            "gap_mm": distribution(rows, "attachment_residual_mm"),
            "medians": {name: STAGE.sample_medians(rows, name) for name in (ARM, "S/independent")},
            "failures": {name: sum(1 for r in rows if not r["variants"][name].get("succeeded"))
                         for name in (ARM, "S/independent")}}
    for bench in FROZEN:
        if bench in given:
            report["benchmarks"][bench] = bench_report(bench, given[bench])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    for bench, entry in report["benchmarks"].items():
        m = entry["medians"]
        print(bench, {name: {k: round(m[name][k], 3) for k in ENDPOINTS if m[name][k] is not None}
                      for name in (ARM, "S/independent", "S/joint")},
              "fits reproduce:", entry["reproduction_of_fits"]["passed"])
    print("wrote", REPORT)


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    {"spec": cmd_spec, "analyze": cmd_analyze}[command](*arguments)
