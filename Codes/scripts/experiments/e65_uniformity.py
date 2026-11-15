"""E65-E67: the temporal and session studies, re-run with the frozen structured configuration.

The rules are fixed in ``Docs/E65_E72_UNIFORMITY_PREREGISTRATION_2026-09-24.md``.
This script writes the driver specs (``e59_unknown_alignment.py`` runs them)
and applies the pre-registered selection and statistics, reusing E62's stage
module so that every interval is computed by the same code as the main tables.

    python e65_uniformity.py e65                 -> v3 validation spec: both temporal families
    python e65_uniformity.py e66 <e65 dirs...>   -> v3 test spec at E65's selections
    python e65_uniformity.py e67                 -> v4 test spec: the session term
    python e65_uniformity.py analyze <e65 dir> <e66 dir> <e67 dir> [e64 v3 dir] [e64 v4 dir]
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PREREGISTRATION = "Docs/E65_E72_UNIFORMITY_PREREGISTRATION_2026-09-24.md"
OUT = Path("Codes/configs/e65")
STAGE = Path("Codes/scripts/experiments/e62_stage_b.py")
ORDERS = (2, 3, 4, 6)
NODES = (4, 6, 8, 12, 16)
MOTION_WEIGHTS = (0.03, 0.1, 0.3, 1.0)


def _stage():
    spec = importlib.util.spec_from_file_location("e62_stage_b", STAGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen(bench: str, split: str, experiment: str) -> tuple[dict, float]:
    """The frozen S configuration on one benchmark, from E62's gate spec."""

    gate = json.loads((Path("Codes/configs/e62") / f"{bench}_gate.json").read_text())
    spec = {
        "experiment": experiment, "preregistration": PREREGISTRATION,
        "benchmark": gate["benchmark"], "split": split, "start": "input_derived",
        "models": {"S": gate["models"]["S"]}, "base_config": dict(gate["base_config"]),
        "variants": [],
    }
    return spec, float(gate["frozen"]["round1"]["S/joint"])


def write(spec: dict, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print("wrote", path, "with", len(spec["variants"]), "variants")
    return path


def periodic_name(order, weight):
    return f"S/periodic/k={order}/wh={weight:g}"


def per_frame_name(nodes, weight):
    return f"S/per_frame/n={nodes}/wh={weight:g}"


def cmd_e65():
    spec, w = frozen("v3", "val", "E65")
    for order in ORDERS:
        for weight in MOTION_WEIGHTS:
            spec["variants"].append({
                "name": periodic_name(order, weight), "model": "S", "arm": "rooted4d_full",
                "overrides": {"identity_prior_weight": w, "fourier_order": order,
                              "motion_prior_weight": weight}})
    for nodes in NODES:
        for weight in MOTION_WEIGHTS:
            spec["variants"].append({
                "name": per_frame_name(nodes, weight), "model": "S", "arm": "rooted4d_full",
                "overrides": {"identity_prior_weight": w, "motion_model": "per_frame",
                              "motion_nodes": nodes, "motion_prior_weight": weight}})
    write(spec, "v3_e65_temporal_selection.json")


def e65_selection(rows):
    """Lowest sum of modality medians per family; ties to fewer motion rows."""

    stage = _stage()
    choices = {}
    for family, grid, name, rows_of in (
            ("periodic", [(k, w) for k in ORDERS for w in MOTION_WEIGHTS], periodic_name,
             lambda k: 2 * k),
            ("per_frame", [(n, w) for n in NODES for w in MOTION_WEIGHTS], per_frame_name,
             lambda n: n)):
        scored = [(stage.score(rows, name(a, w), "sum"), rows_of(a), a, w) for a, w in grid]
        best = min(scored, key=lambda t: (round(t[0], 12), t[1]))
        choices[family] = {"setting": best[2], "motion_weight": best[3], "score": best[0],
                           "grid": {name(a, w): s for s, _, a, w in scored}}
    return choices


def cmd_e66(*directories):
    stage = _stage()
    choices = e65_selection(stage.load_rows(directories))
    spec, w = frozen("v3", "test", "E66")
    spec["e65_selection"] = {k: {"setting": v["setting"], "motion_weight": v["motion_weight"]}
                             for k, v in choices.items()}
    k, wk = choices["periodic"]["setting"], choices["periodic"]["motion_weight"]
    n, wn = choices["per_frame"]["setting"], choices["per_frame"]["motion_weight"]
    periodic = {"identity_prior_weight": w, "fourier_order": k, "motion_prior_weight": wk}
    spec["variants"] = [
        {"name": "S/periodic", "model": "S", "arm": "rooted4d_full", "overrides": periodic},
        {"name": "S/per_frame", "model": "S", "arm": "rooted4d_full",
         "overrides": {"identity_prior_weight": w, "motion_model": "per_frame",
                       "motion_nodes": n, "motion_prior_weight": wn}},
        {"name": "S/fixed_warp", "model": "S", "arm": "rooted4d_full",
         "overrides": dict(periodic, estimate_phase_warp=False)},
    ]
    print("E65 selection:", json.dumps(spec["e65_selection"]))
    write(spec, "v3_e66_temporal_test.json")


def cmd_e67():
    spec, w = frozen("v4", "test", "E67")
    spec["variants"] = [
        {"name": "S/frozen", "model": "S", "arm": "rooted4d_full",
         "overrides": {"identity_prior_weight": w}},
        {"name": "S/oscillatory", "model": "S", "arm": "rooted4d_full",
         "overrides": {"identity_prior_weight": w, "session_includes_offset": False}},
        {"name": "S/no_session", "model": "S", "arm": "rooted4d_full",
         "overrides": {"identity_prior_weight": w, "use_session": False}},
    ]
    write(spec, "v4_e67_session_test.json")


# ------------------------------------------------------------------ analysis ----

DESCRIPTIVE = ("rotation_deg", "translation_mm", "phase_map_cycles", "phase_error_cycles")


def identity_difference(stage, rows, first, second, key):
    """E62's statistic for any endpoint (same seed and draw order as ``paired``)."""

    ids_a, a = stage.per_identity(rows, first, key)
    ids_b, b = stage.per_identity(rows, second, key)
    assert ids_a == ids_b
    rng = np.random.default_rng(stage.SEED)
    draws = np.empty(stage.RESAMPLES)
    for r in range(stage.RESAMPLES):
        t = rng.integers(0, len(a), size=len(a))
        draws[r] = np.nanmedian(a[t]) - np.nanmedian(b[t])
    return {"observed": float(np.nanmedian(a) - np.nanmedian(b)),
            "lower": float(np.percentile(draws, 2.5)), "upper": float(np.percentile(draws, 97.5))}


def compare(stage, rows, first, second):
    out = stage.paired(rows, first, second)
    out["descriptive_intervals"] = {k: identity_difference(stage, rows, first, second, k)
                                    for k in DESCRIPTIVE}
    return out


def reproduction(stage, rows, name, reference_rows, reference_name):
    """Median-level agreement of a re-run with E64's rows for the same configuration."""

    out = {}
    for key in ("heldout_cmr_mm", "heldout_tee_mm", "rotation_deg", "translation_mm"):
        mine = stage.sample_medians(rows, name)[key]
        theirs = stage.sample_medians(reference_rows, reference_name)[key]
        out[key] = {"rerun": mine, "e64": theirs, "difference": mine - theirs}
    out["passed"] = all(abs(v["difference"]) <= 0.01 for k, v in out.items()
                        if k in ("heldout_cmr_mm", "heldout_tee_mm"))
    return out


def cmd_analyze(e65_dir, e66_dir, e67_dir, e64_v3=None, e64_v4=None):
    stage = _stage()
    report = {"experiments": "E65-E67", "preregistration": PREREGISTRATION}
    report["E65"] = e65_selection(stage.load_rows([e65_dir]))
    rows66 = stage.load_rows([e66_dir])
    report["E66"] = {
        "medians": {n: stage.sample_medians(rows66, n)
                    for n in ("S/periodic", "S/per_frame", "S/fixed_warp")},
        "periodic_minus_per_frame": compare(stage, rows66, "S/periodic", "S/per_frame"),
        "periodic_minus_fixed_warp": compare(stage, rows66, "S/periodic", "S/fixed_warp"),
    }
    rows67 = stage.load_rows([e67_dir])
    report["E67"] = {
        "medians": {n: stage.sample_medians(rows67, n)
                    for n in ("S/frozen", "S/oscillatory", "S/no_session")},
        "frozen_minus_no_session": compare(stage, rows67, "S/frozen", "S/no_session"),
        "frozen_minus_oscillatory": compare(stage, rows67, "S/frozen", "S/oscillatory"),
        "oscillatory_minus_no_session": compare(stage, rows67, "S/oscillatory", "S/no_session"),
    }
    if e64_v4:
        report["E67"]["reproduction_of_e64"] = reproduction(
            stage, rows67, "S/frozen", stage.load_rows([e64_v4]), "S/joint")
    if e64_v3:
        spec = json.loads((Path(e66_dir) / "spec.json").read_text())
        chosen = spec["e65_selection"]["periodic"]
        if chosen["setting"] == 3 and abs(chosen["motion_weight"] - 0.3) < 1e-12:
            report["E66"]["reproduction_of_e64"] = reproduction(
                stage, rows66, "S/periodic", stage.load_rows([e64_v3]), "S/joint")
    out = Path("results/benchmark/E65_E67_uniformity.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    {"e65": cmd_e65, "e66": cmd_e66, "e67": cmd_e67, "analyze": cmd_analyze}[command](*arguments)
