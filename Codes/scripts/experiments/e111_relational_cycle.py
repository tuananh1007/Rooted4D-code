"""E111: relational chamber-leaflet measurements over the whole cycle.

Pre-registered in ``Docs/E110_E113_MARGIN_AND_COHORT_PREREGISTRATION_2026-10-21.md``.
E90 measured at end systole; this repeats its free-edge-to-apex distances at
every phase of a 64-point grid and adds the free-edge height above the annular
plane, which late fusion can only answer twice (against its chamber-side ring
or its valve-side ring). Measured on stored fits: Rooted4D (``S/joint``) and
late fusion with trajectory registration (``S/late_fusion_trajectory``),
against each sample's composite truth.

    python e111_relational_cycle.py measure <split> <rows dir>   -> results/benchmark/E111/E111_<split>.json

The validation run is the code check (E113's unthinned validation rows store
the same two arms); the test run reads E100's stored test rows.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E110_E113_MARGIN_AND_COHORT_PREREGISTRATION_2026-10-21.md"
JOINT, LATE = "S/joint", "S/late_fusion_trajectory"
PHASE_GRID = 64
RESAMPLES, SEED = 20000, 20260910
#: Quantities per phase. Late fusion's apex distances are the same under both
#: of its rings; its heights and area are not.
QUANTITIES = ("apex_distance_anterior_mm", "apex_distance_posterior_mm",
              "height_anterior_mm", "height_posterior_mm", "area_mm2")
RELATIONAL = QUANTITIES[:4]
CONTROLS = QUANTITIES[4:]
STATS = ("es", "rms", "excursion")
METHODS = ("rooted4d", "late_fusion_chamber_ring", "late_fusion_valve_ring")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------ geometry ----

def plane_height(ring: np.ndarray, point: np.ndarray, toward: np.ndarray) -> float:
    """Signed distance of ``point`` from the ring's least-squares plane, positive toward ``toward``."""

    centre = ring.mean(axis=0)
    normal = np.linalg.svd(ring - centre)[2][2]
    if np.dot(toward - centre, normal) < 0.0:
        normal = -normal
    return float(np.dot(point - centre, normal))


def ring_area(ring: np.ndarray) -> float:
    closed = np.vstack([ring, ring[:1]])
    return float(0.5 * np.linalg.norm(np.cross(closed[:-1], closed[1:]).sum(axis=0)))


def quantities(chamber: np.ndarray, valve: np.ndarray, ring: np.ndarray, groups: dict) -> dict:
    """One phase: the apex from the chamber side, the free edge from the valve side, heights against ``ring``."""

    apex = chamber[groups["apex"][0]]
    anterior = valve[groups["anterior_free"]].mean(axis=0)
    posterior = valve[groups["posterior_free"]].mean(axis=0)
    return {"apex_distance_anterior_mm": float(np.linalg.norm(anterior - apex)),
            "apex_distance_posterior_mm": float(np.linalg.norm(posterior - apex)),
            "height_anterior_mm": plane_height(ring, anterior, apex),
            "height_posterior_mm": plane_height(ring, posterior, apex),
            "area_mm2": ring_area(ring)}


def series(chamber: np.ndarray, valve: np.ndarray, ring_side: str, groups: dict) -> dict:
    """Every quantity on the phase grid; ``ring_side`` picks the chamber's or the valve's annulus."""

    out = {q: np.empty(len(chamber)) for q in QUANTITIES}
    for t in range(len(chamber)):
        ring = (chamber if ring_side == "chamber" else valve)[t][groups["annulus"]]
        for q, value in quantities(chamber[t], valve[t], ring, groups).items():
            out[q][t] = value
    return out


def statistics(fitted: dict, truth: dict, es_index: int) -> dict:
    out = {}
    for q in QUANTITIES:
        error = fitted[q] - truth[q]
        out[q] = {"es": float(abs(error[es_index])), "rms": float(np.sqrt((error ** 2).mean())),
                  "excursion": float(abs(np.ptp(fitted[q]) - np.ptp(truth[q])))}
    return out


def measure_sample(row: dict, sample: dict, model, config, joint_name: str) -> dict:
    from rooted4d.experiments.fitted_geometry import (chamber_volume, fitted_trajectory, ring_groups,
                                                      truth_trajectory)
    from rooted4d.geometry.rigid import RigidTransform
    from rooted4d.inference.baselines import parameters_from_record
    from rooted4d.inference.trajectory_registration import inverse_phase
    from rooted4d.motion import MonotoneCircularPhaseWarp

    n_vertices = len(sample["template_vertices"])
    n = (n_vertices - 1) // 3
    groups = ring_groups(n)
    truth = truth_trajectory(sample)
    grid = np.linspace(0.0, 1.0, PHASE_GRID, endpoint=False)
    true_geometry = truth(grid)
    volumes = [chamber_volume(frame, n) for frame in true_geometry]
    es_index = int(np.argmin(volumes))

    # E90's code check: the truth plus the CMR half of the session field
    # reproduces the CMR observations up to the generator's noise.
    static, dynamic = sample["session_static"], sample["session_dynamic"]
    times, indices = sample["cmr_times"], sample["cmr_indices"]
    session = static[None] + np.cos(2 * np.pi * times)[:, None, None] * dynamic[None]
    expected = (truth(times) - 0.5 * session)[:, indices, :]
    noise_rms = float(np.sqrt(((sample["cmr_points"] - expected) ** 2).mean()))

    record = {"sample": row["sample"], "identity": row["identity"], "tau_es": float(grid[es_index]),
              "cmr_noise_check_rms_mm": noise_rms, "statistics": {}, "truth": {}}
    true_series = series(true_geometry, true_geometry, "chamber", groups)
    record["truth"] = {q: {"es": float(true_series[q][es_index]), "excursion": float(np.ptp(true_series[q]))}
                       for q in QUANTITIES}

    joint = row["variants"].get(joint_name) or {}
    if joint.get("succeeded"):
        parameters = parameters_from_record(joint["extras"]["parameters"])
        geometry = fitted_trajectory(parameters, model, config, grid, n_vertices)
        record["statistics"]["rooted4d"] = statistics(series(geometry, geometry, "chamber", groups),
                                                      true_series, es_index)
    late = row["variants"].get(LATE) or {}
    if late.get("succeeded"):
        extras = late["extras"]
        cmr, tee = parameters_from_record(extras["cmr"]), parameters_from_record(extras["tee"])
        registration = extras["registration"]
        transform = RigidTransform(np.asarray(registration["rotation"]), np.asarray(registration["translation"]))
        warp = MonotoneCircularPhaseWarp(np.asarray(registration["logits"]), offset=registration["offset"])
        chamber = fitted_trajectory(cmr, model, replace(config, use_tee=False), grid, n_vertices)
        valve = transform.apply(fitted_trajectory(tee, model, config, inverse_phase(warp, grid), n_vertices))
        by_ring = {"chamber": series(chamber, valve, "chamber", groups),
                   "valve": series(chamber, valve, "valve", groups)}
        for side in ("chamber", "valve"):
            record["statistics"][f"late_fusion_{side}_ring"] = statistics(by_ring[side], true_series, es_index)
        # The two answers late fusion gives for a height, and how far apart they are.
        record["late_fusion_disagreement"] = {
            q: {"es": float(abs(by_ring["chamber"][q][es_index] - by_ring["valve"][q][es_index])),
                "rms": float(np.sqrt(((by_ring["chamber"][q] - by_ring["valve"][q]) ** 2).mean()))}
            for q in ("height_anterior_mm", "height_posterior_mm", "area_mm2")}
        record["late_fusion_gap_mm"] = late.get("attachment_residual_mm")
    return record


# ------------------------------------------------------------ summary ----

def _identity_bootstrap(errors_a: dict, errors_b: dict) -> dict:
    """E62's statistic on per-sample values keyed by (identity, sample): median of A minus median of B."""

    identities = sorted({i for i, _ in errors_a} & {i for i, _ in errors_b})

    def per_identity(errors):
        return np.array([np.nanmean([v for (i, _), v in errors.items() if i == identity])
                         for identity in identities])

    a, b = per_identity(errors_a), per_identity(errors_b)
    rng = np.random.default_rng(SEED)
    draws = np.empty(RESAMPLES)
    for r in range(RESAMPLES):
        t = rng.integers(0, len(a), size=len(a))
        draws[r] = np.nanmedian(a[t]) - np.nanmedian(b[t])
    lower, upper = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return {"observed": float(np.nanmedian(a) - np.nanmedian(b)), "lower": lower, "upper": upper,
            "identities": len(identities),
            "reading": ("Rooted4D lower" if upper < 0 else "Rooted4D higher" if lower > 0
                        else "no detectable difference")}


def summarize(records: list[dict]) -> dict:
    def values(method, q, stat):
        return {(r["identity"], r["sample"]): r["statistics"][method][q][stat]
                for r in records if method in r["statistics"]}

    out = {"samples": len(records), "identities": len({r["identity"] for r in records}),
           "median_absolute_error": {}, "paired": {}, "late_fusion_disagreement": {}}
    for q in QUANTITIES:
        out["median_absolute_error"][q] = {}
        out["paired"][q] = {}
        for stat in STATS:
            per_method = {m: values(m, q, stat) for m in METHODS}
            out["median_absolute_error"][q][stat] = {m: float(np.median(list(v.values())))
                                                     for m, v in per_method.items() if v}
            out["paired"][q][stat] = {
                f"rooted4d_minus_{m}": _identity_bootstrap(per_method["rooted4d"], per_method[m])
                for m in METHODS[1:] if per_method[m] and per_method["rooted4d"]}
    late = [r for r in records if "late_fusion_disagreement" in r]
    for q in ("height_anterior_mm", "height_posterior_mm", "area_mm2"):
        out["late_fusion_disagreement"][q] = {
            stat: float(np.median([r["late_fusion_disagreement"][q][stat] for r in late])) for stat in ("es", "rms")}
    out["late_fusion_gap_median_mm"] = float(np.median([r["late_fusion_gap_mm"] for r in late
                                                        if r.get("late_fusion_gap_mm") is not None]))
    out["truth_medians"] = {q: {"es": float(np.median([r["truth"][q]["es"] for r in records])),
                                "excursion": float(np.median([r["truth"][q]["excursion"] for r in records]))}
                            for q in QUANTITIES}
    checks = np.array([r["cmr_noise_check_rms_mm"] for r in records])
    out["code_check_cmr_noise_rms_mm"] = {"median": float(np.median(checks)), "max": float(checks.max())}
    return out


def predictions(summary: dict) -> dict:
    """The pre-registered readings P1 to P3, from the summary."""

    paired = summary["paired"]
    p1 = {q: paired[q]["rms"]["rooted4d_minus_late_fusion_chamber_ring"]["reading"] == "Rooted4D lower"
          for q in ("apex_distance_anterior_mm", "apex_distance_posterior_mm")}
    p2 = {}
    for q in ("height_anterior_mm", "height_posterior_mm"):
        readings = {m: paired[q]["rms"][f"rooted4d_minus_{m}"]["reading"] for m in METHODS[1:]}
        p2[q] = {"readings": readings, "better_than_at_least_one": any(v == "Rooted4D lower" for v in readings.values()),
                 "better_ring_for_late_fusion": min(METHODS[1:], key=lambda m: summary["median_absolute_error"][q]["rms"][m])}
    p3 = {q: paired[q]["rms"]["rooted4d_minus_late_fusion_chamber_ring"]["reading"] == "no detectable difference"
          for q in CONTROLS}
    return {"P1_apex_distance_rms_lower": p1, "P2_heights": p2, "P3_control_no_difference": p3,
            "disagreement_vs_gap": {"height_rms_disagreement_median_mm": {
                q: summary["late_fusion_disagreement"][q]["rms"] for q in ("height_anterior_mm", "height_posterior_mm")},
                "late_fusion_gap_median_mm": summary["late_fusion_gap_median_mm"]}}


def cmd_measure(split: str, rows_dir: str, joint_name: str = JOINT) -> None:
    driver = _load("e59_unknown_alignment")
    from rooted4d.inference import SolverConfig

    spec = json.loads((Path(rows_dir) / "spec.json").read_text())
    assert spec["split"] == split, (spec["split"], split)
    assert not spec.get("observation_thinning") or all(v == 1 for v in spec["observation_thinning"].values()), \
        "E111 measures fits of the full evidence only"
    root = Path(spec["benchmark"])
    model, *_ = driver.build_model(root, spec["models"]["S"])
    joint = next(v for v in spec["variants"] if v["name"] == joint_name)
    config = replace(SolverConfig(**spec["base_config"]), **joint.get("overrides", {}))
    rows = [json.loads(line) for shard in sorted(Path(rows_dir).glob("shard_*.jsonl"))
            for line in shard.read_text().splitlines() if line.strip()]
    names = [r["sample"] for r in rows]
    assert len(names) == len(set(names)), "duplicate rows"
    records = [measure_sample(row, {k: blob[k] for k in blob.files}, model, config, joint_name)
               for row in sorted(rows, key=lambda r: r["sample"])
               for blob in [np.load(root / "samples" / split / row["sample"], allow_pickle=True)]]
    summary = summarize(records)
    report = {"experiment": "E111", "preregistration": PREREGISTRATION, "split": split,
              "benchmark": spec["benchmark"], "rows": rows_dir, "joint_arm": joint_name, "late_arm": LATE,
              "phase_grid": PHASE_GRID, "summary": summary, "predictions": predictions(summary), "samples": records}
    out = Path(f"results/benchmark/E111/E111_{split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    for q in QUANTITIES:
        print(q, json.dumps({s: summary["median_absolute_error"][q][s] for s in STATS}))
        for s in STATS:
            for m, entry in summary["paired"][q][s].items():
                print("   %-9s %-45s %+.3f [%+.3f, %+.3f] %s" % (s, m, entry["observed"], entry["lower"],
                                                               entry["upper"], entry["reading"]))
    print("late fusion's two heights disagree by (median rms):", summary["late_fusion_disagreement"])
    print("code check (CMR noise rms):", summary["code_check_cmr_noise_rms_mm"])
    print(json.dumps(report["predictions"], indent=1))
    print("wrote", out)


def latex(path: Path) -> str:
    """Rows: quantity and statistic, Rooted4D, late fusion (chamber-side ring), difference; then the valve-side ring."""

    tables = _load("e64_manuscript_tables")
    summary = json.loads(path.read_text())["summary"]
    labels = {"apex_distance_anterior_mm": "Free edge to apex, leaflet 1 (mm)",
              "apex_distance_posterior_mm": "Free edge to apex, leaflet 2 (mm)",
              "height_anterior_mm": "Free-edge height, leaflet 1 (mm)",
              "height_posterior_mm": "Free-edge height, leaflet 2 (mm)", "area_mm2": "Annulus area (mm$^2$)"}
    stat_labels = {"es": "end systole", "rms": "over the cycle, RMS", "excursion": "excursion"}
    lines = []
    for m, heading in (("late_fusion_chamber_ring", "chamber-side ring"), ("late_fusion_valve_ring", "valve-side ring")):
        lines.append(r"\multicolumn{4}{@{}l}{\textit{Against late fusion's %s}}\\" % heading)
        for q in QUANTITIES:
            for s in STATS:
                med = summary["median_absolute_error"][q][s]
                entry = summary["paired"][q][s].get(f"rooted4d_minus_{m}")
                if entry is None:
                    continue
                digits = 1 if q == "area_mm2" else 2
                lines.append("%s, %s & %.*f & %.*f & %s\\\\" % (labels[q], stat_labels[s], digits, med["rooted4d"],
                                                              digits, med[m], tables.interval(entry, digits)))
    return "\n".join(lines)


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    if command == "latex":
        print(latex(Path(arguments[0])))
    else:
        {"measure": cmd_measure}[command](*arguments)
