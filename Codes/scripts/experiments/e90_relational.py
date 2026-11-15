"""E90: relational chamber-leaflet measurements against pose-invariant controls.

Pre-registered in ``Docs/E90_RELATIONAL_MEASUREMENTS_PREREGISTRATION_2026-09-28.md``.
Measured on fits E87 stores (``store_parameters``): Rooted4D (``S/joint_rerun``
on test; ``S/joint`` in the validation code check) and late fusion with
trajectory registration, at end systole, against each sample's truth.

    python e90_relational.py spec                          -> Codes/configs/e90/v4_val.json (the code check's fits)
    python e90_relational.py measure <split> <rows dir>    -> results/benchmark/E90/E90_<split>.json
    python e90_relational.py measure test <rows dir> oriented  -> E90_test_oriented.json (E94's test fits)

``measure`` needs SciPy (the convex-hull test), so on Delta it runs in the
nnU-Net environment with the project on ``PYTHONPATH``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E90_RELATIONAL_MEASUREMENTS_PREREGISTRATION_2026-09-28.md"
TEST_SPEC = Path("Codes/configs/e87/v4_test.json")
VAL_SPEC = Path("Codes/configs/e90/v4_val.json")
JOINT = {"test": "S/joint_rerun", "val": "S/joint"}
#: Per the amendment, the measurement is repeated on E94's correctly oriented
#: rebuild: its chain's test run, where Rooted4D's stored fit is ``S/joint``.
INSTANCES = {"original": ("semisynthetic-real-v4", JOINT),
             "oriented": ("semisynthetic-real-v4-oriented", {"test": "S/joint"}),
             # E100: the rebuilt realistic-session primary (96 nodes, oriented valves).
             "v4n96o": ("e100-v4n96o", {"test": "S/joint"})}
LATE = "S/late_fusion_trajectory"
PHASE_GRID = 64
RELATIONAL = ("apex_distance_anterior_mm", "apex_distance_posterior_mm", "axis_angle_deg",
              "clearance_mm")
CONTROLS = ("area_mm2", "perimeter_mm", "intercommissural_mm", "anteroposterior_mm")
ALIGNMENT = ("translation_mm", "rotation_deg", "phase_map_cycles")
THRESHOLDS = {"apex_distance_anterior_mm": (3.0, 5.0), "apex_distance_posterior_mm": (3.0, 5.0),
              "clearance_mm": (3.0, 5.0), "axis_angle_deg": (5.0, 10.0)}
RESAMPLES, SEED = 20000, 20260910


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cmd_spec() -> None:
    """The code check's fits: E87's two stored arms, on the validation split."""

    spec = json.loads(TEST_SPEC.read_text())
    spec.update({"experiment": "E90", "preregistration": PREREGISTRATION, "split": "val",
                 "frozen_from": str(TEST_SPEC)})
    variants = []
    for variant in spec["variants"]:
        variant = json.loads(json.dumps(variant))
        if variant["name"] == "S/joint_rerun":
            variant["name"] = "S/joint"
        variants.append(variant)
    spec["variants"] = variants
    VAL_SPEC.parent.mkdir(parents=True, exist_ok=True)
    VAL_SPEC.write_text(json.dumps(spec, indent=2) + "\n")
    print("wrote", VAL_SPEC, [v["name"] for v in variants])


# ------------------------------------------------------------- geometry ----

def _segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Distance from each point to each segment, shape (points, segments)."""

    direction = end - start
    length = np.maximum((direction ** 2).sum(axis=1), 1e-12)
    t = np.clip(((points[:, None, :] - start[None]) * direction[None]).sum(axis=2) / length, 0.0, 1.0)
    nearest = start[None] + t[..., None] * direction[None]
    return np.linalg.norm(points[:, None, :] - nearest, axis=2)


def quantities(chamber: np.ndarray, valve: np.ndarray, groups: dict, n: int) -> dict:
    """Every E90 quantity for one geometry: chamber side and valve side (the same array for Rooted4D)."""

    from scipy.spatial import Delaunay

    apex = chamber[groups["apex"][0]]
    chamber_ring, valve_ring = chamber[groups["annulus"]], valve[groups["annulus"]]
    free = valve[groups["free"]]
    long_axis = apex - chamber_ring.mean(axis=0)
    leaflet_axis = free.mean(axis=0) - valve_ring.mean(axis=0)
    cosine = np.dot(long_axis, leaflet_axis) / (np.linalg.norm(long_axis) * np.linalg.norm(leaflet_axis))
    lower = chamber[groups["lower"]]
    clearance = _segment_distance(free, lower, np.roll(lower, -1, axis=0)).min()
    hull = Delaunay(chamber[np.concatenate([groups["annulus"], groups["lower"], groups["apex"]])])
    out = {
        "apex_distance_anterior_mm": float(np.linalg.norm(valve[groups["anterior_free"]].mean(axis=0) - apex)),
        "apex_distance_posterior_mm": float(np.linalg.norm(valve[groups["posterior_free"]].mean(axis=0) - apex)),
        "axis_angle_deg": float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))),
        "clearance_mm": float(clearance),
        "penetrating_vertices": int((hull.find_simplex(free) < 0).sum()),
    }
    out.update(ring_controls(valve_ring, n))
    return out


def ring_controls(ring: np.ndarray, n: int) -> dict:
    """Pose-invariant annulus measurements: area, perimeter and the two diameters."""

    half, quarter = n // 2, n // 4
    closed = np.vstack([ring, ring[:1]])
    area = 0.5 * np.linalg.norm(np.cross(closed[:-1], closed[1:]).sum(axis=0))
    commissures = (0.5 * (ring[half - 1] + ring[half]), 0.5 * (ring[n - 1] + ring[0]))
    midpoints = (0.5 * (ring[quarter - 1] + ring[quarter]),
                 0.5 * (ring[half + quarter - 1] + ring[half + quarter]))
    return {"area_mm2": float(area),
            "perimeter_mm": float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum()),
            "intercommissural_mm": float(np.linalg.norm(commissures[0] - commissures[1])),
            "anteroposterior_mm": float(np.linalg.norm(midpoints[0] - midpoints[1]))}


# ------------------------------------------------------------ measuring ----

def measure_sample(row: dict, sample: dict, model, config, joint_name: str) -> dict:
    from dataclasses import replace

    from rooted4d.experiments.fitted_geometry import (chamber_volume, fitted_trajectory, ring_groups,
                                                      truth_trajectory)
    from rooted4d.inference.baselines import parameters_from_record
    from rooted4d.inference.trajectory_registration import inverse_phase
    from rooted4d.geometry.rigid import RigidTransform
    from rooted4d.motion import MonotoneCircularPhaseWarp

    n_vertices = len(sample["template_vertices"])
    n = (n_vertices - 1) // 3
    groups = ring_groups(n)
    truth = truth_trajectory(sample)
    grid = np.linspace(0.0, 1.0, PHASE_GRID, endpoint=False)
    volumes = [chamber_volume(frame, n) for frame in truth(grid)]
    tau = float(grid[int(np.argmin(volumes))])

    # Code check: the truth plus the CMR session half-field must reproduce the
    # CMR observations up to the generator's noise (0.35 mm per coordinate).
    static, dynamic = sample["session_static"], sample["session_dynamic"]
    times, indices = sample["cmr_times"], sample["cmr_indices"]
    session = static[None] + np.cos(2 * np.pi * times)[:, None, None] * dynamic[None]
    expected = (truth(times) - 0.5 * session)[:, indices, :]
    noise_rms = float(np.sqrt(((sample["cmr_points"] - expected) ** 2).mean()))

    record = {"sample": row["sample"], "identity": row["identity"], "tau_es": tau,
              "truth_volume_range_ml": [min(volumes) / 1000.0, max(volumes) / 1000.0],
              "cmr_noise_check_rms_mm": noise_rms}
    true_geometry = truth(np.array([tau]))[0]
    record["truth"] = quantities(true_geometry, true_geometry, groups, n)

    joint = row["variants"][joint_name]
    if joint.get("succeeded"):
        parameters = parameters_from_record(joint["extras"]["parameters"])
        geometry = fitted_trajectory(parameters, model, config, np.array([tau]), n_vertices)[0]
        record["rooted4d"] = quantities(geometry, geometry, groups, n)
    late = row["variants"][LATE]
    if late.get("succeeded"):
        extras = late["extras"]
        cmr, tee = parameters_from_record(extras["cmr"]), parameters_from_record(extras["tee"])
        registration = extras["registration"]
        transform = RigidTransform(np.asarray(registration["rotation"]),
                                   np.asarray(registration["translation"]))
        warp = MonotoneCircularPhaseWarp(np.asarray(registration["logits"]), offset=registration["offset"])
        chamber = fitted_trajectory(cmr, model, replace(config, use_tee=False), np.array([tau]),
                                    n_vertices)[0]
        source = inverse_phase(warp, np.array([tau]))
        valve = transform.apply(fitted_trajectory(tee, model, config, source, n_vertices)[0])
        record["late_fusion"] = quantities(chamber, valve, groups, n)
        # The chamber-side ring, for the controls: the same measurements on the other annulus.
        record["late_fusion_chamber_ring"] = ring_controls(chamber[groups["annulus"]], n)
        record["late_fusion_alignment"] = {k: late.get(k) for k in ALIGNMENT}
    return record


def _identity_bootstrap(errors_a: dict, errors_b: dict) -> dict:
    """E62's statistic on per-sample absolute errors keyed by (identity, sample)."""

    identities = sorted({i for i, _ in errors_a})
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
            "reading": ("Rooted4D lower" if upper < 0 else "Rooted4D higher" if lower > 0
                        else "no detectable difference")}


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    for value in np.unique(values):          # average ranks for ties
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(_ranks(x[keep]), _ranks(y[keep]))[0, 1])


def summarize(records: list[dict]) -> dict:
    def errors(method, key, source=None):
        return {(r["identity"], r["sample"]): abs(r[source or method][key] - r["truth"][key])
                for r in records if method in r}

    out = {"samples": len(records), "identities": len({r["identity"] for r in records}),
           "median_absolute_error": {}, "illustrative_fractions": {}, "paired": {},
           "spearman_late_fusion": {}, "penetration_samples": {}}
    for key in RELATIONAL + CONTROLS:
        methods = {"rooted4d": errors("rooted4d", key), "late_fusion": errors("late_fusion", key)}
        if key in CONTROLS:
            methods["late_fusion_chamber_ring"] = {
                (r["identity"], r["sample"]): abs(r["late_fusion_chamber_ring"][key] - r["truth"][key])
                for r in records if "late_fusion_chamber_ring" in r}
            methods["late_fusion_oracle"] = {
                k: min(methods["late_fusion"][k], methods["late_fusion_chamber_ring"][k])
                for k in methods["late_fusion"]}
        out["median_absolute_error"][key] = {m: float(np.median(list(e.values()))) for m, e in methods.items()}
        if key in THRESHOLDS:
            out["illustrative_fractions"][key] = {
                m: {f">{t:g}": float(np.mean(np.array(list(e.values())) > t)) for t in THRESHOLDS[key]}
                for m, e in methods.items()}
        comparisons = [m for m in methods if m != "rooted4d"]
        out["paired"][key] = {f"rooted4d_minus_{m}": _identity_bootstrap(methods["rooted4d"], methods[m])
                              for m in comparisons}
    late = [r for r in records if "late_fusion" in r]
    for key in RELATIONAL:
        error = np.array([abs(r["late_fusion"][key] - r["truth"][key]) for r in late])
        out["spearman_late_fusion"][key] = {
            alignment: spearman(error, np.array([np.nan if r["late_fusion_alignment"][alignment] is None
                                                 else r["late_fusion_alignment"][alignment] for r in late]))
            for alignment in ALIGNMENT}
    for method in ("truth", "rooted4d", "late_fusion"):
        out["penetration_samples"][method] = int(sum(r[method]["penetrating_vertices"] > 0
                                                     for r in records if method in r))
    checks = np.array([r["cmr_noise_check_rms_mm"] for r in records])
    out["code_check_cmr_noise_rms_mm"] = {"median": float(np.median(checks)), "max": float(checks.max())}
    return out


def cmd_measure(split: str, rows_dir: str, instance: str = "original") -> None:
    driver = _load("e59_driver", HERE / "e59_unknown_alignment.py")
    from rooted4d.inference import SolverConfig

    suffix, joint_names = INSTANCES[instance]
    spec = json.loads((Path(rows_dir) / "spec.json").read_text())
    assert spec["split"] == split and spec["benchmark"].endswith(suffix), spec["benchmark"]
    root = Path(spec["benchmark"])
    model, *_ = driver.build_model(root, spec["models"]["S"])
    joint = next(v for v in spec["variants"] if v["name"] == joint_names[split])
    config = SolverConfig(**spec["base_config"])
    from dataclasses import replace
    joint_config = replace(config, **joint.get("overrides", {}))
    rows = [json.loads(line) for shard in sorted(Path(rows_dir).glob("shard_*.jsonl"))
            for line in shard.read_text().splitlines() if line.strip()]
    names = [r["sample"] for r in rows]
    assert len(names) == len(set(names)) == 60, (len(names), len(set(names)))
    records = []
    for row in sorted(rows, key=lambda r: r["sample"]):
        with np.load(root / "samples" / split / row["sample"], allow_pickle=True) as blob:
            sample = {k: blob[k] for k in blob.files}
        records.append(measure_sample(row, sample, model, joint_config, joint_names[split]))
    report = {"experiment": "E90", "preregistration": PREREGISTRATION, "split": split,
              "instance": instance, "benchmark": spec["benchmark"],
              "rows": rows_dir, "joint_arm": joint_names[split], "summary": summarize(records),
              "samples": records}
    out = Path(f"results/benchmark/E90/E90_{split}{'' if instance == 'original' else '_' + instance}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]["median_absolute_error"], indent=1))
    print("code check (CMR noise rms):", report["summary"]["code_check_cmr_noise_rms_mm"])
    print("wrote", out)


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    {"spec": cmd_spec, "measure": cmd_measure}[command](*arguments)
