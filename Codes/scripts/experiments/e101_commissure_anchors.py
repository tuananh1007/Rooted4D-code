"""E101: the commissure-anchored configuration, E89's follow-up (plan v2, W1.3).

Pre-registered in ``Docs/E101_COMMISSURE_ANCHORS_PREREGISTRATION_2026-10-02.md``.
E89 found Rooted4D's degradation under measured front-end error tracking the
ring disagreement, and most of the largest errors were the valve fit numbering
its ring from an axis of arbitrary sign. Both arms here anchor correspondence at
the two commissures, found from the leaflet labels by ``valve_commissures``:

* **front end** (primary): each predicted valve is fitted with its numbering
  seam placed where the reference valve's seam lies relative to the reference's
  commissures, so the ring and both free edges are numbered from the
  commissures. Rooted4D and late fusion are re-run on that measured-error
  instance.
* **solver** (secondary): curve evidence (``SolverConfig.curve_evidence``) with
  the two commissures as TEE anchors, on the original observations and on E89's
  measured-error observations.

    R4D_MODE=fields    python e101_commissure_anchors.py  -> results/benchmark/E100/E101/E101_{fields,anchors}.npz, E101_fields.json
    R4D_MODE=instances python e101_commissure_anchors.py  -> data/derived/e100-v4n96o-{measured-frontanchored,anchors,measured-anchors}
    R4D_MODE=specs     python e101_commissure_anchors.py  -> Codes/configs/e100/v4n96o/e101/*_test.json
    R4D_MODE=analyze   python e101_commissure_anchors.py <E101 rows root> <E89 rows root>
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("R4D_E89_INSTANCE", "e100")
HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("e89_measured_error", HERE / "e89_measured_error.py")
E89 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E89)

PREREGISTRATION = "Docs/E101_COMMISSURE_ANCHORS_PREREGISTRATION_2026-10-02.md"
OUT = Path("results/benchmark/E100/E101")
#: Chamber fields (E89's) and the front-end-anchored valve fields, in E89's key layout.
FRONT_FIELDS = OUT / "E101_fields.npz"
#: Per valve: the reference fit, the commissures and their places on the reference ring.
ANCHORS = OUT / "E101_anchors.npz"
FRONT = Path("data/derived/e100-v4n96o-measured-frontanchored")
ANCHORED = Path("data/derived/e100-v4n96o-anchors")
MEASURED_ANCHORED = Path("data/derived/e100-v4n96o-measured-anchors")
#: E89's own measured-error instance and specs (fixed before E89 module globals are patched).
E89_MEASURED = E89.MEASURED
E89_SPECS = E89.SPECS
SPECS = Path("Codes/configs/e100/v4n96o/e101")  # written on Delta: under the sync's protected v*/
SPEC_NAMES = ("front", "anchors_original", "anchors_measured")
REPORT = OUT / "E101_analysis.json"
N = E89.N_ANNULUS
CURVES = {"S/joint_curve_ring": "annulus", "S/joint_curves": "annulus+free_edges"}
ENDPOINTS = ("rotation_deg", "translation_mm", "anatomy_rms_mm")
REPORTED = ("rotation_deg", "translation_mm", "phase_map_cycles", "attachment_residual_mm",
            "heldout_cmr_mm", "heldout_tee_mm", "anatomy_rms_mm")
#: A re-run of an identical fit is not bitwise (BLAS threading); medians must agree to this.
REPRODUCTION_TOLERANCE = {"rotation_deg": 0.01, "translation_mm": 0.01, "phase_map_cycles": 1e-4,
                          "attachment_residual_mm": 0.01, "heldout_cmr_mm": 0.01, "heldout_tee_mm": 0.01,
                          "anatomy_rms_mm": 0.01}


# ------------------------------------------------------------ geometry ----

def _wrap(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _at(ring: np.ndarray, u: float) -> np.ndarray:
    """The closed ring at fractional index ``u``, linearly between vertices."""

    n = len(ring)
    k = int(np.floor(u)) % n
    t = float(u - np.floor(u))
    return (1.0 - t) * ring[k] + t * ring[(k + 1) % n]


def _angle(direction: np.ndarray, frame) -> float:
    return float(np.arctan2(direction @ frame.binormal, direction @ frame.reference))


def _ring_position(ring: np.ndarray, frame, angle: float):
    """Fractional index and point where the ring crosses ``angle`` (in ``frame``), or None.

    Where a ring crosses the angle more than once, the crossing farthest out
    along the direction is kept.
    """

    relative = ring - frame.centre
    planar = relative - np.outer(relative @ frame.normal, frame.normal)
    theta = np.arctan2(planar @ frame.binormal, planar @ frame.reference)
    ahead = _wrap(angle - theta)
    direction = frame.direction(angle)
    best = None
    n = len(ring)
    for k in range(n):
        a, b = float(ahead[k]), float(ahead[(k + 1) % n])
        if a == 0.0:
            u = float(k)
        elif a * b < 0.0 and abs(a - b) < np.pi:
            u = k + a / (a - b)
        else:
            continue
        point = _at(ring, u)
        reach = float((point - frame.centre) @ direction)
        if best is None or reach > best[0]:
            best = (reach, u % n, point)
    return None if best is None else (best[1], best[2])


def _flipped(frame):
    """The same commissures read about the opposite normal (angles negate, so they swap)."""

    from rooted4d.donors.valve_fixed import ValveCommissures

    low, high = frame.angles
    return ValveCommissures(centre=frame.centre, normal=-frame.normal, reference=frame.reference,
                            binormal=-frame.binormal, angles=(-high, -low))


def _seam_axis(reference_axis: np.ndarray, reference, prediction) -> np.ndarray:
    """``fit_valve``'s reference axis for the prediction, numbered from its commissures.

    The reference fit's seam (the direction opposite its axis) is placed on the
    prediction at the same fraction of the same arc between the commissures.
    """

    seam = -np.asarray(reference_axis, dtype=np.float64)
    low_r, high_r = reference.angles
    low_p, high_p = prediction.angles
    span_r = float((high_r - low_r) % (2.0 * np.pi))
    span_p = float((high_p - low_p) % (2.0 * np.pi))
    offset = float((_angle(seam, reference) - low_r) % (2.0 * np.pi))
    if offset <= span_r:
        target = low_p + offset / span_r * span_p
    else:
        target = high_p + (offset - span_r) / (2.0 * np.pi - span_r) * (2.0 * np.pi - span_p)
    return -prediction.direction(target)


def _best_shift(first: np.ndarray, second: np.ndarray) -> tuple[int, float]:
    """The cyclic shift of ``first`` that best matches ``second``, and the RMS after it."""

    n = len(first)
    errors = [float(np.sqrt(((first[(np.arange(n) + s) % n] - second) ** 2).sum(axis=1).mean()))
              for s in range(n)]
    shift = int(np.argmin(errors))
    return (shift if shift <= n // 2 else shift - n), errors[shift]


# ------------------------------------------------------------- fields ----

def cmd_fields() -> None:
    import nibabel as nib

    from rooted4d.donors.export import valve_donor_vertices
    from rooted4d.donors.valve import _valve_frame, fit_valve
    from rooted4d.donors.valve_fixed import valve_commissures

    groups = E89._groups()
    ring = np.asarray(groups["annulus"])
    valve = np.concatenate([ring, groups["free"]])
    labels = sorted(E89.MVSEG.rglob("*-label.nii.gz"))
    assert len(labels) == 175
    with np.load(E89.FIELDS) as blob:
        e89_fields = {key: blob[key] for key in blob.files}
    front = {key: value for key, value in e89_fields.items() if not key.endswith("__valve")}
    anchors: dict = {}
    summary = {"experiment": "E101", "preregistration": PREREGISTRATION, "valves": {}}
    for donor in E89._test_donors()[1]:
        index = int(donor.rsplit("-", 1)[-1])
        case = labels[index].name.replace("-label.nii.gz", "")
        reference = nib.load(str(labels[index]))
        zooms = tuple(float(z) for z in reference.header.get_zooms()[:3])
        volumes = {"reference": np.asarray(reference.dataobj),
                   "prediction": np.asarray(nib.load(str(E89.VALVE_PRED / f"valve_{index:04d}.nii.gz")).dataobj)}
        record: dict = {"case": str(labels[index].relative_to(E89.MVSEG))}
        summary["valves"][donor] = record
        fits = {name: fit_valve(volume, zooms, n_annulus=N, n_depth=6, case=case)
                for name, volume in volumes.items()}
        if any(fit is None for fit in fits.values()):
            record["failure"] = "valve fit"
            continue
        vertices = {name: valve_donor_vertices(fit, n_annulus=N) for name, fit in fits.items()}
        record["reference_matches_donor_mm"] = float(
            np.abs(E89._canonical(vertices["reference"])[valve] - E89._stored_donor(donor)[valve]).max())
        frames = {name: valve_commissures(volume, zooms) for name, volume in volumes.items()}
        if any(frame is None for frame in frames.values()):
            record["failure"] = "commissures not found on the " + " and ".join(
                name for name, frame in frames.items() if frame is None)
            continue
        found = {name: [_ring_position(vertices[name][ring], frames[name], a) for a in frames[name].angles]
                 for name in frames}
        if any(p is None for points in found.values() for p in points):
            record["failure"] = "a commissure does not cross its ring"
            continue
        r_points = np.stack([p[1] for p in found["reference"]])
        p_points = np.stack([p[1] for p in found["prediction"]])
        straight = float(np.linalg.norm(p_points - r_points, axis=1).sum())
        crossed = float(np.linalg.norm(p_points[::-1] - r_points, axis=1).sum())
        # The prediction's commissures, named as the reference's are (the nearer pairing).
        prediction_frame = frames["prediction"]
        if crossed < straight:
            prediction_frame = _flipped(prediction_frame)
            p_points = p_points[::-1]
        record["commissures_named_by_flip"] = bool(crossed < straight)
        record["commissure_error_mm"] = [float(v) for v in np.linalg.norm(p_points - r_points, axis=1)]

        # Front end: the prediction numbered from its commissures.
        occupied = np.argwhere(volumes["reference"] > 0) * np.asarray(zooms[:3], dtype=np.float64)
        axis = _seam_axis(_valve_frame(occupied)[2], frames["reference"], prediction_frame)
        anchored = fit_valve(volumes["prediction"], zooms, n_annulus=N, n_depth=6, case=case,
                             frame=(prediction_frame.centre, prediction_frame.normal, axis))
        if anchored is None:
            record["failure"] = "anchored valve fit"
            continue
        anchored_vertices = valve_donor_vertices(anchored, n_annulus=N)
        field = np.zeros_like(vertices["reference"])
        field[valve] = anchored_vertices[valve] - vertices["reference"][valve]
        front[f"{donor}__valve"] = field
        e89_field = e89_fields[f"{donor}__valve"]
        record["rms_mm"] = float(np.sqrt((field[valve] ** 2).sum(axis=1).mean()))
        record["e89_rms_mm"] = float(np.sqrt((e89_field[valve] ** 2).sum(axis=1).mean()))
        for label, predicted in (("anchored", anchored_vertices),
                                 ("e89", vertices["reference"] + e89_field)):
            shift, after = _best_shift(predicted[ring], vertices["reference"][ring])
            record[f"{label}_ring"] = {
                "rms_mm": float(np.sqrt(((predicted[ring] - vertices["reference"][ring]) ** 2).sum(axis=1).mean())),
                "best_shift": shift, "rms_after_best_shift_mm": after}

        # Solver: the reference commissures' places on the reference ring, and the prediction's points.
        anchors[f"{donor}__reference"] = vertices["reference"]
        anchors[f"{donor}__commissure_index"] = np.array([p[0] for p in found["reference"]])
        anchors[f"{donor}__commissures_prediction"] = p_points
    misses = {d: r["reference_matches_donor_mm"] for d, r in summary["valves"].items()
              if r.get("reference_matches_donor_mm", 0.0) > E89.DONOR_CHECK_TOLERANCE_MM}
    OUT.mkdir(parents=True, exist_ok=True)
    FRONT_FIELDS.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    if misses:
        raise SystemExit(f"reference fits do not reproduce their stored donors: {misses}")
    np.savez_compressed(FRONT_FIELDS, **front)
    np.savez_compressed(ANCHORS, **anchors)
    failures = {d: r["failure"] for d, r in summary["valves"].items() if "failure" in r}
    shifts = [r["anchored_ring"]["best_shift"] for r in summary["valves"].values() if "anchored_ring" in r]
    print("valves: %d anchored, failures %s; anchored ring best shifts %s"
          % (len(shifts), failures or "none", sorted(shifts)))
    print("median valve field RMS: E89 %.2f mm, anchored %.2f mm" % (
        np.median([r["e89_rms_mm"] for r in summary["valves"].values() if "rms_mm" in r]),
        np.median([r["rms_mm"] for r in summary["valves"].values() if "rms_mm" in r])))


# ---------------------------------------------------------- instances ----

def cmd_instances() -> None:
    from rooted4d.experiments.benchmark_eval import heldout_partition

    # Front end: E89's own instance builder, with the anchored valve fields.
    E89.FIELDS, E89.MEASURED = FRONT_FIELDS, FRONT
    E89.cmd_instance()

    with np.load(ANCHORS) as blob:
        info = {key: blob[key] for key in blob.files}
    ring = np.asarray(E89._groups()["annulus"])
    position = {int(v): i for i, v in enumerate(ring)}
    for root in (ANCHORED, MEASURED_ANCHORED):
        (root / "samples" / "test").mkdir(parents=True, exist_ok=True)
    report = {}
    for path in sorted((E89.ORIGINAL / "samples" / "test").glob("*.npz")):
        with np.load(path, allow_pickle=True) as blob:
            sample = {k: blob[k] for k in blob.files}
        with np.load(E89_MEASURED / "samples" / "test" / path.name, allow_pickle=True) as blob:
            measured = {k: blob[k] for k in blob.files}
        heart, valve = str(sample["whole_heart_donor_id"]), str(sample["valve_donor_id"])
        if f"{valve}__commissure_index" not in info:
            print("front-end failure, sample not written:", path.name, valve)
            continue
        # Each anchor is the visible TEE ring vertex nearest the reference commissure.
        _, _, _, visible = heldout_partition(sample)
        candidates = np.array([int(v) for v in visible if int(v) in position])
        rotation, scale, order = E89._similarity(heart, valve)
        donor_place = np.array([position[int(order[v])] for v in candidates])
        chosen = []
        for u in info[f"{valve}__commissure_index"]:
            gap = np.abs((donor_place - u + N / 2.0) % N - N / 2.0)
            chosen.append(int(candidates[int(np.argmin(gap))]))
        if chosen[0] == chosen[1]:
            print("anchors coincide, sample not written:", path.name)
            continue
        columns = [int(np.flatnonzero(np.asarray(sample["tee_indices"]) == v)[0]) for v in chosen]
        clean = np.asarray(sample["tee_points"])[:, columns, :]
        reference = info[f"{valve}__reference"]
        errors = np.stack([info[f"{valve}__commissures_prediction"][k] - reference[int(order[v])]
                           for k, v in enumerate(chosen)])
        carried = scale * (errors @ rotation.T)
        posed = carried @ np.asarray(sample["pose_rotation"]).T
        np.savez_compressed(ANCHORED / "samples" / "test" / path.name,
                            **dict(sample, tee_anchor_vertices=np.array(chosen), tee_anchor_points=clean))
        np.savez_compressed(MEASURED_ANCHORED / "samples" / "test" / path.name,
                            **dict(measured, tee_anchor_vertices=np.array(chosen),
                                   tee_anchor_points=clean + posed[None]))
        report[path.name] = {"identity": str(sample["identity_id"]), "anchors": chosen,
                             "anchor_error_mm": [float(v) for v in np.linalg.norm(carried, axis=1)]}
    for root in (ANCHORED, MEASURED_ANCHORED):
        (root / "anchors.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", len(report), "samples to", ANCHORED, "and", MEASURED_ANCHORED)


# -------------------------------------------------------------- specs ----

def cmd_specs() -> None:
    measured = json.loads((E89_SPECS / "measured_test.json").read_text())
    by_name = {v["name"]: v for v in measured["variants"]}
    joint = by_name["S/joint"]

    def curve(name: str, setting: str) -> dict:
        variant = json.loads(json.dumps(joint))
        variant["name"] = name
        variant["overrides"] = dict(variant.get("overrides") or {}, curve_evidence=setting)
        return variant

    def write(name: str, benchmark: Path, variants: list) -> None:
        spec = {key: measured[key] for key in ("start", "models", "base_config", "model_benchmark", "frozen_from")}
        spec.update({"experiment": "E101", "preregistration": PREREGISTRATION, "split": "test",
                     "benchmark": str(benchmark), "variants": variants})
        SPECS.mkdir(parents=True, exist_ok=True)
        (SPECS / f"{name}_test.json").write_text(json.dumps(spec, indent=2) + "\n")
        print("wrote", SPECS / f"{name}_test.json")

    write("front", FRONT, [joint, by_name["S/late_fusion_trajectory"]])
    variants = [joint] + [curve(name, setting) for name, setting in CURVES.items()]
    write("anchors_original", ANCHORED, variants)
    write("anchors_measured", MEASURED_ANCHORED, variants)


# ------------------------------------------------------------ analysis ----

def _rows(directory: Path) -> dict:
    rows = {}
    for shard in sorted(directory.glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["sample"]] = row
    return rows


def _did(sm: dict, so: dict, cm: dict, co: dict) -> dict:
    """(median sm - median so) - (median cm - median co), over identities, with E89's resampling."""

    identities = sorted(set(sm) & set(so) & set(cm) & set(co))
    arrays = [np.array([d[i] for i in identities]) for d in (sm, so, cm, co)]

    def value(t):
        a, b, c, d = (x[t] for x in arrays)
        return (np.nanmedian(a) - np.nanmedian(b)) - (np.nanmedian(c) - np.nanmedian(d))

    rng = np.random.default_rng(E89.SEED)
    draws = np.array([value(rng.integers(0, len(identities), size=len(identities)))
                      for _ in range(E89.RESAMPLES)])
    return {"observed": float(value(np.arange(len(identities)))), "identities": len(identities),
            "lower": float(np.percentile(draws, 2.5)), "upper": float(np.percentile(draws, 97.5))}


def _repairs(reduction: dict, degradation: dict) -> bool:
    """The pre-stated rule: the reduction's interval lies above zero and it is at least half the degradation."""

    return bool(reduction["lower"] > 0.0 and reduction["observed"] >= 0.5 * degradation["observed"])


def cmd_analyze(e101_root: str, e89_root: str) -> None:
    from dataclasses import replace

    sys.path.insert(0, str(HERE))
    import e59_unknown_alignment as driver
    from rooted4d.inference import SolverConfig

    digest = E89._digest
    specs = {name: json.loads((SPECS / f"{name}_test.json").read_text()) for name in SPEC_NAMES}
    e89_specs = {name: json.loads((E89_SPECS / f"{name}_test.json").read_text()) for name in ("measured", "original")}
    rows = {name: _rows(Path(e101_root) / digest(spec)) for name, spec in specs.items()}
    rows.update({f"e89_{name}": _rows(Path(e89_root) / digest(spec)) for name, spec in e89_specs.items()})
    benches = {"front": FRONT, "anchors_original": ANCHORED, "anchors_measured": MEASURED_ANCHORED,
               "e89_measured": E89_MEASURED, "e89_original": E89.ORIGINAL}
    model, *_ = driver.build_model(Path(specs["front"]["model_benchmark"]), specs["front"]["models"]["S"])
    base_config = SolverConfig(**specs["front"]["base_config"])

    def config_of(spec: dict, name: str):
        variant = next(v for v in spec["variants"] if v["name"] == name)
        return replace(base_config, **variant.get("overrides", {}))

    samples = sorted(set(rows["e89_measured"]) & set(rows["front"]) & set(rows["anchors_measured"]))
    records: dict = {}
    for label, methods in (("front", ("S/joint", "S/late_fusion_trajectory")),
                           ("anchors_original", ("S/joint",) + tuple(CURVES)),
                           ("anchors_measured", ("S/joint",) + tuple(CURVES)),
                           ("e89_original", ("S/joint", "S/late_fusion_trajectory")),
                           ("e89_measured", ("S/joint", "S/late_fusion_trajectory"))):
        spec = specs.get(label) or e89_specs[label.split("_", 1)[1]]
        for sample_name in samples:
            row = rows[label][sample_name]
            with np.load(benches[label] / "samples" / "test" / sample_name, allow_pickle=True) as blob:
                sample = {k: blob[k] for k in blob.files}
            for method in methods:
                outcome = row["variants"][method]
                record = {"sample": sample_name, "identity": row["identity"], "method": method,
                          "succeeded": bool(outcome.get("succeeded"))}
                for key in REPORTED[:-1]:
                    record[key] = outcome.get(key) if outcome.get("succeeded") else None
                if outcome.get("succeeded"):
                    kind = "S/late_fusion_trajectory" if method == "S/late_fusion_trajectory" else "S/joint"
                    joint_config = config_of(spec, method) if kind == "S/joint" else config_of(spec, "S/joint")
                    record["anatomy_rms_mm"] = E89.anatomy_rms(kind, outcome, sample, model, joint_config, base_config)
                records.setdefault(label, []).append(record)

    def values(label: str, method: str, key: str) -> dict:
        return E89._identity_values([r for r in records[label] if r["method"] == method], key)

    def median(label: str, method: str, key: str) -> float:
        return float(np.nanmedian([np.nan if r.get(key) is None else r[key]
                                   for r in records[label] if r["method"] == method]))

    report = {"experiment": "E101", "preregistration": PREREGISTRATION, "samples": len(samples),
              "front_end_failures": sorted(set(rows["e89_measured"]) - set(samples)),
              "medians": {label: {method: {key: median(label, method, key) for key in REPORTED}
                                  for method in sorted({r["method"] for r in records[label]})}
                          for label in records}}
    # Code check: the standard fit on the anchored instances reproduces E89's rows.
    check = {}
    for label, e89_label in (("anchors_original", "e89_original"), ("anchors_measured", "e89_measured")):
        check[label] = {key: abs(report["medians"][label]["S/joint"][key] - report["medians"][e89_label]["S/joint"][key])
                        for key in REPORTED}
    report["reproduction_of_e89"] = {"differences": check, "passed": all(
        check[label][key] <= REPRODUCTION_TOLERANCE[key] for label in check for key in REPORTED)}

    standard = {key: E89._paired(values("e89_measured", "S/joint", key), values("e89_original", "S/joint", key))
                for key in REPORTED}
    front = {
        "degradation_standard": standard,
        "degradation_front": {key: E89._paired(values("front", "S/joint", key), values("e89_original", "S/joint", key))
                              for key in REPORTED},
        "reduction": {key: E89._paired(values("e89_measured", "S/joint", key), values("front", "S/joint", key))
                      for key in REPORTED},
        "late_fusion_reduction": {
            key: E89._paired(values("e89_measured", "S/late_fusion_trajectory", key),
                             values("front", "S/late_fusion_trajectory", key)) for key in REPORTED},
        "rooted4d_minus_late_fusion": {
            key: E89._paired(values("front", "S/joint", key), values("front", "S/late_fusion_trajectory", key))
            for key in REPORTED},
    }
    front["repairs_most"] = {key: _repairs(front["reduction"][key], standard[key]) for key in ENDPOINTS}
    report["front_end"] = front
    solver = {}
    for method in CURVES:
        entry = {
            "degradation": {key: E89._paired(values("anchors_measured", method, key),
                                             values("anchors_original", method, key)) for key in REPORTED},
            "reduction": {key: _did(values("e89_measured", "S/joint", key), values("e89_original", "S/joint", key),
                                    values("anchors_measured", method, key), values("anchors_original", method, key))
                          for key in ENDPOINTS},
            "cost_on_original": {key: E89._paired(values("anchors_original", method, key),
                                                  values("e89_original", "S/joint", key)) for key in REPORTED},
            "minus_late_fusion_on_measured": {
                key: E89._paired(values("anchors_measured", method, key),
                                 values("e89_measured", "S/late_fusion_trajectory", key)) for key in REPORTED},
        }
        entry["repairs_most"] = {key: _repairs(entry["reduction"][key], standard[key]) for key in ENDPOINTS}
        solver[method] = entry
    report["solver"] = solver

    def verdict(flags: dict) -> str:
        count = sum(flags.values())
        return "holds" if count == len(flags) else ("holds partly" if count else "does not hold")

    report["reading"] = {"front_end": verdict(front["repairs_most"]),
                         **{f"solver {m}": verdict(solver[m]["repairs_most"]) for m in CURVES}}
    report["records"] = records
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("samples", "front_end_failures", "reproduction_of_e89", "reading")},
                     indent=1))
    for key in ENDPOINTS:
        s, r = standard[key], front["reduction"][key]
        print("%-16s standard %+.2f [%+.2f, %+.2f]  front reduction %+.2f [%+.2f, %+.2f]  %s" % (
            key, s["observed"], s["lower"], s["upper"], r["observed"], r["lower"], r["upper"],
            front["repairs_most"][key]))
        for method in CURVES:
            r = solver[method]["reduction"][key]
            print("%-16s   %-20s reduction %+.2f [%+.2f, %+.2f]  %s" % (
                "", method, r["observed"], r["lower"], r["upper"], solver[method]["repairs_most"][key]))
    print("wrote", REPORT)


def main() -> int:
    mode = os.environ.get("R4D_MODE")
    if mode == "fields":
        cmd_fields()
    elif mode == "instances":
        cmd_instances()
    elif mode == "specs":
        cmd_specs()
    elif mode == "analyze":
        cmd_analyze(*sys.argv[1:3])
    else:
        raise SystemExit("set R4D_MODE to fields, instances, specs or analyze")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
