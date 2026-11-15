"""E89, part 2: fusion under measured front-end error (plan v2, W1.3).

Pre-registered in ``Docs/E89_MEASURED_ERROR_PREREGISTRATION_2026-09-28.md``.

    R4D_MODE=fields   python e89_measured_error.py   -> results/benchmark/E89/E89_fields.{npz,json}
    R4D_MODE=instance python e89_measured_error.py   -> data/derived/semisynthetic-real-v4-measured (test only)
    R4D_MODE=specs    python e89_measured_error.py   -> Codes/configs/e89/{measured,original}_test.json
    R4D_MODE=analyze  python e89_measured_error.py <rows root>

``fields`` fits the donor pool's chamber and valve surfaces to each test donor's
front-end prediction and reference labels; the difference is that donor's
measured error field. ``instance`` adds the fields to the realistic-session
test observations. The chamber side reads E84's test predictions, the valve side
part 1's (``R4D_VALVE_PRED``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

PREREGISTRATION = "Docs/E89_MEASURED_ERROR_PREREGISTRATION_2026-09-28.md"
ORIGINAL = Path("data/derived/semisynthetic-real-v4")
MEASURED = Path("data/derived/semisynthetic-real-v4-measured")
DONORS = Path("data/derived/donors_v3")
ACDC = Path(os.environ.get("R4D_ACDC", "data/raw/acdc"))
MVSEG = Path(os.environ.get("R4D_MVSEG", "data/raw/mvseg2023"))
CHAMBER_PRED = Path(os.environ.get("R4D_CHAMBER_PRED", "data/nnunet/predictions_acdc_test"))
VALVE_PRED = Path(os.environ.get("R4D_VALVE_PRED", "data/nnunet/predictions_mvsegbench_test"))
FIELDS = Path("results/benchmark/E89/E89_fields.npz")
FROZEN = Path("Codes/configs/e75/v4_e75_test.json")
N_ANNULUS = 24
SPECS = Path("Codes/configs/e89")
REPORT = Path("results/benchmark/E89/E89_analysis.json")
#: The donor pool's own fit settings, so a reference fit reproduces its stored donor.
FIT = {"chamber_angles": 32, "valve_annulus": 32}
ORIENT = False
#: The 24-node pool (donors_v3, 4 September) was fitted with MVSeg2023's leaflet
#: labels exchanged (``e89_valve_segmenter.donor_label_convention``).
VALVE_LABELS_EXCHANGED = True
#: A reference fit must reproduce its stored donor (the record's check); above
#: this the fields are not written and the job stops.
DONOR_CHECK_TOLERANCE_MM = 1e-3
#: E100 (Docs/E100_ORIENTED_REBUILD_PREREGISTRATION_2026-09-29.md, its E89 amendment):
#: the same study on the rebuilt realistic-session primary, 96 ring nodes with each
#: valve oriented to its chamber. The fields are fitted at the 96-node pool's
#: settings, and each valve field is re-indexed and carried as the rebuild carries
#: its valve. Selected with ``R4D_E89_INSTANCE=e100``; the defaults are E89's.
if os.environ.get("R4D_E89_INSTANCE") == "e100":
    ORIGINAL = Path("data/derived/e100-v4n96o")
    MEASURED = Path("data/derived/e100-v4n96o-measured")
    DONORS = Path("data/derived/donors_v3_n96")
    FIELDS = Path("results/benchmark/E100/E89/E89_fields.npz")
    FROZEN = Path("Codes/configs/e100/v4n96o/v4n96o_e88_test.json")
    N_ANNULUS = 96
    SPECS = Path("Codes/configs/e100/v4n96o/e89")  # written on Delta: under the sync's protected v*/
    REPORT = Path("results/benchmark/E100/E89/E89_analysis.json")
    FIT = {"chamber_angles": 96, "valve_annulus": 96}
    ORIENT = True
    # The 96-node pool (donors_v3_n96, 28 September) was fitted after the 9 September
    # label fix, with the dataset's own labels (E89 amendment of 2 October).
    VALVE_LABELS_EXCHANGED = False


def _groups():
    from rooted4d.experiments.fitted_geometry import ring_groups

    return ring_groups(N_ANNULUS)


def _test_donors() -> tuple[list[str], list[str]]:
    """Heart and valve donors of the realistic-session test composites."""

    hearts, valves = set(), set()
    for line in (ORIGINAL / "manifests" / "test.jsonl").read_text().splitlines():
        record = json.loads(line)
        hearts.add(record["whole_heart_donor_id"])
        valves.add(record["valve_donor_id"])
    return sorted(hearts), sorted(valves)


def _stored_donor(donor_id: str) -> np.ndarray:
    kind, index = ("heart", donor_id.rsplit("-", 1)[-1]) if "heart" in donor_id else \
                  ("valve", donor_id.rsplit("-", 1)[-1])
    return np.load(DONORS / "fixed_topology" / f"{kind}-{index}.npz")["vertices"]


def _canonical(vertices: np.ndarray) -> np.ndarray:
    from rooted4d.donors.export import _canonicalize

    return _canonicalize(vertices)


def chamber_fields() -> dict:
    """ED and ES error fields for each test chamber donor, with checks."""

    import nibabel as nib

    from rooted4d.active.acdc import fit_lv_surface
    from rooted4d.donors.export import chamber_donor_vertices

    groups = _groups()
    chamber = np.concatenate([groups["annulus"], groups["lower"], groups["apex"]])
    # The donor pool indexes hearts by patient name (build_donor_pool sorts its
    # fits by patient), not by path: ACDC's testing folder would sort first.
    folders = {p.parent.name: p.parent for p in ACDC.rglob("*_gt.nii.gz")}
    by_index = [folders[name] for name in sorted(folders)]
    assert len(by_index) == 150, len(by_index)
    out = {}
    for donor in _test_donors()[0]:
        folder = by_index[int(donor.rsplit("-", 1)[-1])]
        info = dict(line.split(": ", 1) for line in (folder / "Info.cfg").read_text().splitlines() if ": " in line)
        record = {"patient": folder.name}
        for phase, frame in (("ed", int(info["ED"])), ("es", int(info["ES"]))):
            name = f"{folder.name}_frame{frame:02d}"
            reference = nib.load(str(folder / f"{name}_gt.nii.gz"))
            predicted = nib.load(str(CHAMBER_PRED / f"{name}.nii.gz"))
            zooms = tuple(float(z) for z in reference.header.get_zooms()[:3])
            fits = {}
            for label, handle in (("reference", reference), ("prediction", predicted)):
                fit = fit_lv_surface(np.asarray(handle.dataobj), zooms, n_rings=8, n_angles=FIT["chamber_angles"],
                                     patient=folder.name, frame=frame)
                fits[label] = None if fit is None else chamber_donor_vertices(fit, n_annulus=N_ANNULUS)
            record[phase] = {"reference_fitted": fits["reference"] is not None,
                             "prediction_fitted": fits["prediction"] is not None}
            if fits["reference"] is not None and fits["prediction"] is not None:
                field = np.zeros_like(fits["reference"])
                field[chamber] = fits["prediction"][chamber] - fits["reference"][chamber]
                record[phase]["field"] = field
                record[phase]["rms_mm"] = float(np.sqrt((field[chamber] ** 2).sum(axis=1).mean()))
            if phase == "ed" and fits["reference"] is not None:
                # The donor pool fitted the first annotated frame; where that is
                # end diastole the reference fit must reproduce the stored donor.
                stored = _stored_donor(donor)
                record["reference_matches_donor_mm"] = float(
                    np.abs(_canonical(fits["reference"])[chamber] - stored[chamber]).max())
        out[donor] = record
    return out


def valve_fields() -> dict:
    import importlib.util

    import nibabel as nib

    spec = importlib.util.spec_from_file_location(
        "e89_valve_segmenter", Path(__file__).resolve().parent / "e89_valve_segmenter.py")
    segmenter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(segmenter)
    relabel = segmenter.donor_label_convention if VALVE_LABELS_EXCHANGED else (lambda volume: volume)

    from rooted4d.donors.export import valve_donor_vertices
    from rooted4d.donors.valve import fit_valve

    groups = _groups()
    valve = np.concatenate([groups["annulus"], groups["free"]])
    labels = sorted(MVSEG.rglob("*-label.nii.gz"))
    assert len(labels) == 175
    out = {}
    for donor in _test_donors()[1]:
        index = int(donor.rsplit("-", 1)[-1])
        reference = nib.load(str(labels[index]))
        predicted = nib.load(str(VALVE_PRED / f"valve_{index:04d}.nii.gz"))
        zooms = tuple(float(z) for z in reference.header.get_zooms()[:3])
        fits = {}
        for label, handle in (("reference", reference), ("prediction", predicted)):
            # The pool's leaflet convention, so both fits are comparable with the
            # composites' valve (see e89_valve_segmenter.donor_label_convention).
            fit = fit_valve(relabel(np.asarray(handle.dataobj)), zooms, n_annulus=FIT["valve_annulus"],
                            n_depth=6, case=labels[index].name.replace("-label.nii.gz", ""))
            fits[label] = None if fit is None else valve_donor_vertices(fit, n_annulus=N_ANNULUS)
        record = {"case": str(labels[index].relative_to(MVSEG)),
                  "reference_fitted": fits["reference"] is not None,
                  "prediction_fitted": fits["prediction"] is not None}
        if fits["reference"] is not None:
            record["reference_matches_donor_mm"] = float(
                np.abs(_canonical(fits["reference"])[valve] - _stored_donor(donor)[valve]).max())
        if fits["reference"] is not None and fits["prediction"] is not None:
            field = np.zeros_like(fits["reference"])
            field[valve] = fits["prediction"][valve] - fits["reference"][valve]
            record["field"] = field
            record["rms_mm"] = float(np.sqrt((field[valve] ** 2).sum(axis=1).mean()))
        out[donor] = record
    return out


def cmd_fields() -> None:
    chambers, valves = chamber_fields(), valve_fields()
    arrays, summary = {}, {"experiment": "E89", "preregistration": PREREGISTRATION,
                           "chambers": {}, "valves": {}}
    for donor, record in chambers.items():
        for phase in ("ed", "es"):
            if "field" in record[phase]:
                arrays[f"{donor}__{phase}"] = record[phase].pop("field")
        summary["chambers"][donor] = record
    for donor, record in valves.items():
        if "field" in record:
            arrays[f"{donor}__valve"] = record.pop("field")
        summary["valves"][donor] = record
    FIELDS.parent.mkdir(parents=True, exist_ok=True)
    FIELDS.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    misses = {donor: r["reference_matches_donor_mm"] for part in ("chambers", "valves")
              for donor, r in summary[part].items()
              if r.get("reference_matches_donor_mm", 0.0) > DONOR_CHECK_TOLERANCE_MM}
    if misses:
        raise SystemExit("reference fits do not reproduce their stored donors, fields not written: %s"
                         % dict(sorted(misses.items(), key=lambda kv: -kv[1])[:5]))
    np.savez_compressed(FIELDS, **arrays)
    rms = [r[p]["rms_mm"] for r in summary["chambers"].values() for p in ("ed", "es") if "rms_mm" in r[p]]
    vrms = [r["rms_mm"] for r in summary["valves"].values() if "rms_mm" in r]
    print("chamber fields: %d, median rms %.2f mm; valve fields: %d, median rms %.2f mm"
          % (len(rms), np.median(rms), len(vrms), np.median(vrms)))
    print("reference fits against stored donors (max |difference|, mm): chamber %.3g, valve %.3g" % (
        max(r.get("reference_matches_donor_mm", np.nan) for r in summary["chambers"].values()),
        max(r.get("reference_matches_donor_mm", np.nan) for r in summary["valves"].values())))


def _similarity(heart_id: str, valve_id: str):
    """The rotation, scale and vertex order ``combine_donors`` carries this valve with.

    The order is the identity unless ``ORIENT``, where it is the re-indexing
    ``oriented_valve_vertices`` applies (the valve field is re-indexed the same way).
    """

    from rooted4d.geometry import make_toy_complex
    from rooted4d.synthetic.donors import _ring_handedness, _similarity_transform

    heart, valve = _stored_donor(heart_id), _stored_donor(valve_id)
    complex_ = make_toy_complex(n_annulus=N_ANNULUS)
    annulus, free = complex_.groups["annulus"], complex_.groups["leaflet_free"]
    order = np.arange(len(valve))
    if ORIENT:
        chamber = np.setdiff1d(complex_.groups["cmr_visible"], annulus)
        if _ring_handedness(heart, annulus, chamber) != _ring_handedness(valve, annulus, free):
            n = len(annulus)
            ring_order = (n // 2 - 1 - np.arange(n)) % n
            order[annulus], order[free] = annulus[ring_order], free[ring_order]
    rotation, scale = _similarity_transform(valve[order][annulus], heart[annulus])
    return rotation, scale, order


def cmd_instance() -> None:
    from rooted4d.experiments.fitted_geometry import chamber_volume, truth_trajectory

    with np.load(FIELDS) as blob:
        fields = {key: blob[key] for key in blob.files}
    groups = _groups()
    ring = groups["annulus"]
    out_dir = MEASURED / "samples" / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    disagreement = {}
    for path in sorted((ORIGINAL / "samples" / "test").glob("*.npz")):
        with np.load(path, allow_pickle=True) as blob:
            sample = {k: blob[k] for k in blob.files}
        heart, valve = str(sample["whole_heart_donor_id"]), str(sample["valve_donor_id"])
        ed, es = fields.get(f"{heart}__ed"), fields.get(f"{heart}__es")
        valve_field = fields.get(f"{valve}__valve")
        if ed is None or es is None or valve_field is None:
            print("front-end failure, sample not written:", path.name, heart, valve)
            continue
        truth = truth_trajectory(sample)
        grid = np.linspace(0.0, 1.0, 64, endpoint=False)
        volumes = np.array([chamber_volume(frame, N_ANNULUS) for frame in truth(grid)])
        low, high = volumes.min(), volumes.max()
        frame_volumes = np.array([chamber_volume(frame, N_ANNULUS) for frame in truth(sample["cmr_times"])])
        weights = (high - frame_volumes) / (high - low)
        cmr_field = (1.0 - weights)[:, None, None] * ed[None] + weights[:, None, None] * es[None]
        sample["cmr_points"] = sample["cmr_points"] + cmr_field[:, sample["cmr_indices"], :]
        rotation, scale, order = _similarity(heart, valve)
        carried = scale * (valve_field[order] @ rotation.T)
        posed = carried @ np.asarray(sample["pose_rotation"]).T
        sample["tee_points"] = sample["tee_points"] + posed[None, sample["tee_indices"], :]
        np.savez_compressed(out_dir / path.name, **sample)
        # Ring disagreement: TEE front end's ring against CMR front end's end-diastolic ring.
        difference = carried[ring] - ed[ring]
        tangent = np.roll(sample["template_vertices"][ring], -1, axis=0) - np.roll(sample["template_vertices"][ring], 1, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        disagreement[path.name] = {
            "identity": str(sample["identity_id"]),
            "ring_rms_mm": float(np.sqrt((difference ** 2).sum(axis=1).mean())),
            "ring_slide_mm": float((difference * tangent).sum(axis=1).mean()),
        }
    (MEASURED / "ring_disagreement.json").write_text(json.dumps(disagreement, indent=2) + "\n")
    print("wrote", len(disagreement), "samples to", out_dir)


def cmd_specs() -> None:
    frozen = json.loads(FROZEN.read_text())
    by_name = {v["name"]: v for v in frozen["variants"]}
    out = SPECS
    out.mkdir(parents=True, exist_ok=True)
    for name, benchmark in (("measured", str(MEASURED)), ("original", str(ORIGINAL))):
        spec = {key: frozen[key] for key in ("start", "models", "base_config")}
        spec.update({"experiment": "E89", "preregistration": PREREGISTRATION, "split": "test",
                     "benchmark": benchmark, "model_benchmark": str(ORIGINAL), "frozen_from": str(FROZEN)})
        variants = []
        for source in ("S/joint", "S/soft_matched"):
            variant = json.loads(json.dumps(by_name[source]))
            variant["options"] = dict(variant.get("options") or {}, store_parameters=True)
            variants.append(variant)
        variants.append({"name": "S/late_fusion_trajectory", "model": "S", "arm": "late_fusion_trajectory",
                         "options": dict(json.loads(json.dumps(by_name["S/independent"]["options"])),
                                         store_parameters=True)})
        spec["variants"] = variants
        (out / f"{name}_test.json").write_text(json.dumps(spec, indent=2) + "\n")
        print("wrote", out / f"{name}_test.json")


# --------------------------------------------------------------- analysis ----

METHODS = ("S/joint", "S/soft_matched", "S/late_fusion_trajectory")
SCALARS = ("rotation_deg", "translation_mm", "phase_map_cycles", "attachment_residual_mm",
           "heldout_cmr_mm", "heldout_tee_mm")
#: The soft-attachment arm does not store its parameters (``baselines.run_arm``),
#: so anatomy error is computed for these two (amendment of 29 September).
ANATOMY = ("S/joint", "S/late_fusion_trajectory")
#: Endpoints whose degradation decides the anchored follow-up.
DEGRADATION_ENDPOINTS = ("rotation_deg", "translation_mm", "anatomy_rms_mm")
RESAMPLES, SEED = 20000, 20260910


def _digest(spec: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def _spec_rows(rows_root: Path, name: str) -> tuple[dict, dict]:
    spec = json.loads((SPECS / f"{name}_test.json").read_text())
    directory = rows_root / _digest(spec)
    rows = {}
    for shard in sorted(directory.glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["sample"]] = row
    return spec, rows


def anatomy_rms(method: str, outcome: dict, sample: dict, model, joint_config, base_config) -> float:
    """RMS distance of the fitted shared anatomy from the truth, every vertex, 32 phases.

    Late fusion's anatomy is its chamber (annulus, lower ring, apex) from the CMR
    fit and its leaflet free edge from the TEE fit, placed by its registration.
    """

    from dataclasses import replace

    from rooted4d.experiments.fitted_geometry import fitted_trajectory, ring_groups, truth_trajectory
    from rooted4d.geometry.rigid import RigidTransform
    from rooted4d.inference.baselines import parameters_from_record
    from rooted4d.inference.trajectory_registration import inverse_phase
    from rooted4d.motion import MonotoneCircularPhaseWarp

    phases = np.linspace(0.0, 1.0, 32, endpoint=False)
    n = len(sample["template_vertices"])
    truth = truth_trajectory(sample)(phases)
    if method == "S/joint":
        fitted = fitted_trajectory(parameters_from_record(outcome["extras"]["parameters"]), model,
                                   joint_config, phases, n)
    else:
        extras = outcome["extras"]
        cmr, tee = parameters_from_record(extras["cmr"]), parameters_from_record(extras["tee"])
        registration = extras["registration"]
        transform = RigidTransform(np.asarray(registration["rotation"]), np.asarray(registration["translation"]))
        warp = MonotoneCircularPhaseWarp(np.asarray(registration["logits"]), offset=registration["offset"])
        fitted = fitted_trajectory(cmr, model, replace(base_config, use_tee=False), phases, n)
        valve = transform.apply(fitted_trajectory(tee, model, base_config, inverse_phase(warp, phases), n))
        free = ring_groups((n - 1) // 3)["free"]
        fitted[:, free] = valve[:, free]
    return float(np.sqrt(((fitted - truth) ** 2).sum(axis=2).mean()))


def _identity_values(records: list[dict], key: str) -> dict:
    by_identity: dict = {}
    for record in records:
        value = record.get(key)
        by_identity.setdefault(record["identity"], []).append(np.nan if value is None else float(value))
    return {identity: float(np.nanmean(values)) for identity, values in by_identity.items()}


def _paired(first: dict, second: dict) -> dict:
    """E62's statistic on identity-averaged values: difference of medians (first - second)."""

    identities = sorted(set(first) & set(second))
    a = np.array([first[i] for i in identities])
    b = np.array([second[i] for i in identities])
    rng = np.random.default_rng(SEED)
    draws = np.empty(RESAMPLES)
    for r in range(RESAMPLES):
        t = rng.integers(0, len(a), size=len(a))
        draws[r] = np.nanmedian(a[t]) - np.nanmedian(b[t])
    return {"observed": float(np.nanmedian(a) - np.nanmedian(b)), "identities": len(identities),
            "lower": float(np.percentile(draws, 2.5)), "upper": float(np.percentile(draws, 97.5))}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        out = np.empty(len(v))
        out[order] = np.arange(len(v), dtype=float)
        for value in np.unique(v):              # average ranks for ties
            tied = v == value
            out[tied] = out[tied].mean()
        return out

    keep = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(ranks(x[keep]), ranks(y[keep]))[0, 1])


def _correlation(degradation: dict, disagreement: dict) -> dict:
    """Spearman over identities, with an identity-bootstrap interval."""

    identities = sorted(set(degradation) & set(disagreement))
    x = np.array([degradation[i] for i in identities])
    y = np.array([disagreement[i] for i in identities])
    if np.unique(x[np.isfinite(x)]).size < 2 or np.unique(y[np.isfinite(y)]).size < 2:
        # A constant variable has no ranks to correlate.
        return {"rho": float("nan"), "lower": float("nan"), "upper": float("nan"),
                "identities": len(identities)}
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(RESAMPLES):
        t = rng.integers(0, len(x), size=len(x))
        if np.unique(x[t]).size > 1 and np.unique(y[t]).size > 1:
            draws.append(_spearman(x[t], y[t]))
    return {"rho": _spearman(x, y), "lower": float(np.percentile(draws, 2.5)),
            "upper": float(np.percentile(draws, 97.5)), "identities": len(identities)}


def cmd_analyze(rows_root: str) -> None:
    from dataclasses import replace

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import e59_unknown_alignment as driver
    from rooted4d.inference import SolverConfig

    root = Path(rows_root)
    measured_spec, measured = _spec_rows(root, "measured")
    original_spec, original = _spec_rows(root, "original")
    model, *_ = driver.build_model(Path(measured_spec["model_benchmark"]), measured_spec["models"]["S"])
    base_config = SolverConfig(**measured_spec["base_config"])
    joint = next(v for v in measured_spec["variants"] if v["name"] == "S/joint")
    joint_config = replace(base_config, **joint.get("overrides", {}))
    disagreement_rows = json.loads((MEASURED / "ring_disagreement.json").read_text())
    samples = sorted(measured)
    failures = sorted(set(original) - set(measured))
    report = {"experiment": "E89", "preregistration": PREREGISTRATION,
              "rows": {"measured": str(root / _digest(measured_spec)), "original": str(root / _digest(original_spec))},
              "samples": len(samples), "front_end_failures": failures}
    records = {"measured": [], "original": []}
    for name, rows, bench in (("measured", measured, MEASURED), ("original", original, ORIGINAL)):
        for sample_name in samples:
            row = rows[sample_name]
            with np.load(bench / "samples" / "test" / sample_name, allow_pickle=True) as blob:
                sample = {k: blob[k] for k in blob.files}
            for method in METHODS:
                outcome = row["variants"][method]
                record = {"sample": sample_name, "identity": row["identity"], "method": method,
                          "succeeded": bool(outcome.get("succeeded"))}
                for key in SCALARS:
                    record[key] = outcome.get(key) if outcome.get("succeeded") else None
                if method in ANATOMY and outcome.get("succeeded"):
                    record["anatomy_rms_mm"] = anatomy_rms(method, outcome, sample, model, joint_config, base_config)
                records[name].append(record)
    endpoints = SCALARS + ("anatomy_rms_mm",)

    def values(name, method, key):
        return _identity_values([r for r in records[name] if r["method"] == method], key)

    report["medians"] = {
        name: {method: {key: float(np.nanmedian([np.nan if r.get(key) is None else r[key]
                                                 for r in records[name] if r["method"] == method]))
                        for key in endpoints if any(r.get(key) is not None for r in records[name]
                                                    if r["method"] == method)}
               for method in METHODS}
        for name in records}
    report["degradation_measured_minus_original"] = {
        method: {key: _paired(values("measured", method, key), values("original", method, key))
                 for key in endpoints if key in report["medians"]["measured"][method]}
        for method in METHODS}
    report["methods_on_measured"] = {
        f"rooted4d_minus_{other.split('/')[1]}": {
            key: _paired(values("measured", "S/joint", key), values("measured", other, key))
            for key in endpoints if key in report["medians"]["measured"][other]}
        for other in ("S/late_fusion_trajectory", "S/soft_matched")}
    ring_rms = _identity_values([{"identity": v["identity"], "x": v["ring_rms_mm"]}
                                 for s, v in disagreement_rows.items() if s in measured], "x")
    ring_slide = _identity_values([{"identity": v["identity"], "x": abs(v["ring_slide_mm"])}
                                   for s, v in disagreement_rows.items() if s in measured], "x")
    report["ring_disagreement"] = {"rms_mm_median": float(np.median(list(ring_rms.values()))),
                                   "abs_slide_mm_median": float(np.median(list(ring_slide.values())))}
    report["degradation_vs_ring_disagreement"] = {}
    for method in METHODS:
        entry = {}
        for key in DEGRADATION_ENDPOINTS:
            if key not in report["medians"]["measured"][method]:
                continue
            m, o = values("measured", method, key), values("original", method, key)
            degradation = {i: m[i] - o[i] for i in set(m) & set(o)}
            entry[key] = {"ring_rms": _correlation(degradation, ring_rms),
                          "ring_abs_slide": _correlation(degradation, ring_slide)}
        report["degradation_vs_ring_disagreement"][method] = entry
    # The pre-stated condition for designing the commissure-anchored configuration.
    conditions = {}
    for key in DEGRADATION_ENDPOINTS:
        degraded = report["degradation_measured_minus_original"]["S/joint"][key]["lower"] > 0.0
        tracks = report["degradation_vs_ring_disagreement"]["S/joint"][key]["ring_rms"]["lower"] > 0.0
        conditions[key] = {"degrades": bool(degraded), "tracks_ring_disagreement": bool(tracks)}
    report["anchored_configuration_condition"] = {
        "per_endpoint": conditions,
        "holds": any(c["degrades"] and c["tracks_ring_disagreement"] for c in conditions.values())}
    report["records"] = records
    out = REPORT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("samples", "front_end_failures", "medians",
                                              "anchored_configuration_condition")}, indent=1)[:3000])
    print("wrote", out)


if __name__ == "__main__":
    mode = os.environ.get("R4D_MODE")
    if mode == "analyze":
        cmd_analyze(sys.argv[1])
    else:
        {"fields": cmd_fields, "instance": cmd_instance, "specs": cmd_specs}.get(
            mode, lambda: sys.exit(f"unknown R4D_MODE {mode!r}"))()
