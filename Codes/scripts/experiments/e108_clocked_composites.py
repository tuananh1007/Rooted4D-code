"""E108: the main protocol on anatomically clocked composites (weak point 1b).

Pre-registered in ``Docs/E108_CLOCKED_COMPOSITES_PREREGISTRATION_2026-10-04.md``.
The 96-node donor pool numbers each chamber ring from an image axis and each
valve ring from a principal axis of arbitrary sign, so a composite's valve sits
at an arbitrary rotation about the annular axis. This re-encodes the same donors
from anatomy (``rooted4d.donors.anatomy``): every valve numbered from its two
commissures, every chamber sampled about its septum, so that the composites'
vertex matching faces each anterior leaflet to its chamber's septum. The two
96-node primary benchmarks are rebuilt on that pool, with the same split seed,
donor ids, pairing and sample seeds, and taken through E88's unattended chain.

    python e108_clocked_composites.py pool          -> data/derived/donors_v3_n96a, results/benchmark/E108/E108_pool.json
    python e108_clocked_composites.py frames        -> results/benchmark/E108/E108_valve_frames.json
    python e108_clocked_composites.py configs       -> Codes/configs/e108/{v4n96a,v3n96a}.yaml
    python e108_clocked_composites.py check <key>   -> results/benchmark/E108/E108_check_<key>.json (exit 1 on failure)
    python e108_clocked_composites.py hull <root> <name>   -> results/benchmark/E108/E108_hull_<name>.json
    R4D_KEY=<key> python e108_clocked_composites.py chain  (reads R4D_E88_OUT, R4D_WORKERS)
    python e108_clocked_composites.py analyze <key> <test rows dir>   -> results/benchmark/E108/E108_<key>.json
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E108_CLOCKED_COMPOSITES_PREREGISTRATION_2026-10-04.md"
OUT = Path("results/benchmark/E108")
ACDC = Path("data/raw/acdc")
MVSEG = Path("data/raw/mvseg2023")
#: The 96-node pool the E100 primaries were built from, and its fitting settings.
REFERENCE = Path("data/derived/donors_v3_n96")
POOL = Path("data/derived/donors_v3_n96a")
N = 96
N_RINGS, CHAMBER_ANGLES, VALVE_ANNULUS, N_DEPTH = 8, 96, 96, 6
#: A refit must reproduce its stored donor to this (E89's gate).
REPRODUCE_MM = 1e-3
CONFIGS = Path("Codes/configs/e108")
#: key -> (protocol benchmark, E100 counterpart key)
KEYS = {"v4n96a": ("v4", "v4n96o"), "v3n96a": ("v3", "v3n96o")}
WORKERS = int(os.environ.get("R4D_WORKERS", "8"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def benchmark(key: str) -> str:
    return f"data/derived/e108-{key}"


def original(key: str) -> str:
    return f"data/derived/e100-{KEYS[key][1]}"


# ---------------------------------------------------------------- pool

def _heart_sources() -> list[tuple[int, Path]]:
    """(donor index, patient folder): build_donor_pool sorts its chamber donors by patient name."""

    folders = {p.parent.name: p.parent for p in ACDC.rglob("*_gt.nii.gz")}
    names = sorted(folders)
    assert len(names) == 150, len(names)
    return [(index, folders[name]) for index, name in enumerate(names)]


def _valve_sources() -> list[tuple[int, Path]]:
    labels = sorted(MVSEG.rglob("*-label.nii.gz"))
    assert len(labels) == 175, len(labels)
    return list(enumerate(labels))


def _fit_heart(source: tuple[int, Path]) -> dict:
    import nibabel as nib

    from rooted4d.active.acdc import fit_lv_surface
    from rooted4d.donors.anatomy import septal_direction
    from rooted4d.donors.export import _canonicalize, chamber_donor_vertices

    index, folder = source
    # The pool kept the first frame that fits, in sorted path order.
    for path in sorted(folder.glob("*_gt.nii.gz")):
        image = nib.load(str(path))
        labels = np.asarray(image.dataobj)
        zooms = image.header.get_zooms()[:3]
        frame = int(path.name.split("_frame")[-1].split("_")[0])
        fit = fit_lv_surface(labels, zooms, n_rings=N_RINGS, n_angles=CHAMBER_ANGLES,
                             patient=folder.name, frame=frame)
        if fit is not None:
            break
    else:
        return {"index": index, "patient": folder.name, "failure": "no frame fits"}
    record = {"index": index, "patient": folder.name, "frame": frame, "fit": fit}
    reference = REFERENCE / "fixed_topology" / f"heart-{index:04d}.npz"
    if reference.exists():
        from rooted4d.geometry import make_toy_complex

        chamber = make_toy_complex(n_annulus=N).groups["cmr_visible"]
        stored = np.load(reference)["vertices"]
        record["reproduces_mm"] = float(np.abs(_canonicalize(chamber_donor_vertices(fit, n_annulus=N))[chamber]
                                               - stored[chamber]).max())
    septal = septal_direction(labels, zooms)
    if septal is None:
        record["failure"] = "no right-ventricular contact in the basal slices"
        return record
    record["septal"] = septal
    return record


def _fit_valve(source: tuple[int, Path]) -> dict:
    import nibabel as nib

    from rooted4d.donors.anatomy import anterior_span, fit_valve_commissural
    from rooted4d.donors.export import _canonicalize, valve_donor_vertices
    from rooted4d.donors.valve import fit_valve

    index, path = source
    image = nib.load(str(path))
    labels = np.asarray(image.dataobj)
    zooms = tuple(float(z) for z in image.header.get_zooms()[:3])
    case = path.name.replace("-label.nii.gz", "")
    record = {"index": index, "case": str(path.relative_to(MVSEG))}
    reference = REFERENCE / "fixed_topology" / f"valve-{index:04d}.npz"
    if reference.exists():
        from rooted4d.geometry import make_toy_complex

        groups = make_toy_complex(n_annulus=N).groups
        valve_ids = np.concatenate([groups["annulus"], groups["leaflet_free"]])
        old = fit_valve(labels, zooms, n_annulus=VALVE_ANNULUS, n_depth=N_DEPTH, case=case)
        stored = np.load(reference)["vertices"]
        record["reproduces_mm"] = (None if old is None else float(np.abs(
            _canonicalize(valve_donor_vertices(old, n_annulus=N))[valve_ids] - stored[valve_ids]).max()))
    encoded = fit_valve_commissural(labels, zooms, n_per_leaflet=N // 2, n_depth=N_DEPTH, case=case)
    if encoded is None:
        record["failure"] = "commissures not found, or over a quarter of a leaflet's nodes without tissue"
        return record
    fixed, record["bridged_nodes"] = encoded
    record["fit"] = fixed
    record["commissure_angles_deg"] = [float(np.degrees(a)) for a in fixed.commissure_angles]
    record["anterior_span_deg"] = float(np.degrees(anterior_span(fixed)))
    return record


def _sha256(path: Path) -> str:
    from rooted4d.synthetic.reproducibility import sha256_file

    return sha256_file(path)


def cmd_pool() -> None:
    from rooted4d.donors.anatomy import chamber_vertices_clocked, valve_vertices_commissural
    from rooted4d.donors.export import DONOR_SCHEMA, _canonicalize

    if POOL.exists():
        raise SystemExit(f"{POOL} exists; refusing to overwrite a donor pool")
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        valves = list(pool.map(_fit_valve, _valve_sources()))
        hearts = list(pool.map(_fit_heart, _heart_sources()))
    failures = ([("valve", r["index"], r["failure"]) for r in valves if "failure" in r]
                + [("heart", r["index"], r["failure"]) for r in hearts if "failure" in r])
    reproduction = {kind: [r["reproduces_mm"] for r in records if "reproduces_mm" in r]
                    for kind, records in (("hearts", hearts), ("valves", valves))}
    bad = {kind: [v for v in values if v is None or v > REPRODUCE_MM] for kind, values in reproduction.items()}
    report = {"experiment": "E108", "preregistration": PREREGISTRATION, "reference": str(REFERENCE),
              "failures": failures,
              "reproduction": {kind: {"checked": len(values), "above_tolerance": len(bad[kind]),
                                      "max_mm": max((v for v in values if v is not None), default=None)}
                               for kind, values in reproduction.items()}}
    if failures or any(bad.values()):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "E108_pool_failed.json").write_text(json.dumps(report, indent=2) + "\n")
        raise SystemExit(f"pool not written: {report}")

    spans = np.array([r["anterior_span_deg"] for r in valves])
    alpha = float(np.radians(np.median(spans)))
    (POOL / "fixed_topology").mkdir(parents=True)
    entries = []
    for r in hearts:
        name = f"heart-{r['index']:04d}"
        path = POOL / "fixed_topology" / f"{name}.npz"
        np.savez_compressed(path, vertices=_canonicalize(chamber_vertices_clocked(r["fit"], r["septal"].direction,
                                                                                  alpha, n_annulus=N)),
                            septal_direction=r["septal"].direction)
        entries.append({"donor_id": f"acdc-{name}", "global_person_id": f"acdc-person-{r['index']:04d}",
                        "kind": "whole_heart", "path": f"fixed_topology/{name}.npz", "sha256": _sha256(path),
                        "source_dataset": "ACDC"})
    for r in valves:
        name = f"valve-{r['index']:04d}"
        path = POOL / "fixed_topology" / f"{name}.npz"
        np.savez_compressed(path, vertices=_canonicalize(valve_vertices_commissural(r["fit"], n_annulus=N)),
                            commissure_angles=np.array(r["fit"].commissure_angles))
        entries.append({"donor_id": f"mvseg-{name}", "global_person_id": f"mvseg-person-{r['index']:04d}",
                        "kind": "valve", "path": f"fixed_topology/{name}.npz", "sha256": _sha256(path),
                        "source_dataset": "MVSeg2023"})
    manifest = {
        "schema_version": DONOR_SCHEMA, "topology_kind": "procedural_test_v1", "n_annulus": N,
        "provenance": {
            "chamber": "ACDC left-ventricle cavity fits, rings sampled about the septum (E108)",
            "valve": "MVSeg2023 mitral leaflet fits, numbered from the commissures (E108)",
            "construction": (f"{PREREGISTRATION}; the donors of {REFERENCE.name}, same ids and source frames; "
                             f"anterior arc {np.degrees(alpha):.2f} degrees, the median of the valves' spans"),
            "licence_note": (
                "ACDC is CC BY-NC-SA and MVSeg2023 is CC BY-NC-ND. These donors "
                "are derivative works of both, so the pool may be used here and "
                "cannot be redistributed; NoDerivatives blocks release of the "
                "valve side entirely."),
        },
        "donors": entries,
    }
    (POOL / "donors.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report["alpha_deg"] = float(np.degrees(alpha))
    report["anterior_span_deg"] = {"median": float(np.median(spans)), "min": float(spans.min()),
                                   "max": float(spans.max()), "q25": float(np.percentile(spans, 25)),
                                   "q75": float(np.percentile(spans, 75))}
    deviations = [max(r["septal"].deviations_deg) for r in hearts]
    widths = [float(np.median(r["septal"].arc_widths_deg)) for r in hearts]
    report["septum"] = {"slices_used": {"min": min(len(r["septal"].slices) for r in hearts),
                                        "max": max(len(r["septal"].slices) for r in hearts)},
                        "contact_arc_deg": {"median": float(np.median(widths)), "min": float(min(widths)),
                                            "max": float(max(widths))},
                        "largest_slice_deviation_deg": {"median": float(np.median(deviations)),
                                                        "max": float(max(deviations))}}
    report["hearts"] = [{"index": r["index"], "patient": r["patient"], "frame": r["frame"],
                         "septal_direction": r["septal"].direction.tolist(), "slices": list(r["septal"].slices),
                         "arc_widths_deg": list(r["septal"].arc_widths_deg),
                         "deviations_deg": list(r["septal"].deviations_deg),
                         "reproduces_mm": r.get("reproduces_mm")} for r in hearts]
    bridged = np.array([r["bridged_nodes"] for r in valves])
    report["bridged_nodes"] = {"valves_with_any": int((bridged > 0).sum()), "max": int(bridged.max()),
                               "total": int(bridged.sum()), "of_nodes": int(N * len(valves))}
    report["valves"] = [{k: r[k] for k in ("index", "case", "commissure_angles_deg", "anterior_span_deg",
                                           "bridged_nodes")}
                        | {"reproduces_mm": r.get("reproduces_mm")} for r in valves]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "E108_pool.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("hearts", "valves")}, indent=1))


# ---------------------------------------------------------------- configs

def cmd_configs() -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    for key, (_, counterpart) in KEYS.items():
        source = Path("Codes/configs/e100") / f"{counterpart}.yaml"
        text = source.read_text()
        old = "  manifest: ../../../data/derived/donors_v3_n96/donors.json\n"
        assert text.count(old) == 1, source
        text = text.replace(old, "  # E108: the same donors encoded from anatomy (commissures, septum).\n"
                                 "  manifest: ../../../data/derived/donors_v3_n96a/donors.json\n")
        body = text[text.index("schema_version:"):]
        header = (f"# E108 ({PREREGISTRATION}): {key}.\n"
                  f"# {source.name} (E100) with the anatomically encoded pool donors_v3_n96a.\n"
                  "# Written by e108_clocked_composites.py configs; do not edit by hand.\n")
        (CONFIGS / f"{key}.yaml").write_text(header + body)
        print("wrote", CONFIGS / f"{key}.yaml")


# ---------------------------------------------------------------- checks

def _donor(pool: Path, kind: str, donor_id: str) -> dict:
    name = donor_id.split("-", 1)[1]
    with np.load(pool / "fixed_topology" / f"{name}.npz") as blob:
        return {k: blob[k] for k in blob.files}


def _manifest(root: str, split: str) -> list[dict]:
    lines = (Path(root) / "manifests" / f"{split}.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


FRAMES = OUT / "E108_valve_frames.json"


def _valve_frame(source: tuple[int, Path]) -> dict:
    import nibabel as nib

    from rooted4d.donors.valve_fixed import valve_commissures

    index, path = source
    image = nib.load(str(path))
    found = valve_commissures(np.asarray(image.dataobj), tuple(float(z) for z in image.header.get_zooms()[:3]))
    return {"index": index, "case": str(path.relative_to(MVSEG)), "anterior_direction": found.reference.tolist(),
            "normal": found.normal.tolist()}


def cmd_frames() -> None:
    """Each valve's anterior direction in its donor's coordinates (``valve_commissures``), for the clocking measure."""

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        frames = list(pool.map(_valve_frame, _valve_sources()))
    OUT.mkdir(parents=True, exist_ok=True)
    FRAMES.write_text(json.dumps({"experiment": "E108", "preregistration": PREREGISTRATION,
                                  "valves": frames}, indent=1) + "\n")
    print("wrote", FRAMES, len(frames))


def clocking(pairs: list[tuple[str, str]]) -> dict:
    """The anterior leaflet's angle from the septum, as built before (E100's pool) and now (E108's)."""

    from rooted4d.donors.anatomy import valve_clocking

    frames = {f"mvseg-valve-{f['index']:04d}": np.array(f["anterior_direction"])
              for f in json.loads(FRAMES.read_text())["valves"]}
    out = {"before": [], "after": []}
    for heart_id, valve_id in pairs:
        septal = _donor(POOL, "heart", heart_id)["septal_direction"]
        for name, pool in (("before", REFERENCE), ("after", POOL)):
            out[name].append(valve_clocking(_donor(pool, "heart", heart_id)["vertices"],
                                            _donor(pool, "valve", valve_id)["vertices"], frames[valve_id], septal, N))
    summary = {}
    for name, values in out.items():
        a = np.abs(values)
        summary[name] = {"median_abs": float(np.median(a)), "q25_abs": float(np.percentile(a, 25)),
                         "q75_abs": float(np.percentile(a, 75)), "max_abs": float(a.max()),
                         "within_30": int((a <= 30).sum()), "beyond_90": int((a > 90).sum()), "n": int(len(a))}
    return summary


def cmd_check(key: str) -> None:
    """The rebuilt instance against its E100 counterpart: ids, pairing, side, orientation, clocking."""

    from rooted4d.synthetic.donors import _ring_handedness
    from rooted4d.geometry import make_toy_complex

    groups = make_toy_complex(n_annulus=N).groups
    chamber = np.setdiff1d(groups["cmr_visible"], groups["annulus"])
    root, before_root = benchmark(key), original(key)
    report = {"experiment": "E108", "preregistration": PREREGISTRATION, "key": key,
              "instance": root, "original": before_root, "splits": {}}
    ok = True
    for split in ("train", "val", "test"):
        new, old = _manifest(root, split), _manifest(before_root, split)
        same_ids = [r["sample_id"] for r in new] == [r["sample_id"] for r in old]
        pairs = [(r["whole_heart_donor_id"], r["valve_donor_id"]) for r in new]
        same_pairs = pairs == [(r["whole_heart_donor_id"], r["valve_donor_id"]) for r in old]
        heights, reindexed = [], 0
        for record in new:
            with np.load(Path(root) / record["relative_path"]) as blob:
                template = blob["template_vertices"]
            ring = template[groups["annulus"]].mean(axis=0)
            axis = template[chamber[-1]] - ring
            axis /= np.linalg.norm(axis)
            heights.append(float(((template[groups["leaflet_free"]] - ring) @ axis).mean()))
        identities = sorted(set(pairs))
        for heart_id, valve_id in identities:
            heart, valve = _donor(POOL, "heart", heart_id)["vertices"], _donor(POOL, "valve", valve_id)["vertices"]
            reindexed += (_ring_handedness(heart, groups["annulus"], chamber)
                          != _ring_handedness(valve, groups["annulus"], groups["leaflet_free"]))
        entry = {"samples": len(new), "identities": len(identities), "same_sample_ids": same_ids,
                 "same_pairing": same_pairs,
                 "free_edge_height_mm": {"mean": float(np.mean(heights)), "min": float(np.min(heights)),
                                         "max": float(np.max(heights))},
                 "valves_reindexed_by_orientation": int(reindexed),
                 "clocking_deg": clocking(identities)}
        report["splits"][split] = entry
        ok &= same_ids and same_pairs and min(heights) > 0.0 and reindexed == 0
        print(split, entry, flush=True)
    report["passed"] = bool(ok)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"E108_check_{key}.json").write_text(json.dumps(report, indent=2) + "\n")
    if not ok:
        raise SystemExit("E108 instance check failed")


def cmd_hull(root: str, name: str) -> None:
    """E104's free-edge count against the chamber hull, on one instance's test samples (secondary)."""

    e104 = _load("e104_hull_check")
    records = []
    for path in sorted((Path(root) / "samples" / "test").glob("*.npz")):
        with np.load(path, allow_pickle=True) as blob:
            sample = {k: blob[k] for k in blob.files}
        record = e104.measure(sample)
        record["sample"] = path.name
        records.append(record)
    report = {"experiment": "E108", "preregistration": PREREGISTRATION, "root": root,
              "coarse": e104.summarize(records, "coarse"), "per_sample": records}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"E108_hull_{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(name, report["coarse"])


# ---------------------------------------------------------------- protocol

def cmd_chain() -> None:
    key = os.environ["R4D_KEY"]
    bench, _ = KEYS[key]
    e88 = _load("e88_replication")
    chain = e88.Chain(bench, 0, key=key, benchmark=benchmark(key), configs=CONFIGS, experiment="E108",
                      preregistration=PREREGISTRATION, store_parameters=True)
    chain.selection_rounds()
    arms = chain.session_weight()
    directory = chain.test(arms)
    (chain.root / "TEST_DONE").write_text(str(Path(directory).relative_to(chain.root)) + "\n")
    print(f"[{key}] test rows in {directory}", flush=True)


def _samples(directories) -> set:
    return {(json.loads(line)["identity"], json.loads(line)["sample"])
            for directory in directories for shard in sorted(Path(directory).glob("shard_*.jsonl"))
            for line in shard.read_text().splitlines() if line.strip()}


def reading(clocked: dict, before: dict) -> dict:
    """E94's rule: T1 and T3 read as before, and every difference excluding zero keeps its sign."""

    e88 = _load("e88_replication")
    same_t1 = clocked["T1_representation"]["verdict"] == before["T1_representation"]["verdict"]
    same_t3 = all(clocked["T3_fusion_cost"][m] == before["T3_fusion_cost"][m] for m in ("cmr", "tee"))
    kept, lost = [], []
    for name, path in e88.KEY_DIFFERENCES:
        b, c = e88._entry(before, path), e88._entry(clocked, path)
        if b["upper"] < 0 or b["lower"] > 0:
            sign = -1 if b["upper"] < 0 else 1
            (kept if (c["upper"] < 0 if sign < 0 else c["lower"] > 0) else lost).append(name)
    return {"T1_same": same_t1, "T3_same": same_t3, "differences_kept": kept, "differences_lost": lost,
            "robust": bool(same_t1 and same_t3 and not lost)}


def cmd_analyze(key: str, test_dir: str) -> None:
    e88 = _load("e88_replication")
    stage = e88._load("e62_stage_b")
    counterpart = f"results/benchmark/E100/rows/{KEYS[key][1]}_test"
    spec = json.loads((Path(test_dir) / "spec.json").read_text())
    assert spec["benchmark"] == benchmark(key) and spec["split"] == "test", spec["benchmark"]
    clocked, before = stage.load_rows([test_dir]), stage.load_rows([counterpart])
    assert len(clocked) == len(before) == 60, (len(clocked), len(before))
    assert _samples([test_dir]) == _samples([counterpart])
    report = {"experiment": "E108", "preregistration": PREREGISTRATION, "key": key,
              "clocked": e88.test_report(stage, clocked), "original": e88.test_report(stage, before)}
    report["side_by_side"] = {name: {"original": e88._entry(report["original"], path),
                                     "clocked": e88._entry(report["clocked"], path)}
                              for name, path in e88.KEY_DIFFERENCES}
    report["reading"] = reading(report["clocked"], report["original"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"E108_{key}.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, pair in report["side_by_side"].items():
        print("%-34s original %+.3f [%+.3f, %+.3f]   clocked %+.3f [%+.3f, %+.3f]" % (
            name, *(pair["original"][k] for k in ("observed", "lower", "upper")),
            *(pair["clocked"][k] for k in ("observed", "lower", "upper"))))
    print("reading:", report["reading"])


#: (label, path into a test report, decimals) for the supplement table, as Table S (oriented rebuild).
TABLE_ROWS = (
    (r"Rooted4D $-$ late fusion, held-out sum", ("T3_fusion_cost", "comparison", "sum"), 3),
    (r"\quad CMR", ("T3_fusion_cost", "comparison", "cmr"), 3),
    (r"\quad TEE", ("T3_fusion_cost", "comparison", "tee"), 3),
    (r"Rooted4D $-$ late fusion, rotation (\si{\degree})", ("rooted4d_minus_trajectory", "rotation_deg"), 2),
    (r"\quad translation (mm)", ("rooted4d_minus_trajectory", "translation_mm"), 2),
    (r"\quad phase map (cycles)", ("rooted4d_minus_trajectory", "phase_map_cycles"), 3),
    (r"\quad interface gap (mm)", ("rooted4d_minus_trajectory", "attachment_residual_mm"), 2),
    (r"Interface-built $-$ global, held-out sum", ("T1_representation", "comparison", "sum"), 3),
    (r"\quad CMR", ("T1_representation", "comparison", "cmr"), 3),
    (r"\quad TEE", ("T1_representation", "comparison", "tee"), 3),
    (r"Global basis, coupling cost, CMR", ("T4_competitors", "G_fusion_cost", "cmr"), 3),
    (r"\quad TEE", ("T4_competitors", "G_fusion_cost", "tee"), 3),
)


def _cell(entry: dict, decimals: int) -> str:
    def number(value: float) -> str:
        # A bound that would print as zero gets the digit that shows its side of zero.
        places = decimals + 1 if value != 0.0 and round(value, decimals) == 0.0 else decimals
        return f"${value:+.{places}f}$"

    return (f"{number(entry['observed'])} {{\\scriptsize[{number(entry['lower'])}, "
            f"{number(entry['upper'])}]}}")


def cmd_latex() -> None:
    """The supplement table: each 96-node benchmark as built (E100) and clocked (E108)."""

    e88 = _load("e88_replication")
    reports = {key: json.loads((OUT / f"E108_{key}.json").read_text()) for key in ("v4n96a", "v3n96a")}
    lines = []
    for label, path, decimals in TABLE_ROWS:
        cells = [_cell(e88._entry(reports[key][column], path), decimals)
                 for key in ("v4n96a", "v3n96a") for column in ("original", "clocked")]
        lines.append(label + " & " + " & ".join(cells) + "\\\\")
    table = "\n".join(lines)
    (OUT / "E108_table.tex").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    {"pool": cmd_pool, "frames": cmd_frames, "configs": cmd_configs, "check": cmd_check, "hull": cmd_hull, "chain": cmd_chain, "latex": cmd_latex,
     "analyze": cmd_analyze}[command](*arguments)
