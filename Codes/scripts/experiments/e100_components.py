"""E100's component studies: each study's own code, on the rebuilt instances.

Pre-registered in ``Docs/E100_ORIENTED_REBUILD_PREREGISTRATION_2026-09-29.md``,
whose section "Component-study amendments" states, per study, what changes.
Every study keeps its own code, arms and statistics. Its benchmark maps to the
rebuilt primary of the same session regime (v4 -> ``v4n96o``, v3 -> ``v3n96o``),
and it reads that instance's final configuration: S/joint in the chain's test
spec (``Codes/configs/e100/<key>/<key>_e88_test.json``), with the session weight
selected on validation.

    R4D_E88_OUT=<results root> python e100_components.py run <key> <study> [...]
    python e100_components.py e92-configs      -> Codes/configs/e100/e92/*.yaml
    R4D_E88_OUT=<results root> python e100_components.py e95
                                               -> results/benchmark/E100/E95/E95_reliability.json
    R4D_E88_OUT=<results root> python e100_components.py e92-analyze        (on Delta: reads samples)
    python e100_components.py session-field <key> [...]                    (on Delta: reads samples)
    python e100_components.py analyze <rows root> [<key>=<original dirs,...> ...]
                                               -> results/benchmark/E100/E100_<part>.json

``analyze`` reads a copy of the chain roots (E100's; E94's beside it as ``e94``)
and applies each study's own statistics. An original-construction test run can
be given per instance, for the side-by-side record the supplement keeps.

``run`` resumes: a study whose rows are complete is not run again. Specs this
writes go under ``Codes/configs/e100/<key>/``, which the sync protects. Studies:

    gate            E62's donor-grouped cross-validation (the gate spec the chain wrote)
    e65, e66        temporal families on validation; the comparison on test at E65's choice
    e67             the session term's variants on test
    e68             the session-prior weight under thinned evidence, test
    e69             frozen fits on validation and test (calibration, warm starts)
    e70, e71, e74   image evidence, warm-started from e69's test fits
    e72             the registration stress test
    e91, e96, e97   image-domain search, annotation precision, image late fusion
    e92             the nine interventions and the base level
    e89             measured front-end error: fields, instance, both test runs, then E89's
                    and E98's analyses (E89's ``R4D_E89_INSTANCE=e100`` switch)
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E100_ORIENTED_REBUILD_PREREGISTRATION_2026-09-29.md"
CONFIGS = Path("Codes/configs/e100")
#: E94's instance and chain, reused as the realistic-session robustness instance
#: at 24 nodes. Its chain root is E94's (the job points R4D_E88_OUT there).
E94 = {"key": "v4oriented", "benchmark": "data/derived/semisynthetic-real-v4-oriented",
       "configs": Path("Codes/configs/e94"), "experiment": "E94",
       "preregistration": "Docs/E94_ORIENTED_COMPOSITES_PREREGISTRATION_2026-09-28.md"}
#: The amendment's table: which studies run on which instance.
STUDIES = {"v3n96o": ("gate", "e65", "e66", "e69", "e72", "e70", "e71", "e74", "e91", "e96", "e97"),
           "v4n96o": ("gate", "e67", "e68", "e91", "e97", "e92", "e89", "e101"),
           "v3o": ("gate",), "v3n48o": ("gate",), "v4n48o": ("gate",), "v4oriented": ("gate",)}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chain(key: str):
    """The instance's chain object: its specs, results root and row runner."""

    e88 = _load("e88_replication")
    if key == E94["key"]:
        return e88.Chain("v4", 0, key=key, benchmark=E94["benchmark"], configs=E94["configs"],
                         experiment=E94["experiment"], preregistration=E94["preregistration"])
    e100 = _load("e100_rebuild")
    bench, _, k, _ = e100.resolve(key)
    return e88.Chain(bench, k, key=key, benchmark=e100.benchmark(key), configs=CONFIGS,
                     experiment="E100", preregistration=PREREGISTRATION)


def test_spec(c) -> tuple[Path, dict]:
    path = c.specs / f"{c.key}_e88_test.json"
    if not path.exists():
        raise SystemExit(f"{path} does not exist: the instance's chain has not reached its test")
    return path, json.loads(path.read_text())


def frozen_spec(c, split: str, experiment: str) -> tuple[dict, dict]:
    """A spec at the instance's final S configuration, and S/joint's overrides."""

    path, test = test_spec(c)
    joint = next(v for v in test["variants"] if v["name"] == "S/joint")
    spec = {"experiment": experiment, "preregistration": PREREGISTRATION, "benchmark": test["benchmark"],
            "split": split, "start": test["start"], "models": {"S": test["models"]["S"]},
            "base_config": dict(test["base_config"]), "frozen_from": str(path), "variants": []}
    return spec, dict(joint["overrides"])


def _write(c, spec: dict, name: str) -> Path:
    path = c.specs / name
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print("wrote", path, "with", len(spec["variants"]), "variants", flush=True)
    return path


# ------------------------------------------------------- spec-based studies ----

def study_gate(c) -> str:
    path = c.specs / f"{c.key}_gate.json"
    if not path.exists():
        # E94's specs were removed on Delta by a sync before the protect rule
        # existed. The chain's selection rounds rewrite them from its complete
        # rows; nothing may be refitted, so an incomplete stage stops the job.
        run = c.run

        def replay(spec_path, label, *, allow_test=False):
            spec = json.loads(Path(spec_path).read_text())
            digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
            directory = c.root / label / digest
            if not c._complete(spec, directory):
                raise SystemExit(f"{label}: no complete rows at {directory}; not refitting")
            print(f"[{c.key}] {label}: complete, spec rewritten", flush=True)
            return str(directory)

        c.run = replay
        c.selection_rounds()
        c.run = run
    return c.run(path, "gate")


def study_e65(c) -> str:
    u = _load("e65_uniformity")
    spec, joint = frozen_spec(c, "val", "E65")
    for order in u.ORDERS:
        for weight in u.MOTION_WEIGHTS:
            spec["variants"].append({"name": u.periodic_name(order, weight), "model": "S", "arm": "rooted4d_full",
                                     "overrides": dict(joint, fourier_order=order, motion_prior_weight=weight)})
    for nodes in u.NODES:
        for weight in u.MOTION_WEIGHTS:
            spec["variants"].append({"name": u.per_frame_name(nodes, weight), "model": "S", "arm": "rooted4d_full",
                                     "overrides": dict(joint, motion_model="per_frame", motion_nodes=nodes,
                                                       motion_prior_weight=weight)})
    return c.run(_write(c, spec, f"{c.key}_e65_temporal_selection.json"), "e65")


def study_e66(c) -> str:
    u = _load("e65_uniformity")
    choices = u.e65_selection(c.stage.load_rows([study_e65(c)]))
    spec, joint = frozen_spec(c, "test", "E66")
    spec["e65_selection"] = {k: {"setting": v["setting"], "motion_weight": v["motion_weight"]}
                             for k, v in choices.items()}
    periodic = dict(joint, fourier_order=choices["periodic"]["setting"],
                    motion_prior_weight=choices["periodic"]["motion_weight"])
    spec["variants"] = [
        {"name": "S/periodic", "model": "S", "arm": "rooted4d_full", "overrides": periodic},
        {"name": "S/per_frame", "model": "S", "arm": "rooted4d_full",
         "overrides": dict(joint, motion_model="per_frame", motion_nodes=choices["per_frame"]["setting"],
                           motion_prior_weight=choices["per_frame"]["motion_weight"])},
        {"name": "S/fixed_warp", "model": "S", "arm": "rooted4d_full",
         "overrides": dict(periodic, estimate_phase_warp=False)},
    ]
    print("E65 selection:", json.dumps(spec["e65_selection"]), flush=True)
    return c.run(_write(c, spec, f"{c.key}_e66_temporal_test.json"), "e66", allow_test=True)


def study_e67(c) -> str:
    spec, joint = frozen_spec(c, "test", "E67")
    spec["variants"] = [
        {"name": "S/frozen", "model": "S", "arm": "rooted4d_full", "overrides": dict(joint)},
        {"name": "S/oscillatory", "model": "S", "arm": "rooted4d_full",
         "overrides": dict(joint, session_includes_offset=False)},
        {"name": "S/no_session", "model": "S", "arm": "rooted4d_full",
         "overrides": dict(joint, use_session=False)},
    ]
    return c.run(_write(c, spec, f"{c.key}_e67_session_test.json"), "e67", allow_test=True)


def _e92():
    """E92's module, pointed at the rebuilt realistic-session primary."""

    e92 = _load("e92_interventions")
    e92.SOURCE = CONFIGS / "v4n96o.yaml"
    e92.ORIGINAL = "data/derived/e100-v4n96o"
    e92.FROZEN = CONFIGS / "v4n96o" / "v4n96o_e88_test.json"
    e92.INSTANCE_PREFIX = "data/derived/e100-v4n96o-e92"
    e92.REPORT = Path("results/benchmark/E100/E92/E92_interventions.json")
    return e92


def study_e92(c) -> list[str]:
    if c.key != "v4n96o":
        raise SystemExit("E92 runs on v4n96o only")
    e92 = _e92()
    e92.OUT = c.specs / "e92"          # written here, under the sync's protected v*/
    e92.cmd_specs()
    return [c.run(path, "e92", allow_test=True) for path in sorted(e92.OUT.glob("*_test.json"))]


def study_e89(c) -> str:
    if c.key != "v4n96o":
        raise SystemExit("E89 runs on v4n96o only")
    test_spec(c)                        # its specs are built from the chain's test spec
    os.environ["R4D_E89_INSTANCE"] = "e100"
    e89 = _load("e89_measured_error")
    script = str(HERE / "e89_measured_error.py")

    def step(mode: str, *arguments: str) -> None:
        print(f"[{c.key}] e89: {mode}", flush=True)
        subprocess.run([sys.executable, "-u", script, *arguments], env=dict(os.environ, R4D_MODE=mode), check=True)

    if not e89.FIELDS.exists():
        step("fields")
    if not (e89.MEASURED / "ring_disagreement.json").exists():
        step("instance")
    step("specs")
    for name in ("measured", "original"):
        c.run(e89.SPECS / f"{name}_test.json", "e89", allow_test=True)
    root = str(c.root / "e89")
    step("analyze", root)
    print(f"[{c.key}] e98", flush=True)
    subprocess.run([sys.executable, "-u", str(HERE / "e98_measured_reliability.py"), root],
                   env=dict(os.environ), check=True)
    return root


def study_e101(c) -> str:
    """E101, the commissure-anchored configuration (E89's follow-up), on E89's instances."""

    if c.key != "v4n96o":
        raise SystemExit("E101 runs on v4n96o only")
    test_spec(c)
    os.environ["R4D_E89_INSTANCE"] = "e100"
    e101 = _load("e101_commissure_anchors")
    script = str(HERE / "e101_commissure_anchors.py")

    def step(mode: str, *arguments: str) -> None:
        print(f"[{c.key}] e101: {mode}", flush=True)
        subprocess.run([sys.executable, "-u", script, *arguments], env=dict(os.environ, R4D_MODE=mode), check=True)

    if not (c.root / "e89").exists():
        raise SystemExit("E101 needs E89's rows on v4n96o")
    if not (e101.FRONT_FIELDS.exists() and e101.ANCHORS.exists()):
        step("fields")
    if not ((e101.FRONT / "ring_disagreement.json").exists() and (e101.MEASURED_ANCHORED / "anchors.json").exists()):
        step("instances")
    step("specs")
    for name in e101.SPEC_NAMES:
        c.run(e101.SPECS / f"{name}_test.json", "e101", allow_test=True)
    root = str(c.root / "e101")
    step("analyze", root, str(c.root / "e89"))
    return root


# ----------------------------------------------------- script-based studies ----

def _script(c, script: str, name: str, split: str, **extra: str) -> str:
    """One study script through ``uniformity_runner``, at this instance."""

    _, test = test_spec(c)
    expected = {p.name for p in (Path(test["benchmark"]) / "samples" / split).glob("*.npz")}
    out = c.root / "components"
    directory = out / name

    def done() -> set:
        return {json.loads(line)["sample"] for shard in directory.glob("shard_*.jsonl")
                for line in shard.read_text().splitlines() if line.strip()} if directory.exists() else set()

    c._repair_truncated(directory)
    if expected and done() >= expected:
        print(f"[{c.key}] {name}: complete, not re-run", flush=True)
        return str(directory)
    environment = dict(os.environ, R4D_BENCH=c.key, R4D_OUT=str(out), R4D_SHARD="0/1", R4D_SPLIT=split, **extra)
    environment.pop("R4D_ALLOW_TEST", None)
    if split == "test":
        environment["R4D_ALLOW_TEST"] = "1"
    print(f"[{c.key}] {name}: running {script}", flush=True)
    subprocess.run([sys.executable, "-u", str(HERE / script)], env=environment, check=True)
    if not (expected and done() >= expected):
        raise RuntimeError(f"{name} finished without every sample")
    return str(directory)


def study_e68(c) -> str:
    from rooted4d.inference import SolverConfig

    _, joint = frozen_spec(c, "test", "E68")
    weight = float(joint.get("session_prior_weight", SolverConfig().session_prior_weight))
    return _script(c, "e68_session_prior_scarcity.py", f"E68_{c.key}_test", "test",
                   R4D_EXPERIMENT="E68", R4D_FROZEN_SESSION_PRIOR=repr(weight))


def study_e69(c) -> str:
    _script(c, "e69_frozen_fits.py", f"E69_{c.key}_val", "val")
    return _script(c, "e69_frozen_fits.py", f"E69_{c.key}_test", "test")


def _image(study: str):
    def run(c) -> str:
        warm = study_e69(c)
        return _script(c, "e70_image_evidence.py", f"{study.upper()}_{c.key}_test", "test",
                       R4D_STUDY=study, R4D_E69=warm)
    return run


def study_e72(c) -> str:
    return _script(c, "e72_registration_stress.py", f"E72_{c.key}_test", "test")


def study_e91(c) -> str:
    return _script(c, "e91_image_search.py", f"E91_{c.key}_test", "test")


def study_e96(c) -> str:
    return _script(c, "e96_commissure_precision.py", f"E96_{c.key}_test", "test")


def study_e97(c) -> str:
    return _script(c, "e97_image_late_fusion.py", f"E97_{c.key}_test", "test")


FUNCTIONS = {"gate": study_gate, "e65": study_e65, "e66": study_e66, "e67": study_e67, "e68": study_e68,
             "e69": study_e69, "e70": _image("e70"), "e71": _image("e71"), "e74": _image("e74"),
             "e72": study_e72, "e91": study_e91, "e96": study_e96, "e97": study_e97, "e92": study_e92,
             "e89": study_e89, "e101": study_e101}


def cmd_run(key: str, *studies: str) -> None:
    if key not in STUDIES:
        raise SystemExit(f"no component studies on {key}")
    unknown = [s for s in studies if s not in STUDIES[key]]
    if unknown or not studies:
        raise SystemExit(f"{key} runs {', '.join(STUDIES[key])}; not {', '.join(unknown) or 'nothing'}")
    c = chain(key)
    for study in studies:
        print(f"### [{key}] {study}", flush=True)
        FUNCTIONS[study](c)
    print(f"[{key}] finished: {' '.join(studies)}", flush=True)


def chain_test_dir(key: str) -> Path:
    """An E100 chain's test rows, as its TEST_DONE marker names them."""

    root = Path(os.environ["R4D_E88_OUT"]) / key
    marker = root / "TEST_DONE"
    if not marker.exists():
        raise SystemExit(f"{key}: the chain has not finished its test")
    return root / marker.read_text().strip()


def cmd_e95() -> None:
    """E95 on the rebuilt primaries: signals from both chains' test rows, spreads at v4n96o's stored fits."""

    e95 = _load("e95_reliability")
    e95.ROWS = {"v4": chain_test_dir("v4n96o"), "v3": chain_test_dir("v3n96o")}
    e95.STORED, e95.STORED_ARM = chain_test_dir("v4n96o"), "S/joint"
    e95.REPORT = Path("results/benchmark/E100/E95/E95_reliability.json")
    e95.main()


def cmd_e92_configs() -> None:
    e92 = _e92()
    e92.OUT = CONFIGS / "e92"          # committed, like every other E100 config
    e92.cmd_configs()


# ---------------------------------------------------------------- analysis ----

REPORTS = Path("results/benchmark/E100")
INSTANCES = ("v4n96o", "v3n96o", "v4n48o", "v3n48o", "v3o", "v4oriented")


def cmd_e92_analyze() -> None:
    e92 = _e92()
    e92.OUT = CONFIGS / "v4n96o" / "e92"
    e92.cmd_analyze(str(Path(os.environ["R4D_E88_OUT"]) / "v4n96o" / "e92"))


def cmd_session_field(*keys: str) -> None:
    """The session field's size, RMS over samples and vertices (mm), static and dynamic.

    Over every vertex, this reproduces the supplement's figures for the original
    realistic-session benchmark on its test split (2.94 and 0.93 mm). Over the
    topology's CMR- and TEE-visible groups it gives 3.59 and 2.82 mm there; the
    supplement's 3.31 and 2.65 do not reproduce under either definition tried.
    """

    from rooted4d.geometry import toy_complex_for_vertices

    e100 = _load("e100_rebuild")
    report = {"experiment": "E100", "preregistration": PREREGISTRATION, "split": "test", "instances": {}}
    for key in keys:
        paths = sorted((Path(e100.benchmark(key)) / "samples" / "test").glob("*.npz"))
        static = np.array([np.load(p)["session_static"] for p in paths])
        dynamic = np.array([np.load(p)["session_dynamic"] for p in paths])
        groups = toy_complex_for_vertices(static.shape[1]).groups

        def rms(field, index=None):
            field = field if index is None else field[:, index]
            return float(np.sqrt((field ** 2).sum(axis=-1).mean()))

        report["instances"][key] = {
            "samples": len(paths),
            "all_vertices": {"static_mm": rms(static), "dynamic_mm": rms(dynamic)},
            "cmr_visible": {"static_mm": rms(static, groups["cmr_visible"]),
                            "dynamic_mm": rms(dynamic, groups["cmr_visible"])},
            "tee_visible": {"static_mm": rms(static, groups["tee_visible"]),
                            "dynamic_mm": rms(dynamic, groups["tee_visible"])}}
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "E100_session_field.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["instances"], indent=1))
    print("wrote", out)


def _instance_root(root: Path, key: str) -> Path:
    return root.parent / "e94" / key if key == E94["key"] else root / key


def _one(directory: Path) -> Path:
    """The single row directory under a stage label."""

    found = sorted(d for d in directory.iterdir() if d.is_dir()) if directory.exists() else []
    if len(found) != 1:
        raise SystemExit(f"{directory}: expected one row directory, found {len(found)}")
    return found[0]


def cmd_analyze(rows_root: str, *originals: str) -> None:
    from rooted4d.inference import SolverConfig

    root = Path(rows_root)
    e88 = _load("e88_replication")
    stage = e88._load("e62_stage_b")
    uniformity = _load("e65_uniformity")
    images = _load("e68_e72_analysis")
    REPORTS.mkdir(parents=True, exist_ok=True)
    header = {"experiment": "E100", "preregistration": PREREGISTRATION,
              "statistics": "each study's own; E62's are identity-averaged, sum of modality medians, "
                            "%d identity resamples, seed %d" % (stage.RESAMPLES, stage.SEED)}

    def save(name: str, report: dict) -> None:
        out = REPORTS / f"E100_{name}.json"
        out.write_text(json.dumps(dict(header, **report), indent=2) + "\n")
        print("wrote", out, flush=True)

    def test_rows(key):
        base = _instance_root(root, key)
        marker = base / "TEST_DONE"
        if not marker.exists():
            return None
        rows = stage.load_rows([str(base / marker.read_text().strip())])
        assert len(rows) == 60, (key, len(rows))
        return rows

    incomplete = {}

    def component(key, name, expected=60):
        """A component study's rows, only once every sample is in (else listed as incomplete)."""
        directory = _instance_root(root, key) / "components" / name
        if not directory.exists():
            return None
        samples = {json.loads(line)["sample"] for shard in directory.glob("shard_*.jsonl")
                   for line in shard.read_text().splitlines() if line.strip()}
        if len(samples) < expected:
            incomplete[name] = f"{len(samples)} of {expected} samples"
            return None
        return str(directory)

    # The main protocol on each instance, and the original construction beside it.
    main, tests = {"instances": {}, "originals": {}, "side_by_side": {}}, {}
    for key in INSTANCES:
        rows = test_rows(key)
        if rows is None:
            main["instances"][key] = {"status": "not finished"}
            continue
        tests[key] = rows
        entry = e88.test_report(stage, rows)
        selection = json.loads((_instance_root(root, key) / "session_selection.json").read_text())
        entry["session_weights_changed"] = {arm: a["selected"] for arm, a in selection["arms"].items() if a["changed"]}
        main["instances"][key] = entry
    for argument in originals:
        key, directories = argument.split("=", 1)
        rows = stage.load_rows(directories.split(","))
        assert len(rows) == 60, (key, len(rows))
        main["originals"][key] = e88.test_report(stage, rows)
        if "T1_representation" in main["instances"].get(key, {}):
            main["side_by_side"][key] = {
                name: {"original": e88._entry(main["originals"][key], path),
                       "rebuilt": e88._entry(main["instances"][key], path)}
                for name, path in e88.KEY_DIFFERENCES}
    save("main", main)

    # The representation's cross-validation (E62's gate statistics).
    gate = {}
    for key in INSTANCES:
        directory = _instance_root(root, key) / "gate"
        if not directory.exists():
            gate[key] = {"status": "not run"}
            continue
        rows = stage.load_rows([str(_one(directory))])
        gate[key] = {
            "samples": len(rows), "identities": len({r["identity"] for r in rows}),
            "representation": stage.paired(rows, "S/joint", "G/joint"),
            "fusion_cost": {m: stage.paired(rows, f"{m}/joint", f"{m}/independent") for m in stage.MODELS},
            "competitors": {arm: stage.paired(rows, "S/joint", f"S/{arm}")
                            for arm in ("exact_no_gauges", "soft_matched", "exact_repair", "penalty_repair")},
            "S_vs_G_independent": stage.paired(rows, "S/independent", "G/independent"),
            "repair": {f"{m}/{arm}": stage.repair_record(rows, f"{m}/{arm}")
                       for m in stage.MODELS for arm in ("exact_repair", "penalty_repair")}}
    save("gate", {"instances": gate})

    # Temporal (E65, E66, v3n96o) and session (E67, E68, v4n96o) studies.
    temporal = {}
    v3, v4 = _instance_root(root, "v3n96o"), _instance_root(root, "v4n96o")
    if (v3 / "e65").exists():
        temporal["E65"] = uniformity.e65_selection(stage.load_rows([str(_one(v3 / "e65"))]))
    if (v3 / "e66").exists():
        directory = _one(v3 / "e66")
        rows = stage.load_rows([str(directory)])
        temporal["E66"] = {
            "medians": {n: stage.sample_medians(rows, n) for n in ("S/periodic", "S/per_frame", "S/fixed_warp")},
            "periodic_minus_per_frame": uniformity.compare(stage, rows, "S/periodic", "S/per_frame"),
            "periodic_minus_fixed_warp": uniformity.compare(stage, rows, "S/periodic", "S/fixed_warp")}
        chosen = json.loads((directory / "spec.json").read_text())["e65_selection"]["periodic"]
        frozen = json.loads((directory / "spec.json").read_text())["base_config"].get("fourier_order", 3)
        if (chosen["setting"] == frozen and abs(chosen["motion_weight"] - SolverConfig().motion_prior_weight) < 1e-12
                and "v3n96o" in tests):
            temporal["E66"]["reproduction_of_chain_test"] = uniformity.reproduction(
                stage, rows, "S/periodic", tests["v3n96o"], "S/joint")
    if (v4 / "e67").exists():
        rows = stage.load_rows([str(_one(v4 / "e67"))])
        temporal["E67"] = {
            "medians": {n: stage.sample_medians(rows, n) for n in ("S/frozen", "S/oscillatory", "S/no_session")},
            "frozen_minus_no_session": uniformity.compare(stage, rows, "S/frozen", "S/no_session"),
            "frozen_minus_oscillatory": uniformity.compare(stage, rows, "S/frozen", "S/oscillatory"),
            "oscillatory_minus_no_session": uniformity.compare(stage, rows, "S/oscillatory", "S/no_session")}
        if "v4n96o" in tests:
            temporal["E67"]["reproduction_of_chain_test"] = uniformity.reproduction(
                stage, rows, "S/frozen", tests["v4n96o"], "S/joint")
    directory = component("v4n96o", "E68_v4n96o_test")
    if directory:
        arms = json.loads((Path(directory) / "run.json").read_text())["arms"]
        weak = next(name for name in arms if name != "frozen")
        temporal["E68"] = {"arms": arms, "levels": images.analyze_e68(directory, weak)}
        if "v4n96o" in tests:
            rows = stage.load_rows([directory])
            full = {k: stage.sample_medians(rows, "full/frozen")[k] - stage.sample_medians(tests["v4n96o"], "S/joint")[k]
                    for k in ("heldout_cmr_mm", "heldout_tee_mm")}
            temporal["E68"]["reproduction_of_chain_test"] = {
                "differences": full, "passed": all(abs(v) <= 0.01 for v in full.values())}
    save("temporal_session", temporal)

    # Calibration, image evidence and registration (v3n96o), image search and late fusion (both).
    calibration_images = {}
    val, test = component("v3n96o", "E69_v3n96o_val"), component("v3n96o", "E69_v3n96o_test")
    if val and test and "v3n96o" in tests:
        record = images.analyze_e69(val, test, str(_instance_root(root, "v3n96o")
                                                   / (v3 / "TEST_DONE").read_text().strip()))
        record["reproduction_of_chain_test"] = record.pop("reproduction_of_e64")
        calibration_images["E69"] = record
    for study, arms, reference in (
            ("E70", ("point_fit", "image_cold", "image_warm", "landmarks_2", "landmarks_4", "landmarks_8"),
             "image_warm"),
            ("E71", ("point_fit", "cmr_image_tee_point", "cmr_point_tee_image", "both_image"), "point_fit")):
        directory = component("v3n96o", f"{study}_v3n96o_test")
        if directory:
            calibration_images[study] = images.analyze_images(directory, arms, reference)
            records = images._records(directory)
            calibration_images[study]["within_5deg"] = {   # counts, as the tables print them
                arm: sum(1 for r in records if (r["variants"].get(arm) or {}).get("succeeded")
                         and r["variants"][arm]["rotation_deg"] < 5.0) for arm in arms}
            calibration_images[study]["samples"] = len(records)
    directory = component("v3n96o", "E72_v3n96o_test")
    if directory:
        calibration_images["E72"] = images.analyze_e72(directory)
    save("calibration_images", calibration_images)
    e74, e70 = component("v3n96o", "E74_v3n96o_test"), component("v3n96o", "E70_v3n96o_test")
    if e74 and e70:
        _load("e74_analysis").main(e74, e70, str(REPORTS / "E100_E74.json"))
    search = _load("e91_e96_analysis")
    e91 = {key: component(key, f"E91_{key}_test") for key in ("v3n96o", "v4n96o")}
    e91 = {key: directory for key, directory in e91.items() if directory}
    if e91:
        save("E91", search.e91(e91))
    directory = component("v3n96o", "E96_v3n96o_test")
    if directory:
        save("E96", search.e96(directory))
    e97 = {key: (component(key, f"E97_{key}_test"), e91.get(key)) for key in ("v3n96o", "v4n96o")}
    e97 = {key: pair for key, pair in e97.items() if all(pair)}
    if e97:
        _load("e97_image_late_fusion").cmd_analyze(e97, str(REPORTS / "E100_E97.json"))

    # Replication over splits (E88's statistics and summary).
    e100 = _load("e100_rebuild")
    replication = {}
    for base in e100.PRIMARY:
        bench = e100.BASES[base][0]
        if base in main["instances"] and "T1_representation" in main["instances"][base]:
            replication[f"{bench}_original"] = main["instances"][base]
        for k in range(1, len(e100.SPLIT_SEEDS) + 1):
            rows = test_rows(f"{base}s{k}")
            replication[f"{bench}s{k}"] = ({"status": "not finished"} if rows is None
                                           else e88.test_report(stage, rows))
    if incomplete:
        save("incomplete", {"components": incomplete})
        print("incomplete, not analyzed:", incomplete, flush=True)
    try:
        summary = e88.summarize(replication)
    except ValueError:      # a benchmark with no finished test set yet
        summary = {"status": "incomplete: a benchmark has no finished test set yet"}
    save("replication", {"instances": replication, "summary": summary,
                         "names": {"v4": "v4n96o", "v3": "v3n96o"}})


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    {"run": cmd_run, "e92-configs": cmd_e92_configs, "e95": cmd_e95, "e92-analyze": cmd_e92_analyze,
     "session-field": cmd_session_field, "analyze": cmd_analyze}[command](*arguments)
