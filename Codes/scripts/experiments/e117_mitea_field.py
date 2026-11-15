"""E117, part B: the residual field on MITEA's real same-patient missing-frame task.

Pre-registered in ``Docs/E117_FIELD_REPLICATION_AND_MITEA_PREREGISTRATION_2026-10-24.md``.
E99's same-scan pairing (CMR-derived labels against echo predictions), with
E99's model recipe, solver settings, perturbation draws and two-frame start.
Each hidden-frame variant (echo end diastole, CMR end diastole) is fitted by
E116's arm, which scores the joint fit and every field setting from one fit,
and by late fusion with E103's transfer of the other modality's frame.

Two canonicalizations of the subjects' chambers into the model frame:

``reported``
    E99's: ring direction and Procrustes placement from the mean of the scan-1
    ES and ED label chambers. In the CMR end-diastole task the hidden frame
    takes part in it.
``es_only``
    From the scan-1 ES label chamber alone, which both tasks observe; applied
    to the training, validation and test subjects alike, so the model is
    re-learned under it.

Both are computed from E99's stored geometry: any rigid placement and ring
direction of a chamber leave the re-placement unchanged (``place``).

    R4D_MODE=geometry R4D_SPLIT=train|val|test python e117_mitea_field.py
    R4D_MODE=fit      R4D_SPLIT=val|test       python e117_mitea_field.py   (test needs R4D_ALLOW_TEST=1)
    R4D_MODE=analyze  R4D_SPLIT=val|test       python e117_mitea_field.py
    python e117_mitea_field.py latex

With ``R4D_LABELS_AS_PREDICTIONS=1`` the label chambers stand in for the echo
predictions (a code check on the training subjects, which have no predictions).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E117_FIELD_REPLICATION_AND_MITEA_PREREGISTRATION_2026-10-24.md"
CONVENTIONS = ("reported", "es_only")
RULES = {"reported": "es_ed_mean", "es_only": "es"}
VARIANTS = ("echo_ED", "cmr_ED")
#: The held-out endpoint of each variant.
ENDPOINT = {"echo_ED": "heldout_tee_mm", "cmr_ED": "heldout_cmr_mm"}
TRANSFER = {"echo_ED": "transfer_tee_mm", "cmr_ED": "transfer_cmr_mm"}
DERIVED = Path("data/derived/mitea-e117")              # derived MITEA geometry: never committed
OUT = Path("results/benchmark/E117")
E99_ROWS = Path(os.environ.get("R4D_E99_ROWS", "results/benchmark/E99"))
E103_ROWS = Path("results/benchmark/E103/E103_rows.json")
TOLERANCE_MM = 5e-3
POSE_TOLERANCE = 1e-3
GAP_LIMIT_MM = 1e-9
#: Re-placing E99's stored chambers by E99's own rule must return them to this (mm).
IDENTITY_TOLERANCE_MM = 1e-8


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E99 = _load("e99_mitea_fusion")
E103 = _load("e103_mitea_transfer")
E116 = _load("e116_residual_field")
E117 = _load("e117_field_replication")


# -------------------------------------------------------------- geometry ----


def place(chambers: dict, template_chamber: np.ndarray, rule: str) -> tuple[dict, dict]:
    """One subject's chambers in the model frame, by ``rule``: ``es_ed_mean`` (E99's) or ``es``.

    ``chambers`` maps (scan, phase, source) to a (49, 3) chamber or None. The
    ring-direction test is invariant under proper rigid maps and Procrustes
    composes with them, so chambers already placed by any rigid map and ring
    direction give the same result as the raw ones.
    """

    es, ed = chambers[("scan1", "ES", "label")], chambers[("scan1", "ED", "label")]
    reference = es if rule == "es" else 0.5 * (es + ed)
    reverse = E99._handedness(reference) != E99._handedness(template_chamber)
    if reverse:
        chambers = {k: (None if v is None else E99._reverse(v)) for k, v in chambers.items()}
        reference = E99._reverse(reference)
    rotation, translation = E99._procrustes(reference, template_chamber)
    placed = {k: (None if v is None else v @ rotation.T + translation) for k, v in chambers.items()}
    skew = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]])
    angle = float(np.degrees(np.arctan2(np.linalg.norm(skew) / 2.0, (np.trace(rotation) - 1.0) / 2.0)))
    return placed, {"reversed_ring": bool(reverse), "rotation_deg": angle,
                    "translation_mm": float(np.linalg.norm(translation))}


def _by_subject(path: Path) -> dict:
    out = {}
    with np.load(path) as blob:
        for key in blob.files:
            subject, scan, phase, source = key.split("|")
            out.setdefault(subject, {})[(scan, phase, source)] = blob[key]
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _template_chamber() -> np.ndarray:
    return E99._model_template()[: 2 * E99.N_ANNULUS + 1]


def cmd_geometry(split: str) -> None:
    """Both conventions' geometry of one split, from E99's stored geometry."""

    if split == "test" and os.environ.get("R4D_ALLOW_TEST") != "1":
        raise SystemExit("refusing test subjects without R4D_ALLOW_TEST=1")
    source = E99.DERIVED / f"geometry_{split}.npz"
    template = _template_chamber()
    subjects = _by_subject(source)
    record = {"experiment": "E117", "preregistration": PREREGISTRATION, "split": split,
              "source": str(source), "source_sha256": _sha256(source), "subjects": {}}
    arrays = {convention: {} for convention in CONVENTIONS}
    worst_identity = 0.0
    for subject, chambers in sorted(subjects.items()):
        if any(chambers.get(("scan1", p, "label")) is None for p in E99.PHASES):
            record["subjects"][subject] = {"dropped": "no scan-1 ES and ED label chambers"}
            continue
        entry = {}
        for convention in CONVENTIONS:
            placed, moved = place(chambers, template, RULES[convention])
            entry[convention] = moved
            for (scan, phase, src), chamber in placed.items():
                arrays[convention][f"{subject}|{scan}|{phase}|{src}"] = chamber
        worst_identity = max(worst_identity, max(
            float(np.abs(arrays["reported"][f"{subject}|{s}|{p}|{src}"] - chamber).max())
            for (s, p, src), chamber in chambers.items()))
        record["subjects"][subject] = entry
    record["reported_rule_reproduces_stored_mm"] = worst_identity
    record["reported_rule_reproduces_stored"] = bool(worst_identity <= IDENTITY_TOLERANCE_MM)
    for convention in CONVENTIONS:
        directory = DERIVED / convention
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"geometry_{split}.npz"
        if convention == "reported":
            # E99's own arrays, unchanged: the re-placement check above shows they are E99's rule's result.
            shutil.copyfile(source, path)
        else:
            np.savez_compressed(path, **arrays[convention])
        record.setdefault("outputs", {})[convention] = {"path": str(path), "sha256": _sha256(path)}
    moved = [e["es_only"] for e in record["subjects"].values() if "es_only" in e]
    record["es_only_against_reported"] = {
        "subjects": len(moved), "ring_direction_differs": int(sum(
            e["es_only"]["reversed_ring"] != e["reported"]["reversed_ring"]
            for e in record["subjects"].values() if "es_only" in e))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"E117_mitea_geometry_{split}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"[geometry/{split}] {len(moved)} subjects | E99's rule reproduces its stored chambers to "
          f"{worst_identity:.2e} mm | ring direction differs for "
          f"{record['es_only_against_reported']['ring_direction_differs']}", flush=True)
    if not record["reported_rule_reproduces_stored"]:
        raise SystemExit("E99's placement rule does not reproduce its stored chambers")


# ----------------------------------------------------------------- fits ----

#: Built in the parent before the worker pool forks: convention -> (model, base, joint, late options, chambers).
CONTEXT: dict = {}


def _prepare(convention: str, split: str, stand_in: bool) -> None:
    from rooted4d.inference import SolverConfig

    E99.DERIVED = DERIVED / convention       # E99's model and chamber readers follow this directory
    model, spec, record = E99.build_model()
    base = replace(SolverConfig(**spec["base_config"]), estimate_phase_warp=False)     # as E99
    by_name = {v["name"]: v for v in spec["variants"]}
    joint = replace(base, **by_name["S/joint"]["overrides"])
    late_options = dict(by_name["S/independent"].get("options") or {}, store_parameters=True)
    chambers = E99._chambers(split)
    if stand_in:
        chambers.update({(s, scan, p, "prediction"): v for (s, scan, p, src), v in list(chambers.items())
                         if src == "label"})
    CONTEXT[convention] = (model, base, joint, late_options, chambers, record)


def _task(arguments: tuple) -> dict:
    """One subject, convention and hidden frame: the field arm and late fusion with transfer."""

    from rooted4d.inference.baselines import run_arm

    convention, subject, held_out, configs = arguments
    model, base, joint, late_options, chambers, _ = CONTEXT[convention]
    values = {side: {p: chambers.get((subject, "scan1", p, source)) for p in E99.PHASES}
              for side, source in (("cmr", "label"), ("echo", "prediction"))}
    record = {"subject": subject, "convention": convention, "held_out": held_out}
    if any(v is None for side in values.values() for v in side.values()):
        record["front_end_failure"] = True
        return record
    rng = np.random.default_rng(E99._seed(subject, "same_scan"))          # E99's draw, in E99's order
    rotation = rng.normal(0.0, E99.POSE_ROTATION_SD, size=3)
    translation = rng.normal(0.0, E99.POSE_TRANSLATION_SD, size=3)
    offset = float(rng.uniform(-E99.OFFSET_RANGE, E99.OFFSET_RANGE))
    problem = E99.make_problem(values["cmr"], values["echo"], model, joint, rotation=rotation,
                               translation=translation, offset=offset, held_out=held_out)
    alignment = E99.two_frame_alignment(problem.observations)
    started = time.perf_counter()
    field = run_arm("rooted4d_residual_field", problem, joint, alignment=alignment,
                    options={"configs": configs}).as_dict()
    record["field_seconds"] = round(time.perf_counter() - started, 3)
    extras = field.get("extras") or {}
    endpoint = ENDPOINT[held_out]
    record["joint_fit"] = {k: field.get(k) for k in ("rotation_deg", "translation_mm", "phase_error_cycles",
                                                     "phase_map_cycles")}
    # The arm's own flag asks for both modalities' held-out errors, and each task hides one. The
    # joint fit fails only with a non-finite cost, which the arm passes on as its failure reason.
    record["joint_fit"]["succeeded"] = bool(extras) and field.get("failure_reason") in (None, "non_finite_score")
    record["joint_fit"][endpoint] = (extras.get("rooted4d") or {}).get(endpoint)
    record["by_config"] = {name: {k: scores.get(k) for k in (endpoint, "attachment_residual_mm", "field_rms_mm")}
                           for name, scores in (extras.get("by_config") or {}).items()}
    started = time.perf_counter()
    late = run_arm("late_fusion_trajectory", problem, base, alignment=alignment, options=late_options).as_dict()
    record["late_fusion"] = {k: late.get(k) for k in ("succeeded", endpoint, "rotation_deg", "translation_mm",
                                                      "phase_error_cycles")}
    if late.get("succeeded"):
        record["late_fusion"].update(E103.transfer_errors(late["extras"], problem, base, late_options))
    record["late_seconds"] = round(time.perf_counter() - started, 3)
    return record


def _selection() -> dict:
    report = json.loads((OUT / "E117_mitea_val.json").read_text())
    if not report["check"]["passed"]:
        raise SystemExit("the validation checks did not pass")
    return report["selection"]


def configs(split: str, convention: str) -> list:
    if split == "val":
        return E116.GRID
    chosen = _selection()[convention]
    out = []
    for config in [*E117.frozen_settings().values(), chosen[E116.OURS["shared"]], chosen[E116.OURS["session"]]]:
        if E116._name(config) not in {E116._name(c) for c in out}:
            out.append(config)
    return out


def cmd_fit(split: str) -> None:
    if split == "test" and os.environ.get("R4D_ALLOW_TEST") != "1":
        raise SystemExit("refusing test subjects without R4D_ALLOW_TEST=1")
    stand_in = os.environ.get("R4D_LABELS_AS_PREDICTIONS") == "1"
    limit = int(os.environ.get("R4D_LIMIT", "0"))       # code checks only
    for convention in CONVENTIONS:
        _prepare(convention, split, stand_in)
    tasks = []
    for convention in CONVENTIONS:
        chambers = CONTEXT[convention][4]
        subjects = sorted({k[0] for k in chambers})
        if limit:
            subjects = subjects[:limit]
        settings = configs(split, convention) if not os.environ.get("R4D_CONFIGS") else json.loads(
            os.environ["R4D_CONFIGS"])
        tasks += [(convention, subject, held_out, settings) for subject in subjects for held_out in VARIANTS]
    workers = max(1, int(os.environ.get("R4D_WORKERS", "1")))
    started = time.perf_counter()
    if workers == 1:
        rows = [_task(t) for t in tasks]
    else:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            rows = pool.map(_task, tasks, chunksize=1)
    for row in rows:
        if not row.get("front_end_failure"):
            print(row["convention"], row["subject"], row["held_out"],
                  round(row["joint_fit"].get(ENDPOINT[row["held_out"]]) or float("nan"), 3),
                  round(row["late_fusion"].get(TRANSFER[row["held_out"]]) or float("nan"), 3), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "_standin" if stand_in else ""
    path = OUT / f"E117_mitea_rows_{split}{suffix}.json"
    path.write_text(json.dumps(
        {"experiment": "E117", "preregistration": PREREGISTRATION, "split": split,
         "labels_as_predictions": stand_in, "models": {c: CONTEXT[c][5] for c in CONVENTIONS},
         "seconds": round(time.perf_counter() - started, 1), "rows": rows}, default=float) + "\n")
    print("wrote", path, f"({len(rows)} tasks)")


# -------------------------------------------------------------- analysis ----


def _values(rows: list, convention: str, held_out: str, arm: str) -> dict:
    """Per-subject held-out error of one arm: ``joint_fit``, ``transfer``, ``late_own`` or a setting's name."""

    endpoint, out = ENDPOINT[held_out], {}
    for row in rows:
        if row.get("front_end_failure") or row["convention"] != convention or row["held_out"] != held_out:
            continue
        if arm == "joint_fit":
            value = row["joint_fit"].get(endpoint) if row["joint_fit"].get("succeeded") else None
        elif arm in ("transfer", "late_own"):
            late = row["late_fusion"]
            value = late.get(TRANSFER[held_out] if arm == "transfer" else endpoint) if late.get("succeeded") else None
        else:
            value = (row["by_config"].get(arm) or {}).get(endpoint) if row["joint_fit"].get("succeeded") else None
        if value is not None and np.isfinite(value):
            out[row["subject"]] = float(value)
    return out


def _stored_e99(split: str) -> dict:
    """E99's rows of the joint fit and late fusion: (subject, held_out) -> {arm: record}."""

    rows = json.loads((E99_ROWS / f"E99_fits_{split}.json").read_text())["rows"]
    out = {}
    for row in rows:
        if row["pairing"] != "same_scan" or row.get("front_end_failure"):
            continue
        for held_out in VARIANTS:
            out[(row["subject"], held_out)] = {arm: row["variants"].get(f"{arm}/{held_out}") or {}
                                               for arm in ("rooted4d_full", "late_fusion_trajectory")}
    return out


def code_check(rows: list, split: str) -> dict:
    """The reported convention against E99's (and, on test, E103's) rows; every setting's gap."""

    stored = _stored_e99(split)
    transfer = {}
    if split == "test":
        for row in json.loads(E103_ROWS.read_text())["rows"]:
            for held_out, record in (row.get("variants") or {}).items():
                transfer[(row["subject"], held_out)] = record.get(TRANSFER[held_out])
    joint, late, moved, pose, gaps, missing = [], [], [], [], [], 0
    for row in rows:
        if row.get("front_end_failure"):
            missing += row["convention"] == "reported"
            continue
        gaps += [v.get("attachment_residual_mm", np.inf) for v in row["by_config"].values()]
        if row["convention"] != "reported":
            continue
        reference = stored.get((row["subject"], row["held_out"]))
        endpoint = ENDPOINT[row["held_out"]]
        if not (reference and reference["rooted4d_full"].get("succeeded") and row["joint_fit"].get("succeeded")
                and row["late_fusion"].get("succeeded")):
            missing += 1
            continue
        joint.append(abs(row["joint_fit"][endpoint] - reference["rooted4d_full"][endpoint]))
        pose.append(max(abs(row["joint_fit"][k] - reference["rooted4d_full"][k])
                        for k in ("rotation_deg", "translation_mm", "phase_error_cycles")))
        late.append(abs(row["late_fusion"][endpoint] - reference["late_fusion_trajectory"][endpoint]))
        if split == "test":
            moved.append(abs(row["late_fusion"][TRANSFER[row["held_out"]]]
                             - transfer[(row["subject"], row["held_out"])]))
    worst = {"joint_fit_mm": max(joint, default=np.nan), "joint_fit_pose": max(pose, default=np.nan),
             "late_fusion_mm": max(late, default=np.nan), "transfer_mm": max(moved, default=None),
             "gap_any_setting_mm": max(gaps, default=np.nan)}
    passed = (bool(joint) and missing == 0 and worst["joint_fit_mm"] <= TOLERANCE_MM
              and worst["joint_fit_pose"] <= POSE_TOLERANCE and worst["late_fusion_mm"] <= TOLERANCE_MM
              and worst["gap_any_setting_mm"] <= GAP_LIMIT_MM
              and (split != "test" or (bool(moved) and worst["transfer_mm"] <= TOLERANCE_MM)))
    return {"compared": len(joint), "not_comparable": missing, "max_abs_difference": worst, "passed": bool(passed)}


def _sum_of_medians(rows: list, convention: str, arm: str) -> float:
    medians = [np.median(list(v.values())) if (v := _values(rows, convention, h, arm)) else np.nan
               for h in VARIANTS]
    return float(np.sum(medians))


def analyze_val(rows: list) -> dict:
    report = {"experiment": "E117", "preregistration": PREREGISTRATION, "split": "val",
              "rule": "sum over the two hidden-frame tasks of the median held-out error, per variant and convention",
              "selection": {}, "at_grid_edge": {}, "scores": {}, "joint_fit_sum": {},
              "check": code_check(rows, "val")}
    for convention in CONVENTIONS:
        names = [E116._name(c) for c in E116.GRID]
        scores = {n: _sum_of_medians(rows, convention, n) for n in names}
        selection, edges = {}, {}
        for sharing, variant in E116.OURS.items():
            candidates = {n: v for n, v in scores.items() if n.startswith(sharing + "/") and np.isfinite(v)}
            chosen = E116._parse(min(candidates, key=candidates.get))
            selection[variant], edges[variant] = chosen, E116._edges(chosen)
        report["scores"][convention] = scores
        report["selection"][convention] = selection
        report["at_grid_edge"][convention] = edges
        report["joint_fit_sum"][convention] = _sum_of_medians(rows, convention, "joint_fit")
        print(f"[val/{convention}] selected {selection} | sums "
              f"{[round(scores[E116._name(c)], 3) for c in selection.values()]} | joint fit "
              f"{report['joint_fit_sum'][convention]:.3f} | frozen "
              f"{[round(scores[E116._name(c)], 3) for c in E117.frozen_settings().values()]}", flush=True)
    return report


def _arms(selection: dict) -> dict:
    frozen = E117.frozen_settings()
    return {"Rooted4D": E116._name(frozen["R4D+GP-frozen"]),
            "Rooted4D, per-modality field": E116._name(frozen["R4D+GP-session-frozen"]),
            "Rooted4D, selected field": E116._name(selection[E116.OURS["shared"]]),
            "Rooted4D, selected per-modality field": E116._name(selection[E116.OURS["session"]]),
            "joint fit": "joint_fit", "late fusion, frame transfer": "transfer", "late fusion, own fit": "late_own"}


def _below(entry: dict) -> bool:
    return bool(np.isfinite(entry["upper"]) and entry["upper"] < 0)


def analyze_test(rows: list) -> dict:
    selection = _selection()
    report = {"experiment": "E117", "preregistration": PREREGISTRATION, "split": "test", "selection": selection,
              "frozen": E117.frozen_settings(),
              "statistic": "E99's: per-subject values, difference of medians, %d resamples, seed %d" % (
                  E99.RESAMPLES, E99.SEED),
              "check": code_check(rows, "test"), "medians": {}, "minus": {}, "between_conventions": {}}
    comparisons = (("Rooted4D", "joint fit"), ("Rooted4D", "late fusion, frame transfer"),
                   ("joint fit", "late fusion, frame transfer"), ("Rooted4D", "Rooted4D, per-modality field"),
                   ("Rooted4D, per-modality field", "joint fit"), ("Rooted4D, selected field", "joint fit"),
                   ("Rooted4D, selected per-modality field", "joint fit"), ("joint fit", "late fusion, own fit"))
    for convention in CONVENTIONS:
        arms = _arms(selection[convention])
        values = {(arm, h): _values(rows, convention, h, name) for arm, name in arms.items() for h in VARIANTS}
        report["medians"][convention] = {arm: {h: float(np.median(list(values[(arm, h)].values())))
                                               for h in VARIANTS} for arm in arms}
        report["minus"][convention] = {f"{a} - {b}": {h: E99._paired(values[(a, h)], values[(b, h)])
                                                      for h in VARIANTS} for a, b in comparisons}
    for arm in ("joint fit", "late fusion, frame transfer", "Rooted4D"):
        report["between_conventions"][arm] = {h: E99._paired(
            _values(rows, "es_only", h, _arms(selection["es_only"])[arm]),
            _values(rows, "reported", h, _arms(selection["reported"])[arm])) for h in VARIANTS}
    primary = report["minus"]["es_only"]
    report["predictions"] = {
        "M1_field_lower_than_joint_fit": {h: _below(primary["Rooted4D - joint fit"][h]) for h in VARIANTS},
        "M2_field_lower_than_frame_transfer": {
            h: _below(primary["Rooted4D - late fusion, frame transfer"][h]) for h in VARIANTS},
        "M3_joint_fit_lower_than_frame_transfer": {
            h: _below(primary["joint fit - late fusion, frame transfer"][h]) for h in VARIANTS},
    }
    return report


def cmd_analyze(split: str) -> None:
    rows = json.loads((OUT / f"E117_mitea_rows_{split}.json").read_text())["rows"]
    report = analyze_val(rows) if split == "val" else analyze_test(rows)
    path = OUT / f"E117_mitea_{split}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    if split == "test":
        print(json.dumps(report["medians"], indent=1))
        print(json.dumps(report["predictions"], indent=1))
    print("wrote", path, "| checks passed:", report["check"]["passed"])


def latex() -> str:
    """Rows of the missing-frame table, available-evidence canonicalization first, then E99's."""

    tables = _load("e64_manuscript_tables")
    report = json.loads((OUT / "E117_mitea_test.json").read_text())
    lines = []
    for convention, label in (("es_only", "Available-evidence placement"), ("reported", "Placement as reported")):
        lines.append(r"\multicolumn{3}{@{}l}{\textit{%s}}\\" % label)
        for arm in ("late fusion, frame transfer", "joint fit", "Rooted4D", "Rooted4D, per-modality field"):
            m = report["medians"][convention][arm]
            lines.append("%s & %s & %s\\\\" % (arm, tables.number(m["echo_ED"]), tables.number(m["cmr_ED"])))
    return "\n".join(lines)


if __name__ == "__main__":
    if sys.argv[1:] == ["latex"]:
        print(latex())
    else:
        mode, split_name = os.environ.get("R4D_MODE"), os.environ.get("R4D_SPLIT", "val")
        if mode == "geometry":
            cmd_geometry(split_name)
        elif mode == "fit":
            cmd_fit(split_name)
        elif mode == "analyze":
            cmd_analyze(split_name)
        else:
            sys.exit(f"unknown R4D_MODE {mode!r}")
