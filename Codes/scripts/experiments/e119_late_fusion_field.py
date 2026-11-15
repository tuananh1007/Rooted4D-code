"""E119: late fusion with E116's residual field, on the 16 test sets.

Pre-registered in ``Docs/E119_LATE_FUSION_FIELD_PREREGISTRATION_2026-11-13.md``.
Late fusion's two reconstructions and trajectory registration as each set's
chain froze them (``S/late_fusion_trajectory``), then E116's residual field
(``late_fusion_residual_field``): one field per reconstruction (``LF+GP-session``,
primary) or one field shared after the registration (``LF+GP``). The sets are
the two 96-node primaries (E100's chain, E116's field rows) and the 14 sets of
E117 (their chains, E117's field rows).

    R4D_RESULTS=<results root> R4D_E119_OUT=<dir> python e119_late_fusion_field.py run <split> [key ...]
    python e119_late_fusion_field.py analyze <split>   -> results/benchmark/E119/E119_<split>.json
    python e119_late_fusion_field.py latex

Validation runs E116's 802 settings on every set and selects, per variant and
set, by E116's rule. Test scores the selected settings and, reported without
prediction, Rooted4D's own selected settings on the same set, once. ``run
test`` refuses unless the validation report exists and its checks passed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E119_LATE_FUSION_FIELD_PREREGISTRATION_2026-11-13.md"
#: The smoke run writes its reports elsewhere (``R4D_E119_REPORTS``), never here.
REPORTS = Path(os.environ.get("R4D_E119_REPORTS", "results/benchmark/E119"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E116 = _load("e116_residual_field")
E117 = _load("e117_field_replication")
ROOTED, LATE = E116.ROOTED, E116.LATE
R4D, R4D_SESSION = E116.OURS["shared"], E116.OURS["session"]
#: key -> (results root of its chain, session level, what distinguishes it)
SETS = {"v4n96o": ("e100", "realistic", "96 nodes, primary"), "v3n96o": ("e100", "small", "96 nodes, primary")}
SETS.update(E117.SETS)
PRIMARIES = ("v4n96o", "v3n96o")
#: The arm's variant in the spec, and the names its settings are read under.
ARM_NAME = "LF+field"
OURS = {"session": "LF+GP-session", "shared": "LF+GP"}
#: Late fusion at Rooted4D's own selected setting on the same set (reported without prediction).
AT_ROOTED = {"LF+GP-session@R4D": R4D_SESSION, "LF+GP@R4D": R4D}
TOLERANCE_MM = 5e-3
POSE_TOLERANCE = 1e-3
EXACT = 1e-9


def _results() -> Path:
    return Path(os.environ["R4D_RESULTS"])


def _out() -> Path:
    return Path(os.environ["R4D_E119_OUT"])


def _root(key: str) -> Path:
    return _results() / SETS[key][0] / key


def _frozen(key: str) -> tuple[dict, Path]:
    directory = _root(key) / (_root(key) / "TEST_DONE").read_text().strip()
    return json.loads((directory / "spec.json").read_text()), directory


def _late(frozen: dict) -> dict:
    return next(v for v in frozen["variants"] if v["name"] == LATE)


def _selection() -> dict:
    report = json.loads((REPORTS / "E119_val.json").read_text())
    if not report["check"]["passed"]:
        raise SystemExit("the validation checks did not pass")
    return report["selection"]


def rooted_selection(key: str) -> dict:
    """Rooted4D's selected field settings on this set: E116's on the primaries, E117's on the others."""

    if key in PRIMARIES:
        return json.loads((E116.REPORTS / "E116_val.json").read_text())["selection"][key]
    return json.loads((E117.REPORTS / "E117_sets_val.json").read_text())["selection"][key]


def rooted_rows(key: str) -> Path:
    """Rooted4D's field rows on this set's test split (E116's or E117's)."""

    if key in PRIMARIES:
        index = json.loads((_results() / "e116" / "test" / "index.json").read_text())
        return _results() / "e116" / "test" / index[f"{key}/full"]
    index = json.loads((_results() / "e117" / "test" / "index.json").read_text())
    return _results() / "e117" / "test" / index[key]


def configs(key: str, split: str) -> list:
    """Every setting on validation; on test the selected ones, then Rooted4D's, once each."""

    if split == "val":
        # The per-modality zero setting first: it is the arm's primary, late fusion itself.
        return [c for c in E116.GRID if c[0] == "session"] + [c for c in E116.GRID if c[0] == "shared"]
    chosen, theirs = _selection()[key], rooted_selection(key)
    out = []
    for config in [chosen[OURS["session"]], chosen[OURS["shared"]], theirs[R4D_SESSION], theirs[R4D]]:
        if E116._name(config) not in {E116._name(c) for c in out}:
            out.append(config)
    return out


def specs(split: str, keys) -> list[tuple[str, dict]]:
    out = []
    for key in keys:
        frozen, directory = _frozen(key)
        late = _late(frozen)
        spec = {"experiment": "E119", "preregistration": PREREGISTRATION, "key": key, "level": "full",
                "benchmark": frozen["benchmark"], "split": split, "start": frozen["start"],
                "models": frozen["models"], "base_config": frozen["base_config"], "frozen_from": str(directory),
                "variants": [{"name": ARM_NAME, "arm": "late_fusion_residual_field", "model": late["model"],
                              "options": {"modality_overrides": late["options"]["modality_overrides"],
                                          "configs": configs(key, split)}}]}
        out.append((key, spec))
    return out


def _index(split: str) -> dict:
    """Set -> rows directory, from the spec files: several jobs run sets at once, so
    no shared index file is trusted."""

    stage_dir = _out() / split
    return {key: E116._digest(json.loads((stage_dir / f"{key}.json").read_text()))
            for key in SETS if (stage_dir / f"{key}.json").exists()}


def cmd_run(split: str, *keys: str) -> None:
    keys = keys or tuple(SETS)
    unknown = [k for k in keys if k not in SETS]
    if unknown:
        raise SystemExit(f"unknown sets: {unknown}")
    if split == "test":
        _selection()
    elif split != "val":
        raise SystemExit("split must be val or test")
    stage_dir = _out() / split
    stage_dir.mkdir(parents=True, exist_ok=True)
    # Smoke only (a separate output root): one shard of the samples and a few settings.
    smoke = os.environ.get("R4D_SMOKE_SHARD")
    for name, spec in specs(split, keys):
        if smoke and os.environ.get("R4D_CONFIGS"):
            spec["variants"][0]["options"]["configs"] = json.loads(os.environ["R4D_CONFIGS"])
        digest = E116._digest(spec)
        spec_path = stage_dir / f"{name}.json"
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
        if not smoke and E116._complete(spec, stage_dir / digest):
            print(f"[{split}/{name}] complete, not re-run", flush=True)
            continue
        environment = dict(os.environ, R4D_SPEC=str(spec_path), R4D_OUT=str(stage_dir), R4D_SHARD=smoke or "0/1")
        environment.pop("R4D_ALLOW_TEST", None)
        if split == "test":
            environment["R4D_ALLOW_TEST"] = "1"
        print(f"[{split}/{name}] running {spec_path}", flush=True)
        subprocess.run([sys.executable, "-u", str(HERE / "e59_unknown_alignment.py")], env=environment, check=True)
        if not smoke and not E116._complete(spec, stage_dir / digest):
            raise RuntimeError(f"{name} finished without every sample")


# ---------------------------------------------------------------- analysis


def expanded(rows: list, chosen: dict | None = None, theirs: dict | None = None) -> list:
    """One pseudo-variant per scored setting (``LF+field/<setting>``) and, given the
    selections, the named variants."""

    out = []
    for row in rows:
        variants = dict(row["variants"])
        outcome = variants.get(ARM_NAME) or {}
        by_config = (outcome.get("extras") or {}).get("by_config") or {}
        for config, scores in by_config.items():
            merged = dict(outcome, **scores)
            merged["succeeded"] = bool(outcome.get("extras")) and all(
                merged.get(k) is not None and np.isfinite(merged[k]) for k in ("heldout_cmr_mm", "heldout_tee_mm"))
            variants[f"{ARM_NAME}/{config}"] = merged
        if chosen:
            for variant in OURS.values():
                variants[variant] = variants.get(f"{ARM_NAME}/{E116._name(chosen[variant])}", {"succeeded": False})
        if theirs:
            for variant, source in AT_ROOTED.items():
                variants[variant] = variants.get(f"{ARM_NAME}/{E116._name(theirs[source])}", {"succeeded": False})
        out.append(dict(row, variants=variants))
    return out


def unimodal_reference(key: str, frozen: dict) -> tuple[dict, dict]:
    """The chain's validation rows of the two unimodal fits at late fusion's frozen weights.

    Late fusion's held-out CMR and TEE errors are those fits' (identical on every
    test sample of every set, checked when E119 was written).
    """

    weights = {m: _late(frozen)["options"]["modality_overrides"][m]["identity_prior_weight"] for m in ("cmr", "tee")}
    found = {}
    for spec_path in sorted(_root(key).glob("round1*/*/spec.json")):
        spec = json.loads(spec_path.read_text())
        if spec["split"] != "val":
            continue
        for variant in spec["variants"]:
            for modality, weight in weights.items():
                if (variant["name"].startswith(f"S/{modality}_only/")
                        and variant.get("overrides", {}).get("identity_prior_weight") == weight):
                    found.setdefault(modality, []).append((spec_path.parent, variant["name"]))
    if any(len(found.get(m, [])) != 1 for m in weights):
        raise FileNotFoundError(f"{key}: unimodal validation rows not found exactly once: {found}")
    rows = {}
    for modality, [(directory, name)] in found.items():
        rows[modality] = {r["sample"]: (r["variants"].get(name) or {}) for r in E116._rows(directory)}
    return rows, {m: f"{found[m][0][0]} {found[m][0][1]}" for m in found}


def check_val(key: str, rows: list) -> dict:
    """The arm's own late fusion against the chain's unimodal rows, and the zero settings against it."""

    frozen, _ = _frozen(key)
    reference, names = unimodal_reference(key, frozen)
    differences, zero, missing = [], [], 0
    for row in rows:
        outcome = row["variants"].get(ARM_NAME) or {}
        extras = outcome.get("extras") or {}
        cmr, tee = reference["cmr"].get(row["sample"]) or {}, reference["tee"].get(row["sample"]) or {}
        if not (extras and cmr.get("succeeded") and tee.get("succeeded")):
            missing += 1
            continue
        base = extras["late_fusion"]
        differences.append(max(abs(base["heldout_cmr_mm"] - cmr["heldout_cmr_mm"]),
                               abs(base["heldout_tee_mm"] - tee["heldout_tee_mm"])))
        for sharing in ("session", "shared"):
            scores = extras["by_config"][E116._name([sharing, 0.0, 0.0, E116.NO_FIELD_WIDTH])]
            zero.append(max(abs(scores[k] - base[k]) for k in
                            ("heldout_cmr_mm", "heldout_tee_mm", "attachment_residual_mm")))
    worst = float(max(differences)) if differences else float("nan")
    worst_zero = float(max(zero)) if zero else float("nan")
    passed = bool(differences) and missing == 0 and worst <= TOLERANCE_MM and worst_zero <= EXACT
    return {"reference": names, "samples_compared": len(differences), "not_comparable": missing,
            "max_abs_difference_mm": worst, "max_zero_setting_difference": worst_zero, "passed": bool(passed)}


def check_test(rows: list, stored: list) -> dict:
    """The arm's own late fusion against the chain's stored late-fusion test rows."""

    reference = {r["sample"]: (r["variants"].get(LATE) or {}) for r in stored}
    errors, pose, gaps, missing = [], [], [], 0
    for row in rows:
        extras = (row["variants"].get(ARM_NAME) or {}).get("extras") or {}
        theirs = reference.get(row["sample"]) or {}
        if not (extras and theirs.get("succeeded")):
            missing += 1
            continue
        base = extras["late_fusion"]
        errors.append(max(abs(base[k] - theirs[k]) for k in ("heldout_cmr_mm", "heldout_tee_mm")))
        gaps.append(abs(base["attachment_residual_mm"] - theirs["attachment_residual_mm"]))
        pose.append(max(abs(base[k] - theirs[k]) for k in ("rotation_deg", "translation_mm", "phase_map_cycles")))
    worst = float(max(errors)) if errors else float("nan")
    worst_gap = float(max(gaps)) if gaps else float("nan")
    worst_pose = float(max(pose)) if pose else float("nan")
    passed = (bool(errors) and missing == 0 and worst <= TOLERANCE_MM and worst_gap <= TOLERANCE_MM
              and worst_pose <= POSE_TOLERANCE)
    return {"samples_compared": len(errors), "not_comparable": missing, "max_abs_difference_mm": worst,
            "max_gap_difference_mm": worst_gap, "max_pose_timing_difference": worst_pose, "passed": bool(passed)}


def analyze_val(stage) -> dict:
    index = _index("val")
    report = {"experiment": "E119", "preregistration": PREREGISTRATION, "split": "val",
              "rule": "E116's: sum of the median held-out CMR and TEE errors, per variant and set",
              "selection": {}, "at_grid_edge": {}, "scores": {}, "late_fusion_sum": {}, "field_rms_mm": {},
              "check": {"arms": {}}}
    missing = [k for k in SETS if k not in index]
    passed = not missing
    report["check"]["missing_sets"] = missing
    for key in [k for k in SETS if k in index]:
        rows = expanded(E116._rows(_out() / "val" / index[key]))
        scores = {E116._name(c): stage.score(rows, f"{ARM_NAME}/{E116._name(c)}", "sum") for c in E116.GRID}
        selection, edges = {}, {}
        for sharing, variant in OURS.items():
            candidates = {n: v for n, v in scores.items() if n.startswith(sharing + "/") and np.isfinite(v)}
            chosen = E116._parse(min(candidates, key=candidates.get))
            selection[variant], edges[variant] = chosen, E116._edges(chosen)
        report["scores"][key] = scores
        report["selection"][key] = selection
        report["at_grid_edge"][key] = edges
        report["late_fusion_sum"][key] = scores[E116._name(["session", 0.0, 0.0, E116.NO_FIELD_WIDTH])]
        report["field_rms_mm"][key] = {variant: float(np.median([
            r["variants"][f"{ARM_NAME}/{E116._name(selection[variant])}"]["field_rms_mm"] for r in rows
            if r["variants"][f"{ARM_NAME}/{E116._name(selection[variant])}"].get("succeeded")]))
            for variant in OURS.values()}
        check = check_val(key, rows)
        report["check"]["arms"][key] = check
        passed &= check["passed"]
        print(f"[val/{key}] selected {selection} | sums "
              f"{[round(scores[E116._name(c)], 4) for c in selection.values()]} vs late fusion "
              f"{report['late_fusion_sum'][key]:.4f} | at edge {edges} | check {check['passed']}", flush=True)
    report["check"]["passed"] = bool(passed)
    return report


#: (first, second) for every paired comparison on test; first minus second.
COMPARISONS = ((R4D, OURS["session"]), (R4D, OURS["shared"]), (R4D_SESSION, OURS["session"]),
               (R4D, OURS["shared"]), (OURS["session"], LATE), (OURS["shared"], LATE),
               (OURS["shared"], OURS["session"]), ("LF+GP-session@R4D", LATE), (R4D, LATE), (ROOTED, LATE))
COMPARISONS = tuple(dict.fromkeys(COMPARISONS))
ARMS = (LATE, OURS["session"], OURS["shared"], *AT_ROOTED, ROOTED, R4D_SESSION, R4D)


def analyze_test(stage) -> dict:
    index = _index("test")
    selection = _selection()
    paired_endpoint = _load("e115_recent_fusion").paired_endpoint
    report = {"experiment": "E119", "preregistration": PREREGISTRATION, "split": "test", "selection": selection,
              "statistic": "E62, %d identity resamples, seed %d" % (stage.RESAMPLES, stage.SEED),
              "sets": {}, "check": {"arms": {}}}
    missing = [k for k in SETS if k not in index]
    passed = not missing
    report["check"]["missing_sets"] = missing
    for key in [k for k in SETS if k in index]:
        ours_dir = _out() / "test" / index[key]
        _, chain = _frozen(key)
        theirs_dir = rooted_rows(key)
        check = check_test(expanded(E116._rows(ours_dir)), E116._rows(chain))
        rows = stage.load_rows([ours_dir, chain, theirs_dir])
        rows = expanded(E116._expanded(rows, rooted_selection(key)), selection[key], rooted_selection(key))
        rooted_missing = sum(1 for r in rows if not (r["variants"].get(R4D) or {}).get("succeeded"))
        check["rooted4d_rows_missing"] = rooted_missing
        check["shared_gap_minus_late_fusion_max_mm"] = float(max(
            abs(r["variants"][OURS["shared"]]["attachment_residual_mm"]
                - r["variants"][LATE]["attachment_residual_mm"]) for r in rows))
        check["passed"] = bool(check["passed"] and rooted_missing == 0)
        report["check"]["arms"][key] = check
        passed &= check["passed"]
        entry = {"rows": str(ours_dir), "comparator_rows": [str(chain), str(theirs_dir)],
                 "medians": {arm: E116._medians(rows, arm) for arm in ARMS},
                 "field_rms_mm": {v: float(np.median([r["variants"][v]["field_rms_mm"] for r in rows
                                                      if r["variants"][v].get("succeeded")]))
                                  for v in OURS.values()},
                 "minus": {}}
        for first, second in COMPARISONS:
            comparison = stage.paired(rows, first, second)
            if (first, second) == (R4D, OURS["session"]):
                for endpoint in ("phase_map_cycles", "rotation_deg", "translation_mm", "attachment_residual_mm"):
                    comparison[endpoint] = paired_endpoint(rows, first, second, endpoint, stage)
            entry["minus"][f"{first} - {second}"] = comparison
        report["sets"][key] = entry
        m = entry["medians"]
        print(f"[test/{key}] " + " | ".join(
            "%s cmr %.3f tee %.3f gap %.3g fail %d" % (
                arm, m[arm]["heldout_cmr_mm"], m[arm]["heldout_tee_mm"], m[arm]["attachment_residual_mm"],
                m[arm]["failures"]) for arm in ARMS) + f" | check {check['passed']}", flush=True)
    report["check"]["passed"] = bool(passed)
    report["predictions"] = predictions(report["sets"])
    return report


def _below(entry: dict) -> bool:
    return bool(np.isfinite(entry["upper"]) and entry["upper"] < 0)


def _above(entry: dict) -> bool:
    return bool(np.isfinite(entry["lower"]) and entry["lower"] > 0)


def predictions(sets: dict) -> dict:
    """P1 to P5 of the pre-registration."""

    def minus(key, a, b):
        return sets[key]["minus"][f"{a} - {b}"]

    primaries = [k for k in PRIMARIES if k in sets]
    per_set = {
        "P1_field_lowers_late_fusion_sum": {k: _below(minus(k, OURS["session"], LATE)["sum"]) for k in sets},
        "P2_rooted4d_not_lower_than_late_fusion_with_field": {
            k: not _below(minus(k, R4D, OURS["session"])["sum"]) for k in primaries},
        "P3_late_fusion_with_field_lower_on_realistic_primary": {
            "v4n96o": _above(minus("v4n96o", R4D, OURS["session"])["sum"])} if "v4n96o" in sets else {},
        "P4_same_field_same_accuracy": {
            k: not (_below(e) or _above(e)) for k in primaries
            for e in [minus(k, R4D_SESSION, OURS["session"])["sum"]]},
        "P5_field_does_not_attach_late_fusion": {
            k: sets[k]["medians"][OURS["session"]]["attachment_residual_mm"] > 0.5 for k in primaries},
    }
    return {name: {"per_set": values, "holds_on": sum(values.values()), "sets": len(values),
                   "holds": bool(values) and all(values.values())}
            for name, values in per_set.items()}


def cmd_analyze(split: str) -> None:
    stage = _load("e62_stage_b")
    report = analyze_val(stage) if split == "val" else analyze_test(stage)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"E119_{split}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    if split == "test":
        print(json.dumps({k: {"holds_on": v["holds_on"], "sets": v["sets"], "per_set": v["per_set"]}
                          for k, v in report["predictions"].items()}, indent=1))
    print("wrote", out, "| checks passed:", report["check"]["passed"])


PAPER = {LATE: "Late fusion", OURS["session"]: "Late fusion, per-modality field",
         OURS["shared"]: "Late fusion, shared field", ROOTED: "Rooted4D, joint fit",
         R4D_SESSION: "Rooted4D, per-modality field", R4D: "Rooted4D"}


def latex() -> str:
    """Table rows (realistic-session primary, Table 5 layout), then one row per set of paired sums."""

    tables = _load("e64_manuscript_tables")
    report = json.loads((REPORTS / "E119_test.json").read_text())
    lines = ["% Table 5 rows: method & CMR & TEE & gap & phase & rotation"]
    m = report["sets"]["v4n96o"]["medians"]
    for arm in (LATE, OURS["session"], OURS["shared"], ROOTED, R4D_SESSION, R4D):
        lines.append("%s & %s & %s & %s & %s & %s\\\\" % (
            PAPER[arm], tables.number(m[arm]["heldout_cmr_mm"]), tables.number(m[arm]["heldout_tee_mm"]),
            tables.gap(m[arm]["attachment_residual_mm"]), tables.number(m[arm]["phase_map_cycles"], 3),
            tables.number(m[arm]["rotation_deg"])))
    lines.append("% per set: R4D - LF+GP-session, R4D-session - LF+GP-session, LF+GP-session - LF, R4D - LF+GP (sums)")
    for key, (_, session, label) in SETS.items():
        if key not in report["sets"]:
            continue
        minus = report["sets"][key]["minus"]
        cells = [tables.interval(minus[f"{a} - {b}"]["sum"], 2) for a, b in (
            (R4D, OURS["session"]), (R4D_SESSION, OURS["session"]), (OURS["session"], LATE), (R4D, OURS["shared"]))]
        lines.append("%s, %s & %s\\\\" % (session, label, " & ".join(cells)))
    return "\n".join(lines)


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    if command == "latex":
        print(latex())
    else:
        {"run": cmd_run, "analyze": cmd_analyze}[command](*arguments)
