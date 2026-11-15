"""E117, part A: the residual field on the other benchmark variants and partitions.

Pre-registered in ``Docs/E117_FIELD_REPLICATION_AND_MITEA_PREREGISTRATION_2026-10-24.md``.
E116's arm (``rooted4d_residual_field``) on the 14 test sets E116 did not
cover: the 24- and 48-node benchmarks at both session levels, and the five
further donor partitions of each 96-node benchmark. Each set keeps its own
frozen joint fit (E100's chain, E94's for the 24-node realistic-session set),
start and masks. The field is selected on each set's validation split by
E116's rule; on test the frozen main configuration's settings
(``Codes/configs/rooted4d_main.json``) are scored beside the selected ones,
from the same fit.

    R4D_RESULTS=<results root> R4D_E117_OUT=<dir> python e117_field_replication.py run <split> [key ...]
    python e117_field_replication.py analyze <split>   -> results/benchmark/E117/E117_sets_<split>.json
    python e117_field_replication.py latex

``run test`` refuses unless the validation report exists and its checks passed.
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
PREREGISTRATION = "Docs/E117_FIELD_REPLICATION_AND_MITEA_PREREGISTRATION_2026-10-24.md"
MAIN_CONFIGURATION = Path("Codes/configs/rooted4d_main.json")
#: key -> (results root of its chain, session level, what distinguishes it)
SETS = {"v4oriented": ("e94", "realistic", "24 nodes"), "v4n48o": ("e100", "realistic", "48 nodes"),
        "v3o": ("e100", "small", "24 nodes"), "v3n48o": ("e100", "small", "48 nodes")}
SETS.update({f"v4n96os{k}": ("e100", "realistic", f"partition {k}") for k in range(1, 6)})
SETS.update({f"v3n96os{k}": ("e100", "small", f"partition {k}") for k in range(1, 6)})
REPORTS = Path("results/benchmark/E117")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E116 = _load("e116_residual_field")
ROOTED, LATE, OURS = E116.ROOTED, E116.LATE, E116.OURS
#: The frozen main configuration's settings, scored on test beside each set's selection.
FROZEN = {"R4D+GP-frozen": "field", "R4D+GP-session-frozen": "reported_variant"}


def frozen_settings() -> dict:
    main = json.loads(MAIN_CONFIGURATION.read_text())["benchmarks"]["v4n96o"]
    return {name: [main[entry][k] for k in ("sharing", "static", "dynamic", "width")]
            for name, entry in FROZEN.items()}


def _results() -> Path:
    return Path(os.environ["R4D_RESULTS"])


def _out() -> Path:
    return Path(os.environ["R4D_E117_OUT"])


def _root(key: str) -> Path:
    return _results() / SETS[key][0] / key


def _frozen(key: str) -> tuple[dict, Path]:
    directory = _root(key) / (_root(key) / "TEST_DONE").read_text().strip()
    return json.loads((directory / "spec.json").read_text()), directory


def _joint(frozen: dict) -> dict:
    return next(v for v in frozen["variants"] if v["name"] == ROOTED)


def session_weight(frozen: dict) -> float:
    """The frozen joint fit's session prior weight: its override, else the resolved base config's."""

    from rooted4d.inference import SolverConfig

    return float(_joint(frozen)["overrides"].get(
        "session_prior_weight", SolverConfig(**frozen["base_config"]).session_prior_weight))


def validation_reference(key: str, frozen: dict) -> tuple[Path, str]:
    """The chain's validation rows of the joint fit at its frozen configuration (a session round)."""

    name = f"{ROOTED}/bs={session_weight(frozen):g}"
    identity = _joint(frozen)["overrides"].get("identity_prior_weight")
    for spec_path in sorted(_root(key).glob("session*/*/spec.json")):
        spec = json.loads(spec_path.read_text())
        variant = next((v for v in spec["variants"] if v["name"] == name), None)
        if variant is None or spec["split"] != "val":
            continue
        if variant["overrides"].get("identity_prior_weight") != identity:
            raise ValueError(f"{key}: {name} in {spec_path} has another identity weight than the frozen fit")
        return spec_path.parent, name
    raise FileNotFoundError(f"no validation rows of {name} for {key}")


def _selection() -> dict:
    report = json.loads((REPORTS / "E117_sets_val.json").read_text())
    if not report["check"]["passed"]:
        raise SystemExit("the validation checks did not pass")
    return report["selection"]


def configs(key: str, split: str) -> list:
    """Every setting on validation; on test the selected ones first, then the frozen ones, once each."""

    if split == "val":
        return E116.GRID
    chosen = _selection()[key]
    out = []
    for config in [chosen[OURS["shared"]], chosen[OURS["session"]], *frozen_settings().values()]:
        if E116._name(config) not in {E116._name(c) for c in out}:
            out.append(config)
    return out


def specs(split: str, keys) -> list[tuple[str, dict]]:
    out = []
    for key in keys:
        frozen, directory = _frozen(key)
        rooted = _joint(frozen)
        spec = {"experiment": "E117", "preregistration": PREREGISTRATION, "key": key, "level": "full",
                "benchmark": frozen["benchmark"], "split": split, "start": frozen["start"],
                "models": frozen["models"], "base_config": frozen["base_config"],
                "frozen_from": str(directory),
                "variants": [{"name": OURS["shared"], "arm": "rooted4d_residual_field", "model": rooted["model"],
                              "overrides": rooted["overrides"], "options": {"configs": configs(key, split)}}]}
        out.append((key, spec))
    return out


def _index(split: str) -> dict:
    path = _out() / split / "index.json"
    return json.loads(path.read_text()) if path.exists() else {}


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
    for name, spec in specs(split, keys):
        digest = E116._digest(spec)
        spec_path = stage_dir / f"{name}.json"
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
        index = _index(split)
        index[name] = digest
        (stage_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
        if E116._complete(spec, stage_dir / digest):
            print(f"[{split}/{name}] complete, not re-run", flush=True)
            continue
        environment = dict(os.environ, R4D_SPEC=str(spec_path), R4D_OUT=str(stage_dir), R4D_SHARD="0/1")
        environment.pop("R4D_ALLOW_TEST", None)
        if split == "test":
            environment["R4D_ALLOW_TEST"] = "1"
        print(f"[{split}/{name}] running {spec_path}", flush=True)
        subprocess.run([sys.executable, "-u", str(HERE / "e59_unknown_alignment.py")], env=environment, check=True)
        if not E116._complete(spec, stage_dir / digest):
            raise RuntimeError(f"{name} finished without every sample")


# ---------------------------------------------------------------- analysis


def analyze_val(stage) -> dict:
    index = _index("val")
    report = {"experiment": "E117", "preregistration": PREREGISTRATION, "split": "val",
              "rule": "E116's: sum of the median held-out CMR and TEE errors, per variant and set",
              "selection": {}, "at_grid_edge": {}, "scores": {}, "rooted4d_sum": {}, "field_rms_mm": {},
              "check": {"arms": {}}}
    missing = [k for k in SETS if k not in index]
    passed = not missing
    report["check"]["missing_sets"] = missing
    for key in [k for k in SETS if k in index]:
        rows = E116._expanded(E116._rows(_out() / "val" / index[key]))
        scores = {E116._name(c): stage.score(rows, f"R4D+GP/{E116._name(c)}", "sum") for c in E116.GRID}
        selection, edges = {}, {}
        for sharing, variant in OURS.items():
            candidates = {n: v for n, v in scores.items() if n.startswith(sharing + "/") and np.isfinite(v)}
            chosen = E116._parse(min(candidates, key=candidates.get))
            selection[variant], edges[variant] = chosen, E116._edges(chosen)
        report["scores"][key] = scores
        report["selection"][key] = selection
        report["at_grid_edge"][key] = edges
        report["rooted4d_sum"][key] = scores[E116._name(E116.GRID[0])]
        report["field_rms_mm"][key] = {variant: float(np.median([
            r["variants"][f"R4D+GP/{E116._name(selection[variant])}"]["field_rms_mm"] for r in rows
            if r["variants"][f"R4D+GP/{E116._name(selection[variant])}"].get("succeeded")]))
            for variant in OURS.values()}
        frozen, _ = _frozen(key)
        directory, name = validation_reference(key, frozen)
        check = E116._rooted_check(rows, E116._rows(directory), name, with_pose=False)
        check["reference_rows"] = str(directory)
        report["check"]["arms"][key] = check
        passed &= check["passed"]
        print(f"[val/{key}] selected {selection} | sums "
              f"{[round(scores[E116._name(c)], 4) for c in selection.values()]} vs no field "
              f"{report['rooted4d_sum'][key]:.4f} | at edge {edges} | check {check['passed']}", flush=True)
    report["check"]["passed"] = bool(passed)
    return report


#: (first, second) for every paired comparison on test; first minus second.
COMPARISONS = ((OURS["shared"], ROOTED), (OURS["shared"], LATE), (OURS["shared"], OURS["session"]),
               ("R4D+GP-frozen", ROOTED), ("R4D+GP-frozen", LATE), (OURS["session"], ROOTED),
               ("R4D+GP-session-frozen", ROOTED))


def _with_frozen(rows: list) -> list:
    """The frozen settings under their own names (each scored once by the arm)."""

    settings = frozen_settings()
    for row in rows:
        for name, config in settings.items():
            row["variants"][name] = row["variants"].get(f"R4D+GP/{E116._name(config)}", {"succeeded": False})
    return rows


def analyze_test(stage) -> dict:
    index = _index("test")
    selection = _selection()
    report = {"experiment": "E117", "preregistration": PREREGISTRATION, "split": "test", "selection": selection,
              "frozen": frozen_settings(),
              "statistic": "E62, %d identity resamples, seed %d" % (stage.RESAMPLES, stage.SEED),
              "sets": {}, "check": {"arms": {}}}
    missing = [k for k in SETS if k not in index]
    passed = not missing
    report["check"]["missing_sets"] = missing
    arms = (ROOTED, OURS["shared"], OURS["session"], *FROZEN, LATE)
    for key in [k for k in SETS if k in index]:
        ours = E116._rows(_out() / "test" / index[key])
        _, directory = _frozen(key)
        check = E116._rooted_check(E116._expanded(ours), E116._rows(directory), ROOTED, with_pose=True)
        report["check"]["arms"][key] = check
        passed &= check["passed"]
        rows = _with_frozen(E116._expanded(stage.load_rows([_out() / "test" / index[key], directory]),
                                           selection[key]))
        entry = {"rows": str(_out() / "test" / index[key]), "comparator_rows": str(directory),
                 "medians": {arm: E116._medians(rows, arm) for arm in arms},
                 "field_rms_mm": {v: float(np.median([r["variants"][v]["field_rms_mm"] for r in rows
                                                      if r["variants"][v].get("succeeded")]))
                                  for v in (*OURS.values(), *FROZEN)},
                 "minus": {f"{a} - {b}": stage.paired(rows, a, b) for a, b in COMPARISONS}}
        report["sets"][key] = entry
        m = entry["medians"]
        print(f"[test/{key}] " + " | ".join(
            "%s cmr %.3f tee %.3f gap %.2g fail %d" % (
                arm, m[arm]["heldout_cmr_mm"], m[arm]["heldout_tee_mm"], m[arm]["attachment_residual_mm"],
                m[arm]["failures"]) for arm in arms) + f" | check {check['passed']}", flush=True)
    report["check"]["passed"] = bool(passed)
    report["predictions"] = predictions(report["sets"])
    return report


def _below(entry: dict) -> bool:
    return bool(np.isfinite(entry["upper"]) and entry["upper"] < 0)


def _above(entry: dict) -> bool:
    return bool(np.isfinite(entry["lower"]) and entry["lower"] > 0)


def predictions(sets: dict) -> dict:
    """A1 to A5 of the pre-registration, per set, with the number of sets on which each holds."""

    shared, session = OURS["shared"], OURS["session"]

    def minus(key, a, b):
        return sets[key]["minus"][f"{a} - {b}"]

    realistic = [k for k in sets if SETS[k][1] == "realistic"]
    per_set = {
        "A1_sum_lower_than_joint_fit": {k: _below(minus(k, shared, ROOTED)["sum"]) for k in sets},
        "A2_cmr_and_tee_lower_than_joint_fit": {
            k: _below(minus(k, shared, ROOTED)["cmr"]) and _below(minus(k, shared, ROOTED)["tee"]) for k in sets},
        "A3_sum_lower_than_late_fusion": {k: _below(minus(k, shared, LATE)["sum"]) for k in sets},
        "A4_per_modality_sum_lower_on_realistic_session": {
            k: _above(minus(k, shared, session)["sum"]) for k in realistic},
        "A5_frozen_setting_sum_lower_than_joint_fit": {
            k: _below(minus(k, "R4D+GP-frozen", ROOTED)["sum"]) for k in sets},
    }
    return {name: {"per_set": values, "holds_on": sum(values.values()), "sets": len(values),
                   "holds": bool(values) and all(values.values())}
            for name, values in per_set.items()}


def cmd_analyze(split: str) -> None:
    stage = _load("e62_stage_b")
    report = analyze_val(stage) if split == "val" else analyze_test(stage)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"E117_sets_{split}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    if split == "test":
        print(json.dumps({k: {"holds_on": v["holds_on"], "sets": v["sets"]}
                          for k, v in report["predictions"].items()}, indent=1))
    print("wrote", out, "| checks passed:", report["check"]["passed"])


def latex() -> str:
    """One row per set: the selected shared setting, then the shared field minus the joint fit, minus
    late fusion and minus the per-modality field, and the frozen setting minus the joint fit (sums)."""

    tables = _load("e64_manuscript_tables")
    report = json.loads((REPORTS / "E117_sets_test.json").read_text())
    lines = []
    for level in ("realistic", "small"):
        lines.append(r"\multicolumn{6}{@{}l}{\textit{%s-session benchmark}}\\" % level.capitalize())
        for key, (_, session, label) in SETS.items():
            if session != level or key not in report["sets"]:
                continue
            _, static, dynamic, width = report["selection"][key][OURS["shared"]]
            minus = report["sets"][key]["minus"]
            cells = [tables.interval(minus[f"{a} - {b}"]["sum"], 2) for a, b in (
                (OURS["shared"], ROOTED), (OURS["shared"], LATE), (OURS["shared"], OURS["session"]),
                ("R4D+GP-frozen", ROOTED))]
            lines.append("%s & %g, %g, %g & %s\\\\" % (label, static, dynamic, width, " & ".join(cells)))
    return "\n".join(lines)


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    if command == "latex":
        print(latex())
    else:
        {"run": cmd_run, "analyze": cmd_analyze}[command](*arguments)
