"""E116: Rooted4D with a Gaussian-kernel residual field.

Pre-registered in ``Docs/E116_RESIDUAL_FIELD_PREREGISTRATION_2026-10-23.md``.
Rooted4D's fit at the configuration E100 froze, then a residual field on the
physical vertices with GPMM fusion's Gaussian kernel in space and Rooted4D's
periodic basis in time (``rooted4d/inference/residual_field.py``): one field
shared by both modalities (``R4D+GP``, primary) or one per modality
(``R4D+GP-session``). Same benchmarks, masks and input-derived start as E100,
E113 and E115.

    R4D_RESULTS=<results root> R4D_E116_OUT=<dir> python e116_residual_field.py run <split>
    python e116_residual_field.py analyze <split>   -> results/benchmark/E116/E116_<split>.json
    python e116_residual_field.py latex test <which>

Validation runs every setting on both primaries at full evidence and selects,
per variant and primary, by the sum of the median held-out errors. Test runs
the selected settings once, on both primaries at full evidence and on the
realistic-session primary at four thinned levels; the comparators' rows are
E100's, E113's and E115's stored test rows. ``run test`` refuses unless the
validation report exists and its checks passed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E116_RESIDUAL_FIELD_PREREGISTRATION_2026-10-23.md"
KEYS = ("v4n96o", "v3n96o")
THINNED = {"cmr4": (4, 1), "cmr3": (6, 1), "tee6": (1, 4), "tee3": (1, 8)}
#: Field scales in units of a 1 mm noise (only the ratio matters; see residual_field.py).
SCALES = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
WIDTHS = (5.0, 10.0, 20.0, 40.0, 80.0)
SHARINGS = ("shared", "session")
#: With both scales zero the field vanishes whatever the width; that setting,
#: Rooted4D itself, is in each variant's grid once.
NO_FIELD_WIDTH = 10.0
GRID = [[sharing, 0.0, 0.0, NO_FIELD_WIDTH] for sharing in SHARINGS] + [
    [sharing, s0, sd, w] for sharing, w, s0, sd in itertools.product(SHARINGS, WIDTHS, SCALES, SCALES)
    if s0 > 0 or sd > 0]
ROOTED, LATE = "S/joint", "S/late_fusion_trajectory"
OURS = {"shared": "R4D+GP", "session": "R4D+GP-session"}
E115_ARMS = ("GPMM-G", "GPMM-S", "LF+BCPD")
COMPARATORS = (ROOTED, "GPMM-G", "GPMM-S", "LF+BCPD", LATE, OURS["session"])
TOLERANCE_MM = 5e-3
POSE_TOLERANCE = 1e-3
GAP_LIMIT_MM = 1e-9
REPORTS = Path("results/benchmark/E116")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _results() -> Path:
    return Path(os.environ["R4D_RESULTS"])


def _out() -> Path:
    return Path(os.environ["R4D_E116_OUT"])


def _frozen(key: str) -> tuple[dict, Path]:
    directory = _results() / "e100" / key / (_results() / "e100" / key / "TEST_DONE").read_text().strip()
    return json.loads((directory / "spec.json").read_text()), directory


def _selection() -> dict:
    report = json.loads((REPORTS / "E116_val.json").read_text())
    if not report["check"]["passed"]:
        raise SystemExit("the validation checks did not pass")
    return report["selection"]


def _variants(key: str, frozen: dict, split: str) -> list[dict]:
    rooted = next(v for v in frozen["variants"] if v["name"] == ROOTED)
    if split == "val":
        configs = GRID
    else:
        chosen = _selection()[key]
        configs = [chosen[OURS["shared"]], chosen[OURS["session"]]]
    return [{"name": OURS["shared"], "arm": "rooted4d_residual_field", "model": rooted["model"],
             "overrides": rooted["overrides"], "options": {"configs": configs}}]


def specs(split: str) -> list[tuple[str, dict]]:
    out = []
    for key in KEYS:
        frozen, directory = _frozen(key)
        levels = [("full", None)]
        if split == "test" and key == "v4n96o":
            levels += [(name, {"cmr_keep_every": c, "tee_keep_every": t}) for name, (c, t) in THINNED.items()]
        for level, thinning in levels:
            spec = {"experiment": "E116", "preregistration": PREREGISTRATION, "key": key, "level": level,
                    "benchmark": frozen["benchmark"], "split": split, "start": frozen["start"],
                    "models": frozen["models"], "base_config": frozen["base_config"],
                    "frozen_from": str(directory), "variants": _variants(key, frozen, split)}
            if thinning:
                spec["observation_thinning"] = thinning
            out.append((f"{key}/{level}", spec))
    return out


def _digest(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def _rows(directory: Path) -> list:
    return [json.loads(line) for shard in sorted(Path(directory).glob("shard_*.jsonl"))
            for line in shard.read_text().splitlines() if line.strip()]


def _complete(spec: dict, directory: Path) -> bool:
    from rooted4d.experiments.benchmark_eval import sample_paths

    expected = len(sample_paths(Path(spec["benchmark"]), spec["split"]))
    return directory.exists() and expected > 0 and len({r["sample"] for r in _rows(directory)}) == expected


def _index(split: str) -> dict:
    path = _out() / split / "index.json"
    return json.loads(path.read_text()) if path.exists() else {}


def cmd_run(split: str) -> None:
    if split == "test":
        _selection()
    stage_dir = _out() / split
    stage_dir.mkdir(parents=True, exist_ok=True)
    index = _index(split)
    for name, spec in specs(split):
        digest = _digest(spec)
        spec_path = stage_dir / f"{name.replace('/', '_')}.json"
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
        index[name] = digest
        (stage_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
        if _complete(spec, stage_dir / digest):
            print(f"[{split}/{name}] complete, not re-run", flush=True)
            continue
        environment = dict(os.environ, R4D_SPEC=str(spec_path), R4D_OUT=str(stage_dir), R4D_SHARD="0/1")
        environment.pop("R4D_ALLOW_TEST", None)
        if split == "test":
            environment["R4D_ALLOW_TEST"] = "1"
        print(f"[{split}/{name}] running {spec_path}", flush=True)
        subprocess.run([sys.executable, "-u", str(HERE / "e59_unknown_alignment.py")], env=environment, check=True)
        if not _complete(spec, stage_dir / digest):
            raise RuntimeError(f"{name} finished without every sample")


# ---------------------------------------------------------------- analysis


def _name(config) -> str:
    sharing, static, dynamic, width = config
    return f"{sharing}/{float(static):g},{float(dynamic):g},{float(width):g}"


def _parse(name: str) -> list:
    sharing, values = name.split("/")
    return [sharing] + [float(v) for v in values.split(",")]


def _expanded(rows: list, chosen: dict | None = None) -> list:
    """One pseudo-variant per scored setting (``R4D+GP/<setting>``) and, given the
    selection, the two variants under their own names."""

    out = []
    for row in rows:
        variants = dict(row["variants"])
        outcome = variants.get(OURS["shared"]) or {}
        by_config = (outcome.get("extras") or {}).get("by_config") or {}
        for config, scores in by_config.items():
            merged = dict(outcome, **scores)
            # A setting stands on its own: the arm's flag describes its first one.
            merged["succeeded"] = bool(outcome.get("extras")) and all(
                merged.get(k) is not None and np.isfinite(merged[k]) for k in ("heldout_cmr_mm", "heldout_tee_mm"))
            variants[f"R4D+GP/{config}"] = merged
        if chosen:
            for sharing, variant in OURS.items():
                variants[variant] = variants.get(f"R4D+GP/{_name(chosen[variant])}", {"succeeded": False})
        out.append(dict(row, variants=variants))
    return out


def _rooted_check(rows: list, reference_rows: list, reference_name: str, with_pose: bool) -> dict:
    """The arm's own Rooted4D fit against Rooted4D's stored rows, and the gap of every setting."""

    reference = {r["sample"]: (r["variants"].get(reference_name) or {}) for r in reference_rows}
    differences, pose, missing, gaps = [], [], 0, []
    for row in rows:
        ours, theirs = row["variants"].get(OURS["shared"]) or {}, reference.get(row["sample"]) or {}
        extras = ours.get("extras") or {}
        if not (extras and theirs.get("succeeded")):
            missing += 1
            continue
        base = extras["rooted4d"]
        differences.append(max(abs(base["heldout_cmr_mm"] - theirs["heldout_cmr_mm"]),
                               abs(base["heldout_tee_mm"] - theirs["heldout_tee_mm"])))
        if with_pose:
            pose.append(max(abs(ours[k] - theirs[k]) for k in ("rotation_deg", "translation_mm", "phase_map_cycles")))
        gaps.append(max(s.get("attachment_residual_mm", np.inf) for s in extras["by_config"].values()))
    worst = float(max(differences)) if differences else float("nan")
    worst_pose = float(max(pose)) if pose else None
    worst_gap = float(max(gaps)) if gaps else float("nan")
    passed = (bool(differences) and missing == 0 and worst <= TOLERANCE_MM and worst_gap <= GAP_LIMIT_MM
              and (not with_pose or (worst_pose is not None and worst_pose <= POSE_TOLERANCE)))
    return {"reference_name": reference_name, "samples_compared": len(differences), "not_comparable": missing,
            "max_abs_difference_mm": worst, "max_pose_timing_difference": worst_pose,
            "max_gap_any_setting_mm": worst_gap, "passed": bool(passed)}


def _validation_reference(key: str) -> Path:
    """E100's validation rows of Rooted4D at its frozen configuration (session round, weight 0.5)."""

    for spec_path in sorted((_results() / "e100" / key / "session").glob("*/spec.json")):
        spec = json.loads(spec_path.read_text())
        if any(v["name"] == "S/joint/bs=0.5" for v in spec["variants"]) and spec["split"] == "val":
            return spec_path.parent
    raise FileNotFoundError(f"no E100 validation rows of Rooted4D for {key}")


def _edges(config: list) -> list:
    _, static, dynamic, width = config
    field = static > 0 or dynamic > 0
    return [static == SCALES[-1], dynamic == SCALES[-1], field and width in (WIDTHS[0], WIDTHS[-1])]


def analyze_val(stage) -> dict:
    index = _index("val")
    report = {"experiment": "E116", "preregistration": PREREGISTRATION, "split": "val",
              "rule": "sum of the median held-out CMR and TEE errors (E62 score), per variant and primary",
              "selection": {}, "at_grid_edge": {}, "scores": {}, "field_rms_mm": {}, "check": {"arms": {}}}
    passed = True
    for key in KEYS:
        rows = _expanded(_rows(_out() / "val" / index[f"{key}/full"]))
        names = [_name(c) for c in GRID]
        scores = {n: stage.score(rows, f"R4D+GP/{n}", "sum") for n in names}
        report["scores"][key] = scores
        selection, edges = {}, {}
        for sharing, variant in OURS.items():
            candidates = {n: v for n, v in scores.items() if n.startswith(sharing + "/") and np.isfinite(v)}
            chosen = _parse(min(candidates, key=candidates.get))
            selection[variant], edges[variant] = chosen, _edges(chosen)
        report["selection"][key] = selection
        report["at_grid_edge"][key] = edges
        report["field_rms_mm"][key] = {variant: float(np.median([
            r["variants"][f"R4D+GP/{_name(selection[variant])}"]["field_rms_mm"] for r in rows
            if r["variants"][f"R4D+GP/{_name(selection[variant])}"].get("succeeded")]))
            for variant in OURS.values()}
        report.setdefault("rooted4d_sum", {})[key] = scores[_name(GRID[0])]
        check = _rooted_check(rows, _rows(_validation_reference(key)), "S/joint/bs=0.5", with_pose=False)
        report["check"]["arms"][key] = check
        passed &= check["passed"]
        print(f"[val/{key}] selected {selection} | sums {[round(scores[_name(c)], 4) for c in selection.values()]} "
              f"vs no field {scores[_name(GRID[0])]:.4f} | at edge {edges} | check {check['passed']}", flush=True)
    report["check"]["passed"] = bool(passed)
    return report


def _stored(key: str, level: str) -> tuple[list, str]:
    """Rooted4D's and late fusion's stored rows, and E115's comparators', for one benchmark and level."""

    if level == "full":
        _, rooted_dir = _frozen(key)
        source = "E100 test rows"
    else:
        e113 = _results() / "e113" / "test"
        rooted_dir = e113 / json.loads((e113 / "index.json").read_text())[level]
        source = "E113 test rows"
    e115 = _results() / "e115" / "test"
    e115_dir = e115 / json.loads((e115 / "index.json").read_text())[f"{key}/{level}"]
    return [rooted_dir, e115_dir], f"{rooted_dir} ({source}); {e115_dir} (E115 test rows)"


def analyze_test(stage) -> dict:
    index = _index("test")
    selection = _selection()
    report = {"experiment": "E116", "preregistration": PREREGISTRATION, "split": "test", "selection": selection,
              "statistic": "E62, %d identity resamples, seed %d" % (stage.RESAMPLES, stage.SEED),
              "benchmarks": {}, "check": {"arms": {}}}
    passed = True
    paired_endpoint = _load("e115_recent_fusion").paired_endpoint
    for name, digest in index.items():
        key, level = name.split("/")
        chosen = selection[key if level == "full" else "v4n96o"]
        ours = _rows(_out() / "test" / digest)
        directories, source = _stored(key, level)
        check = _rooted_check(_expanded(ours), _rows(directories[0]), ROOTED, with_pose=True)
        report["check"]["arms"][name] = check
        passed &= check["passed"]
        rows = _expanded(stage.load_rows([_out() / "test" / digest] + directories), chosen)
        arms = (ROOTED, OURS["shared"], OURS["session"], LATE) + E115_ARMS
        entry = {"rows": str(_out() / "test" / digest), "comparator_rows": source,
                 "medians": {arm: _medians(rows, arm) for arm in arms},
                 "field_rms_mm": {v: float(np.median([r["variants"][v]["field_rms_mm"] for r in rows
                                                      if r["variants"][v].get("succeeded")]))
                                  for v in OURS.values()},
                 "ours_minus": {}}
        for comparator in COMPARATORS:
            comparison = stage.paired(rows, OURS["shared"], comparator)
            for endpoint in ("phase_map_cycles", "rotation_deg", "translation_mm"):
                comparison[endpoint] = paired_endpoint(rows, OURS["shared"], comparator, endpoint, stage)
            entry["ours_minus"][comparator] = comparison
        report["benchmarks"][name] = entry
        m = entry["medians"]
        print(f"[test/{name}] " + " | ".join(
            "%s cmr %.3f tee %.3f gap %.2g phase %.3f rot %.2f fail %d" % (
                arm, m[arm]["heldout_cmr_mm"], m[arm]["heldout_tee_mm"], m[arm]["attachment_residual_mm"],
                m[arm]["phase_map_cycles"], m[arm]["rotation_deg"], m[arm]["failures"])
            for arm in arms) + f" | check {check['passed']}", flush=True)
    report["check"]["passed"] = bool(passed)
    report["predictions"] = _predictions(report["benchmarks"])
    return report


def _medians(rows: list, name: str) -> dict:
    succeeded = [r["variants"][name] for r in rows if (r["variants"].get(name) or {}).get("succeeded")]
    keys = ("heldout_cmr_mm", "heldout_tee_mm", "attachment_residual_mm", "phase_map_cycles",
            "rotation_deg", "translation_mm")
    out = {k: (float(np.median([o[k] for o in succeeded])) if succeeded else float("nan")) for k in keys}
    out["failures"] = len(rows) - len(succeeded)
    out["samples"] = len(rows)
    return out


def _excludes(comparison: dict, endpoint: str) -> bool:
    entry = comparison[endpoint]
    return bool(np.isfinite(entry["upper"]) and entry["upper"] < 0)


def _predictions(benchmarks: dict) -> dict:
    """P1 to P5 of the pre-registration."""

    full = {key: benchmarks[f"{key}/full"]["ours_minus"] for key in KEYS}
    return {
        "P1_sum_lower_than_rooted4d": {k: _excludes(full[k][ROOTED], "sum") for k in KEYS},
        "P2_sum_lower_than_GPMMG": {k: _excludes(full[k]["GPMM-G"], "sum") for k in KEYS},
        "P3_cmr_and_tee_lower_than_GPMMG": {k: {e: _excludes(full[k]["GPMM-G"], e) for e in ("cmr", "tee")}
                                             for k in KEYS},
        "P4_thinned_sum_lower_than_GPMMG": {
            level: _excludes(benchmarks[f"v4n96o/{level}"]["ours_minus"]["GPMM-G"], "sum") for level in THINNED},
        "P5_shared_sum_lower_than_session": {k: _excludes(full[k][OURS["session"]], "sum") for k in KEYS},
    }


def cmd_analyze(split: str) -> None:
    stage = _load("e62_stage_b")
    report = analyze_val(stage) if split == "val" else analyze_test(stage)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"E116_{split}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    if split == "test":
        print(json.dumps(report["predictions"], indent=1))
    print("wrote", out, "| checks passed:", report["check"]["passed"])


PAPER = {ROOTED: "Rooted4D", OURS["shared"]: "Rooted4D + residual field",
         OURS["session"]: "Rooted4D + per-session residual field", LATE: "Late fusion",
         "LF+BCPD": r"Late fusion + BCPD~\cite{hirose2021bcpd}",
         "GPMM-G": r"GPMM fusion~\cite{luthi2018gpmm}", "GPMM-S": r"GPMM fusion, structured space"}
BENCHMARK_NAMES = {"v4n96o": "realistic-session", "v3n96o": "small-session"}
LEVEL_NAMES = {"cmr4": ("CMR", 4), "cmr3": ("CMR", 3), "tee6": ("TEE", 6), "tee3": ("TEE", 3)}


def latex(split: str = "test", which: str = "main") -> str:
    """Rows for the paper: ``main`` (every arm on both primaries), ``paired``
    (the shared field minus each comparator in held-out error), ``alignment``
    (the same for phase map, rotation and translation) and ``thinned``."""

    tables = _load("e64_manuscript_tables")
    report = json.loads((REPORTS / f"E116_{split}.json").read_text())
    lines = []
    if which == "main":
        for key in KEYS:
            entry = report["benchmarks"][f"{key}/full"]
            lines.append(r"\multicolumn{8}{@{}l}{\textit{%s benchmark}}\\" % BENCHMARK_NAMES[key].capitalize())
            for arm in (LATE, "LF+BCPD", "GPMM-G", "GPMM-S", ROOTED, OURS["session"], OURS["shared"]):
                m = entry["medians"][arm]
                lines.append("%s & %s & %s & %s & %s & %s & %s & %d\\\\" % (
                    PAPER[arm], tables.number(m["heldout_cmr_mm"]), tables.number(m["heldout_tee_mm"]),
                    tables.gap(m["attachment_residual_mm"]), tables.number(m["phase_map_cycles"], 3),
                    tables.number(m["rotation_deg"]), tables.number(m["translation_mm"]),
                    round(100.0 * m["failures"] / max(m["samples"], 1))))
    elif which in ("paired", "alignment"):
        endpoints = (("sum", 2), ("cmr", 2), ("tee", 2)) if which == "paired" else (
            ("phase_map_cycles", 3), ("rotation_deg", 2), ("translation_mm", 2))
        for key in KEYS:
            entry = report["benchmarks"][f"{key}/full"]["ours_minus"]
            lines.append(r"\multicolumn{4}{@{}l}{\textit{%s benchmark}}\\" % BENCHMARK_NAMES[key].capitalize())
            for arm in COMPARATORS:
                lines.append("%s & %s\\\\" % (PAPER[arm], " & ".join(
                    tables.interval(entry[arm][e], d) for e, d in endpoints)))
    elif which == "thinned":
        for level, (modality, frames) in LEVEL_NAMES.items():
            entry = report["benchmarks"][f"v4n96o/{level}"]
            m = entry["medians"]
            cells = []
            for arm in (ROOTED, OURS["shared"], "GPMM-G"):
                cells += [tables.number(m[arm]["heldout_cmr_mm"]), tables.number(m[arm]["heldout_tee_mm"])]
            lines.append("%s, %d & %s & %s & %s\\\\" % (
                modality, frames, " & ".join(cells), tables.interval(entry["ours_minus"][ROOTED]["sum"], 2),
                tables.interval(entry["ours_minus"]["GPMM-G"]["sum"], 2)))
    return "\n".join(lines)


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    if command == "latex":
        print(latex(*arguments))
    else:
        {"run": cmd_run, "analyze": cmd_analyze}[command](*arguments)
