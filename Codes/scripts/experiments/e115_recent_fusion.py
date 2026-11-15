"""E115: fusion designs built from recent registration and shape-model methods.

Pre-registered in ``Docs/E115_RECENT_FUSION_COMPARATORS_PREREGISTRATION_2026-10-22.md``.
Late fusion followed by a BCPD attachment (Hirose 2021) or an NDP attachment
(Li and Harada 2022), and model-based fusion with a GPMM (Luthi et al. 2018) on
the global basis (GPMM-G, primary) and on the structured space (GPMM-S). All
see the frozen E100 benchmark, masks and input-derived start.

    R4D_RESULTS=<results root> R4D_E115_OUT=<dir> python e115_recent_fusion.py run <split>
    python e115_recent_fusion.py analyze <split>   -> results/benchmark/E115/E115_<split>.json
    python e115_recent_fusion.py latex test <which>

Validation runs every configuration on both primaries at full evidence and
selects by the sum of the median held-out errors; the two attachment
comparators select only among configurations that attach (median curve gap
at most 0.5 mm). Test runs the selected configurations once, on both primaries
at full evidence and on the realistic-session primary at four thinned levels.
``run test`` refuses unless the validation report exists and its checks passed.
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
PREREGISTRATION = "Docs/E115_RECENT_FUSION_COMPARATORS_PREREGISTRATION_2026-10-22.md"
KEYS = ("v4n96o", "v3n96o")
THINNED = {"cmr4": (4, 1), "cmr3": (6, 1), "tee6": (1, 4), "tee3": (1, 8)}
BETAS = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
LAMBDAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
BCPD_GRID = [list(pair) for pair in itertools.product(BETAS, LAMBDAS)]
LEVELS = 11
NOISES = (0.25, 0.5, 1.0, 2.0, 4.0)
SCALES = (0.5, 1.0, 2.0, 4.0, 8.0)
WIDTHS = (5.0, 10.0, 20.0, 40.0, 80.0)
GPMM_GRID = ([[0.0, 0.0, n] for n in NOISES]
             + [[s, w, n] for s, w, n in itertools.product(SCALES, WIDTHS, NOISES)])
ATTACH_LIMIT_MM = 0.5
ROOTED, LATE = "S/joint", "S/late_fusion_trajectory"
ATTACHERS = ("LF+BCPD", "LF+NDP")
BASELINES = ATTACHERS + ("GPMM-G", "GPMM-S")
TOLERANCE_MM = 5e-3
#: Pose and timing come from late fusion's own registration; re-runs are not
#: bitwise, so they are compared at reporting precision (degrees, mm, cycles).
POSE_TOLERANCE = 1e-3
REPORTS = Path("results/benchmark/E115")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _results() -> Path:
    return Path(os.environ["R4D_RESULTS"])


def _out() -> Path:
    return Path(os.environ["R4D_E115_OUT"])


def _frozen(key: str) -> tuple[dict, Path]:
    directory = _results() / "e100" / key / (_results() / "e100" / key / "TEST_DONE").read_text().strip()
    return json.loads((directory / "spec.json").read_text()), directory


def _late_overrides(frozen: dict) -> dict:
    return next(v for v in frozen["variants"] if v["name"] == LATE)["options"]["modality_overrides"]


def _selection() -> dict:
    report = json.loads((REPORTS / "E115_val.json").read_text())
    if not report["check"]["passed"]:
        raise SystemExit("the validation checks did not pass")
    return report["selection"]


def _variants(key: str, frozen: dict, split: str) -> list[dict]:
    overrides = _late_overrides(frozen)
    if split == "val":
        bcpd, levels, primary_levels = BCPD_GRID, LEVELS, LEVELS
        gpmm = {"GPMM-G": GPMM_GRID, "GPMM-S": GPMM_GRID}
    else:
        chosen = _selection()[key]
        bcpd = [chosen["LF+BCPD"]]
        levels = primary_levels = int(chosen["LF+NDP"])
        gpmm = {name: [chosen[name]] for name in ("GPMM-G", "GPMM-S")}
    return [
        {"name": "LF+BCPD", "arm": "late_fusion_bcpd", "model": "S",
         "options": {"modality_overrides": overrides, "grid": bcpd, "gamma": 0.1}},
        {"name": "LF+NDP", "arm": "late_fusion_ndp", "model": "S",
         "options": {"modality_overrides": overrides, "levels": levels, "primary_levels": primary_levels,
                     "seed": 0}},
        {"name": "GPMM-G", "arm": "gpmm_fusion", "model": "G",
         "options": {"grid": gpmm["GPMM-G"], "iterations": 25}},
        {"name": "GPMM-S", "arm": "gpmm_fusion", "model": "S",
         "options": {"grid": gpmm["GPMM-S"], "iterations": 25}},
    ]


def specs(split: str) -> list[tuple[str, dict]]:
    out = []
    for key in KEYS:
        frozen, directory = _frozen(key)
        levels = [("full", None)]
        if split == "test" and key == "v4n96o":
            levels += [(name, {"cmr_keep_every": c, "tee_keep_every": t}) for name, (c, t) in THINNED.items()]
        for level, thinning in levels:
            spec = {"experiment": "E115", "preregistration": PREREGISTRATION, "key": key, "level": level,
                    "benchmark": frozen["benchmark"], "split": split, "start": frozen["start"],
                    "models": frozen["models"],
                    "base_config": frozen["base_config"], "frozen_from": str(directory),
                    "variants": _variants(key, frozen, split)}
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


def _expanded(rows: list) -> list:
    """Add one pseudo-variant per scored configuration, read from the arms' extras."""

    out = []
    for row in rows:
        variants = dict(row["variants"])
        for name in BASELINES:
            outcome = variants.get(name) or {}
            for config, scores in ((outcome.get("extras") or {}).get("by_config") or {}).items():
                merged = dict(outcome, **scores)
                # A configuration stands on its own: the arm's flag describes its primary one.
                merged["succeeded"] = all(merged.get(k) is not None and np.isfinite(merged[k])
                                          for k in ("heldout_cmr_mm", "heldout_tee_mm"))
                variants[f"{name}/{config}"] = merged
        out.append(dict(row, variants=variants))
    return out


def _median_of(rows: list, name: str, key: str) -> float:
    values = [r["variants"][name][key] for r in rows if (r["variants"].get(name) or {}).get("succeeded")
              and np.isfinite(r["variants"][name].get(key, np.nan))]
    return float(np.median(values)) if values else float("nan")


def _reference_check(rows: list, reference_rows: list, reference_name: str, with_pose: bool) -> dict:
    """The attachment arms' CMR error, TEE error before attachment, and (on test) pose and timing,
    against late fusion's stored rows."""

    reference = {r["sample"]: (r["variants"].get(reference_name) or {}) for r in reference_rows}
    out = {"reference_name": reference_name}
    passed = True
    for arm in ATTACHERS:
        differences, pose, missing = [], [], 0
        for row in rows:
            ours, theirs = row["variants"].get(arm) or {}, reference.get(row["sample"]) or {}
            if not (ours.get("extras") and theirs.get("succeeded")):
                missing += 1
                continue
            differences.append(max(abs(ours["heldout_cmr_mm"] - theirs["heldout_cmr_mm"]),
                                   abs(ours["extras"]["undeformed_heldout_tee_mm"] - theirs["heldout_tee_mm"])))
            if with_pose:
                pose.append(max(abs(ours[k] - theirs[k]) for k in ("rotation_deg", "translation_mm",
                                                                     "phase_map_cycles")))
        worst = float(max(differences)) if differences else float("nan")
        worst_pose = float(max(pose)) if pose else None
        ok = bool(differences) and missing == 0 and worst <= TOLERANCE_MM and (
            not with_pose or (worst_pose is not None and worst_pose <= POSE_TOLERANCE))
        out[arm] = {"samples_compared": len(differences), "not_comparable": missing,
                    "max_abs_difference_mm": worst, "max_pose_timing_difference": worst_pose, "passed": ok}
        passed &= ok
    out["passed"] = bool(passed)
    return out


def paired_endpoint(rows: list, first: str, second: str, key: str, stage) -> dict:
    """First minus second in one endpoint's median, identity-averaged, identities resampled."""

    ids_a, a = stage.per_identity(rows, first, key)
    ids_b, b = stage.per_identity(rows, second, key)
    assert ids_a == ids_b
    rng = np.random.default_rng(stage.SEED)
    draws = np.empty(stage.RESAMPLES)
    for r in range(stage.RESAMPLES):
        t = rng.integers(0, len(a), size=len(a))
        draws[r] = np.nanmedian(a[t]) - np.nanmedian(b[t])
    return {"observed": float(np.nanmedian(a) - np.nanmedian(b)),
            "lower": float(np.percentile(draws, 2.5)), "upper": float(np.percentile(draws, 97.5))}


def _medians(rows: list, name: str) -> dict:
    succeeded = [r["variants"][name] for r in rows if (r["variants"].get(name) or {}).get("succeeded")]
    keys = ("heldout_cmr_mm", "heldout_tee_mm", "attachment_residual_mm", "phase_map_cycles",
            "rotation_deg", "translation_mm")
    out = {k: (float(np.median([o[k] for o in succeeded])) if succeeded else float("nan")) for k in keys}
    out["failures"] = len(rows) - len(succeeded)
    out["samples"] = len(rows)
    return out


def _parse(name: str, config: str):
    values = [float(v) for v in config.split(",")]
    return int(values[0]) if name == "LF+NDP" else values


def _edges(name: str, chosen) -> list:
    if name == "LF+BCPD":
        return [chosen[0] in (BETAS[0], BETAS[-1]), chosen[1] in (LAMBDAS[0], LAMBDAS[-1])]
    if name == "LF+NDP":
        return [chosen in (1, LEVELS)]
    scale, width, noise = chosen
    return [scale in (0.0, SCALES[-1]), scale > 0 and width in (WIDTHS[0], WIDTHS[-1]),
            noise in (NOISES[0], NOISES[-1])]


def analyze_val(stage) -> dict:
    index = _index("val")
    report = {"experiment": "E115", "preregistration": PREREGISTRATION, "split": "val",
              "rule": "sum of the median held-out CMR and TEE errors (E62 score); the attachment comparators "
                      "only among configurations whose median curve gap is at most %g mm" % ATTACH_LIMIT_MM,
              "selection": {}, "attaches": {}, "at_grid_edge": {}, "scores": {}, "gaps": {}, "hinge_shear_mm": {},
              "check": {"arms": {}}}
    passed = True
    bcpd_reference = REPORTS / "E115_bcpd_reference.json"
    reference = json.loads(bcpd_reference.read_text()) if bcpd_reference.exists() else {"passed": False}
    report["check"]["bcpd_binary"] = {k: reference.get(k) for k in ("binary", "compared", "max_abs_difference_mm",
                                                                     "tolerance_mm", "passed")}
    passed &= bool(reference.get("passed"))
    for key in KEYS:
        rows = _expanded(_rows(_out() / "val" / index[f"{key}/full"]))
        configs = {"LF+BCPD": [f"{b:g},{l:g}" for b, l in BCPD_GRID],
                   "LF+NDP": [f"{m}" for m in range(1, LEVELS + 1)],
                   "GPMM-G": [f"{s:g},{w:g},{n:g}" for s, w, n in GPMM_GRID]}
        configs["GPMM-S"] = configs["GPMM-G"]
        scores = {name: {c: stage.score(rows, f"{name}/{c}", "sum") for c in configs[name]} for name in BASELINES}
        gaps = {name: {c: {"curve_gap_mm": _median_of(rows, f"{name}/{c}", "curve_gap_mm"),
                           "attachment_residual_mm": _median_of(rows, f"{name}/{c}", "attachment_residual_mm")}
                       for c in configs[name]} for name in ATTACHERS}
        report["scores"][key], report["gaps"][key] = scores, gaps
        report["hinge_shear_mm"][key] = {name: {c: _median_of(rows, f"{name}/{c}", "hinge_shear_mm")
                                                for c in configs[name]} for name in ATTACHERS}
        selection, attaches = {}, {}
        for name in BASELINES:
            candidates = {c: v for c, v in scores[name].items() if np.isfinite(v)}
            if name in ATTACHERS:
                eligible = {c: v for c, v in candidates.items() if gaps[name][c]["curve_gap_mm"] <= ATTACH_LIMIT_MM}
                attaches[name] = bool(eligible)
                candidates = eligible or candidates
            selection[name] = _parse(name, min(candidates, key=candidates.get))
        report["selection"][key] = selection
        report["attaches"][key] = attaches
        report["at_grid_edge"][key] = {name: _edges(name, selection[name]) for name in BASELINES}
        reference_dir = next(p.parent for p in sorted((_results() / "e100" / key / "round2").glob("*/spec.json")))
        check = _reference_check(rows, _rows(reference_dir), "S/independent", with_pose=False)
        report["check"]["arms"][key] = check
        passed &= check["passed"]
        print(f"[val/{key}] selected {selection} | attaches {attaches} | at edge {report['at_grid_edge'][key]} "
              f"| check {check['passed']}", flush=True)
    report["check"]["passed"] = bool(passed)
    return report


def _rooted_dir(key: str, level: str) -> tuple[Path, str]:
    """Rooted4D's (and late fusion's) stored rows for one benchmark and level."""

    if level == "full":
        frozen, directory = _frozen(key)
        return directory, "E100 test rows"
    e113 = _results() / "e113" / "test"
    return e113 / json.loads((e113 / "index.json").read_text())[level], "E113 test rows"


def analyze_test(stage) -> dict:
    index = _index("test")
    report = {"experiment": "E115", "preregistration": PREREGISTRATION, "split": "test",
              "selection": _selection(), "statistic": "E62, %d identity resamples, seed %d" % (
                  stage.RESAMPLES, stage.SEED), "benchmarks": {}, "check": {"arms": {}}}
    passed = True
    for name, digest in index.items():
        key, level = name.split("/")
        ours = _rows(_out() / "test" / digest)
        reference_dir, source = _rooted_dir(key, level)
        check = _reference_check(ours, _rows(reference_dir), LATE, with_pose=True)
        report["check"]["arms"][name] = check
        passed &= check["passed"]
        rows = stage.load_rows([_out() / "test" / digest, reference_dir])
        extras = {arm: [((r["variants"].get(arm) or {}).get("extras") or {}) for r in ours] for arm in ATTACHERS}
        entry = {"rows": str(_out() / "test" / digest), "rooted4d_rows": f"{reference_dir} ({source})",
                 "attachment": {arm: {
                     "curve_gap_mm": float(np.median([next(iter(e["by_config"].values()))["curve_gap_mm"]
                                                      for e in extras[arm] if e])),
                     "hinge_shear_mm": float(np.median([next(iter(e["by_config"].values()))["hinge_shear_mm"]
                                                        for e in extras[arm] if e])),
                     "late_fusion_curve_gap_mm": float(np.median([e["late_fusion_curve_gap_mm"]
                                                                  for e in extras[arm] if e])),
                     "hinge_template_distance_mm": float(np.median([e["hinge_template_distance_mm"]
                                                                    for e in extras[arm] if e]))}
                     for arm in ATTACHERS},
                 "medians": {arm: _medians(rows, arm) for arm in (ROOTED, LATE) + BASELINES},
                 "rooted4d_minus": {}}
        for baseline in BASELINES:
            comparison = stage.paired(rows, ROOTED, baseline)
            for endpoint in ("phase_map_cycles", "rotation_deg", "translation_mm"):
                comparison[endpoint] = paired_endpoint(rows, ROOTED, baseline, endpoint, stage)
            entry["rooted4d_minus"][baseline] = comparison
        report["benchmarks"][name] = entry
        m = entry["medians"]
        print(f"[test/{name}] " + " | ".join(
            "%s cmr %.3f tee %.3f gap %.2f phase %.3f rot %.2f fail %d" % (
                arm, m[arm]["heldout_cmr_mm"], m[arm]["heldout_tee_mm"], m[arm]["attachment_residual_mm"],
                m[arm]["phase_map_cycles"], m[arm]["rotation_deg"], m[arm]["failures"])
            for arm in (ROOTED, LATE) + BASELINES) + f" | check {check['passed']}", flush=True)
    report["check"]["passed"] = bool(passed)
    report["predictions"] = _predictions(report["benchmarks"])
    return report


def _excludes(comparison: dict, endpoint: str) -> bool:
    entry = comparison[endpoint]
    return bool(np.isfinite(entry["upper"]) and entry["upper"] < 0)


def _predictions(benchmarks: dict) -> dict:
    """P1 to P5 of the pre-registration."""

    full = {key: benchmarks[f"{key}/full"]["rooted4d_minus"] for key in KEYS}
    out = {
        "P1_tee_lower_than_attachers": {k: {b: _excludes(full[k][b], "tee") for b in ATTACHERS} for k in KEYS},
        "P2_phase_lower_than_attachers_and_GPMMG": {
            k: {b: _excludes(full[k][b], "phase_map_cycles") for b in ATTACHERS + ("GPMM-G",)} for k in KEYS},
        "P3_cmr_and_tee_lower_than_GPMMG": {k: {e: _excludes(full[k]["GPMM-G"], e) for e in ("cmr", "tee")}
                                             for k in KEYS},
        "P4_rotation_lower_realistic": {b: _excludes(full["v4n96o"][b], "rotation_deg")
                                        for b in ATTACHERS + ("GPMM-G",)},
        "P5_thinned": {},
    }
    for level, endpoint in (("cmr4", "cmr"), ("cmr3", "cmr"), ("tee6", "tee"), ("tee3", "tee")):
        comparisons = benchmarks[f"v4n96o/{level}"]["rooted4d_minus"]
        out["P5_thinned"][level] = {b: _excludes(comparisons[b], endpoint) for b in ATTACHERS}
    return out


def cmd_analyze(split: str) -> None:
    stage = _load("e62_stage_b")
    report = analyze_val(stage) if split == "val" else analyze_test(stage)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"E115_{split}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    if split == "test":
        print(json.dumps(report["predictions"], indent=1))
    print("wrote", out, "| checks passed:", report["check"]["passed"])


PAPER = {ROOTED: "Rooted4D", LATE: "Late fusion",
         "LF+BCPD": r"Late fusion + BCPD~\cite{hirose2021bcpd}",
         "LF+NDP": r"Late fusion + NDP~\cite{li2022ndp}",
         "GPMM-G": r"GPMM fusion~\cite{luthi2018gpmm}",
         "GPMM-S": r"GPMM fusion, structured space"}
BENCHMARK_NAMES = {"v4n96o": "realistic-session", "v3n96o": "small-session"}
LEVEL_NAMES = {"cmr4": ("CMR", 4), "cmr3": ("CMR", 3), "tee6": ("TEE", 6), "tee3": ("TEE", 3)}


def latex(split: str = "test", which: str = "main") -> str:
    """Rows for the paper: ``main`` (the main fusion table's new rows), ``supp``
    (every arm on both primaries), ``paired`` (Rooted4D minus each comparator in
    held-out error), ``alignment`` (the same for phase map, rotation and
    translation) and ``thinned`` (the thinned-evidence comparison)."""

    tables = _load("e64_manuscript_tables")
    report = json.loads((REPORTS / f"E115_{split}.json").read_text())
    lines = []
    if which == "main":
        m = report["benchmarks"]["v4n96o/full"]["medians"]
        for arm in ("LF+BCPD", "LF+NDP", "GPMM-G"):
            lines.append("%s & %s & %s & %s & %s & %s\\\\" % (
                PAPER[arm], tables.number(m[arm]["heldout_cmr_mm"]), tables.number(m[arm]["heldout_tee_mm"]),
                tables.gap(m[arm]["attachment_residual_mm"]), tables.number(m[arm]["phase_map_cycles"], 3),
                tables.number(m[arm]["rotation_deg"])))
    elif which == "supp":
        for key in KEYS:
            entry = report["benchmarks"][f"{key}/full"]
            lines.append(r"\multicolumn{8}{@{}l}{\textit{%s benchmark}}\\" % BENCHMARK_NAMES[key].capitalize())
            for arm in (LATE,) + BASELINES + (ROOTED,):
                m = entry["medians"][arm]
                lines.append("%s & %s & %s & %s & %s & %s & %s & %d\\\\" % (
                    PAPER[arm], tables.number(m["heldout_cmr_mm"]), tables.number(m["heldout_tee_mm"]),
                    tables.gap(m["attachment_residual_mm"]), tables.number(m["phase_map_cycles"], 3),
                    tables.number(m["rotation_deg"]), tables.number(m["translation_mm"]),
                    round(100.0 * m["failures"] / max(m["samples"], 1))))
    elif which == "paired":
        for key in KEYS:
            entry = report["benchmarks"][f"{key}/full"]["rooted4d_minus"]
            lines.append(r"\multicolumn{4}{@{}l}{\textit{%s benchmark}}\\" % BENCHMARK_NAMES[key].capitalize())
            for arm in BASELINES:
                c = entry[arm]
                lines.append("%s & %s & %s & %s\\\\" % (
                    PAPER[arm], tables.interval(c["sum"]), tables.interval(c["cmr"]), tables.interval(c["tee"])))
    elif which == "alignment":
        for key in KEYS:
            entry = report["benchmarks"][f"{key}/full"]["rooted4d_minus"]
            lines.append(r"\multicolumn{4}{@{}l}{\textit{%s benchmark}}\\" % BENCHMARK_NAMES[key].capitalize())
            for arm in BASELINES:
                c = entry[arm]
                lines.append("%s & %s & %s & %s\\\\" % (
                    PAPER[arm], tables.interval(c["phase_map_cycles"], 3), tables.interval(c["rotation_deg"], 2),
                    tables.interval(c["translation_mm"], 2)))
    elif which == "thinned":
        for level, (modality, frames) in LEVEL_NAMES.items():
            entry = report["benchmarks"][f"v4n96o/{level}"]
            key = "heldout_cmr_mm" if modality == "CMR" else "heldout_tee_mm"
            axis = modality.lower()
            m = entry["medians"]
            lines.append("%s, %d & %s & %s & %s & %s & %s & %s & %s & %s\\\\" % (
                modality, frames, tables.number(m[ROOTED][key]), tables.number(m[LATE][key]),
                tables.number(m["LF+BCPD"][key]), tables.number(m["LF+NDP"][key]), tables.number(m["GPMM-G"][key]),
                tables.interval(entry["rooted4d_minus"]["LF+BCPD"][axis], 2),
                tables.interval(entry["rooted4d_minus"]["LF+NDP"][axis], 2),
                tables.interval(entry["rooted4d_minus"]["GPMM-G"][axis], 2)))
    elif which == "attachment":
        for key in KEYS:
            entry = report["benchmarks"][f"{key}/full"]
            lines.append(r"\multicolumn{5}{@{}l}{\textit{%s benchmark}}\\" % BENCHMARK_NAMES[key].capitalize())
            for arm in ATTACHERS:
                a, m = entry["attachment"][arm], entry["medians"][arm]
                lines.append("%s & %s & %s & %s & %s\\\\" % (
                    PAPER[arm], tables.gap(m["attachment_residual_mm"]), tables.number(a["curve_gap_mm"]),
                    tables.number(a["late_fusion_curve_gap_mm"]), tables.number(a["hinge_shear_mm"])))
    return "\n".join(lines)


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    if command == "latex":
        print(latex(*arguments))
    else:
        {"run": cmd_run, "analyze": cmd_analyze}[command](*arguments)
