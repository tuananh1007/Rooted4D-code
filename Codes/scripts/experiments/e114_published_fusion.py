"""E114: two published fusion designs on the fusion benchmark.

Pre-registered in ``Docs/E114_PUBLISHED_FUSION_BASELINES_PREREGISTRATION_2026-10-22.md``.
B1 is spatio-temporal registration adapted from Zhang et al. (2011); B2 is
model-based fusion adapted from Grbic et al. (2011), on the global basis (B2-G,
faithful) and on the structured space (B2-S). Both see the frozen E100
benchmark, masks and input-derived start.

    R4D_RESULTS=<results root> R4D_E114_OUT=<dir> python e114_published_fusion.py run <split>
    python e114_published_fusion.py analyze <split>   -> results/benchmark/E114/E114_<split>.json
    python e114_published_fusion.py latex test

Validation runs every configuration and selects by the sum of the median
held-out errors; test runs the selected configurations once, on both primaries
at full evidence and on the realistic-session primary at four thinned levels.
``run test`` refuses unless the validation report exists and its check passed.

Addendum (22 October, after the validation run, before any test fit): B1's
width and two of B2's weights were selected at the lower edge of their grids,
so the validation grids are extended once by E62's edge rule, two further
values beyond each edge at the grid's own ratio (widths 2.5 and 1.25 mm, the
latter about the ring-node spacing at 96 nodes; shape weights 1/16 and 1/64;
warp ridges 0.1 and 0.01), and validation is rerun on the union. Selection is
from the union; the registered-grid selection is reported beside it. A
narrow field can close the gap by moving the ring alone, so B1 also reports
its hinge shear, reported without prediction. The predictions are unchanged.

Second addendum (22 October, before any test fit): B1's exact Gaussian
interpolation was ill-conditioned for widths above the ring-node spacing and
swung far from the ring; the field is now built in two levels
(``published_fusion.RadialField``) and validation is rerun on the union grid.
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
PREREGISTRATION = "Docs/E114_PUBLISHED_FUSION_BASELINES_PREREGISTRATION_2026-10-22.md"
KEYS = ("v4n96o", "v3n96o")
THINNED = {"cmr4": (4, 1), "cmr3": (6, 1), "tee6": (1, 4), "tee3": (1, 8)}
WIDTHS = (5.0, 10.0, 20.0, 40.0)
GRID = [list(pair) for pair in itertools.product((0.25, 1.0, 4.0, 16.0), (1.0, 10.0, 100.0, 1000.0))]
ROOTED, LATE = "S/joint", "S/late_fusion_trajectory"
BASELINES = ("B1", "B2-G", "B2-S")
LABELS = {"B1": "Spatio-temporal registration (Zhang et al.)",
          "B2-G": "Model-based fusion (Grbic et al.)",
          "B2-S": "Model-based fusion, structured space"}
TOLERANCE_MM = 5e-3
REPORTS = Path("results/benchmark/E114")
ADDENDUM = PREREGISTRATION + ", addenda of 22 October 2026 (grid extension; two-level radial field)"
EXTENDED_WIDTHS = (1.25, 2.5) + WIDTHS
EXTENDED_GRID = [list(pair) for pair in itertools.product((0.015625, 0.0625, 0.25, 1.0, 4.0, 16.0),
                                                          (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0))]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _results() -> Path:
    return Path(os.environ["R4D_RESULTS"])


def _out() -> Path:
    return Path(os.environ["R4D_E114_OUT"])


def _frozen(key: str) -> tuple[dict, Path]:
    directory = _results() / "e100" / key / (_results() / "e100" / key / "TEST_DONE").read_text().strip()
    return json.loads((directory / "spec.json").read_text()), directory


def _late_overrides(frozen: dict) -> dict:
    return next(v for v in frozen["variants"] if v["name"] == LATE)["options"]["modality_overrides"]


def _selection() -> dict:
    report = json.loads((REPORTS / "E114_val.json").read_text())
    if not report["check"]["passed"]:
        raise SystemExit("the validation check did not pass")
    return report["selection"]


def _variants(key: str, frozen: dict, split: str) -> list[dict]:
    if split == "val":
        widths, grids = list(EXTENDED_WIDTHS), {"B2-G": EXTENDED_GRID, "B2-S": EXTENDED_GRID}
    else:
        chosen = _selection()[key]
        widths = [chosen["B1"]]
        grids = {name: [chosen[name]] for name in ("B2-G", "B2-S")}
    return [
        {"name": "B1", "arm": "spatiotemporal_registration", "model": "S",
         "options": {"modality_overrides": _late_overrides(frozen), "widths_mm": widths}},
        {"name": "B2-G", "arm": "model_based_fusion", "model": "G",
         "options": {"grid": grids["B2-G"], "centers": 12, "iterations": 25}},
        {"name": "B2-S", "arm": "model_based_fusion", "model": "S",
         "options": {"grid": grids["B2-S"], "centers": 12, "iterations": 25}},
    ]


def specs(split: str) -> list[tuple[str, dict]]:
    out = []
    for key in KEYS:
        frozen, directory = _frozen(key)
        levels = [("full", None)]
        if split == "test" and key == "v4n96o":
            levels += [(name, {"cmr_keep_every": c, "tee_keep_every": t}) for name, (c, t) in THINNED.items()]
        for level, thinning in levels:
            spec = {"experiment": "E114", "preregistration": PREREGISTRATION, "key": key, "level": level,
                    "benchmark": frozen["benchmark"], "split": split, "start": frozen["start"],
                    "models": frozen["models"],
                    "base_config": frozen["base_config"], "frozen_from": str(directory),
                    "variants": _variants(key, frozen, split), "amendment": ADDENDUM}
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
        b1 = variants.get("B1") or {}
        for width, scores in ((b1.get("extras") or {}).get("by_width") or {}).items():
            variants[f"B1/w={width}"] = dict(b1, **scores)
        for name in ("B2-G", "B2-S"):
            outcome = variants.get(name) or {}
            for config, scores in ((outcome.get("extras") or {}).get("by_config") or {}).items():
                variants[f"{name}/{config}"] = dict(outcome, **scores)
        out.append(dict(row, variants=variants))
    return out


def _reference_check(rows: list, reference_rows: list, reference_name: str) -> dict:
    """B1's CMR error, and its TEE error before the deformation, against late fusion's."""

    reference = {r["sample"]: (r["variants"].get(reference_name) or {}) for r in reference_rows}
    differences, missing = [], 0
    for row in rows:
        ours, theirs = row["variants"].get("B1") or {}, reference.get(row["sample"]) or {}
        if not (ours.get("succeeded") and theirs.get("succeeded")):
            missing += 1
            continue
        differences.append(max(abs(ours["heldout_cmr_mm"] - theirs["heldout_cmr_mm"]),
                               abs(ours["extras"]["undeformed_heldout_tee_mm"] - theirs["heldout_tee_mm"])))
    worst = float(max(differences)) if differences else float("nan")
    return {"reference_name": reference_name, "samples_compared": len(differences), "not_comparable": missing,
            "max_abs_difference_mm": worst,
            "passed": bool(differences) and missing == 0 and worst <= TOLERANCE_MM}


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


def analyze_val(stage) -> dict:
    index = _index("val")
    report = {"experiment": "E114", "preregistration": PREREGISTRATION, "amendment": ADDENDUM, "split": "val",
              "rule": "sum of the median held-out CMR and TEE errors (E62 score)",
              "selection": {}, "selection_registered_grid": {}, "at_extended_edge": {}, "scores": {},
              "b1_hinge_shear_mm": {}, "check": {"arms": {}}}
    passed = True
    registered = {"B1": {f"{w:g}" for w in WIDTHS}, "B2-G": {f"{s:g},{d:g}" for s, d in GRID}}
    registered["B2-S"] = registered["B2-G"]
    for key in KEYS:
        rows = _expanded(_rows(_out() / "val" / index[f"{key}/full"]))
        scores = {"B1": {f"{w:g}": stage.score(rows, f"B1/w={w:g}", "sum") for w in EXTENDED_WIDTHS}}
        for name in ("B2-G", "B2-S"):
            scores[name] = {f"{s:g},{d:g}": stage.score(rows, f"{name}/{s:g},{d:g}", "sum")
                            for s, d in EXTENDED_GRID}
        best = {name: min(values, key=values.get) for name, values in scores.items()}
        first = {name: min(registered[name], key=values.get) for name, values in scores.items()}
        report["scores"][key] = scores

        def parsed(chosen):
            return {"B1": float(chosen["B1"]), "B2-G": [float(v) for v in chosen["B2-G"].split(",")],
                    "B2-S": [float(v) for v in chosen["B2-S"].split(",")]}

        report["selection"][key] = parsed(best)
        report["selection_registered_grid"][key] = parsed(first)
        lowest = {"B1": min(EXTENDED_WIDTHS), "s": EXTENDED_GRID[0][0], "d": EXTENDED_GRID[0][1]}
        highest = {"B1": max(EXTENDED_WIDTHS), "s": EXTENDED_GRID[-1][0], "d": EXTENDED_GRID[-1][1]}
        chosen = report["selection"][key]
        report["at_extended_edge"][key] = {
            "B1": chosen["B1"] in (lowest["B1"], highest["B1"]),
            **{name: [chosen[name][0] in (lowest["s"], highest["s"]), chosen[name][1] in (lowest["d"], highest["d"])]
               for name in ("B2-G", "B2-S")}}
        report["b1_hinge_shear_mm"][key] = {
            f"{w:g}": float(np.median([r["variants"][f"B1/w={w:g}"]["hinge_shear_mm"] for r in rows
                                       if r["variants"].get(f"B1/w={w:g}", {}).get("succeeded")]))
            for w in EXTENDED_WIDTHS}
        reference_dir = next(p.parent for p in sorted((_results() / "e100" / key / "round2").glob("*/spec.json")))
        check = _reference_check(rows, _rows(reference_dir), "S/independent")
        report["check"]["arms"][key] = check
        passed &= check["passed"]
        print(f"[val/{key}] selected {report['selection'][key]} (registered grid "
              f"{report['selection_registered_grid'][key]}, at extended edge {report['at_extended_edge'][key]}) "
              f"| B1 hinge shear {report['b1_hinge_shear_mm'][key]} | check {check}")
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
    report = {"experiment": "E114", "preregistration": PREREGISTRATION, "amendment": ADDENDUM, "split": "test",
              "selection": _selection(), "statistic": "E62, %d identity resamples, seed %d" % (
                  stage.RESAMPLES, stage.SEED), "benchmarks": {}, "check": {"arms": {}}}
    passed = True
    for name, digest in index.items():
        key, level = name.split("/")
        ours = _rows(_out() / "test" / digest)
        reference_dir, source = _rooted_dir(key, level)
        check = _reference_check(ours, _rows(reference_dir), LATE)
        report["check"]["arms"][name] = check
        passed &= check["passed"]
        rows = stage.load_rows([_out() / "test" / digest, reference_dir])
        hinge = [(r["variants"].get("B1") or {}).get("extras", {}) for r in ours]
        entry = {"rows": str(_out() / "test" / digest), "rooted4d_rows": f"{reference_dir} ({source})",
                 "b1_hinge_shear_mm": float(np.median([next(iter(e["by_width"].values()))["hinge_shear_mm"]
                                                        for e in hinge if e and e.get("by_width")])),
                 "hinge_template_distance_mm": float(np.median([e["hinge_template_distance_mm"]
                                                                for e in hinge if e])),
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
            for arm in (ROOTED, LATE) + BASELINES) + f" | check {check['passed']}")
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
        "P1_tee_lower_than_B1": {k: _excludes(full[k]["B1"], "tee") for k in KEYS},
        "P2_phase_lower_than_B1_and_B2G": {k: {b: _excludes(full[k][b], "phase_map_cycles") for b in ("B1", "B2-G")}
                                            for k in KEYS},
        "P3_cmr_and_tee_lower_than_B2G": {k: {e: _excludes(full[k]["B2-G"], e) for e in ("cmr", "tee")}
                                           for k in KEYS},
        "P4_rotation_lower_than_B1_realistic": _excludes(full["v4n96o"]["B1"], "rotation_deg"),
        "P5_thinned": {},
    }
    for level, endpoint, baselines in (("cmr4", "cmr", ("B1", "B2-G")), ("cmr3", "cmr", ("B1", "B2-G")),
                                       ("tee6", "tee", ("B1",)), ("tee3", "tee", ("B1",))):
        comparisons = benchmarks[f"v4n96o/{level}"]["rooted4d_minus"]
        out["P5_thinned"][level] = {b: _excludes(comparisons[b], endpoint) for b in baselines}
    return out


def cmd_analyze(split: str) -> None:
    stage = _load("e62_stage_b")
    report = analyze_val(stage) if split == "val" else analyze_test(stage)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"E114_{split}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    if split == "test":
        print(json.dumps(report["predictions"], indent=1))
    print("wrote", out, "| check passed:", report["check"]["passed"])


PAPER = {ROOTED: "Rooted4D", LATE: "Late fusion",
         "B1": r"Spatio-temporal registration$^{a}$~\cite{zhang2011registration}",
         "B2-G": r"Model-based fusion$^{a}$~\cite{grbic2011fusion}",
         "B2-S": r"Model-based fusion, structured space$^{a}$"}
BENCHMARK_NAMES = {"v4n96o": "realistic-session", "v3n96o": "small-session"}
LEVEL_NAMES = {"cmr4": ("CMR", 4), "cmr3": ("CMR", 3), "tee6": ("TEE", 6), "tee3": ("TEE", 3)}


def latex(split: str = "test", which: str = "main") -> str:
    """Rows for the paper: ``main`` (two rows of the main fusion table), ``supp``
    (every arm on both primaries), ``paired`` (Rooted4D minus each baseline in
    held-out error), ``alignment`` (the same for phase map, rotation and
    translation) and ``thinned`` (the thinned-evidence comparison)."""

    tables = _load("e64_manuscript_tables")
    report = json.loads((REPORTS / f"E114_{split}.json").read_text())
    lines = []
    if which == "main":
        m = report["benchmarks"]["v4n96o/full"]["medians"]
        for arm in ("B1", "B2-G"):
            lines.append("%s & %s & %s & %s & %s & %s\\\\" % (
                PAPER[arm], tables.number(m[arm]["heldout_cmr_mm"]), tables.number(m[arm]["heldout_tee_mm"]),
                tables.gap(m[arm]["attachment_residual_mm"]), tables.number(m[arm]["phase_map_cycles"], 3),
                tables.number(m[arm]["rotation_deg"])))
    elif which == "supp":
        for key in KEYS:
            entry = report["benchmarks"][f"{key}/full"]
            lines.append(r"\multicolumn{8}{@{}l}{\textit{%s benchmark}}\\" % BENCHMARK_NAMES[key].capitalize())
            for arm in (LATE, "B1", "B2-G", "B2-S", ROOTED):
                m = entry["medians"][arm]
                lines.append("%s & %s & %s & %s & %s & %s & %s & %d\\\\" % (
                    PAPER[arm], tables.number(m["heldout_cmr_mm"]), tables.number(m["heldout_tee_mm"]),
                    tables.gap(m["attachment_residual_mm"]), tables.number(m["phase_map_cycles"], 3),
                    tables.number(m["rotation_deg"]), tables.number(m["translation_mm"]),
                    round(100.0 * m["failures"] / max(m["samples"], 1))))
    elif which == "paired":
        # The layout of the supplement's paired table: Sum, CMR, TEE.
        for key in KEYS:
            entry = report["benchmarks"][f"{key}/full"]["rooted4d_minus"]
            lines.append(r"\multicolumn{4}{@{}l}{\textit{%s benchmark}}\\" % BENCHMARK_NAMES[key].capitalize())
            for arm in ("B1", "B2-G", "B2-S"):
                c = entry[arm]
                lines.append("%s & %s & %s & %s\\\\" % (
                    PAPER[arm], tables.interval(c["sum"]), tables.interval(c["cmr"]), tables.interval(c["tee"])))
    elif which == "alignment":
        for key in KEYS:
            entry = report["benchmarks"][f"{key}/full"]["rooted4d_minus"]
            lines.append(r"\multicolumn{4}{@{}l}{\textit{%s benchmark}}\\" % BENCHMARK_NAMES[key].capitalize())
            for arm in ("B1", "B2-G", "B2-S"):
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
            lines.append("%s, %d & %s & %s & %s & %s & %s & %s & %s\\\\" % (
                modality, frames, tables.number(m[ROOTED][key]), tables.number(m[LATE][key]),
                tables.number(m["B1"][key]), tables.number(m["B2-G"][key]), tables.number(m["B2-S"][key]),
                tables.interval(entry["rooted4d_minus"]["B1"][axis], 2),
                tables.interval(entry["rooted4d_minus"]["B2-G"][axis], 2)))
    return "\n".join(lines)


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    if command == "latex":
        print(latex(*arguments))
    else:
        {"run": cmd_run, "analyze": cmd_analyze}[command](*arguments)
