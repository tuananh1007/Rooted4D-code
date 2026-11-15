"""Reproduce Rooted4D's main configuration from its frozen file.

``Codes/configs/rooted4d_main.json`` holds everything the method needs: the
resolved solver settings and model recipe per benchmark, the residual field's
setting, the dataset fingerprints, and per-sample reference rows from the run
the paper reports (E116's test run). This script re-runs that configuration
through the same runner and compares sample by sample. Run it from the
workspace root, with BLAS pinned to one thread.

    python reproduce_rooted4d_main.py check-data
    python reproduce_rooted4d_main.py run <out_dir> [--keys v4n96o,v3n96o] [--levels full,cmr4,...] [--samples N]
    python reproduce_rooted4d_main.py compare <out_dir> <report.json>

``run`` uses ``R4D_WORKERS`` processes (default 1); ``--samples N`` runs only the
first N test samples of each set, in this process, for a quick check.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG = Path("Codes/configs/rooted4d_main.json")
VARIANT = "R4D+GP"
ENDPOINTS = ("heldout_cmr_mm", "heldout_tee_mm", "heldout_cmr_p95_mm", "heldout_tee_p95_mm",
             "attachment_residual_mm", "rotation_deg", "translation_mm", "phase_map_cycles")


def _config() -> dict:
    return json.loads(CONFIG.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cmd_check_data() -> bool:
    """The datasets and the session subspace are the ones the reference rows were computed on."""

    passed = True
    for key, entry in _config()["benchmarks"].items():
        root = Path(entry["benchmark"])
        checks = {"dataset.json": (_sha256(root / "dataset.json"), entry["dataset_json_sha256"])}
        for split, expected in entry["manifest_sha256"].items():
            checks[f"manifests/{split}.jsonl"] = (_sha256(root / "manifests" / f"{split}.jsonl"), expected)
        if entry.get("session_subspace"):
            subspace = entry["session_subspace"]
            checks[subspace["path"]] = (_sha256(Path(subspace["path"])), subspace["sha256"])
        for name, (found, expected) in checks.items():
            ok = found == expected
            passed &= ok
            print(f"[{key}] {name}: {'ok' if ok else 'MISMATCH ' + found}", flush=True)
    return passed


def spec(key: str, level: str) -> dict:
    entry = _config()["benchmarks"][key]
    settings = [[f["sharing"], f["static"], f["dynamic"], f["width"]]
                for f in (entry["field"], entry["reported_variant"])]
    out = {"experiment": "rooted4d_main reproduction", "config": str(CONFIG), "key": key, "level": level,
           "benchmark": entry["benchmark"], "split": "test", "start": _config()["start"],
           "models": {"S": entry["model"]}, "base_config": entry["solver"],
           "variants": [{"name": VARIANT, "arm": "rooted4d_residual_field", "model": "S", "overrides": {},
                         "options": {"configs": settings}}]}
    if level != "full":
        out["observation_thinning"] = entry["thinned_levels"][level]
    return out


def _sets(keys: str | None, levels: str | None) -> list[tuple[str, str]]:
    config = _config()["benchmarks"]
    wanted_keys = keys.split(",") if keys else list(config)
    wanted_levels = levels.split(",") if levels else None
    return [(key, level) for key in wanted_keys for level in config[key]["reference"]
            if wanted_levels is None or level in wanted_levels]


def _runner():
    module_spec = importlib.util.spec_from_file_location("runner", HERE / "e59_unknown_alignment.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def cmd_run(out: Path, keys: str | None, levels: str | None, samples: int | None) -> None:
    from rooted4d.experiments.benchmark_eval import sample_paths

    out.mkdir(parents=True, exist_ok=True)
    for key, level in _sets(keys, levels):
        name = f"{key}_{level}"
        definition = spec(key, level)
        spec_path = out / f"{name}.json"
        spec_path.write_text(json.dumps(definition, indent=2) + "\n")
        print(f"[{name}] running", flush=True)
        if samples is None:
            environment = dict(os.environ, R4D_SPEC=str(spec_path), R4D_OUT=str(out / name), R4D_SHARD="0/1",
                               R4D_ALLOW_TEST="1")
            subprocess.run([sys.executable, "-u", str(HERE / "e59_unknown_alignment.py")], env=environment,
                           check=True)
            continue
        runner = _runner()
        runner._initialise(definition)
        directory = out / name / "partial"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "shard_000_of_001.jsonl").open("w") as handle:
            for path in sample_paths(Path(definition["benchmark"]), "test")[:samples]:
                handle.write(json.dumps(runner.run_sample(str(path))) + "\n")


def _ours(directory: Path) -> dict:
    rows = {}
    for shard in sorted(directory.glob("*/shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["sample"]] = row["variants"][VARIANT]
    return rows


def _flatten(record: dict) -> dict:
    """Every compared number of one sample: the method's endpoints, both settings, and Rooted4D before the field."""

    extras = record.get("extras") or {}
    by_config = extras.get("by_config", record.get("by_config", {}))
    rooted = extras.get("rooted4d", record.get("rooted4d", {}))
    values = {k: record.get(k) for k in ENDPOINTS}
    for setting, scores in by_config.items():
        for k in ("heldout_cmr_mm", "heldout_tee_mm", "attachment_residual_mm", "field_rms_mm"):
            values[f"{setting}:{k}"] = scores.get(k)
    for k in ("heldout_cmr_mm", "heldout_tee_mm"):
        values[f"rooted4d:{k}"] = rooted.get(k)
    return values


def cmd_compare(out: Path, report_path: Path) -> bool:
    config = _config()
    tolerance = config["reproduction"]["tolerance_mm"]
    report = {"config": str(CONFIG), "rows": str(out), "tolerance_mm": tolerance, "sets": {}}
    passed = True
    for key, entry in config["benchmarks"].items():
        for level, reference in entry["reference"].items():
            directory = out / f"{key}_{level}"
            if not directory.exists():
                continue
            ours = _ours(directory)
            expected = {}
            for line in Path(reference["rows"]).read_text().splitlines():
                row = json.loads(line)
                expected[row["sample"]] = row[VARIANT]
            differences, missing = [], [s for s in ours if s not in expected]
            for sample, record in ours.items():
                if sample not in expected:
                    continue
                a, b = _flatten(record), _flatten(expected[sample])
                for k in b:
                    if b[k] is None and a.get(k) is None:
                        continue
                    differences.append(abs(float(a[k]) - float(b[k])) if a.get(k) is not None else np.inf)
            differences = np.asarray(differences, dtype=np.float64)
            medians = {}
            if len(ours) == len(expected):
                for k in ("heldout_cmr_mm", "heldout_tee_mm"):
                    medians[k] = [round(float(np.median([r[k] for r in ours.values()])), 2),
                                  round(float(np.median([r[k] for r in expected.values()])), 2)]
            ok = (bool(len(differences)) and not missing and float(differences.max()) <= tolerance["per_sample_max"]
                  and float(np.median(differences)) <= tolerance["median_abs"]
                  and all(x == y for x, y in medians.values()))
            report["sets"][f"{key}/{level}"] = {
                "samples": len(ours), "reference_samples": len(expected), "not_in_reference": missing,
                "values_compared": int(len(differences)),
                "max_abs_difference": float(differences.max()) if len(differences) else None,
                "median_abs_difference": float(np.median(differences)) if len(differences) else None,
                "medians_at_reporting_precision": medians, "passed": bool(ok)}
            passed &= ok
            print(f"[{key}/{level}] samples {len(ours)} max |d| {report['sets'][f'{key}/{level}']['max_abs_difference']} "
                  f"medians {medians} {'passed' if ok else 'FAILED'}", flush=True)
    report["passed"] = bool(passed and report["sets"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", report_path, "| passed:", report["passed"])
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check-data")
    run = commands.add_parser("run")
    run.add_argument("out", type=Path)
    run.add_argument("--keys")
    run.add_argument("--levels")
    run.add_argument("--samples", type=int)
    compare = commands.add_parser("compare")
    compare.add_argument("out", type=Path)
    compare.add_argument("report", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "check-data":
        sys.exit(0 if cmd_check_data() else 1)
    if arguments.command == "run":
        cmd_run(arguments.out, arguments.keys, arguments.levels, arguments.samples)
    else:
        sys.exit(0 if cmd_compare(arguments.out, arguments.report) else 1)
