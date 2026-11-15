"""E113: thinning the evidence of one modality, the held-out evidence unchanged.

Pre-registered in ``Docs/E110_E113_MARGIN_AND_COHORT_PREREGISTRATION_2026-10-21.md``.
No benchmark is rebuilt: each stored sample of the realistic-session primary
is thinned at problem construction (``thin_observations``), keeping every k-th
visible CMR frame (16 -> 8, 4, 2; TEE complete) or every k-th TEE frame
(24 -> 12, 6, 3; CMR complete). Level 0 is the full evidence, rerun as a
check. Rooted4D and late fusion with trajectory registration run at the
configuration E100 froze; nothing is reselected per level.

    R4D_E100_ROWS=<E100 test rows dir> R4D_E113_OUT=<results dir> python e113_sparsity.py run <split>
    python e113_sparsity.py check <split>      (level 0 against the stored rows)
    python e113_sparsity.py analyze <split>    -> results/benchmark/E113/E113_<split>.json
    python e113_sparsity.py latex <split>

``run test`` refuses unless the validation report exists and its check passed.

Addendum (22 October, after the validation run, before any test fit): the
2-frame CMR level cannot start (circular DTW needs three phases), so one
level is added, ``cmr3`` (every 6th visible CMR frame, 3 frames). It runs on
both splits, is reported under ``amendment`` and enters no prediction; the
registered levels and predictions are unchanged.
The validation reference (``R4D_E113_VAL_REFERENCE``) is E100's session-weight
stage rows, where the selected joint fit is ``S/joint/bs=0.5``.
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
PREREGISTRATION = "Docs/E110_E113_MARGIN_AND_COHORT_PREREGISTRATION_2026-10-21.md"
JOINT, LATE = "S/joint", "S/late_fusion_trajectory"
#: (level, CMR keep-every, TEE keep-every); the first is the full evidence.
LEVELS = (("full", 1, 1), ("cmr8", 2, 1), ("cmr4", 4, 1), ("cmr2", 8, 1),
          ("tee12", 1, 2), ("tee6", 1, 4), ("tee3", 1, 8))
AXES = {"cmr": ("full", "cmr8", "cmr4", "cmr2"), "tee": ("full", "tee12", "tee6", "tee3")}
#: Added after the validation run (pre-registration addendum, 22 October); in no prediction.
AMENDMENT_LEVELS = (("cmr3", 6, 1),)
ADDENDUM = PREREGISTRATION + ", addendum of 22 October 2026"
HELDOUT_CMR_FRAMES = 4
TOLERANCE_MM = 5e-3
ENDPOINTS = ("heldout_cmr_mm", "heldout_tee_mm", "rotation_deg", "translation_mm", "phase_map_cycles",
             "attachment_residual_mm")
REPORTS = Path("results/benchmark/E113")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _out() -> Path:
    return Path(os.environ["R4D_E113_OUT"])


def _reference_spec() -> tuple[dict, Path]:
    directory = Path(os.environ["R4D_E100_ROWS"])
    return json.loads((directory / "spec.json").read_text()), directory


def _frames(benchmark: Path, split: str) -> dict:
    from rooted4d.experiments.benchmark_eval import sample_paths

    with np.load(sample_paths(benchmark, split)[0], allow_pickle=True) as blob:
        return {"cmr_visible": int(len(blob["cmr_times"])) - HELDOUT_CMR_FRAMES, "tee": int(len(blob["tee_times"]))}


def specs(split: str) -> list[tuple[str, dict]]:
    """One spec per level, from E100's frozen test spec: the two arms, parameters stored."""

    reference, directory = _reference_spec()
    frames = _frames(Path(reference["benchmark"]), split)
    arms = {v["name"]: v for v in reference["variants"]}
    out = []
    for level, cmr_every, tee_every in LEVELS + AMENDMENT_LEVELS:
        variants = []
        for name in (JOINT, LATE):
            variant = json.loads(json.dumps(arms[name]))
            variant["options"] = dict(variant.get("options") or {}, store_parameters=True)
            variants.append(variant)
        spec = {
            "experiment": "E113", "preregistration": PREREGISTRATION, "level": level,
            "benchmark": reference["benchmark"], "split": split, "start": reference["start"],
            "models": {"S": reference["models"]["S"]}, "base_config": reference["base_config"],
            "observation_thinning": {"cmr_keep_every": cmr_every, "tee_keep_every": tee_every},
            "visible_frames": {"cmr": -(-frames["cmr_visible"] // cmr_every), "tee": -(-frames["tee"] // tee_every)},
            "frozen_from": str(directory), "variants": variants}
        if level in dict((name, k) for name, k, _ in AMENDMENT_LEVELS):
            spec["amendment"] = ADDENDUM
        out.append((level, spec))
    return out


def _digest(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def _rows(directory: Path) -> list:
    return [json.loads(line) for shard in sorted(directory.glob("shard_*.jsonl"))
            for line in shard.read_text().splitlines() if line.strip()]


def _complete(spec: dict, directory: Path) -> bool:
    from rooted4d.experiments.benchmark_eval import sample_paths

    expected = len(sample_paths(Path(spec["benchmark"]), spec["split"]))
    done = {row["sample"] for row in _rows(directory)} if directory.exists() else set()
    return expected > 0 and len(done) == expected


def _index(split: str) -> dict:
    path = _out() / split / "index.json"
    return json.loads(path.read_text()) if path.exists() else {}


def cmd_run(split: str) -> None:
    if split == "test":
        report = REPORTS / "E113_val.json"
        if not report.exists() or not json.loads(report.read_text())["check"]["passed"]:
            raise SystemExit("the validation run and its check must pass before test")
    stage_dir = _out() / split
    stage_dir.mkdir(parents=True, exist_ok=True)
    index = _index(split)
    for level, spec in specs(split):
        digest = _digest(spec)
        spec_path = stage_dir / f"{level}.json"
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
        directory = stage_dir / digest
        index[level] = digest
        (stage_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
        if _complete(spec, directory):
            print(f"[{split}/{level}] complete, not re-run", flush=True)
            continue
        environment = dict(os.environ, R4D_SPEC=str(spec_path), R4D_OUT=str(stage_dir), R4D_SHARD="0/1")
        environment.pop("R4D_ALLOW_TEST", None)
        if split == "test":
            environment["R4D_ALLOW_TEST"] = "1"
        print(f"[{split}/{level}] running {spec_path} (CMR x{spec['observation_thinning']['cmr_keep_every']}, "
              f"TEE x{spec['observation_thinning']['tee_keep_every']}; visible {spec['visible_frames']})", flush=True)
        subprocess.run([sys.executable, "-u", str(HERE / "e59_unknown_alignment.py")], env=environment, check=True)
        if not _complete(spec, directory):
            raise RuntimeError(f"{level} finished without every sample")


def cmd_check(split: str) -> dict:
    """Level 0 against the stored rows: both held-out errors per sample within the tolerance."""

    rows = {r["sample"]: r for r in _rows(_out() / split / _index(split)["full"])}
    if split == "test":
        reference_dir = Path(os.environ["R4D_E100_ROWS"])
        pairs = {JOINT: JOINT, LATE: LATE}
    else:
        reference_dir = Path(os.environ["R4D_E113_VAL_REFERENCE"])
        pairs = {JOINT: os.environ.get("R4D_E113_VAL_REFERENCE_NAME", "S/joint/bs=0.5")}
    reference = {r["sample"]: r for r in _rows(reference_dir)}
    result = {"split": split, "reference": str(reference_dir), "tolerance_mm": TOLERANCE_MM, "arms": {}}
    passed = True
    for ours, theirs in pairs.items():
        differences, missing = [], 0
        for sample, row in rows.items():
            a, b = row["variants"].get(ours) or {}, (reference.get(sample) or {}).get("variants", {}).get(theirs) or {}
            if not (a.get("succeeded") and b.get("succeeded")):
                missing += 1
                continue
            differences.append(max(abs(a["heldout_cmr_mm"] - b["heldout_cmr_mm"]),
                                   abs(a["heldout_tee_mm"] - b["heldout_tee_mm"])))
        worst = float(max(differences)) if differences else float("nan")
        ok = bool(differences) and worst <= TOLERANCE_MM and missing == 0
        passed &= ok
        result["arms"][ours] = {"reference_name": theirs, "samples_compared": len(differences),
                                "not_comparable": missing, "max_abs_difference_mm": worst, "passed": ok}
        print(f"[{split}] {ours} vs {theirs}: {len(differences)} samples, max |diff| {worst:.2e} mm, "
              f"{missing} not comparable -> {'pass' if ok else 'FAIL'}")
    result["passed"] = bool(passed)
    return result


def cmd_analyze(split: str) -> None:
    stage = _load("e62_stage_b")
    index = _index(split)
    levels = {}
    for level, cmr_every, tee_every in LEVELS + AMENDMENT_LEVELS:
        directory = _out() / split / index[level]
        spec = json.loads((directory / "spec.json").read_text())
        rows = stage.load_rows([directory])
        medians = {}
        for arm in (JOINT, LATE):
            succeeded = [r["variants"][arm] for r in rows if (r["variants"].get(arm) or {}).get("succeeded")]
            medians[arm] = {k: float(np.median([o[k] for o in succeeded])) for k in ENDPOINTS}
            medians[arm]["failures"] = len(rows) - len(succeeded)
            medians[arm]["samples"] = len(rows)
        levels[level] = {"thinning": spec["observation_thinning"], "visible_frames": spec["visible_frames"],
                         "rows": str(directory), "samples": len(rows),
                         "start_failures": sum(1 for r in rows if "failure" in (r.get("start") or {})),
                         "medians": medians, "rooted4d_minus_late_fusion": stage.paired(rows, JOINT, LATE)}
        p = levels[level]["rooted4d_minus_late_fusion"]
        print("[%s/%-5s] visible %s | R4D cmr %.3f tee %.3f | LF cmr %.3f tee %.3f | sum %+.3f [%+.3f, %+.3f] "
              "cmr %+.3f [%+.3f, %+.3f] tee %+.3f [%+.3f, %+.3f] | failures %d/%d" % (
                  split, level, spec["visible_frames"], medians[JOINT]["heldout_cmr_mm"], medians[JOINT]["heldout_tee_mm"],
                  medians[LATE]["heldout_cmr_mm"], medians[LATE]["heldout_tee_mm"],
                  p["sum"]["observed"], p["sum"]["lower"], p["sum"]["upper"],
                  p["cmr"]["observed"], p["cmr"]["lower"], p["cmr"]["upper"],
                  p["tee"]["observed"], p["tee"]["lower"], p["tee"]["upper"],
                  medians[JOINT]["failures"], medians[LATE]["failures"]))
    report = {"experiment": "E113", "preregistration": PREREGISTRATION, "split": split,
              "arms": {"rooted4d": JOINT, "late_fusion": LATE},
              "statistic": "E62: identity-averaged, %d identity resamples, seed %d" % (stage.RESAMPLES, stage.SEED),
              "check": cmd_check(split), "levels": levels, "predictions": _predictions(levels),
              "amendment": _amendment(levels)}
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"E113_{split}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["predictions"], indent=1))
    print("wrote", out)


def _predictions(levels: dict) -> dict:
    """P1 and P2 of the pre-registration, read on the thinned modality's own held-out error."""

    out = {}
    for axis, names in AXES.items():
        entries = [levels[n]["rooted4d_minus_late_fusion"][axis] for n in names]
        sparsest = entries[-2:]
        p1 = all(e["upper"] < 0 for e in sparsest)
        observed = [e["observed"] for e in entries]
        p2 = all(observed[i + 1] < observed[i] for i in range(len(observed) - 1))
        out[axis] = {"levels": list(names), "observed": observed,
                     "P1_interval_excludes_zero_at_two_sparsest": p1,
                     "P2_magnitude_grows_monotonically": p2}
    return out


def _amendment(levels: dict) -> dict:
    """The level added after validation: P1's reading only, outside every prediction."""

    out = {"source": ADDENDUM, "in_predictions": False, "levels": {}}
    for level, _, _ in AMENDMENT_LEVELS:
        if level in levels:
            entry = levels[level]["rooted4d_minus_late_fusion"]["cmr"]
            out["levels"][level] = {"visible_frames": levels[level]["visible_frames"], "cmr": entry,
                                    "interval_excludes_zero_favoring_rooted4d": bool(entry["upper"] < 0)}
    return out


def latex(split: str) -> str:
    tables = _load("e64_manuscript_tables")
    report = json.loads((REPORTS / f"E113_{split}.json").read_text())
    lines = []
    for axis, label in (("cmr", "CMR frames thinned, TEE complete"), ("tee", "TEE frames thinned, CMR complete")):
        lines.append(r"\multicolumn{8}{@{}l}{\textit{%s}}\\" % label)
        names = AXES[axis] + (tuple(n for n, _, _ in AMENDMENT_LEVELS) if axis == "cmr" else ())
        for name in sorted(names, key=lambda n: -report["levels"][n]["visible_frames"][axis]):
            level = report["levels"][name]
            m, p = level["medians"], level["rooted4d_minus_late_fusion"]
            if m[JOINT]["failures"] == m[JOINT]["samples"] and m[LATE]["failures"] == m[LATE]["samples"]:
                lines.append(r"%d & %d & \multicolumn{6}{l}{no input-derived start for either method}\\" % (
                    level["visible_frames"]["cmr"], level["visible_frames"]["tee"]))
                continue
            lines.append("%d & %d & %s & %s & %s & %s & %s & %s\\\\" % (
                level["visible_frames"]["cmr"], level["visible_frames"]["tee"],
                tables.number(m[JOINT]["heldout_cmr_mm"]), tables.number(m[JOINT]["heldout_tee_mm"]),
                tables.number(m[LATE]["heldout_cmr_mm"]), tables.number(m[LATE]["heldout_tee_mm"]),
                tables.interval(p["cmr"]), tables.interval(p["tee"])))
    return "\n".join(lines)


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    if command == "latex":
        print(latex(*arguments))
    elif command == "check":
        print(json.dumps(cmd_check(*arguments), indent=1))
    else:
        {"run": cmd_run, "analyze": cmd_analyze}[command](*arguments)
