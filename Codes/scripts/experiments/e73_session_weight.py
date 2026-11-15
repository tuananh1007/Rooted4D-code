"""E73: the session-prior weight, selected on validation for every arm with a session term.

Pre-registered in ``Docs/E73_SESSION_WEIGHT_PREREGISTRATION_2026-09-25.md``.
E68 found on v4 test that the frozen configuration's session weight
(beta_s = 12) did worse than beta_s = 1 at full evidence. beta_s was fixed
rather than selected, so this study selects it on the 60 validation samples of
each benchmark, for each arm whose fit carries a session term, with every other
setting at its frozen value. No test sample is read.

    python e73_session_weight.py spec <bench>               -> validation spec over the grid
    python e73_session_weight.py extend <bench> <dirs...>   -> edge-extension spec, if an optimum is at an edge
    python e73_session_weight.py analyze v4=<dirs,...> v3=<dirs,...> v3n48=<dirs,...>

The spec runs with the E59 driver, like E62's rounds. Scores, the paired
statistic, and the edge mechanics are E62's (``e62_stage_b``).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

PREREGISTRATION = "Docs/E73_SESSION_WEIGHT_PREREGISTRATION_2026-09-25.md"
OUT = Path("Codes/configs/e73")
GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 32.0, 64.0]
LIMITS = (1.0 / 64.0, 1024.0)
FROZEN = 12.0
#: The arms whose fits carry a session term. Exact assembly without gauges drops
#: the session prior, and the independent, unimodal, and exact-repair fits each
#: see one session, so beta_s does not enter them.
ARMS = ("S/joint", "G/joint", "S/soft_matched", "G/soft_matched", "S/penalty_repair", "G/penalty_repair")


def _stage():
    spec = importlib.util.spec_from_file_location(
        "e62_stage_b", Path(__file__).resolve().parent / "e62_stage_b.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE = _stage()


def name(arm: str, value: float) -> str:
    return f"{arm}/bs={value:g}"


def variants(bench: str, values_by_arm: dict) -> dict:
    """The frozen test spec's session-term arms, on validation, at the given beta_s values."""

    frozen = json.loads((Path("Codes/configs/e62") / f"{bench}_test.json").read_text())
    by_name = {v["name"]: v for v in frozen["variants"]}
    spec = {key: frozen[key] for key in ("benchmark", "start", "models", "base_config")}
    spec.update({"experiment": "E73", "preregistration": PREREGISTRATION, "split": "val",
                 "frozen_from": f"Codes/configs/e62/{bench}_test.json", "variants": []})
    for arm, values in values_by_arm.items():
        template = by_name[arm]
        for value in values:
            variant = json.loads(json.dumps(template))
            variant["name"] = name(arm, value)
            variant["overrides"] = dict(template.get("overrides") or {}, session_prior_weight=value)
            spec["variants"].append(variant)
    return spec


def write(spec: dict, filename: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print("wrote", path, "with", len(spec["variants"]), "variants")
    return path


def cmd_spec(bench: str) -> None:
    write(variants(bench, {arm: GRID for arm in ARMS}), f"{bench}_e73_session_weight_val.json")


def choose(rows: list, arm: str) -> dict:
    """E62's score over every value run; ties go toward the frozen value; E62's edge rule."""

    present = sorted({float(key.split("/bs=")[1]) for r in rows for key in r["variants"]
                      if key.startswith(arm + "/bs=")})
    scores = {v: STAGE.score(rows, name(arm, v), "sum") for v in present}
    best = min(present, key=lambda v: (round(scores[v], 12), abs(math.log(v / FROZEN))))
    extension = []
    if best == present[-1] and best * 2 <= LIMITS[1]:
        extension = [v for v in (best * 2, best * 4) if v <= LIMITS[1]]
    elif best == present[0] and best / 2 >= LIMITS[0]:
        extension = [v for v in (best / 2, best / 4) if v >= LIMITS[0]]
    out = {"scores": {f"{v:g}": s for v, s in scores.items()}, "best": best, "extension_needed": extension}
    if best != FROZEN:
        comparison = STAGE.paired(rows, name(arm, best), name(arm, FROZEN))
        out["best_minus_frozen"] = comparison
        out["changed"] = bool(comparison["sum"]["upper"] < 0.0)
    else:
        out["changed"] = False
    out["selected"] = best if out["changed"] else FROZEN
    return out


def cmd_extend(bench: str, *directories: str) -> None:
    rows = STAGE.load_rows(directories)
    needed = {arm: choose(rows, arm)["extension_needed"] for arm in ARMS}
    needed = {arm: values for arm, values in needed.items() if values}
    if not needed:
        print("no optimum at an edge; nothing to extend")
        return
    round_index = len(directories)
    write(variants(bench, needed), f"{bench}_e73_session_weight_val_extension{round_index}.json")


def cmd_analyze(*arguments: str) -> None:
    report = {"experiment": "E73", "preregistration": PREREGISTRATION, "grid": GRID, "benchmarks": {}}
    for argument in arguments:
        bench, directories = argument.split("=", 1)
        rows = STAGE.load_rows(directories.split(","))
        entry = {"samples": len(rows), "identities": len({r["identity"] for r in rows}), "arms": {}}
        for arm in ARMS:
            entry["arms"][arm] = choose(rows, arm)
        entry["any_change"] = any(a["changed"] for a in entry["arms"].values())
        entry["pending_extensions"] = {a: v["extension_needed"] for a, v in entry["arms"].items()
                                       if v["extension_needed"]}
        report["benchmarks"][bench] = entry
    out = Path("results/benchmark/E73_session_weight.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out)
    for bench, entry in report["benchmarks"].items():
        print(bench, {arm: (a["best"], a["selected"], a["changed"]) for arm, a in entry["arms"].items()},
              "pending extensions:", entry["pending_extensions"] or "none")


if __name__ == "__main__":
    command, *rest = sys.argv[1:]
    {"spec": cmd_spec, "extend": cmd_extend, "analyze": cmd_analyze}[command](*rest)
