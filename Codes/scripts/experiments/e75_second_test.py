"""E75: the second test evaluation, with the session-prior weights E73 selected on validation.

Its design was fixed in advance by ``Docs/E73_SESSION_WEIGHT_PREREGISTRATION_2026-09-25.md``
and is recorded, with the authors' approval, in ``Docs/E75_SECOND_TEST_2026-09-26.md``:

* every arm of the benchmark's E64 test spec is run again; arms E73 changed take
  their selected beta_s, each model's descriptive soft arm at 1e4 takes its joint
  arm's value, and every other arm is re-run unchanged as a same-run reference;
* E64's rules T1-T4 apply unchanged, through E62's own statistics;
* the arms re-run unchanged must reproduce E64 (0.01 mm on the medians).

    python e75_second_test.py spec <bench>             -> Codes/configs/e75/<bench>_e75_test.json
    python e75_second_test.py e82                      -> E67 repeated at the new v4 weight
    python e75_second_test.py analyze v4=<dir> v3n48=<dir> [e64v4=<dir>] [e64v3n48=<dir>]
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path("Codes/configs/e75")
RECORD = "Docs/E75_SECOND_TEST_2026-09-26.md"
SELECTION = Path("results/benchmark/E73_session_weight.json")


def _stage():
    spec = importlib.util.spec_from_file_location("e62_stage_b", HERE / "e62_stage_b.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE = _stage()


def selected(bench: str) -> dict:
    """The arms whose beta_s E73 changed, with their selected values."""

    arms = json.loads(SELECTION.read_text())["benchmarks"][bench]["arms"]
    return {arm: float(entry["selected"]) for arm, entry in arms.items() if entry["changed"]}


def write(spec: dict, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print("wrote", path, "with", len(spec["variants"]), "variants")
    return path


def cmd_spec(bench: str) -> None:
    frozen = json.loads((Path("Codes/configs/e62") / f"{bench}_test.json").read_text())
    changed = selected(bench)
    spec = dict(frozen, experiment="E75", preregistration=RECORD,
                session_weights={arm: value for arm, value in changed.items()})
    variants = []
    for variant in frozen["variants"]:
        variant = json.loads(json.dumps(variant))
        model = variant["model"]
        weight = changed.get(variant["name"])
        if variant["name"].endswith("/soft_matched_1e4"):
            weight = changed.get(f"{model}/joint")  # the descriptive arm follows its joint arm
        if weight is not None:
            variant["overrides"] = dict(variant.get("overrides") or {}, session_prior_weight=weight)
        variants.append(variant)
    spec["variants"] = variants
    print(bench, "changed:", {v["name"]: v["overrides"]["session_prior_weight"] for v in variants
                              if "session_prior_weight" in (v.get("overrides") or {})})
    write(spec, f"{bench}_e75_test.json")


def cmd_e82() -> None:
    """E67's three session arms on v4 test, at the structured joint arm's new weight."""

    weight = selected("v4")["S/joint"]
    spec = json.loads(Path("Codes/configs/e65/v4_e67_session_test.json").read_text())
    spec.update(experiment="E82", preregistration=RECORD, session_prior_weight=weight)
    for variant in spec["variants"]:
        variant["overrides"] = dict(variant.get("overrides") or {}, session_prior_weight=weight)
    write(spec, "v4_e82_session_test.json")


def cmd_analyze(*arguments: str) -> None:
    given = dict(argument.split("=", 1) for argument in arguments)
    report = {"experiment": "E75", "record": RECORD, "selection": str(SELECTION), "benchmarks": {}}

    def label(entry):
        return ("joint better" if entry["upper"] < 0 else
                "cost remains" if entry["lower"] > 0 else "no detectable cost")

    for bench in ("v4", "v3n48"):
        if bench not in given:
            continue
        rows = STAGE.load_rows([given[bench]])
        representation = STAGE.paired(rows, "S/joint", "G/joint")
        fusion = STAGE.paired(rows, "S/joint", "S/independent")
        entry = {
            "identities": len({r["identity"] for r in rows}), "samples": len(rows),
            "session_weights": selected(bench),
            "medians": {f"{m}/{a}": STAGE.sample_medians(rows, f"{m}/{a}")
                        for m in STAGE.MODELS for a in STAGE.TEST_ARMS},
            "T1_representation": {"verdict": STAGE.verdict(representation["sum"]), "comparison": representation},
            "T2_modalities": {m: STAGE.verdict(representation[m]) for m in ("cmr", "tee")},
            "T3_fusion_cost": {"cmr": label(fusion["cmr"]), "tee": label(fusion["tee"]),
                               "cost_removed": all(label(fusion[m]) != "cost remains" for m in ("cmr", "tee")),
                               "comparison": fusion},
            "T4_competitors": {arm: STAGE.paired(rows, "S/joint", f"S/{arm}")
                               for arm in ("exact_no_gauges", "soft_matched", "soft_matched_1e4",
                                           "exact_repair", "penalty_repair")}
                              | {"S_vs_G_independent": STAGE.paired(rows, "S/independent", "G/independent"),
                                 "G_fusion_cost": STAGE.paired(rows, "G/joint", "G/independent")},
            "repair": {f"{m}/{a}": STAGE.repair_record(rows, f"{m}/{a}")
                       for m in STAGE.MODELS for a in ("exact_repair", "penalty_repair")},
        }
        reference = given.get(f"e64{bench}")
        if reference:
            e64 = STAGE.load_rows([reference])
            unchanged = [f"{m}/{a}" for m in STAGE.MODELS for a in STAGE.TEST_ARMS
                         if f"{m}/{a}" not in selected(bench)
                         and not (a == "soft_matched_1e4" and f"{m}/joint" in selected(bench))]
            check = {}
            for name in unchanged:
                mine, theirs = STAGE.sample_medians(rows, name), STAGE.sample_medians(e64, name)
                check[name] = {key: mine[key] - theirs[key] for key in ("heldout_cmr_mm", "heldout_tee_mm")
                               if mine.get(key) is not None and theirs.get(key) is not None}
            entry["reproduction_of_unchanged_arms"] = {
                "differences": check,
                "passed": all(abs(d) <= 0.01 for arm in check.values() for d in arm.values())}
        report["benchmarks"][bench] = entry
    out = Path("results/benchmark/E75_test.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({b: {"T1": v["T1_representation"]["verdict"], "T2": v["T2_modalities"],
                          "T3": {k: v["T3_fusion_cost"][k] for k in ("cmr", "tee", "cost_removed")},
                          "reproduction": (v.get("reproduction_of_unchanged_arms") or {}).get("passed")}
                      for b, v in report["benchmarks"].items()}, indent=2))
    print("wrote", out)


def cmd_components(e82_dir: str, e83_dir: str, e75_v4_dir: str) -> None:
    """E82 and E83 by E67's and E68's own statistics, with their frozen arms checked against E75."""

    spec = importlib.util.spec_from_file_location("e65_uniformity", HERE / "e65_uniformity.py")
    uniformity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(uniformity)
    spec = importlib.util.spec_from_file_location("e68_e72_analysis", HERE / "e68_e72_analysis.py")
    components = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(components)
    rows82, e75 = STAGE.load_rows([e82_dir]), STAGE.load_rows([e75_v4_dir])
    report = {"experiments": "E82-E83", "record": RECORD, "E82": {
        "medians": {n: STAGE.sample_medians(rows82, n) for n in ("S/frozen", "S/oscillatory", "S/no_session")},
        "frozen_minus_no_session": uniformity.compare(STAGE, rows82, "S/frozen", "S/no_session"),
        "frozen_minus_oscillatory": uniformity.compare(STAGE, rows82, "S/frozen", "S/oscillatory"),
        "oscillatory_minus_no_session": uniformity.compare(STAGE, rows82, "S/oscillatory", "S/no_session"),
        "reproduction_of_e75": uniformity.reproduction(STAGE, rows82, "S/frozen", e75, "S/joint")}}
    report["E83"] = components.analyze_e68(e83_dir)
    rows83 = STAGE.load_rows([e83_dir])
    full = {key: STAGE.sample_medians(rows83, "full/frozen")[key] - STAGE.sample_medians(e75, "S/joint")[key]
            for key in ("heldout_cmr_mm", "heldout_tee_mm")}
    report["E83"]["reproduction_of_e75"] = {"differences": full, "passed": all(abs(v) <= 0.01 for v in full.values())}
    out = Path("results/benchmark/E82_E83_components.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    command, *rest = sys.argv[1:]
    {"spec": cmd_spec, "e82": cmd_e82, "analyze": cmd_analyze, "components": cmd_components}[command](*rest)
