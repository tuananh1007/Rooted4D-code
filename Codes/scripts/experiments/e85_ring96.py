"""E85: the main comparison at the ring density the resolution study selected, 96 nodes.

Pre-registered in ``Docs/E85_RING96_PREREGISTRATION_2026-09-28.md``. Every rule
is E62's (selection rounds, edge extensions, cross-validation, statistics),
E73's (the session weight) or E64's (T1-T4). This script only points that code
at the 96-node small-session benchmark and writes its specs to
``Codes/configs/e85``; the E59 driver runs them, as for every other benchmark.

    python e85_ring96.py round1                           -> round-1 spec
    python e85_ring96.py round2   <round-1 dirs...>       -> round-2 spec, or a round-1 edge extension
    python e85_ring96.py gate     <round-2 dirs...>       -> cross-validation spec, or a round-2 extension
    python e85_ring96.py template                         -> E64-style test template (beta_s = 12; never run)
    python e85_ring96.py session                          -> session-weight spec over E73's grid (validation)
    python e85_ring96.py session  <session dirs...>       -> E73's edge extension, or the frozen selection
    python e85_ring96.py test                             -> the single test spec at the selected weights
    python e85_ring96.py analyze  gate=<dir> test=<dir>   -> the cross-validation check and T1-T4
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E85_RING96_PREREGISTRATION_2026-09-28.md"
BENCH = "v3n96"
OUT = Path("Codes/configs/e85")
SELECTION = Path("results/benchmark/E85/E85_session_weight.json")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE = _load("e62_stage_b")
STAGE.BENCHMARKS[BENCH] = {"benchmark": "data/derived/semisynthetic-real-v3-n96", "session_includes_offset": False}
STAGE.OUT = OUT
_stage_write = STAGE.write


def _write(spec: dict, name: str) -> Path:
    """E62's writer, relabelled: the specs are E85's and point at its record."""

    spec["experiment"], spec["preregistration"] = "E85", PREREGISTRATION
    return _stage_write(spec, name)


STAGE.write = _write

SESSION = _load("e73_session_weight")
SESSION.OUT, SESSION.PREREGISTRATION = OUT, PREREGISTRATION


def _session_variants(bench: str, values_by_arm: dict) -> dict:
    """E73's validation spec, templated on E85's test template instead of E62's test spec."""

    template = json.loads((OUT / f"{bench}_test_template.json").read_text())
    by_name = {v["name"]: v for v in template["variants"]}
    spec = {key: template[key] for key in ("benchmark", "start", "models", "base_config")}
    spec.update({"experiment": "E85", "preregistration": PREREGISTRATION, "split": "val",
                 "frozen_from": str(OUT / f"{bench}_test_template.json"), "variants": []})
    for arm, values in values_by_arm.items():
        for value in values:
            variant = json.loads(json.dumps(by_name[arm]))
            variant["name"] = SESSION.name(arm, value)
            variant["overrides"] = dict(by_name[arm].get("overrides") or {}, session_prior_weight=value)
            spec["variants"].append(variant)
    return spec


SESSION.variants = _session_variants


def cmd_session(*directories: str) -> None:
    if not directories:
        SESSION.write(_session_variants(BENCH, {arm: SESSION.GRID for arm in SESSION.ARMS}),
                      f"{BENCH}_session_weight_val.json")
        return
    rows = STAGE.load_rows(directories)
    arms = {arm: SESSION.choose(rows, arm) for arm in SESSION.ARMS}
    pending = {arm: a["extension_needed"] for arm, a in arms.items() if a["extension_needed"]}
    if pending:
        SESSION.write(_session_variants(BENCH, pending), f"{BENCH}_session_weight_val_extension{len(directories)}.json")
        print("edges to extend:", pending)
        return
    report = {"experiment": "E85", "preregistration": PREREGISTRATION, "rule": "E73", "grid": SESSION.GRID,
              "samples": len(rows), "identities": len({r["identity"] for r in rows}), "arms": arms}
    SELECTION.parent.mkdir(parents=True, exist_ok=True)
    SELECTION.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", SELECTION, {arm: (a["best"], a["selected"], a["changed"]) for arm, a in arms.items()})


def cmd_test() -> None:
    """E75's construction: changed arms take their selected beta_s; the 1e4 soft arm follows its joint arm."""

    template = json.loads((OUT / f"{BENCH}_test_template.json").read_text())
    arms = json.loads(SELECTION.read_text())["arms"]
    changed = {arm: float(a["selected"]) for arm, a in arms.items() if a["changed"]}
    spec = dict(template, experiment="E85", preregistration=PREREGISTRATION, round="test", session_weights=changed)
    variants = []
    for variant in template["variants"]:
        variant = json.loads(json.dumps(variant))
        weight = changed.get(variant["name"])
        if variant["name"].endswith("/soft_matched_1e4"):
            weight = changed.get(f"{variant['model']}/joint")
        if weight is not None:
            variant["overrides"] = dict(variant.get("overrides") or {}, session_prior_weight=weight)
        variants.append(variant)
    spec["variants"] = variants
    _stage_write(spec, f"{BENCH}_e85_test.json")
    print("changed:", changed or "none")


def cmd_analyze(*arguments: str) -> None:
    given = dict(argument.split("=", 1) for argument in arguments)
    report = {"experiment": "E85", "preregistration": PREREGISTRATION, "benchmark": BENCH}

    def label(entry):
        return ("joint better" if entry["upper"] < 0 else
                "cost remains" if entry["lower"] > 0 else "no detectable cost")

    if "gate" in given:
        rows = STAGE.load_rows([given["gate"]])
        d1 = STAGE.paired(rows, "S/joint", "G/joint")
        report["cross_validation"] = {"identities": len({r["identity"] for r in rows}),
                                      "structured_minus_global": d1, "verdict": STAGE.verdict(d1["sum"])}
    if "test" in given:
        rows = STAGE.load_rows([given["test"]])
        representation = STAGE.paired(rows, "S/joint", "G/joint")
        fusion = STAGE.paired(rows, "S/joint", "S/independent")
        t3 = {m: label(fusion[m]) for m in ("cmr", "tee")}
        report["test"] = {
            "identities": len({r["identity"] for r in rows}), "samples": len(rows),
            "medians": {f"{m}/{a}": STAGE.sample_medians(rows, f"{m}/{a}")
                        for m in STAGE.MODELS for a in STAGE.TEST_ARMS},
            "T1_representation": {"verdict": STAGE.verdict(representation["sum"]), "comparison": representation},
            "T2_modalities": {m: STAGE.verdict(representation[m]) for m in ("cmr", "tee")},
            "T3_fusion_cost": dict(t3, comparison=fusion),
            "T4_competitors": {arm: STAGE.paired(rows, "S/joint", f"S/{arm}")
                               for arm in ("exact_no_gauges", "soft_matched", "soft_matched_1e4",
                                           "exact_repair", "penalty_repair")}
                              | {"S_vs_G_independent": STAGE.paired(rows, "S/independent", "G/independent"),
                                 "G_fusion_cost": STAGE.paired(rows, "G/joint", "G/independent")},
            "repair": {f"{m}/{a}": STAGE.repair_record(rows, f"{m}/{a}")
                       for m in STAGE.MODELS for a in ("exact_repair", "penalty_repair")},
        }
        # Descriptive, by E62's identity statistic (as E64 and E75 reported): the quantities late fusion gets wrong.
        uniformity = _load("e65_uniformity")
        report["test"]["descriptive_vs_late_fusion"] = {
            key: uniformity.identity_difference(STAGE, rows, "S/joint", "S/independent", key)
            for key in ("rotation_deg", "translation_mm", "phase_map_cycles")}
        holds = (report["test"]["T1_representation"]["verdict"] == "confirmed"
                 and all(v != "cost remains" for v in t3.values()))
        report["ordering_holds_at_96"] = holds
    out = Path("results/benchmark/E85/E85_ring96.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    summary = {k: report[k] for k in ("ordering_holds_at_96",) if k in report}
    if "test" in report:
        summary.update(T1=report["test"]["T1_representation"]["verdict"], T2=report["test"]["T2_modalities"],
                       T3={k: v for k, v in report["test"]["T3_fusion_cost"].items() if k != "comparison"})
    if "cross_validation" in report:
        summary["cross_validation"] = report["cross_validation"]["verdict"]
    print(json.dumps(summary, indent=2))
    print("wrote", out)


def cmd_template() -> None:
    """E64's test spec from the frozen gate spec, kept as the template: at beta_s = 12 it is never run."""

    STAGE.cmd_test(BENCH)
    written = OUT / f"{BENCH}_test.json"
    spec = json.loads(written.read_text())
    spec["round"] = "template (never run; the test spec is v3n96_e85_test.json)"
    (OUT / f"{BENCH}_test_template.json").write_text(json.dumps(spec, indent=2) + "\n")
    written.unlink()
    print("kept as", OUT / f"{BENCH}_test_template.json")


COMMANDS = {
    "round1": lambda: STAGE.cmd_round1(BENCH),
    "round2": lambda *d: STAGE.cmd_round2(BENCH, *d),
    "gate": lambda *d: STAGE.cmd_gate(BENCH, *d),
    "template": cmd_template,
    "session": cmd_session,
    "test": cmd_test,
    "analyze": cmd_analyze,
}

if __name__ == "__main__":
    command, *rest = sys.argv[1:]
    COMMANDS[command](*rest)
