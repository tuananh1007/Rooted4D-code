"""E62 stage logic: specs, mechanical selections, and the gate analysis.

Every rule here is fixed by ``Docs/E62_PREREGISTRATION_2026-09-24.md``. The
driver (``e59_unknown_alignment.py``) runs whatever spec this writes. This
script writes those specs and applies the selections, so nothing between rounds
is chosen by hand.

    python e62_stage_b.py round1  <bench>                  -> round-1 tuning spec
    python e62_stage_b.py round2  <bench> <round1 dirs...> -> round-2 spec, or an edge extension
    python e62_stage_b.py gate    <bench> <round2 dirs...> -> cross-validation gate spec
    python e62_stage_b.py analyze <v4 gate> <v3 gate> <v3n48 gate> -> D1-D5
    python e62_stage_b.py test    <bench>                  -> E64 test spec from the frozen gate spec
    python e62_stage_b.py analyze_test <v4> <v3> <v3n48>   -> E64 test analysis (T1-T4)

Directories hold the driver's shard files. Selections read every shard in
every directory given, so an edge extension simply adds a directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BENCHMARKS = {
    "v4": {"benchmark": "data/derived/semisynthetic-real-v4", "session_includes_offset": True},
    "v3": {"benchmark": "data/derived/semisynthetic-real-v3", "session_includes_offset": False},
    "v3n48": {"benchmark": "data/derived/semisynthetic-real-v3-n48", "session_includes_offset": False},
}
MODELS = {
    "G": {"builder": "global", "n_modes": 48},
    "S": {"builder": "structured", "interface_dim": 12, "interior_dim": 12, "coordinates": "covariance"},
}
WEIGHTS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
SOFT_PENALTIES = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
REPAIR_PENALTIES = [1e2, 1e3, 1e4, 1e5, 1e6]
WEIGHT_LIMITS = (1.0 / 64.0, 1024.0)
SOFT_PENALTY_LIMITS = (1.0, 1e9)
#: The repair penalty never extends below its grid: as it approaches zero the
#: arm turns into the joint fit it is supposed to compete with.
REPAIR_PENALTY_LIMITS = (1e2, 1e9)
CV = {"folds": 5, "seed": 20260924}
RESAMPLES, SEED = 20000, 20260910
OUT = Path("Codes/configs/e62")


def base_spec(bench: str, split: str) -> dict:
    return {
        "experiment": "E62", "preregistration": "Docs/E62_PREREGISTRATION_2026-09-24.md",
        "benchmark": BENCHMARKS[bench]["benchmark"], "split": split,
        "start": "input_derived", "models": MODELS,
        "base_config": {
            "fourier_order": 3, "session_order": 1, "max_iterations": 40,
            "covariance_aware_identity_prior": True, "identity_prior_scale_floor": 0.005,
            "session_includes_offset": BENCHMARKS[bench]["session_includes_offset"],
        },
        "variants": [],
    }


# Round 1: identity weights for the joint arms and the two independent fits.
ROUND1_ARMS = {"joint": "rooted4d_full", "exact_no_gauges": "joint_exact_no_gauges",
               "cmr_only": "cmr_only", "tee_only": "tee_only"}


def round1_variants(weights_by_key: dict) -> list:
    variants = []
    for (model, role), weights in weights_by_key.items():
        for w in weights:
            variants.append({"name": f"{model}/{role}/w={w:g}", "model": model,
                             "arm": ROUND1_ARMS[role], "overrides": {"identity_prior_weight": w}})
    return variants


# ---------------------------------------------------------------- reading ----

def load_rows(directories) -> list:
    rows = {}
    for directory in directories:
        for shard in sorted(Path(directory).glob("shard_*.jsonl")):
            for line in shard.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows.setdefault(row["sample"], {"identity": row["identity"], "variants": {}})
                    rows[row["sample"]]["variants"].update(row["variants"])
                    if "fold" in row:
                        rows[row["sample"]]["fold"] = row["fold"]
    return list(rows.values())


def per_identity(rows, name, key):
    """Identity-averaged values of one endpoint for one variant (NaN on failure)."""

    by_identity = {}
    for row in rows:
        outcome = row["variants"].get(name)
        value = outcome.get(key) if outcome and outcome.get("succeeded") else None
        by_identity.setdefault(row["identity"], []).append(
            np.nan if value is None else float(value))
    identities = sorted(by_identity)
    return identities, np.array([np.mean(by_identity[i]) for i in identities])


def score(rows, name, rule):
    _, cmr = per_identity(rows, name, "heldout_cmr_mm")
    _, tee = per_identity(rows, name, "heldout_tee_mm")
    if rule == "cmr":
        return float(np.nanmedian(cmr))
    if rule == "tee":
        return float(np.nanmedian(tee))
    return float(np.nanmedian(cmr) + np.nanmedian(tee))


def select(rows, prefix, values, rule, limits, factor=2.0):
    """Best value on the grid, or the next two values beyond an edge."""

    names = {v: f"{prefix}{v:g}" for v in values}
    present = {v: score(rows, n, rule) for v, n in names.items()
               if any(n in r["variants"] for r in rows)}
    ordered = sorted(present)
    best = min(ordered, key=lambda v: present[v])
    index = ordered.index(best)
    if index == len(ordered) - 1:
        if best * factor > limits[1]:
            return best, []
        return best, [v for v in (best * factor, best * factor * factor) if v <= limits[1]]
    if index == 0:
        nxt = [best / factor, best / factor / factor]
        if best / factor < limits[0]:
            return best, []
        return best, [v for v in nxt if v >= limits[0]]
    return best, []


# ------------------------------------------------------------- the rounds ----

def write(spec: dict, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print("wrote", path, "with", len(spec["variants"]), "variants")
    return path


def cmd_round1(bench):
    spec = base_spec(bench, "val")
    spec["round"] = "round1"
    spec["variants"] = round1_variants({(m, role): WEIGHTS for m in MODELS for role in ROUND1_ARMS})
    write(spec, f"{bench}_round1.json")


def round1_selection(rows):
    chosen, extensions = {}, {}
    for model in MODELS:
        for role, rule in (("joint", "sum"), ("exact_no_gauges", "sum"),
                           ("cmr_only", "cmr"), ("tee_only", "tee")):
            present = sorted({float(n.split("w=")[1]) for r in rows for n in r["variants"]
                              if n.startswith(f"{model}/{role}/w=")})
            best, more = select(rows, f"{model}/{role}/w=", present, rule, WEIGHT_LIMITS)
            chosen[(model, role)] = best
            if more:
                extensions[(model, role)] = more
    return chosen, extensions


def cmd_round2(bench, *directories):
    rows = load_rows(directories)
    chosen, extensions = round1_selection(rows)
    if extensions:
        spec = base_spec(bench, "val")
        spec["round"] = "round1-extension"
        spec["variants"] = round1_variants(extensions)
        write(spec, f"{bench}_round1_extension_{len(directories)}.json")
        print("edges to extend:", {f"{m}/{r}": v for (m, r), v in extensions.items()})
        return
    spec = base_spec(bench, "val")
    spec["round"] = "round2"
    spec["round1_selection"] = {f"{m}/{r}": w for (m, r), w in chosen.items()}
    for model in MODELS:
        spec["variants"].extend(round2_variants(model, chosen))
    write(spec, f"{bench}_round2.json")


def _pair(model, chosen):
    return {"cmr": {"identity_prior_weight": chosen[(model, "cmr_only")]},
            "tee": {"identity_prior_weight": chosen[(model, "tee_only")]}}


def round2_variants(model, chosen, soft=SOFT_PENALTIES, repair=REPAIR_PENALTIES):
    w = chosen[(model, "joint")]
    variants = [{"name": f"{model}/soft_matched/l={l:g}", "model": model,
                 "arm": "joint_soft_attachment_matched",
                 "overrides": {"identity_prior_weight": w}, "soft_attachment_weight": l}
                for l in soft]
    variants += [{"name": f"{model}/penalty_repair/l={l:g}", "model": model,
                  "arm": "independent_penalty_repair", "overrides": {"identity_prior_weight": w},
                  "options": {"modality_overrides": _pair(model, chosen), "ring_weight": l}}
                 for l in repair]
    variants += [{"name": f"{model}/exact_repair", "model": model, "arm": "independent_exact_repair",
                  "options": {"modality_overrides": _pair(model, chosen)}},
                 {"name": f"{model}/independent", "model": model, "arm": "late_fusion",
                  "options": {"modality_overrides": _pair(model, chosen)}}]
    return variants


def cmd_gate(bench, *directories):
    rows = load_rows(directories)
    round2 = json.loads((OUT / f"{bench}_round2.json").read_text())
    chosen = {tuple(k.split("/")): v for k, v in round2["round1_selection"].items()}
    penalties, extensions = {}, []
    for model in MODELS:
        for role, grid in (("soft_matched", SOFT_PENALTIES), ("penalty_repair", REPAIR_PENALTIES)):
            present = sorted({float(n.split("l=")[1]) for r in rows for n in r["variants"]
                              if n.startswith(f"{model}/{role}/l=")})
            limits = SOFT_PENALTY_LIMITS if role == "soft_matched" else REPAIR_PENALTY_LIMITS
            best, more = select(rows, f"{model}/{role}/l=", present, "sum", limits, 10.0)
            penalties[(model, role)] = best
            if more:
                extensions.append((model, role, more))
    if extensions:
        spec = base_spec(bench, "val")
        spec["round"] = "round2-extension"
        for model, role, more in extensions:
            variants = round2_variants(model, chosen, soft=more if role == "soft_matched" else [],
                                       repair=more if role == "penalty_repair" else [])
            spec["variants"] += [v for v in variants if f"/{role}/" in v["name"]]
        write(spec, f"{bench}_round2_extension_{len(directories)}.json")
        print("edges to extend:", extensions)
        return
    spec = base_spec(bench, "train")
    spec["round"] = "gate"
    spec["cv"] = CV
    spec["frozen"] = {"round1": round2["round1_selection"],
                      "penalties": {f"{m}/{r}": v for (m, r), v in penalties.items()}}
    for model in MODELS:
        w, pair = chosen[(model, "joint")], _pair(model, chosen)
        spec["variants"] += [
            {"name": f"{model}/joint", "model": model, "arm": "rooted4d_full",
             "overrides": {"identity_prior_weight": w}},
            {"name": f"{model}/exact_no_gauges", "model": model, "arm": "joint_exact_no_gauges",
             "overrides": {"identity_prior_weight": chosen[(model, "exact_no_gauges")]}},
            {"name": f"{model}/soft_matched", "model": model, "arm": "joint_soft_attachment_matched",
             "overrides": {"identity_prior_weight": w},
             "soft_attachment_weight": penalties[(model, "soft_matched")]},
            {"name": f"{model}/independent", "model": model, "arm": "late_fusion",
             "options": {"modality_overrides": pair}},
            {"name": f"{model}/exact_repair", "model": model, "arm": "independent_exact_repair",
             "options": {"modality_overrides": pair}},
            {"name": f"{model}/penalty_repair", "model": model, "arm": "independent_penalty_repair",
             "overrides": {"identity_prior_weight": w},
             "options": {"modality_overrides": pair,
                         "ring_weight": penalties[(model, "penalty_repair")]}},
        ]
    write(spec, f"{bench}_gate.json")


# ------------------------------------------------------------- statistics ----

def sum_of_medians(cmr, tee):
    return float(np.nanmedian(cmr) + np.nanmedian(tee))


def paired(rows, first, second):
    """First minus second, paired at the identity, resampling identities."""

    ids_a, cmr_a = per_identity(rows, first, "heldout_cmr_mm")
    _, tee_a = per_identity(rows, first, "heldout_tee_mm")
    ids_b, cmr_b = per_identity(rows, second, "heldout_cmr_mm")
    _, tee_b = per_identity(rows, second, "heldout_tee_mm")
    assert ids_a == ids_b
    rng = np.random.default_rng(SEED)
    n = len(ids_a)
    observed = {"sum": sum_of_medians(cmr_a, tee_a) - sum_of_medians(cmr_b, tee_b),
                "cmr": float(np.nanmedian(cmr_a) - np.nanmedian(cmr_b)),
                "tee": float(np.nanmedian(tee_a) - np.nanmedian(tee_b))}
    draws = {k: np.empty(RESAMPLES) for k in observed}
    for r in range(RESAMPLES):
        t = rng.integers(0, n, size=n)
        draws["sum"][r] = sum_of_medians(cmr_a[t], tee_a[t]) - sum_of_medians(cmr_b[t], tee_b[t])
        draws["cmr"][r] = np.nanmedian(cmr_a[t]) - np.nanmedian(cmr_b[t])
        draws["tee"][r] = np.nanmedian(tee_a[t]) - np.nanmedian(tee_b[t])
    out = {k: {"observed": observed[k], "lower": float(np.percentile(v, 2.5)),
               "upper": float(np.percentile(v, 97.5))} for k, v in draws.items()}

    def failures(name):
        return sum(1 for r in rows if not (r["variants"].get(name) or {}).get("succeeded"))

    out["descriptive"] = {
        "identities": n,
        "cmr_better_worse": [int(np.nansum(cmr_a < cmr_b)), int(np.nansum(cmr_a > cmr_b))],
        "tee_better_worse": [int(np.nansum(tee_a < tee_b)), int(np.nansum(tee_a > tee_b))],
        "tee_p90": [float(np.nanpercentile(tee_a, 90)), float(np.nanpercentile(tee_b, 90))],
        "cmr_p90": [float(np.nanpercentile(cmr_a, 90)), float(np.nanpercentile(cmr_b, 90))],
        "tee_max": [float(np.nanmax(tee_a)), float(np.nanmax(tee_b))],
        "failures": [failures(first), failures(second)],
    }
    for key in ("rotation_deg", "translation_mm", "phase_map_cycles", "attachment_residual_mm",
                "seconds"):
        a = per_identity(rows, first, key)[1]
        b = per_identity(rows, second, key)[1]
        out["descriptive"][key + "_medians"] = [float(np.nanmedian(a)), float(np.nanmedian(b))]
    return out


def repair_record(rows, name):
    """One repair arm's registration and projection error, and what it left free."""

    outcomes = [r["variants"].get(name) or {} for r in rows]
    record = {}
    for key in ("repair_registration_rms_mm", "repair_projection_rms_mm"):
        values = np.array([o[key] for o in outcomes if o.get(key) is not None], dtype=float)
        record[key] = ({"median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
                        "max": float(values.max()), "samples": int(values.size)}
                       if values.size else None)
    for key in ("repair_free_identity_directions", "repair_free_shape_directions"):
        record[key] = sorted({o[key] for o in outcomes if o.get(key) is not None})
    return record


def classify_d1(v4, v3):
    s = v4["sum"]
    if s["upper"] < 0.0:
        return "A" if v3["sum"]["observed"] <= 0.0 else "B"
    return "D" if s["lower"] > 0.0 else "C"


def cmd_analyze(v4_dir, v3_dir, n48_dir):
    gates = {"v4": load_rows([v4_dir]), "v3": load_rows([v3_dir]), "v3n48": load_rows([n48_dir])}
    report = {"experiment": "E62", "preregistration": "Docs/E62_PREREGISTRATION_2026-09-24.md"}
    d1 = {b: paired(rows, "S/joint", "G/joint") for b, rows in gates.items()}
    cls = classify_d1(d1["v4"], d1["v3"])
    desc = d1["v4"]["descriptive"]
    capped = desc["failures"][0] > desc["failures"][1] + 2
    if capped and cls in ("A", "B"):
        cls = "C"
    claim = [m for m in ("cmr", "tee") if d1["v4"][m]["upper"] < 0.0]
    if desc["tee_p90"][0] > desc["tee_p90"][1]:
        claim = [m for m in claim if m != "tee"]
    n48 = d1["v3n48"]["sum"]
    report["D1"] = {"class": cls, "failure_cap_applied": bool(capped), "comparisons": d1}
    report["D2"] = {"claimable": claim}
    report["D3"] = {"ordering_preserved": n48["observed"] <= 0.0, "ordering_confirmed": n48["upper"] < 0.0}
    chosen = "S" if cls in ("A", "B") else "G"
    fusion = paired(gates["v4"], f"{chosen}/joint", f"{chosen}/independent")

    def label(entry):
        return ("joint better" if entry["upper"] < 0 else
                "cost remains" if entry["lower"] > 0 else "no detectable cost")

    report["D4"] = {"representation": chosen, "cmr": label(fusion["cmr"]), "tee": label(fusion["tee"]),
                    "cost_removed": all(label(fusion[m]) != "cost remains" for m in ("cmr", "tee")),
                    "comparison": fusion}
    report["D5"] = {bench: {arm: paired(rows, f"{chosen}/joint", f"{chosen}/{arm}")
                            for arm in ("exact_no_gauges", "soft_matched", "exact_repair", "penalty_repair")}
                    | {"S_vs_G_independent": paired(rows, "S/independent", "G/independent")}
                    for bench, rows in gates.items()}
    report["D5_repair"] = {bench: {f"{m}/{arm}": repair_record(rows, f"{m}/{arm}")
                                   for m in MODELS for arm in ("exact_repair", "penalty_repair")}
                           for bench, rows in gates.items()}
    out = Path("results/benchmark/E62_stage_b_gate.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"D1": cls, "D2": claim, "D3": report["D3"],
                      "D4": {k: report["D4"][k] for k in ("representation", "cmr", "tee", "cost_removed")}},
                     indent=2))
    print("wrote", out)


# ------------------------------------------------------------------- test ----
# E64: the single test evaluation of the configuration E62 froze. The authors
# decided on 24 September 2026 to spend the synthetic test split now; the rules
# are pre-registered in TEST_PREREGISTRATION, committed before any test fit.

TEST_PREREGISTRATION = "Docs/E64_TEST_PREREGISTRATION_2026-09-24.md"
TEST_ENDPOINTS = ("heldout_cmr_mm", "heldout_tee_mm", "rotation_deg", "translation_mm",
                  "phase_error_cycles", "phase_map_cycles", "attachment_residual_mm",
                  "heldout_cmr_p95_mm", "heldout_tee_p95_mm", "seconds")
TEST_ARMS = ("joint", "exact_no_gauges", "soft_matched", "soft_matched_1e4", "independent",
             "exact_repair", "penalty_repair", "cmr_only", "tee_only")


def cmd_test(bench):
    """Write the test spec from the gate spec, changing only what test needs.

    The gate's twelve arms are copied unchanged, with their frozen settings.
    Bases are learned from the full training split rather than per fold, which
    is how validation tuning ran. Three rows per model that the manuscript's
    table needs are added at settings that were already frozen: CMR-only and
    TEE-only at their selected weights, and matched soft attachment at the
    published penalty of 1e4 beside the tuned one.
    """
    gate = json.loads((OUT / f"{bench}_gate.json").read_text())
    chosen = {tuple(k.split("/")): v for k, v in gate["frozen"]["round1"].items()}
    spec = base_spec(bench, "test")
    spec["experiment"] = "E64"
    spec["preregistration"] = TEST_PREREGISTRATION
    spec["round"] = "test"
    spec["frozen"] = gate["frozen"]
    spec["variants"] = [dict(v) for v in gate["variants"]]
    for model in MODELS:
        spec["variants"] += [
            {"name": f"{model}/cmr_only", "model": model, "arm": "cmr_only",
             "overrides": {"identity_prior_weight": chosen[(model, "cmr_only")]}},
            {"name": f"{model}/tee_only", "model": model, "arm": "tee_only",
             "overrides": {"identity_prior_weight": chosen[(model, "tee_only")]}},
            {"name": f"{model}/soft_matched_1e4", "model": model,
             "arm": "joint_soft_attachment_matched",
             "overrides": {"identity_prior_weight": chosen[(model, "joint")]},
             "soft_attachment_weight": 1e4},
        ]
    write(spec, f"{bench}_test.json")


def sample_medians(rows, name):
    """Medians over samples, the statistic the manuscript's tables report."""

    out = {}
    for key in TEST_ENDPOINTS:
        values = [(r["variants"].get(name) or {}).get(key) for r in rows
                  if (r["variants"].get(name) or {}).get("succeeded")]
        values = [float(v) for v in values if v is not None]
        out[key] = float(np.median(values)) if values else None
    out["failures"] = sum(1 for r in rows if not (r["variants"].get(name) or {}).get("succeeded"))
    out["samples"] = len(rows)
    return out


def verdict(entry):
    """T1 and T2 wording: an interval below 0 confirms, above 0 contradicts."""

    return ("confirmed" if entry["upper"] < 0 else
            "contradicted" if entry["lower"] > 0 else "not confirmed")


def cmd_analyze_test(v4_dir, v3_dir, n48_dir):
    tests = {"v4": load_rows([v4_dir]), "v3": load_rows([v3_dir]), "v3n48": load_rows([n48_dir])}
    report = {"experiment": "E64", "preregistration": TEST_PREREGISTRATION,
              "statistics": "as E62: identity-averaged, sum of modality medians, "
                            "differences of medians, %d identity resamples, seed %d"
                            % (RESAMPLES, SEED),
              "benchmarks": {}}

    def label(entry):
        return ("joint better" if entry["upper"] < 0 else
                "cost remains" if entry["lower"] > 0 else "no detectable cost")

    for bench, rows in tests.items():
        representation = paired(rows, "S/joint", "G/joint")
        fusion = paired(rows, "S/joint", "S/independent")
        report["benchmarks"][bench] = {
            "identities": len({r["identity"] for r in rows}), "samples": len(rows),
            "medians": {f"{m}/{a}": sample_medians(rows, f"{m}/{a}")
                        for m in MODELS for a in TEST_ARMS},
            "T1_representation": {"verdict": verdict(representation["sum"]),
                                  "comparison": representation},
            "T2_modalities": {m: verdict(representation[m]) for m in ("cmr", "tee")},
            "T3_fusion_cost": {"cmr": label(fusion["cmr"]), "tee": label(fusion["tee"]),
                               "cost_removed": all(label(fusion[m]) != "cost remains"
                                                   for m in ("cmr", "tee")),
                               "comparison": fusion},
            "T4_competitors": {arm: paired(rows, "S/joint", f"S/{arm}")
                               for arm in ("exact_no_gauges", "soft_matched", "soft_matched_1e4",
                                           "exact_repair", "penalty_repair")}
                              | {"S_vs_G_independent": paired(rows, "S/independent", "G/independent"),
                                 "G_fusion_cost": paired(rows, "G/joint", "G/independent")},
            "repair": {f"{m}/{a}": repair_record(rows, f"{m}/{a}")
                       for m in MODELS for a in ("exact_repair", "penalty_repair")},
        }
    out = Path("results/benchmark/E64_test.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({b: {"T1": v["T1_representation"]["verdict"], "T2": v["T2_modalities"],
                          "T3": {k: v["T3_fusion_cost"][k] for k in ("cmr", "tee", "cost_removed")}}
                      for b, v in report["benchmarks"].items()}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    {"round1": cmd_round1, "round2": cmd_round2, "gate": cmd_gate, "analyze": cmd_analyze,
     "test": cmd_test, "analyze_test": cmd_analyze_test}[command](*arguments)
