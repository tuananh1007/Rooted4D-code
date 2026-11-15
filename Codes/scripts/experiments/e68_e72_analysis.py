"""Analysis of E68-E72, by the rules of ``Docs/E65_E72_UNIFORMITY_PREREGISTRATION_2026-09-24.md``.

Every paired statistic is E62's (``e62_stage_b.paired`` and the same bootstrap
for other endpoints), so the component studies read on the same scale as the
main tables.

    python e68_e72_analysis.py <E68 dir> <E69 val dir> <E69 test dir> <E70 dir> <E71 dir> <E72 dir> <E64 v3 test dir>
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE = _module("e62_stage_b", HERE / "e62_stage_b.py")
UNIFORMITY = _module("e65_uniformity", HERE / "e65_uniformity.py")
CONFORMAL_TARGETS = (0.8, 0.9)
PERMUTATIONS = 20000


def rows_of(directory):
    return STAGE.load_rows([directory])


def medians(rows, names, keys):
    out = {}
    for name in names:
        values = {}
        for key in keys:
            v = [r["variants"][name][key] for r in rows
                 if (r["variants"].get(name) or {}).get("succeeded") and r["variants"][name].get(key) is not None]
            values[key] = float(np.median(v)) if v else None
        values["failures"] = sum(1 for r in rows if not (r["variants"].get(name) or {}).get("succeeded"))
        out[name] = values
    return out


def difference(rows, first, second, key):
    return UNIFORMITY.identity_difference(STAGE, rows, first, second, key)


# --------------------------------------------------------------------- E68 ----

def analyze_e68(directory, weak_arm="session_prior_1"):
    rows = rows_of(directory)
    levels = ("full", "half", "quarter", "eighth", "eighth_noisy")
    out = {}
    for level in levels:
        frozen, weak = f"{level}/frozen", f"{level}/{weak_arm}"
        comparison = STAGE.paired(rows, frozen, weak)
        comparison["descriptive_intervals"] = {k: difference(rows, frozen, weak, k)
                                               for k in ("translation_mm", "rotation_deg")}
        starts = [r["variants"][frozen].get("start") for r in rows if r["variants"].get(frozen)]
        out[level] = {"medians": medians(rows, (frozen, weak), ("heldout_cmr_mm", "heldout_tee_mm",
                                                               "translation_mm", "rotation_deg")),
                      f"frozen_minus_{weak_arm}": comparison,
                      "zero_starts": int(sum(1 for s in starts if s == "zero")),
                      "visible": {k: float(np.median([r["variants"][frozen][k] for r in rows
                                                      if r["variants"][frozen].get("succeeded")]))
                                  for k in ("visible_cmr_frames", "visible_tee_vertices")}}
    return out


# --------------------------------------------------------------------- E69 ----

def _records(directory):
    records = []
    for shard in sorted(Path(directory).glob("shard_*.jsonl")):
        records += [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]
    return records


def _score(name, pair):
    predicted, true = pair
    if name == "phase_offset_cycles":
        delta = (predicted - true + 0.5) % 1.0 - 0.5
        return abs(delta)
    return abs(predicted - true)


def _one_per_identity(records):
    chosen = {}
    for record in sorted(records, key=lambda r: r["sample"]):
        chosen.setdefault(record["identity"], record)
    return list(chosen.values())


def analyze_e69(val_dir, test_dir, e64_test_dir):
    val, test = _records(val_dir), _records(test_dir)
    calibration = _one_per_identity(val)
    out = {"calibration_identities": len(calibration), "test_samples": len(test),
           "test_identities": len({r["identity"] for r in test}), "quantities": {}}
    for name in ("phase_offset_cycles", "translation_norm_mm", "rotation_angle_deg", "annulus_diameter_mm"):
        scores = np.sort([_score(name, r["scalars"][name]) for r in calibration])
        entry = {}
        for target in CONFORMAL_TARGETS:
            rank = int(np.ceil((len(scores) + 1) * target))
            half = float(scores[rank - 1]) if rank <= len(scores) else float("inf")
            covered = [_score(name, r["scalars"][name]) <= half for r in test]
            identity_covered = [_score(name, r["scalars"][name]) <= half for r in _one_per_identity(test)]
            entry[str(target)] = {"half_width": half, "coverage_samples": float(np.mean(covered)),
                                  "coverage_identities": float(np.mean(identity_covered))}
        out["quantities"][name] = entry
    # Exchangeability probe for the annulus diameter, as E15 reported it.
    diam_val = np.array([r["scalars"]["annulus_diameter_mm"][1] for r in calibration])
    diam_test = np.array([r["scalars"]["annulus_diameter_mm"][1] for r in _one_per_identity(test)])
    res_val = np.array([_score("annulus_diameter_mm", r["scalars"]["annulus_diameter_mm"]) for r in calibration])
    res_test = np.array([_score("annulus_diameter_mm", r["scalars"]["annulus_diameter_mm"])
                         for r in _one_per_identity(test)])
    pooled = np.concatenate([res_val, res_test])
    rng = np.random.default_rng(STAGE.SEED)
    observed = res_test.mean() - res_val.mean()
    null = np.empty(PERMUTATIONS)
    for i in range(PERMUTATIONS):
        permuted = rng.permutation(pooled)
        null[i] = permuted[len(res_val):].mean() - permuted[:len(res_val)].mean()
    out["exchangeability_probe"] = {
        "true_diameter_mean_mm": {"val": float(diam_val.mean()), "test": float(diam_test.mean())},
        "residual_mean_mm": {"val": float(res_val.mean()), "test": float(res_test.mean())},
        "permutation_p": float(np.mean(np.abs(null) >= abs(observed)))}
    # Reproduction of E64's frozen-configuration test fits.
    e64 = STAGE.load_rows([e64_test_dir])
    mine = {r["sample"]: r["errors"] for r in test}
    rep = {}
    for key in ("heldout_cmr_mm", "heldout_tee_mm", "rotation_deg", "translation_mm"):
        a = float(np.median([v[key] for v in mine.values()]))
        b = STAGE.sample_medians(e64, "S/joint")[key]
        rep[key] = {"e69": a, "e64": b, "difference": a - b}
    rep["passed"] = all(abs(rep[k]["difference"]) <= 0.01 for k in ("heldout_cmr_mm", "heldout_tee_mm"))
    out["reproduction_of_e64"] = rep
    return out


# ----------------------------------------------------------------- E70, E71 ----

def analyze_images(directory, arms, reference):
    rows = rows_of(directory)
    keys = ("rotation_deg", "translation_mm", "phase_map_cycles")
    out = {"medians": medians(rows, arms, keys), "against_" + reference: {}}
    for arm in arms:
        if arm == reference:
            continue
        out["against_" + reference][arm] = {k: difference(rows, arm, reference, k) for k in keys}
    return out


# --------------------------------------------------------------------- E72 ----

def analyze_e72(directory):
    rows = rows_of(directory)
    arms = ("full", "position_only", "no_directed_arc", "neither",
            "full_commensurate", "position_only_commensurate")
    out = {}
    for slide in (0, 30, 60, 90):
        entry = {"median_rotation_deg": {}, "recovered_below_5deg": {}, "paired": {}}
        for arm in arms:
            name = f"slide{slide}/{arm}"
            values = [r["variants"][name]["rotation_deg"] for r in rows
                      if (r["variants"].get(name) or {}).get("succeeded")]
            entry["median_rotation_deg"][arm] = float(np.median(values))
            entry["recovered_below_5deg"][arm] = float(np.mean(np.asarray(values) < 5.0))
        for first, second in (("full", "no_directed_arc"), ("full", "position_only"),
                              ("full_commensurate", "no_directed_arc"),
                              ("position_only_commensurate", "neither")):
            entry["paired"][f"{first}_minus_{second}"] = difference(
                rows, f"slide{slide}/{first}", f"slide{slide}/{second}", "rotation_deg")
        out[str(slide)] = entry
    # Per-sample fields that load_rows does not keep.
    records = _records(directory)
    out["normal_fallbacks_median"] = [float(np.median([r["normal_fallbacks"][i] for r in records]))
                                      for i in (0, 1)]
    out["shared_vertices_median"] = float(np.median([r["shared_vertices"] for r in records]))
    return out


def main(e68, e69_val, e69_test, e70, e71, e72, e64_v3):
    report = {"experiments": "E68-E72", "preregistration": UNIFORMITY.PREREGISTRATION,
              "E68": analyze_e68(e68), "E69": analyze_e69(e69_val, e69_test, e64_v3),
              "E70": analyze_images(e70, ("point_fit", "image_cold", "image_warm", "landmarks_2",
                                          "landmarks_4", "landmarks_8"), "image_warm"),
              "E71": analyze_images(e71, ("point_fit", "cmr_image_tee_point", "cmr_point_tee_image",
                                          "both_image"), "point_fit"),
              "E72": analyze_e72(e72)}
    out = Path("results/benchmark/E68_E72_uniformity.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
