"""Analysis of E74, by ``Docs/E74_LANDMARKS_COLD_START_PREREGISTRATION_2026-09-25.md``.

    python e74_analysis.py <E74 dir> <E70 dir>

Medians, the counts within 5 degrees and beyond 90, E62's identity statistic
for the pre-registered comparisons (descriptive), and the reproduction of E70's
zero-start arm. E70's rows are read separately and joined by sample, because
both runs contain an ``image_cold`` arm.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYSIS = _module("e68_e72_analysis")
STAGE = ANALYSIS.STAGE
KEYS = ("rotation_deg", "translation_mm", "phase_map_cycles")
COLD = ("image_cold", "landmarks_2_cold", "landmarks_4_cold", "landmarks_8_cold", "commissures_cold")


def main(e74_dir: str, e70_dir: str, out: str = "results/benchmark/E74_landmarks_cold.json") -> int:
    e74 = ANALYSIS._records(e74_dir)  # raw rows keep the sample name, which the join needs
    e70 = {r["sample"]: r for r in ANALYSIS._records(e70_dir)}
    records = {r["sample"]: r for r in e74}
    arms = COLD + ("commissures_warm",)
    report = {"experiment": "E74", "samples": len(e74), "identities": len({r["identity"] for r in e74}),
              "medians": ANALYSIS.medians(e74, arms, KEYS), "counts": {}, "landmarks_used": {}, "paired": {}}
    for arm in arms:
        rotation = np.array([r["variants"][arm]["rotation_deg"] for r in e74
                             if r["variants"][arm].get("succeeded")])
        report["counts"][arm] = {"within_5deg": int((rotation < 5.0).sum()), "beyond_90deg": int((rotation > 90.0).sum()),
                                 "fitted": int(rotation.size), "resolves": bool((rotation < 5.0).sum() > len(e74) / 2)}
        used = {tuple(records[s]["variants"][arm].get("landmarks_used") or []) for s in records}
        report["landmarks_used"][arm] = sorted({str(list(u)) for u in used})
    for arm in COLD[1:]:
        report["paired"][f"{arm}_minus_image_cold"] = {k: ANALYSIS.difference(e74, arm, "image_cold", k) for k in KEYS}
    report["paired"]["commissures_cold_minus_landmarks_2_cold"] = {
        k: ANALYSIS.difference(e74, "commissures_cold", "landmarks_2_cold", k) for k in KEYS}
    # The warm commissure arm against E70's warm arms, joined by sample.
    joined = []
    for row in e74:
        other = e70.get(row["sample"])
        if other is None:
            continue
        variants = {"commissures_warm": row["variants"]["commissures_warm"],
                    "e70_image_warm": other["variants"]["image_warm"],
                    "e70_landmarks_2": other["variants"]["landmarks_2"]}
        joined.append({"sample": row["sample"], "identity": row["identity"], "variants": variants})
    for reference in ("e70_image_warm", "e70_landmarks_2"):
        report["paired"][f"commissures_warm_minus_{reference}"] = {
            k: ANALYSIS.difference(joined, "commissures_warm", reference, k) for k in KEYS}
    # Reproduction of E70's zero-start arm.
    gaps = [abs(r["variants"]["image_cold"]["rotation_deg"] - e70[r["sample"]]["variants"]["image_cold"]["rotation_deg"])
            for r in e74 if r["sample"] in e70]
    report["reproduction_of_e70_image_cold"] = {"samples": len(gaps), "max_abs_deg": float(max(gaps)),
                                                "over_0.01deg": int(sum(g > 0.01 for g in gaps))}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out)
    for arm in arms:
        m, c = report["medians"][arm], report["counts"][arm]
        print("%-18s rot %7.2f  trans %6.2f  phase %.4f  <5deg %2d  >90deg %2d  resolves %s  landmarks %s" % (
            arm, m["rotation_deg"], m["translation_mm"], m["phase_map_cycles"], c["within_5deg"],
            c["beyond_90deg"], c["resolves"], report["landmarks_used"][arm][:3]))
    for key, entry in report["paired"].items():
        d = entry["rotation_deg"]
        print("  %-44s rotation %+.2f [%+.2f, %+.2f]" % (key, d["observed"], d["lower"], d["upper"]))
    print("reproduction:", report["reproduction_of_e70_image_cold"])
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
