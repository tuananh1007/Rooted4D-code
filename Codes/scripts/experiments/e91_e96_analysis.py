"""E91 and E96 analyses, as pre-registered.

``Docs/E91_IMAGE_SEARCH_PREREGISTRATION_2026-09-28.md`` and
``Docs/E96_COMMISSURE_PRECISION_PREREGISTRATION_2026-09-29.md``: medians,
fractions within 5 degrees and in the wrong basin (above 90), and paired
identity-level differences by E62's statistic (identity-averaged, difference of
medians, 20,000 resamples, seed 20260910); for E96 also where the commissures
stop placing the rotation and whether the objective detects a swap.

    python e91_e96_analysis.py      -> results/benchmark/E91/E91_analysis.json, results/benchmark/E96/E96_analysis.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

KEYS = ("rotation_deg", "translation_mm", "phase_map_cycles")
RESAMPLES, SEED = 20000, 20260910


def _rows(directory: Path) -> list:
    return [json.loads(line) for shard in sorted(directory.glob("shard_*.jsonl"))
            for line in shard.read_text().splitlines() if line.strip()]


def summary(rows, arm) -> dict:
    outcomes = [r["variants"][arm] for r in rows if arm in r["variants"]]
    ok = [o for o in outcomes if o.get("succeeded")]
    rotation = np.array([o["rotation_deg"] for o in ok])
    out = {k: float(np.median([o[k] for o in ok])) for k in KEYS}
    out.update(samples=len(outcomes), failures=len(outcomes) - len(ok),
               within_5deg=float(np.mean(rotation <= 5.0)), wrong_basin=float(np.mean(rotation > 90.0)))
    return out


def paired(rows, first, second, key) -> dict:
    identities = sorted({r["identity"] for r in rows})

    def per_identity(arm):
        return np.array([np.nanmean([r["variants"][arm][key] if r["variants"][arm].get("succeeded") else np.nan
                                     for r in rows if r["identity"] == i]) for i in identities])

    a, b = per_identity(first), per_identity(second)
    rng = np.random.default_rng(SEED)
    draws = np.empty(RESAMPLES)
    for r in range(RESAMPLES):
        t = rng.integers(0, len(a), size=len(a))
        draws[r] = np.nanmedian(a[t]) - np.nanmedian(b[t])
    return {"observed": float(np.nanmedian(a) - np.nanmedian(b)),
            "lower": float(np.percentile(draws, 2.5)), "upper": float(np.percentile(draws, 97.5))}


def e91(directories: dict | None = None) -> dict:
    directories = directories or {b: Path(f"results/benchmark/E91/E91_{b}_test") for b in ("v3", "v4")}
    report = {"experiment": "E91", "preregistration": "Docs/E91_IMAGE_SEARCH_PREREGISTRATION_2026-09-28.md",
              "benchmarks": {}}
    for bench, directory in directories.items():
        rows = _rows(Path(directory))
        arms = ("image_search", "commissures_search", "image_uninformed", "commissures_uninformed")
        report["benchmarks"][bench] = {
            "samples": len(rows),
            "arms": {arm: summary(rows, arm) for arm in arms},
            "search_minus_uninformed": {
                evidence: {k: paired(rows, f"{evidence}_search", f"{evidence}_uninformed", k) for k in KEYS}
                for evidence in ("image", "commissures")},
        }
    return report


def e96(directory: str = "results/benchmark/E96/E96_v3_test") -> dict:
    rows = _rows(Path(directory))
    arms = sorted({a for r in rows for a in r["variants"]})
    report = {"experiment": "E96", "preregistration": "Docs/E96_COMMISSURE_PRECISION_PREREGISTRATION_2026-09-29.md",
              "samples": len(rows), "arms": {arm: summary(rows, arm) for arm in arms},
              "against_zero_error": {arm: {k: paired(rows, arm, "noise_0mm", k) for k in KEYS}
                                     for arm in arms if arm != "noise_0mm"}}
    base = report["arms"]["noise_0mm"]["within_5deg"]
    stops = [arm for arm in ("noise_1mm", "noise_2mm", "noise_4mm", "noise_8mm")
             if report["arms"][arm]["within_5deg"] < 0.5 * base]
    report["stops_placing_rotation_at"] = stops[0] if stops else None
    report["commissure_search_places_rotation"] = bool(base >= 0.5)
    swapped = [(r["variants"]["swap"].get("final_objective"), r["variants"]["noise_0mm"].get("final_objective"))
               for r in rows if r["variants"]["swap"].get("succeeded") and r["variants"]["noise_0mm"].get("succeeded")]
    detected = [s > c for s, c in swapped]
    report["swap_detection"] = {"samples": len(detected), "fraction_detected": float(np.mean(detected)),
                                "detected_by_rule": bool(np.mean(detected) >= 0.9)}
    return report


def main() -> int:
    for name, report in (("E91", e91()), ("E96", e96())):
        out = Path(f"results/benchmark/{name}/{name}_analysis.json")
        out.write_text(json.dumps(report, indent=2) + "\n")
        print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
