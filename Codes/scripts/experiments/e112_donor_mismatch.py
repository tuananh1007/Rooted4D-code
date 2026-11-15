"""E112: does a composite's donor mismatch drive the interface gap or the held-out difference?

Pre-registered as exploratory in
``Docs/E110_E113_MARGIN_AND_COHORT_PREREGISTRATION_2026-10-21.md``. For every
test identity of the realistic-session primary and its five split replications,
the mismatch between its two donors is measured from the donors themselves,
exactly as ``combine_donors`` carries the valve onto the chamber's ring: the
similarity scale, and the RMS residual of the valve's ring after that transform
against the chamber's ring. Each is related to the identity's outcomes in the
stored test rows by Spearman correlation with identity-bootstrap intervals,
and by tertiles of the ring residual.

    python e112_donor_mismatch.py <results root>   -> results/benchmark/E112/E112_donor_mismatch.json

Runs where the benchmarks and the donor pool are (``data/derived``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREREGISTRATION = "Docs/E110_E113_MARGIN_AND_COHORT_PREREGISTRATION_2026-10-21.md"
KEYS = ["v4n96o"] + [f"v4n96os{k}" for k in range(1, 6)]
DONORS = Path("data/derived/donors_v3_n96")
N_ANNULUS = 96
JOINT, LATE = "S/joint", "S/late_fusion_trajectory"
RESAMPLES, SEED = 20000, 20260910
OUT = Path("results/benchmark/E112/E112_donor_mismatch.json")
MISMATCH = ("scale", "log_scale_abs", "ring_residual_rms_mm")
OUTCOMES = ("late_fusion_gap_mm", "rooted4d_sum_mm", "late_fusion_sum_mm", "difference_sum_mm",
            "rooted4d_rotation_deg", "late_fusion_rotation_deg")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stored_donor(donor_id: str) -> np.ndarray:
    kind = "heart" if "heart" in donor_id else "valve"
    return np.load(DONORS / "fixed_topology" / f"{kind}-{donor_id.rsplit('-', 1)[-1]}.npz")["vertices"]


def mismatch(heart_id: str, valve_id: str) -> dict:
    """The similarity ``combine_donors`` applies (oriented, as E100 builds), and what it leaves."""

    from rooted4d.geometry import make_toy_complex
    from rooted4d.synthetic.donors import _ring_handedness, _similarity_transform, oriented_valve_vertices

    heart, valve = _stored_donor(heart_id), _stored_donor(valve_id)
    complex_ = make_toy_complex(n_annulus=N_ANNULUS)
    annulus, free = complex_.groups["annulus"], complex_.groups["leaflet_free"]
    chamber = np.setdiff1d(complex_.groups["cmr_visible"], annulus)
    reindexed = _ring_handedness(heart, annulus, chamber) != _ring_handedness(valve, annulus, free)
    valve = oriented_valve_vertices(heart, valve, annulus, chamber, free)
    rotation, scale = _similarity_transform(valve[annulus], heart[annulus])
    carried = scale * ((valve[annulus] - valve[annulus].mean(axis=0)) @ rotation.T) + heart[annulus].mean(axis=0)
    residual = float(np.sqrt(((carried - heart[annulus]) ** 2).sum(axis=1).mean()))
    perimeter = lambda ring: float(np.linalg.norm(np.diff(np.vstack([ring, ring[:1]]), axis=0), axis=1).sum())
    return {"scale": float(scale), "log_scale_abs": float(abs(np.log(scale))), "ring_residual_rms_mm": residual,
            "reindexed": bool(reindexed), "chamber_ring_perimeter_mm": perimeter(heart[annulus]),
            "valve_ring_perimeter_mm": perimeter(valve[annulus])}


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    return float(np.corrcoef(_ranks(x[keep]), _ranks(y[keep]))[0, 1])


def spearman_interval(x: np.ndarray, y: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    draws = np.empty(RESAMPLES)
    for r in range(RESAMPLES):
        t = rng.integers(0, len(x), size=len(x))
        draws[r] = spearman(x[t], y[t])
    return {"rho": spearman(x, y), "lower": float(np.nanpercentile(draws, 2.5)),
            "upper": float(np.nanpercentile(draws, 97.5)), "n": int(len(x))}


def identities(key: str, results: Path, stage) -> list[dict]:
    """One record per test identity: donors, mismatch, outcomes (variants averaged)."""

    benchmark = Path(f"data/derived/e100-{key}")
    donors = {}
    for line in (benchmark / "manifests" / "test.jsonl").read_text().splitlines():
        record = json.loads(line)
        donors[record["identity_id"]] = (record["whole_heart_donor_id"], record["valve_donor_id"])
    done = results / "e100" / key / "TEST_DONE"
    rows = stage.load_rows([results / "e100" / key / done.read_text().strip()])
    by_identity: dict[str, list] = {}
    for row in rows.values() if isinstance(rows, dict) else rows:
        by_identity.setdefault(row["identity"], []).append(row["variants"])
    out = []
    for identity in sorted(by_identity):
        heart, valve = donors[identity]
        record = {"instance": key, "identity": identity, "whole_heart_donor_id": heart, "valve_donor_id": valve,
                  **mismatch(heart, valve)}
        values = {name: [] for name in OUTCOMES}
        for variants in by_identity[identity]:
            joint, late = variants.get(JOINT) or {}, variants.get(LATE) or {}
            ok_j, ok_l = joint.get("succeeded"), late.get("succeeded")
            j_sum = joint["heldout_cmr_mm"] + joint["heldout_tee_mm"] if ok_j else np.nan
            l_sum = late["heldout_cmr_mm"] + late["heldout_tee_mm"] if ok_l else np.nan
            values["late_fusion_gap_mm"].append(late["attachment_residual_mm"] if ok_l else np.nan)
            values["rooted4d_sum_mm"].append(j_sum)
            values["late_fusion_sum_mm"].append(l_sum)
            values["difference_sum_mm"].append(j_sum - l_sum)
            values["rooted4d_rotation_deg"].append(joint["rotation_deg"] if ok_j else np.nan)
            values["late_fusion_rotation_deg"].append(late["rotation_deg"] if ok_l else np.nan)
        record.update({name: float(np.mean(v)) for name, v in values.items()})
        record["samples"] = len(by_identity[identity])
        out.append(record)
    return out


def summarize(records: list[dict]) -> dict:
    def table(subset):
        out = {"identities": len(subset), "spearman": {}}
        for m in MISMATCH:
            x = np.array([r[m] for r in subset])
            out["spearman"][m] = {o: spearman_interval(x, np.array([r[o] for r in subset])) for o in OUTCOMES}
        residual = np.array([r["ring_residual_rms_mm"] for r in subset])
        edges = np.percentile(residual, [100 / 3, 200 / 3])
        tertile = np.digitize(residual, edges)
        out["tertiles_by_ring_residual"] = []
        for k in range(3):
            members = [r for r, t in zip(subset, tertile) if t == k]
            out["tertiles_by_ring_residual"].append({
                "identities": len(members),
                "ring_residual_rms_mm": [float(min(r["ring_residual_rms_mm"] for r in members)),
                                         float(max(r["ring_residual_rms_mm"] for r in members))],
                "median_scale": float(np.median([r["scale"] for r in members])),
                **{o: float(np.nanmedian([r[o] for r in members])) for o in OUTCOMES}})
        out["mismatch_medians"] = {m: float(np.median([r[m] for r in subset])) for m in MISMATCH}
        out["mismatch_ranges"] = {m: [float(min(r[m] for r in subset)), float(max(r[m] for r in subset))]
                                  for m in MISMATCH}
        out["reindexed_valves"] = int(sum(r["reindexed"] for r in subset))
        return out

    return {"pooled": table(records),
            "per_instance": {key: table([r for r in records if r["instance"] == key]) for key in KEYS
                             if any(r["instance"] == key for r in records)}}


def main(root: str) -> int:
    stage = _load("e62_stage_b")
    records = []
    for key in KEYS:
        if not (Path(root) / "e100" / key / "TEST_DONE").exists():
            print(f"{key}: no test rows, skipped")
            continue
        records += identities(key, Path(root), stage)
        print(f"{key}: {sum(r['instance'] == key for r in records)} identities")
    report = {"experiment": "E112", "preregistration": PREREGISTRATION, "kind": "exploratory",
              "arms": {"rooted4d": JOINT, "late_fusion": LATE}, "mismatch_measures": MISMATCH,
              "outcomes": OUTCOMES, "resamples": RESAMPLES, "seed": SEED,
              "summary": summarize(records), "identities": records}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    pooled = report["summary"]["pooled"]
    print("pooled identities:", pooled["identities"], "| mismatch medians:", pooled["mismatch_medians"])
    for m in MISMATCH:
        for o in OUTCOMES:
            s = pooled["spearman"][m][o]
            print("  rho(%s, %s) = %+.2f [%+.2f, %+.2f]" % (m, o, s["rho"], s["lower"], s["upper"]))
    for t in pooled["tertiles_by_ring_residual"]:
        print("  tertile residual %.2f-%.2f mm: gap %.2f, R4D sum %.2f, LF sum %.2f, diff %+.3f" % (
            *t["ring_residual_rms_mm"], t["late_fusion_gap_mm"], t["rooted4d_sum_mm"], t["late_fusion_sum_mm"],
            t["difference_sum_mm"]))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
