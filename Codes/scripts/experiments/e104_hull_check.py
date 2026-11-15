"""E104: the true free edge against the chamber hull, coarse and fine (weak point 1a).

Pre-registered in ``Docs/E103_E107_REVIEWER_ANALYSES_PREREGISTRATION_2026-10-03.md``.
On the test samples of E102's richer instance (six chamber rings), the true
shared anatomy at end systole is tested against two convex hulls of its own
chamber: the coarse hull (annulus, the fourth level, which is the ring the
reduced instance carries, and the apex) and the fine hull (annulus, all six
levels, apex). The reduced instance ``v3o`` (same donors, identities and seeds)
is counted with its own coarse hull, as the link to the manuscript's figure.

    python e104_hull_check.py <richer instance root> <reduced instance root>
        -> results/benchmark/E104/E104_hull.json

Needs SciPy (Delaunay and ConvexHull, as E90).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PREREGISTRATION = "Docs/E103_E107_REVIEWER_ANALYSES_PREREGISTRATION_2026-10-03.md"
OUT = Path("results/benchmark/E104")
PHASE_GRID = 64
#: The reduced instance's lower ring is the chamber fit's ring round(7/2) = 4, which
#: the richer instance carries as its fourth level (rings 1..6 of the same fit).
COARSE_LEVEL = 3


def _groups(sample) -> dict:
    from rooted4d.geometry import make_toy_complex, sample_topology

    n_vertices = len(sample["template_vertices"])
    topology = sample_topology(sample)
    levels, rows = topology.get("chamber_levels", 1), topology.get("leaflet_rows", 1)
    n = (n_vertices - 1) // (1 + levels + rows)
    complex_ = make_toy_complex(n, levels, rows)
    assert len(complex_.global_vertices) == n_vertices, (n_vertices, n, levels, rows)
    level_ids = [np.arange((1 + k) * n, (2 + k) * n) for k in range(levels)]
    apex = np.array([(1 + levels) * n])
    first_row = int(apex[0]) + 1
    free = np.arange(first_row + (rows - 1) * n, first_row + rows * n)
    return {"n": n, "levels": levels, "annulus": np.arange(n), "level_ids": level_ids,
            "apex": apex, "free": free}


def _point_triangle_distance(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest of the triangles, exact (Ericson's closest point)."""

    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    best = np.full(len(points), np.inf)
    for i, p in enumerate(points):
        ab, ac, ap = b - a, c - a, p - a
        d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
        bp = p - b
        d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
        cp = p - c
        d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
        va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
        closest = np.empty_like(a)
        denominator = va + vb + vc
        safe = np.where(np.abs(denominator) > 1e-15, denominator, 1.0)
        v, w = vb / safe, vc / safe
        closest[:] = a + ab * v[:, None] + ac * w[:, None]          # face region
        region = (d1 <= 0) & (d2 <= 0)
        closest[region] = a[region]
        region = (d3 >= 0) & (d4 <= d3)
        closest[region] = b[region]
        region = (d6 >= 0) & (d5 <= d6)
        closest[region] = c[region]
        region = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        t = d1 / np.where(np.abs(d1 - d3) > 1e-15, d1 - d3, 1.0)
        closest[region] = (a + ab * t[:, None])[region]
        region = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        t = d2 / np.where(np.abs(d2 - d6) > 1e-15, d2 - d6, 1.0)
        closest[region] = (a + ac * t[:, None])[region]
        region = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        t = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) > 1e-15, (d4 - d3) + (d5 - d6), 1.0)
        closest[region] = (b + (c - b) * t[:, None])[region]
        best[i] = np.linalg.norm(closest - p, axis=1).min()
    return best


def hull_record(chamber: np.ndarray, free: np.ndarray) -> dict:
    from scipy.spatial import ConvexHull, Delaunay

    outside = Delaunay(chamber).find_simplex(free) < 0
    record = {"outside_vertices": int(outside.sum()), "free_vertices": int(len(free))}
    if outside.any():
        hull = ConvexHull(chamber)
        depths = _point_triangle_distance(free[outside], chamber[hull.simplices])
        record["max_depth_mm"] = float(depths.max())
        record["depths_mm"] = depths.tolist()
    return record


def measure(sample) -> dict:
    from scipy.spatial import ConvexHull

    from rooted4d.experiments.fitted_geometry import truth_trajectory

    groups = _groups(sample)
    grid = np.linspace(0.0, 1.0, PHASE_GRID, endpoint=False)
    frames = truth_trajectory(sample)(grid)
    fine_ids = np.concatenate([groups["annulus"], *groups["level_ids"], groups["apex"]])
    volumes = [ConvexHull(frame[fine_ids]).volume for frame in frames]
    tau = int(np.argmin(volumes))
    geometry = frames[tau]
    free = geometry[groups["free"]]
    out = {"tau_es": float(grid[tau]), "levels": groups["levels"]}
    if groups["levels"] == 1:
        coarse_ids = fine_ids
    else:
        coarse_ids = np.concatenate([groups["annulus"], groups["level_ids"][COARSE_LEVEL], groups["apex"]])
        out["fine"] = hull_record(geometry[fine_ids], free)
    out["coarse"] = hull_record(geometry[coarse_ids], free)
    return out


def summarize(records: list, hull: str) -> dict:
    present = [r[hull] for r in records if hull in r]
    depths = [d for r in present for d in r.get("depths_mm", [])]
    return {"samples": len(present),
            "samples_outside": int(sum(r["outside_vertices"] > 0 for r in present)),
            "vertex_fraction_outside": float(sum(r["outside_vertices"] for r in present)
                                             / sum(r["free_vertices"] for r in present)),
            "median_depth_mm": float(np.median(depths)) if depths else 0.0,
            "p90_depth_mm": float(np.percentile(depths, 90)) if depths else 0.0}


def main(rich_root: str, reduced_root: str) -> None:
    report = {"experiment": "E104", "preregistration": PREREGISTRATION, "instances": {}}
    for name, root in (("richer", rich_root), ("reduced", reduced_root)):
        records = []
        for path in sorted((Path(root) / "samples" / "test").glob("*.npz")):
            with np.load(path, allow_pickle=True) as blob:
                sample = {k: blob[k] for k in blob.files}
            record = measure(sample)
            record["sample"] = path.name
            records.append(record)
        entry = {"root": root, "coarse": summarize(records, "coarse")}
        if any("fine" in r for r in records):
            entry["fine"] = summarize(records, "fine")
        entry["per_sample"] = [{k: v for k, v in r.items()} for r in records]
        report["instances"][name] = entry
        print(name, {k: v for k, v in entry.items() if k in ("coarse", "fine")}, flush=True)
    richer = report["instances"]["richer"]
    coarse, fine = richer["coarse"]["samples_outside"], richer["fine"]["samples_outside"]
    report["reading"] = ("the coarse hull accounts for it" if fine <= coarse / 2
                         else "the composites place part of the free edge outside the chamber")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "E104_hull.json").write_text(json.dumps(report, indent=2) + "\n")
    print("reading:", report["reading"])


if __name__ == "__main__":
    main(*sys.argv[1:3])
