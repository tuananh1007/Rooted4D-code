"""E79 and E80: the structured shape space on the real valve (Stage C).

Pre-registered in ``Docs/E79_E80_VALVE_PREREGISTRATION_2026-09-26.md``.

The valve counterpart of E76 and E77. Each labelled valve is encoded in the
fixed topology of ``donors.valve_fixed`` (32 annular nodes, 12 on the anterior
arc and 20 on the posterior, each with a six-row leaflet profile), so that
valves share vertex meaning. S is the structured space with the annulus as the
interface and each leaflet a component (12 ring modes and 12 interior modes per
leaflet); G is principal components at the same rank. Both are learned from the
105 challenge training cases' reference labels. The fit is the same
probabilistic-PCA MAP estimate as the chamber's, so beta = 0 is the direct
encoding.

Predictions come from the frozen fold-0 nnU-Net, which never saw the 30
validation or 40 test cases. Scoring is E42's: each leaflet's sheet against its
own reference label voxels (symmetric distance, six samples per face).

    R4D_MODE=select python e79_valve_shape_fit.py                    E79, validation: select beta
    R4D_MODE=test R4D_ALLOW_TEST=1 python e79_valve_shape_fit.py     E80, the single test evaluation
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PREREGISTRATION = "Docs/E79_E80_VALVE_PREREGISTRATION_2026-09-26.md"
BETAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
BETA_EXTENSION, BETA_LIMIT = 3.0, 1000.0
RING_DIM, INTERIOR_DIM = 12, 12
N_ANTERIOR, N_POSTERIOR, N_DEPTH = 12, 20, 6
RESAMPLES, SEED = 20000, 20260926


def topology():
    from rooted4d.donors.valve_fixed import fixed_valve_topology

    return fixed_valve_topology(N_ANTERIOR, N_POSTERIOR, N_DEPTH)


def encode(labels, zooms, case=""):
    from rooted4d.donors.valve_fixed import fit_valve_fixed

    fit = fit_valve_fixed(labels, zooms, n_anterior=N_ANTERIOR, n_posterior=N_POSTERIOR, n_depth=N_DEPTH, case=case)
    return None if fit is None else fit.vertices()


def load(path: Path):
    import nibabel as nib

    handle = nib.load(str(path))
    return np.asarray(handle.dataobj), tuple(float(z) for z in handle.header.get_zooms()[:3])


def build_spaces(root: Path):
    from rooted4d.experiments.component_shape import fit_global_space, fit_structured_space

    top = topology()
    shapes, used = [], []
    for path in sorted((root / "train").glob("*-label.nii.gz")):
        labels, zooms = load(path)
        vertices = encode(labels, zooms, path.name)
        if vertices is not None:
            shapes.append(vertices)
            used.append(path.name)
    shapes = np.stack(shapes)
    structured = fit_structured_space(shapes, top["faces"], top["components"], top["annulus"],
                                      ring_dim=RING_DIM, interior_dim=INTERIOR_DIM)
    return {"S": structured, "G": fit_global_space(shapes, rank=structured.rank)}, used


def score(vertices, reference, zooms) -> dict:
    """E42's metric: each leaflet's sheet against its own reference voxels."""

    from rooted4d.donors.valve import ANTERIOR_LABEL, POSTERIOR_LABEL
    from rooted4d.experiments.component_eval import symmetric_surface_distance

    top = topology()
    out = {}
    for name, label in (("anterior", ANTERIOR_LABEL), ("posterior", POSTERIOR_LABEL)):
        grid = top[f"{name}_grid"]
        local = np.arange(grid.size).reshape(grid.shape)
        faces = []
        for k in range(grid.shape[0] - 1):
            for d in range(grid.shape[1] - 1):
                faces += [(local[k, d], local[k + 1, d], local[k + 1, d + 1]),
                          (local[k, d], local[k + 1, d + 1], local[k, d + 1])]
        points = np.argwhere(reference == label) * np.asarray(zooms)
        distances = symmetric_surface_distance(points, vertices[grid.ravel()], np.asarray(faces), sample_faces=6)
        out[name] = {"symmetric_mm": distances["symmetric_mean_mm"], "hausdorff_mm": distances["hausdorff_mm"]}
    out["mean_symmetric_mm"] = 0.5 * (out["anterior"]["symmetric_mm"] + out["posterior"]["symmetric_mm"])
    return out


def evaluate(root: Path, split: str, predictions: Path, spaces: dict, betas_by_space: dict):
    from rooted4d.experiments.component_shape import map_estimate

    rows, cache = [], []
    for path in sorted((root / split).glob("*-label.nii.gz")):
        case = path.name.replace("-label.nii.gz", "")
        predicted_path = predictions / f"{case}.nii.gz"
        reference, zooms = load(path)
        row = {"case": case, "fitted": False}
        if not predicted_path.exists():
            row["failure_reason"] = "no prediction"
            rows.append(row)
            continue
        predicted, _ = load(predicted_path)
        direct = encode(predicted, zooms, case)
        if direct is None:
            row["failure_reason"] = "prediction could not be encoded"
            rows.append(row)
            continue
        row.update(fitted=True, direct=score(direct, reference, zooms))
        reference_encoding = encode(reference, zooms, case)
        row["reference_encoding"] = None if reference_encoding is None else score(reference_encoding, reference, zooms)
        cache.append((row, direct, reference, zooms))
        rows.append(row)
        print("  %s direct %.3f" % (case, row["direct"]["mean_symmetric_mm"]), flush=True)
    add_betas(cache, spaces, betas_by_space)
    return rows, cache


def add_betas(cache, spaces, betas_by_space):
    from rooted4d.experiments.component_shape import map_estimate

    for row, vertices, reference, zooms in cache:
        for key, betas in betas_by_space.items():
            for beta in betas:
                row[f"{key}/beta={beta:g}"] = score(map_estimate(vertices, spaces[key], beta, complement=True),
                                                    reference, zooms)


def median_of(rows, arm, key="mean_symmetric_mm"):
    return float(np.median([r[arm][key] for r in rows if r.get("fitted")]))


def select(rows, cache, spaces, key):
    """Lowest median mean-leaflet distance, ties to the smaller beta; the upper edge extended by 3."""

    betas = list(BETAS)
    while True:
        medians = {b: median_of(rows, f"{key}/beta={b:g}") for b in betas}
        best = min(betas, key=lambda b: (round(medians[b], 12), b))
        if best != max(betas) or best * BETA_EXTENSION > BETA_LIMIT:
            break
        extra = [v for v in (best * BETA_EXTENSION, best * BETA_EXTENSION**2) if v <= BETA_LIMIT]
        add_betas(cache, spaces, {key: extra})
        betas += extra
    return {"medians": {f"{b:g}": m for b, m in medians.items()}, "best": best}


def case_difference(rows, first, second, key="mean_symmetric_mm"):
    """Difference of medians over cases, first minus second, resampling cases."""

    pairs = np.array([(r[first][key], r[second][key]) for r in rows if r.get("fitted")])
    rng = np.random.default_rng(SEED)
    observed = float(np.median(pairs[:, 0]) - np.median(pairs[:, 1]))
    index = rng.integers(0, len(pairs), size=(RESAMPLES, len(pairs)))
    draws = np.median(pairs[index, 0], axis=1) - np.median(pairs[index, 1], axis=1)
    return {"observed": observed, "lower": float(np.percentile(draws, 2.5)),
            "upper": float(np.percentile(draws, 97.5)), "cases": int(len(pairs))}


def main() -> int:
    mode = os.environ.get("R4D_MODE", "select")
    root = Path(os.environ.get("R4D_MVSEG", "data/raw/mvseg2023"))
    out_dir = Path(os.environ.get("R4D_OUT", "results/benchmark"))
    started = time.time()
    spaces, used = build_spaces(root)
    print("spaces from %d training cases: S rank %d, G rank %d" % (len(used), spaces["S"].rank, spaces["G"].rank),
          flush=True)
    report = {"preregistration": PREREGISTRATION, "training_cases": len(used),
              "ranks": {k: v.rank for k, v in spaces.items()},
              "residual_scales": {k: v.residual_scale for k, v in spaces.items()},
              "scales": {k: v.scales.tolist() for k, v in spaces.items()}}
    if mode == "select":
        rows, cache = evaluate(root, "val", Path(os.environ["R4D_NNUNET_VAL"]), spaces,
                               {"S": list(BETAS), "G": list(BETAS)})
        selection = {key: select(rows, cache, spaces, key) for key in spaces}
        report.update(experiment="E79", split="val", rows=rows, selection=selection,
                      direct_median_mm=median_of(rows, "direct"),
                      frozen={key: entry["best"] for key, entry in selection.items()})
        out = out_dir / "E79_valve_selection.json"
    elif mode == "test":
        if os.environ.get("R4D_ALLOW_TEST") != "1":
            print("refusing to read the test split without R4D_ALLOW_TEST=1", flush=True)
            return 2
        frozen = json.loads(Path(os.environ["R4D_E79"]).read_text())["frozen"]
        betas = {key: [float(frozen[key])] for key in spaces}
        rows, _ = evaluate(root, "test", Path(os.environ["R4D_NNUNET_TEST"]), spaces, betas)
        arms = {"direct": "direct", **{key: f"{key}/beta={betas[key][0]:g}" for key in spaces}}
        report.update(
            experiment="E80", split="test", frozen=frozen, rows=rows,
            medians={name: {"mean_symmetric_mm": median_of(rows, arm),
                            "anterior_symmetric_mm": float(np.median([r[arm]["anterior"]["symmetric_mm"]
                                                                       for r in rows if r.get("fitted")])),
                            "posterior_symmetric_mm": float(np.median([r[arm]["posterior"]["symmetric_mm"]
                                                                        for r in rows if r.get("fitted")]))}
                     for name, arm in arms.items()},
            paired={f"{a}_minus_{b}": case_difference(rows, arms[a], arms[b])
                    for a, b in (("S", "direct"), ("G", "direct"), ("S", "G"))})
        out = out_dir / "E80_valve_test.json"
    else:
        raise SystemExit(f"unknown mode {mode}")
    report["wall_seconds"] = round(time.time() - started, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
