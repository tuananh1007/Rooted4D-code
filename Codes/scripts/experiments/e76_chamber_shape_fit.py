"""E76 and E77: the structured shape space on the real chamber (Stage C).

Pre-registered in ``Docs/E76_E77_CHAMBER_PREREGISTRATION_2026-09-26.md``.

The image-input chamber row (E36) resamples the frozen E31 segmenter's
prediction directly into the 8-ring by 32-angle surface. Here that surface is
regularized by a shape space learned from the training patients' reference-label
fits: the structured space S (the basal ring plus the interior residual, 12 + 12
modes on covariance axes) and the global comparator G (principal components at
the same rank). Each is fitted by its closed-form MAP estimate under the
covariance-aware prior, in the probabilistic-PCA form that shrinks the part of
the input outside the span instead of discarding it
(``component_shape.map_estimate(..., complement=True)``), so beta = 0 returns
the direct resampling itself.

    R4D_MODE=select python e76_chamber_shape_fit.py                      E76, validation: select beta
    R4D_MODE=test R4D_ALLOW_TEST=1 python e76_chamber_shape_fit.py       E77, the single test evaluation

The test mode reads the frozen betas from the committed E76 result and refuses
to run without ``R4D_ALLOW_TEST=1``. Its direct arm is E36's pipeline, so it must
reproduce E36's per-frame scores.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PREREGISTRATION = "Docs/E76_E77_CHAMBER_PREREGISTRATION_2026-09-26.md"
BETAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
BETA_EXTENSION = 3.0          # the grid's spacing, for an optimum at its upper edge
BETA_LIMIT = 1000.0
RING_DIM, INTERIOR_DIM = 12, 12
N_RINGS, N_ANGLES, CAVITY = 8, 32, 3
RESAMPLES, SEED = 20000, 20260926


def frames_of(root: Path, splits: dict, split: str) -> list[Path]:
    return [p for p in sorted(root.rglob("*_gt.nii.gz")) if splits.get(p.parent.name) == split]


def reference_fit(path: Path):
    import nibabel as nib

    from rooted4d.active.acdc import fit_lv_surface

    handle = nib.load(str(path))
    labels = np.asarray(handle.dataobj)
    zooms = tuple(float(z) for z in handle.header.get_zooms()[:3])
    frame = int(path.name.split("_frame")[-1].split("_")[0])
    fit = fit_lv_surface(labels, zooms, n_rings=N_RINGS, n_angles=N_ANGLES, patient=path.parent.name,
                         frame=frame, base_first=True)
    return labels, zooms, frame, fit


def build_spaces(root: Path, splits: dict):
    """S and G from the training patients' reference-label fits, both frames each."""

    from rooted4d.experiments.component_shape import chamber_structure, fit_global_space, fit_structured_space

    shapes, used = [], []
    for path in frames_of(root, splits, "train"):
        _, _, frame, fit = reference_fit(path)
        if fit is not None:
            shapes.append(fit.vertices)
            used.append(f"{path.parent.name}/{frame}")
    shapes = np.stack(shapes)
    structured = fit_structured_space(shapes, **chamber_structure(N_RINGS, N_ANGLES),
                                      ring_dim=RING_DIM, interior_dim=INTERIOR_DIM)
    comparator = fit_global_space(shapes, rank=structured.rank)
    return {"S": structured, "G": comparator}, used


class Segmenter:
    """The frozen E31 network, applied exactly as E31 and E36 apply it."""

    def __init__(self, checkpoint: Path):
        import torch

        from rooted4d.segmentation import CompactUNet

        torch.manual_seed(0)
        torch.set_num_threads(int(os.environ.get("R4D_THREADS", 8)))
        self.net = CompactUNet()
        state = torch.load(checkpoint, map_location="cpu")
        self.net.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        self.net.eval()

    def __call__(self, image_path: Path, reference_shape, spacing):
        """E36's ``predict_volume``, unchanged: segment each slice and return it on the original grid."""

        import nibabel as nib
        import torch
        import torch.nn.functional as functional

        from rooted4d.segmentation.data import PATCH, TARGET_SPACING_MM

        volume = np.asarray(nib.load(str(image_path)).dataobj, dtype=np.float32)
        output = np.zeros(reference_shape, dtype=np.int64)
        for index in range(volume.shape[2]):
            plane = volume[:, :, index]
            finite = plane[np.isfinite(plane)]
            low, high = np.percentile(finite, [1.0, 99.0]) if finite.size else (0.0, 1.0)
            plane = np.clip((plane - low) / max(high - low, 1e-6), 0.0, 1.0)
            tensor = torch.as_tensor(plane, dtype=torch.float32)[None, None]
            scale = (spacing[0] / TARGET_SPACING_MM, spacing[1] / TARGET_SPACING_MM)
            size = (max(1, int(round(plane.shape[0] * scale[0]))),
                    max(1, int(round(plane.shape[1] * scale[1]))))
            resampled = functional.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
            canvas = torch.zeros((1, 1, PATCH, PATCH))
            rows_, cols = min(PATCH, size[0]), min(PATCH, size[1])
            sr, sc = (size[0] - rows_) // 2, (size[1] - cols) // 2
            dr, dc = (PATCH - rows_) // 2, (PATCH - cols) // 2
            canvas[0, 0, dr:dr + rows_, dc:dc + cols] = resampled[0, 0, sr:sr + rows_, sc:sc + cols]
            with torch.no_grad():
                logits = self.net(canvas)
            patch = torch.zeros((1, 4, size[0], size[1]))
            patch[0, :, sr:sr + rows_, sc:sc + cols] = logits[0, :, dr:dr + rows_, dc:dc + cols]
            back = functional.interpolate(patch, size=plane.shape, mode="bilinear", align_corners=False)
            output[:, :, index] = back.argmax(1)[0].numpy()
        return output


def score(points, vertices, faces) -> dict:
    from rooted4d.experiments.component_eval import symmetric_surface_distance
    from rooted4d.metrics.topology import signed_enclosed_volume

    distances = symmetric_surface_distance(points, vertices, faces)
    return {"symmetric_mm": distances["symmetric_mean_mm"], "hausdorff_mm": distances["hausdorff_mm"],
            "volume_ml": abs(signed_enclosed_volume(vertices, faces)) / 1000.0}


def evaluate(root: Path, splits: dict, split: str, spaces: dict, betas_by_space: dict, segmenter):
    """Score every frame of a split; returns the rows and, per fitted frame, what later betas need."""

    from rooted4d.active.acdc import fit_lv_surface, lv_topology_faces
    from rooted4d.experiments.component_eval import cavity_boundary_points

    faces = lv_topology_faces(N_RINGS, N_ANGLES)
    rows, cache = [], []
    for path in frames_of(root, splits, split):
        image_path = Path(str(path).replace("_gt.nii.gz", ".nii.gz"))
        if not image_path.exists():
            continue
        truth, zooms, frame, truth_fit = reference_fit(path)
        predicted = segmenter(image_path, truth.shape, zooms[:2])
        direct = fit_lv_surface(predicted, zooms, n_rings=N_RINGS, n_angles=N_ANGLES,
                                patient=path.parent.name, frame=frame, base_first=True)
        tested = fit_lv_surface(predicted, zooms, n_rings=N_RINGS, n_angles=N_ANGLES)
        row = {"patient": path.parent.name, "frame": frame, "fitted": truth_fit is not None and direct is not None}
        if direct is not None:
            # Whether E36's radius test would have agreed on this prediction; where it
            # does not, E36 capped the ends the other way round and cannot be reproduced.
            row["radius_test_agrees"] = bool(np.array_equal(tested.vertices, direct.vertices))
        if not row["fitted"]:
            rows.append(row)
            continue
        points = cavity_boundary_points(truth, zooms, cavity_label=CAVITY)
        row["reference_volume_ml"] = score(points, truth_fit.vertices, faces)["volume_ml"]
        row["direct"] = score(points, direct.vertices, faces)
        cache.append((row, points, direct.vertices))
        add_betas(cache[-1:], spaces, betas_by_space, faces)
        rows.append(row)
        print("  %s frame %d  direct %.3f" % (row["patient"], frame, row["direct"]["symmetric_mm"]), flush=True)
    return rows, cache


def add_betas(cache, spaces, betas_by_space, faces=None):
    """Score MAP fits at the given betas on already-predicted frames."""

    from rooted4d.active.acdc import lv_topology_faces
    from rooted4d.experiments.component_shape import map_estimate

    faces = lv_topology_faces(N_RINGS, N_ANGLES) if faces is None else faces
    for row, points, vertices in cache:
        for key, betas in betas_by_space.items():
            for beta in betas:
                row[f"{key}/beta={beta:g}"] = score(points, map_estimate(vertices, spaces[key], beta, complement=True),
                                                     faces)


def median_of(rows, arm, key="symmetric_mm"):
    return float(np.median([r[arm][key] for r in rows if r.get("fitted")]))


def select(rows, cache, spaces, key):
    """Lowest median symmetric distance, ties to the smaller beta; an optimum at the upper edge
    is extended by the grid's factor until two further values are worse, up to the limit."""

    betas = list(BETAS)
    while True:
        medians = {beta: median_of(rows, f"{key}/beta={beta:g}") for beta in betas}
        best = min(betas, key=lambda b: (round(medians[b], 12), b))
        if best != max(betas) or best * BETA_EXTENSION > BETA_LIMIT:
            break
        extra = [v for v in (best * BETA_EXTENSION, best * BETA_EXTENSION**2) if v <= BETA_LIMIT]
        add_betas(cache, spaces, {key: extra})
        betas += extra
    return {"medians": {f"{b:g}": m for b, m in medians.items()}, "best": best}


def patient_difference(rows, first, second, key="symmetric_mm"):
    """Difference of medians over frames, first minus second, resampling whole patients."""

    by_patient = {}
    for r in rows:
        if r.get("fitted"):
            by_patient.setdefault(r["patient"], []).append((r[first][key], r[second][key]))
    patients = sorted(by_patient)
    rng = np.random.default_rng(SEED)
    def statistic(chosen):
        pairs = [p for c in chosen for p in by_patient[c]]
        return float(np.median([a for a, _ in pairs]) - np.median([b for _, b in pairs]))
    observed = statistic(patients)
    draws = np.array([statistic([patients[i] for i in rng.integers(0, len(patients), len(patients))])
                      for _ in range(RESAMPLES)])
    return {"observed": observed, "lower": float(np.percentile(draws, 2.5)),
            "upper": float(np.percentile(draws, 97.5)), "patients": len(patients)}


def ejection_fraction_agreement(rows, arm):
    """EF from each fitter's two surfaces against EF from the reference-label fits."""

    by_patient = {}
    for r in rows:
        if r.get("fitted"):
            by_patient.setdefault(r["patient"], []).append(r)
    errors = []
    for frames in by_patient.values():
        if len(frames) != 2:
            continue
        ref = sorted(f["reference_volume_ml"] for f in frames)
        fit = sorted(f[arm]["volume_ml"] for f in frames)
        errors.append(100.0 * (fit[1] - fit[0]) / fit[1] - 100.0 * (ref[1] - ref[0]) / ref[1])
    errors = np.asarray(errors)
    return {"patients": int(errors.size), "bias": float(errors.mean()),
            "median_absolute": float(np.median(np.abs(errors))),
            "limits_of_agreement": [float(errors.mean() - 1.96 * errors.std(ddof=1)),
                                    float(errors.mean() + 1.96 * errors.std(ddof=1))]}


def main() -> int:
    from rooted4d.segmentation import load_patient_splits

    mode = os.environ.get("R4D_MODE", "select")
    root = Path(os.environ.get("R4D_ACDC", "data/raw/acdc"))
    splits = load_patient_splits(os.environ.get("R4D_SPLITS", "results/benchmark/acdc_patient_splits.json"))
    checkpoint = Path(os.environ.get("R4D_SEGMENTER", "results/21900728/E31_segmenter.pt"))
    out_dir = Path(os.environ.get("R4D_OUT", "results/benchmark"))
    started = time.time()
    spaces, used = build_spaces(root, splits)
    print("spaces from %d training frames: S rank %d, G rank %d" % (len(used), spaces["S"].rank, spaces["G"].rank),
          flush=True)
    segmenter = Segmenter(checkpoint)
    report = {"preregistration": PREREGISTRATION, "checkpoint": str(checkpoint), "training_frames": len(used),
              "ranks": {k: v.rank for k, v in spaces.items()},
              "residual_scales": {k: v.residual_scale for k, v in spaces.items()},
              "scales": {k: v.scales.tolist() for k, v in spaces.items()}}
    if mode == "select":
        betas = {"S": list(BETAS), "G": list(BETAS)}
        rows, cache = evaluate(root, splits, "val", spaces, betas, segmenter)
        selection = {key: select(rows, cache, spaces, key) for key in spaces}
        report.update(experiment="E76", split="val", rows=rows, selection=selection,
                      direct_median_mm=median_of(rows, "direct"),
                      frozen={key: entry["best"] for key, entry in selection.items()})
        out = out_dir / "E76_chamber_selection.json"
    elif mode == "test":
        if os.environ.get("R4D_ALLOW_TEST") != "1":
            print("refusing to read the test split without R4D_ALLOW_TEST=1", flush=True)
            return 2
        frozen = json.loads(Path(os.environ.get("R4D_E76", "results/benchmark/E76_chamber_selection.json"))
                            .read_text())["frozen"]
        betas = {key: [float(frozen[key])] for key in spaces}
        rows, _ = evaluate(root, splits, "test", spaces, betas, segmenter)
        arms = {"direct": "direct", **{key: f"{key}/beta={betas[key][0]:g}" for key in spaces}}
        report.update(
            experiment="E77", split="test", frozen=frozen, rows=rows,
            medians={name: {k: median_of(rows, arm, k) for k in ("symmetric_mm", "hausdorff_mm")}
                     for name, arm in arms.items()},
            paired={f"{a}_minus_{b}": {k: patient_difference(rows, arms[a], arms[b], k)
                                       for k in ("symmetric_mm", "hausdorff_mm")}
                    for a, b in (("S", "direct"), ("G", "direct"), ("S", "G"))},
            ejection_fraction={name: ejection_fraction_agreement(rows, arm) for name, arm in arms.items()})
        out = out_dir / "E77_chamber_test.json"
    else:
        raise SystemExit(f"unknown mode {mode}")
    report["wall_seconds"] = round(time.time() - started, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
