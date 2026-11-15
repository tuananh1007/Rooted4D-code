"""E86: a connected-component constraint on the predicted cavity before the chamber fit.

Pre-registered in ``Docs/E86_CAVITY_COMPONENTS_PREREGISTRATION_2026-09-28.md``.
Everything except the constraint is E84's: the two front ends, the direct surface
fit, the scoring, and the ejection-fraction definition.

    R4D_MODE=select python e86_cavity_components.py                   validation: one candidate per front end
    R4D_MODE=test R4D_ALLOW_TEST=1 python e86_cavity_components.py    the single test evaluation

Environment: R4D_SPLITS, R4D_SEGMENTER (E31), R4D_NNUNET_PRED (nnU-Net labels for
the split), R4D_OUT; for ``test`` also R4D_SELECTION and R4D_E84 (E84's result file).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PREREGISTRATION = "Docs/E86_CAVITY_COMPONENTS_PREREGISTRATION_2026-09-28.md"
CANDIDATES = ("none", "largest_3d", "largest_3d_slice")
FRONT_ENDS = ("compact_unet", "nnunet")
CAVITY = 3
RESAMPLES, SEED = 20000, 20260929
SURFACE_TOLERANCE_MM, TIE = 0.01, 1e-9


def constrain(candidate: str, labels: np.ndarray) -> np.ndarray:
    from rooted4d.segmentation.cleanup import keep_largest_cavity_component

    if candidate == "none":
        return labels
    return keep_largest_cavity_component(labels, cavity_label=CAVITY, per_slice=candidate == "largest_3d_slice")


def score_split(split: str, candidates: dict) -> list:
    """Every frame of a split, through each front end and each of its candidates."""

    import nibabel as nib

    import e76_chamber_shape_fit as stage_c
    from rooted4d.active.acdc import fit_lv_surface, lv_topology_faces
    from rooted4d.experiments.component_eval import cavity_boundary_points

    root = Path(os.environ.get("R4D_ACDC", "data/raw/acdc"))
    splits = json.loads(Path(os.environ["R4D_SPLITS"]).read_text())
    predictions = Path(os.environ["R4D_NNUNET_PRED"])
    segmenter = stage_c.Segmenter(Path(os.environ["R4D_SEGMENTER"]))
    faces = lv_topology_faces(stage_c.N_RINGS, stage_c.N_ANGLES)
    rows = []
    for path in stage_c.frames_of(root, splits, split):
        truth, zooms, frame, truth_fit = stage_c.reference_fit(path)
        case = path.name.replace("_gt.nii.gz", "")
        image_path = Path(str(path).replace("_gt.nii.gz", ".nii.gz"))
        points = cavity_boundary_points(truth, zooms, cavity_label=CAVITY)
        row = {"patient": path.parent.name, "frame": frame, "fitted": truth_fit is not None}
        if truth_fit is not None:
            row["reference_volume_ml"] = stage_c.score(points, truth_fit.vertices, faces)["volume_ml"]
        predicted = {"compact_unet": segmenter(image_path, truth.shape, zooms[:2]),
                     "nnunet": np.asarray(nib.load(str(predictions / f"{case}.nii.gz")).dataobj)}
        for front, labels in predicted.items():
            assert labels.shape == truth.shape, (front, labels.shape, truth.shape)
            for candidate in candidates[front]:
                cleaned = constrain(candidate, labels)
                overlap = np.logical_and(cleaned == CAVITY, truth == CAVITY).sum()
                dice = 2.0 * overlap / max(int((cleaned == CAVITY).sum() + (truth == CAVITY).sum()), 1)
                fit = fit_lv_surface(cleaned, zooms, n_rings=stage_c.N_RINGS, n_angles=stage_c.N_ANGLES,
                                     patient=path.parent.name, frame=frame, base_first=True)
                entry = {"removed_voxels": int(np.sum((labels == CAVITY) & (cleaned != CAVITY))),
                         "dice_lv_cavity": float(dice), "fitted": fit is not None}
                if fit is not None:
                    entry.update(stage_c.score(points, fit.vertices, faces))
                    if truth_fit is not None:
                        entry["surface_displacement_median_mm"] = float(
                            np.median(np.linalg.norm(fit.vertices - truth_fit.vertices, axis=1)))
                row[f"{front}/{candidate}"] = entry
                row["fitted"] = row["fitted"] and entry["fitted"]
        rows.append(row)
        print(json.dumps({"patient": row["patient"], "frame": frame,
                          **{k: (round(v["symmetric_mm"], 3), v["removed_voxels"]) for k, v in row.items()
                             if isinstance(v, dict) and "symmetric_mm" in v}}), flush=True)
    return rows


def _patients(rows):
    groups = {}
    for r in rows:
        if r.get("fitted"):
            groups.setdefault(r["patient"], []).append(r)
    return {p: f for p, f in groups.items() if len(f) == 2}


def _ef_errors(frames: dict, arm: str) -> dict:
    """E31's definition: EF of the fit to the prediction minus EF of the fit to the reference labels."""

    out = {}
    for patient, pair in frames.items():
        ref = sorted(f["reference_volume_ml"] for f in pair)
        fit = sorted(f[arm]["volume_ml"] for f in pair)
        out[patient] = 100.0 * (fit[1] - fit[0]) / fit[1] - 100.0 * (ref[1] - ref[0]) / ref[1]
    return out


def _arm_summary(frames: dict, arm: str) -> dict:
    errors = np.array(list(_ef_errors(frames, arm).values()))
    sd = float(errors.std(ddof=1))
    values = lambda key: [f[arm][key] for pair in frames.values() for f in pair]
    return {"median_symmetric_mm": float(np.median(values("symmetric_mm"))),
            "median_hausdorff_mm": float(np.median(values("hausdorff_mm"))),
            "median_dice": float(np.median(values("dice_lv_cavity"))),
            "median_surface_displacement_mm": float(np.median(values("surface_displacement_median_mm"))),
            "frames_changed": int(sum(1 for pair in frames.values() for f in pair if f[arm]["removed_voxels"] > 0)),
            "ef": {"bias": float(errors.mean()), "median_absolute": float(np.median(np.abs(errors))),
                   "limits_of_agreement": [float(errors.mean() - 1.96 * sd), float(errors.mean() + 1.96 * sd)],
                   "width": float(2 * 1.96 * sd)}}


def choose(scores: dict) -> str:
    """The pre-registered rule; ``scores`` maps each candidate to its validation summary."""

    none = scores["none"]
    eligible = [c for c in ("largest_3d", "largest_3d_slice")
                if scores[c]["median_symmetric_mm"] <= none["median_symmetric_mm"] + SURFACE_TOLERANCE_MM
                and scores[c]["ef"]["width"] <= none["ef"]["width"] + TIE]
    if not eligible:
        return "none"
    if len(eligible) == 2:
        slice_better = scores["largest_3d_slice"]["ef"]["width"] < scores["largest_3d"]["ef"]["width"] - TIE
        return "largest_3d_slice" if slice_better else "largest_3d"
    return eligible[0]


def select() -> None:
    rows = score_split("val", {front: CANDIDATES for front in FRONT_ENDS})
    frames = _patients(rows)
    report = {"experiment": "E86", "preregistration": PREREGISTRATION, "split": "val",
              "patients": len(frames), "front_ends": {}}
    for front in FRONT_ENDS:
        scores = {c: _arm_summary(frames, f"{front}/{c}") for c in CANDIDATES}
        report["front_ends"][front] = {"scores": scores, "chosen": choose(scores)}
    out = Path(os.environ.get("R4D_OUT", "results/benchmark/E86")) / "E86_selection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report | {"rows": rows}, indent=2) + "\n")
    for front, entry in report["front_ends"].items():
        print(front, "chosen:", entry["chosen"],
              {c: (round(s["ef"]["width"], 2), round(s["median_symmetric_mm"], 4), s["frames_changed"])
               for c, s in entry["scores"].items()})
    print("wrote", out)


def _paired(frames: dict, first: str, second: str) -> dict:
    patients = sorted(frames)
    rng = np.random.default_rng(SEED)
    draws = [[patients[i] for i in rng.integers(0, len(patients), len(patients))] for _ in range(RESAMPLES)]
    errors = {arm: _ef_errors(frames, arm) for arm in (first, second)}

    def symmetric(arm, chosen):
        return np.median([np.mean([f[arm]["symmetric_mm"] for f in frames[p]]) for p in chosen])

    def width(arm, chosen):
        return 2 * 1.96 * np.std([errors[arm][p] for p in chosen], ddof=1)

    def medae(arm, chosen):
        return np.median([abs(errors[arm][p]) for p in chosen])

    out = {}
    for key, statistic in (("ef_limits_width", width), ("ef_median_absolute_error", medae),
                           ("symmetric_mm_patient_means", symmetric)):
        observed = statistic(first, patients) - statistic(second, patients)
        boot = np.array([statistic(first, d) - statistic(second, d) for d in draws])
        entry = {"observed": float(observed), "lower": float(np.percentile(boot, 2.5)),
                 "upper": float(np.percentile(boot, 97.5))}
        entry["verdict"] = ("constraint lower" if entry["upper"] < 0 else
                            "constraint higher" if entry["lower"] > 0 else "no detectable difference")
        out[key] = entry
    out["per_patient_ef_change"] = {p: {"none": errors[second][p], "constrained": errors[first][p]}
                                    for p in patients if abs(errors[first][p] - errors[second][p]) > 1e-9}
    return out


def test() -> None:
    if os.environ.get("R4D_ALLOW_TEST") != "1":
        raise SystemExit("refusing to score test frames without R4D_ALLOW_TEST=1")
    selection = json.loads(Path(os.environ["R4D_SELECTION"]).read_text())
    chosen = {front: selection["front_ends"][front]["chosen"] for front in FRONT_ENDS}
    candidates = {front: ("none",) + ((chosen[front],) if chosen[front] != "none" else ()) for front in FRONT_ENDS}
    start = time.time()
    rows = score_split("test", candidates)
    e84 = {(r["patient"], int(r["frame"])): r for r in json.loads(Path(os.environ["R4D_E84"]).read_text())["rows"]}
    differences = [abs(r[f"{front}/none"]["symmetric_mm"] - e84[(r["patient"], r["frame"])][front]["symmetric_mm"])
                   for r in rows for front in FRONT_ENDS
                   if "symmetric_mm" in r[f"{front}/none"] and "symmetric_mm" in e84[(r["patient"], r["frame"])][front]]
    frames = _patients(rows)
    report = {"experiment": "E86", "preregistration": PREREGISTRATION, "split": "test", "chosen": chosen,
              "reproduction_of_e84": {"comparisons": len(differences), "largest_difference_mm": max(differences),
                                      "passed": max(differences) <= 1e-6},
              "patients": len(frames), "front_ends": {}}
    for front in FRONT_ENDS:
        entry = {"arms": {c: _arm_summary(frames, f"{front}/{c}") for c in candidates[front]}}
        if chosen[front] != "none":
            paired = _paired(frames, f"{front}/{chosen[front]}", f"{front}/none")
            entry["paired_constrained_minus_none"] = paired
            entry["rules"] = {"K1_ef_interval": paired["ef_limits_width"]["verdict"],
                              "K2_surface": paired["symmetric_mm_patient_means"]["verdict"]}
        report["front_ends"][front] = entry
    report["wall_seconds"] = time.time() - start
    out = Path(os.environ.get("R4D_OUT", "results/benchmark/E86")) / "E86_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report | {"rows": rows}, indent=2) + "\n")
    print(json.dumps({"chosen": chosen, "reproduction": report["reproduction_of_e84"],
                      "rules": {f: e.get("rules") for f, e in report["front_ends"].items()}}, indent=2))
    for front, entry in report["front_ends"].items():
        for c, s in entry["arms"].items():
            print(front, c, "EF LoA", [round(v, 2) for v in s["ef"]["limits_of_agreement"]], "width %.1f" % s["ef"]["width"],
                  "medAE %.2f" % s["ef"]["median_absolute"], "sym %.3f" % s["median_symmetric_mm"],
                  "frames changed", s["frames_changed"])
    print("wrote", out)


def main() -> int:
    mode = os.environ.get("R4D_MODE", "")
    if mode == "select":
        select()
    elif mode == "test":
        test()
    else:
        raise SystemExit("set R4D_MODE to select or test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
