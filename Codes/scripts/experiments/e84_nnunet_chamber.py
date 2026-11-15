"""E84: a competitive chamber segmenter (nnU-Net) behind the same fitting.

Pre-registered in ``Docs/E84_NNUNET_CHAMBER_PREREGISTRATION_2026-09-28.md``.

The chamber front end the paper reports is E31's compact U-Net. This trains
nnU-Net on the same 90 training patients and scores it behind the unchanged
fitting (E77's direct surface), beside the compact U-Net on the same frames.

    R4D_MODE=convert python e84_nnunet_chamber.py        ACDC -> nnU-Net raw (Dataset502_ACDC)
    R4D_MODE=splits  python e84_nnunet_chamber.py        fold 0 = the project split, after preprocessing
    R4D_MODE=score R4D_ALLOW_TEST=1 python e84_nnunet_chamber.py   the single test scoring

``convert`` and ``splits`` need only the standard library, so they run in the
nnU-Net environment. ``score`` needs the project environment.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

PREREGISTRATION = "Docs/E84_NNUNET_CHAMBER_PREREGISTRATION_2026-09-28.md"
DATASET = "Dataset502_ACDC"
LABELS = {"background": 0, "RV": 1, "myocardium": 2, "LV_cavity": 3}
CAVITY = 3
RESAMPLES, SEED = 20000, 20260928
ARMS = ("compact_unet", "nnunet")


def splits_of() -> dict:
    """The project's split lock: patient -> train, val or test."""

    return json.loads(Path(os.environ.get("R4D_SPLITS", "results/benchmark/acdc_patient_splits.json")).read_text())


def labelled_frames(root: Path, splits: dict, wanted: tuple) -> list:
    return [p for p in sorted(root.rglob("*_gt.nii.gz")) if splits.get(p.parent.name) in wanted]


def case_id(path: Path) -> str:
    return path.name.replace("_gt.nii.gz", "")


# ---------------------------------------------------------------- convert ----

def convert(root: Path, splits: dict) -> None:
    raw = Path(os.environ["nnUNet_raw"]) / DATASET
    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        (raw / sub).mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0, "test": 0}
    for path in labelled_frames(root, splits, ("train", "val", "test")):
        split = splits[path.parent.name]
        image = Path(str(path).replace("_gt.nii.gz", ".nii.gz"))
        case = case_id(path)
        if split == "test":
            # Test frames are images only: nnU-Net never sees their labels.
            shutil.copy(image, raw / "imagesTs" / f"{case}_0000.nii.gz")
        else:
            shutil.copy(image, raw / "imagesTr" / f"{case}_0000.nii.gz")
            shutil.copy(path, raw / "labelsTr" / f"{case}.nii.gz")
        counts[split] += 1
    dataset = {
        "channel_names": {"0": "cineMRI"},
        "labels": LABELS,
        "numTraining": counts["train"] + counts["val"],
        "file_ending": ".nii.gz",
        "name": "ACDC",
        "description": ("ACDC end-diastolic and end-systolic frames under the project split lock. imagesTr holds "
                        "the training and validation patients; fold 0 (splits_final.json) trains on the training "
                        "patients and validates on the validation patients. The test patients are imagesTs only."),
        "reference": "https://www.creatis.insa-lyon.fr/Challenge/acdc/",
        "licence": "CC BY-NC-SA 4.0",
    }
    (raw / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    print(json.dumps({"frames": counts, "raw": str(raw)}, indent=2))


def write_splits(root: Path, splits: dict) -> None:
    folder = Path(os.environ["nnUNet_preprocessed"]) / DATASET
    fold0 = {name: [case_id(p) for p in labelled_frames(root, splits, (name,))] for name in ("train", "val")}
    (folder / "splits_final.json").write_text(json.dumps([fold0], indent=2) + "\n")
    print("fold 0: %d training frames, %d validation frames -> %s"
          % (len(fold0["train"]), len(fold0["val"]), folder / "splits_final.json"))


# ------------------------------------------------------------------ score ----

def score_frames(root: Path, splits: dict) -> list:
    import nibabel as nib
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import e76_chamber_shape_fit as stage_c
    from rooted4d.active.acdc import fit_lv_surface, lv_topology_faces
    from rooted4d.experiments.component_eval import cavity_boundary_points

    predictions = Path(os.environ["R4D_NNUNET_PRED"])
    segmenter = stage_c.Segmenter(Path(os.environ["R4D_SEGMENTER"]))
    faces = lv_topology_faces(stage_c.N_RINGS, stage_c.N_ANGLES)
    rows = []
    for path in stage_c.frames_of(root, splits, "test"):
        truth, zooms, frame, truth_fit = stage_c.reference_fit(path)
        image_path = Path(str(path).replace("_gt.nii.gz", ".nii.gz"))
        points = cavity_boundary_points(truth, zooms, cavity_label=CAVITY)
        row = {"patient": path.parent.name, "frame": frame, "fitted": truth_fit is not None}
        if truth_fit is not None:
            row["reference_volume_ml"] = stage_c.score(points, truth_fit.vertices, faces)["volume_ml"]
        predicted = {"compact_unet": segmenter(image_path, truth.shape, zooms[:2]),
                     "nnunet": np.asarray(nib.load(str(predictions / f"{case_id(path)}.nii.gz")).dataobj)}
        for arm, labels in predicted.items():
            assert labels.shape == truth.shape, (arm, labels.shape, truth.shape)
            overlap = np.logical_and(labels == CAVITY, truth == CAVITY).sum()
            dice = 2.0 * overlap / max(int((labels == CAVITY).sum() + (truth == CAVITY).sum()), 1)
            fit = fit_lv_surface(labels, zooms, n_rings=stage_c.N_RINGS, n_angles=stage_c.N_ANGLES,
                                 patient=path.parent.name, frame=frame, base_first=True)
            entry = {"dice_lv_cavity": float(dice), "fitted": fit is not None}
            if fit is not None:
                entry.update(stage_c.score(points, fit.vertices, faces))
                if truth_fit is not None:
                    displacement = np.linalg.norm(fit.vertices - truth_fit.vertices, axis=1)
                    entry["surface_displacement_median_mm"] = float(np.median(displacement))
            row[arm] = entry
            row["fitted"] = row["fitted"] and entry["fitted"]
        rows.append(row)
        print(json.dumps({k: (v if not isinstance(v, dict) else {m: round(x, 3) for m, x in v.items()
                                                                 if isinstance(x, float)})
                          for k, v in row.items()}), flush=True)
    return rows


def _by_patient(rows):
    groups = {}
    for r in rows:
        if r.get("fitted"):
            groups.setdefault(r["patient"], []).append(r)
    return {p: f for p, f in groups.items() if len(f) == 2}


def ef_errors(frames_by_patient: dict, arm: str) -> dict:
    """E31's definition: EF of the fit to the prediction minus EF of the fit to the reference labels."""

    out = {}
    for patient, frames in frames_by_patient.items():
        ref = sorted(f["reference_volume_ml"] for f in frames)
        fit = sorted(f[arm]["volume_ml"] for f in frames)
        out[patient] = 100.0 * (fit[1] - fit[0]) / fit[1] - 100.0 * (ref[1] - ref[0]) / ref[1]
    return out


def agreement(errors) -> dict:
    import numpy as np

    e = np.asarray(list(errors), dtype=float)
    sd = float(e.std(ddof=1))
    return {"patients": int(e.size), "bias": float(e.mean()), "median_absolute": float(np.median(np.abs(e))),
            "limits_of_agreement": [float(e.mean() - 1.96 * sd), float(e.mean() + 1.96 * sd)],
            "width": float(2 * 1.96 * sd)}


def summarize(rows: list) -> dict:
    import numpy as np

    frames = _by_patient(rows)
    patients = sorted(frames)
    rng = np.random.default_rng(SEED)
    draws = [[patients[i] for i in rng.integers(0, len(patients), len(patients))] for _ in range(RESAMPLES)]

    def frame_values(arm, key, chosen):
        return [f[arm][key] for p in chosen for f in frames[p]]

    def median_with_interval(arm, key):
        observed = float(np.median(frame_values(arm, key, patients)))
        boot = np.array([np.median(frame_values(arm, key, d)) for d in draws])
        return {"median": observed, "lower": float(np.percentile(boot, 2.5)), "upper": float(np.percentile(boot, 97.5))}

    report = {"patients": len(patients), "frames": 2 * len(patients), "arms": {}}
    errors = {arm: ef_errors(frames, arm) for arm in ARMS}
    for arm in ARMS:
        report["arms"][arm] = {
            "dice_lv_cavity_median": float(np.median(frame_values(arm, "dice_lv_cavity", patients))),
            "symmetric_mm": median_with_interval(arm, "symmetric_mm"),
            "hausdorff_mm": median_with_interval(arm, "hausdorff_mm"),
            "surface_displacement_median_mm": float(np.median(frame_values(arm, "surface_displacement_median_mm",
                                                                            patients))),
            "ejection_fraction": agreement(errors[arm].values()),
        }

    def paired(statistic):
        observed = statistic(patients)
        boot = np.array([statistic(d) for d in draws])
        entry = {"observed": float(observed), "lower": float(np.percentile(boot, 2.5)),
                 "upper": float(np.percentile(boot, 97.5))}
        entry["verdict"] = ("nnU-Net lower" if entry["upper"] < 0 else
                            "nnU-Net higher" if entry["lower"] > 0 else "no detectable difference")
        return entry

    def patient_mean_symmetric(arm, chosen):
        return [np.mean([f[arm]["symmetric_mm"] for f in frames[p]]) for p in chosen]

    def width(arm, chosen):
        return 2 * 1.96 * np.std([errors[arm][p] for p in chosen], ddof=1)

    report["paired_nnunet_minus_compact"] = {
        "symmetric_mm_patient_means": paired(lambda c: np.median(patient_mean_symmetric("nnunet", c))
                                             - np.median(patient_mean_symmetric("compact_unet", c))),
        "ef_median_absolute_error": paired(lambda c: np.median([abs(errors["nnunet"][p]) for p in c])
                                           - np.median([abs(errors["compact_unet"][p]) for p in c])),
        "ef_limits_width": paired(lambda c: width("nnunet", c) - width("compact_unet", c)),
    }
    report["rules"] = {
        "F1_ef_interval": report["paired_nnunet_minus_compact"]["ef_limits_width"]["verdict"],
        "F2_surface": report["paired_nnunet_minus_compact"]["symmetric_mm_patient_means"]["verdict"],
    }
    return report


def reproduction(rows: list, e77_path: Path) -> dict:
    e77 = {(r["patient"], int(r["frame"])): r for r in json.loads(e77_path.read_text())["rows"]}
    differences = []
    for r in rows:
        mine, theirs = r.get("compact_unet") or {}, (e77.get((r["patient"], r["frame"])) or {}).get("direct") or {}
        if "symmetric_mm" in mine and "symmetric_mm" in theirs:
            differences.append(abs(mine["symmetric_mm"] - theirs["symmetric_mm"]))
    return {"frames_compared": len(differences), "largest_difference_mm": max(differences) if differences else None,
            "passed": bool(differences) and max(differences) <= 1e-6}


def score() -> None:
    if os.environ.get("R4D_ALLOW_TEST") != "1":
        raise SystemExit("refusing to score test frames without R4D_ALLOW_TEST=1")
    root, splits = Path(os.environ.get("R4D_ACDC", "data/raw/acdc")), splits_of()
    start = time.time()
    rows = score_frames(root, splits)
    report = {"experiment": "E84", "preregistration": PREREGISTRATION,
              "segmenters": {"compact_unet": os.environ["R4D_SEGMENTER"],
                             "nnunet": "nnU-Net v2 3d_fullres fold 0 (project split), final checkpoint, "
                                       "--disable_tta; predictions " + os.environ["R4D_NNUNET_PRED"]},
              "reproduction_of_e77_direct": reproduction(rows, Path(os.environ["R4D_E77"])),
              "summary": summarize(rows), "wall_seconds": None, "rows": rows}
    report["wall_seconds"] = time.time() - start
    out = Path(os.environ.get("R4D_OUT", "results/benchmark")) / "E84_nnunet_chamber.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    s = report["summary"]
    for arm in ARMS:
        a = s["arms"][arm]
        print("%-13s dice %.3f  symmetric %.3f [%.3f, %.3f]  hausdorff %.2f  EF LoA [%.2f, %+.2f] width %.1f"
              % (arm, a["dice_lv_cavity_median"], a["symmetric_mm"]["median"], a["symmetric_mm"]["lower"],
                 a["symmetric_mm"]["upper"], a["hausdorff_mm"]["median"],
                 *a["ejection_fraction"]["limits_of_agreement"], a["ejection_fraction"]["width"]))
    print(json.dumps({"rules": s["rules"], "reproduction": report["reproduction_of_e77_direct"]}, indent=2))
    print("wrote", out)


def main() -> int:
    mode = os.environ.get("R4D_MODE", "")
    if mode == "convert":
        convert(Path(os.environ.get("R4D_ACDC", "data/raw/acdc")), splits_of())
    elif mode == "splits":
        write_splits(Path(os.environ.get("R4D_ACDC", "data/raw/acdc")), splits_of())
    elif mode == "score":
        score()
    else:
        raise SystemExit("set R4D_MODE to convert, splits or score")
    return 0


if __name__ == "__main__":
    sys.exit(main())
