"""E36: the chamber component scored with images as input, not labels.

Panel A of the paper reports the template fitted to reference segmentations and
scored against those same segmentations. That is a representation floor -- how
well the frozen topology can approximate real anatomy -- and it is the wrong
quantity to compare against a method that consumes images, because it never
passes through a segmentation error. Leaving the external rows pending while
our own row is measured differently would invite exactly that comparison.

This measures the composed quantity: predict labels from the image with the
trained front end, fit the template to the prediction, and score the result
against the *reference* label surface, using the identical metric as panel A.
That is protocol-compatible with an image-input method evaluated against the
same reference labels on the same held-out patients.

Reuses the checkpoint from E31 rather than retraining, so the segmenter is
exactly the one whose Dice and downstream numbers the paper already reports.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from rooted4d.active.acdc import fit_lv_surface, lv_topology_faces
from rooted4d.experiments.component_eval import (
    cavity_boundary_points,
    symmetric_surface_distance,
)
from rooted4d.segmentation import CompactUNet, load_patient_splits
from rooted4d.segmentation.data import PATCH, TARGET_SPACING_MM

torch.manual_seed(0)
torch.set_num_threads(int(os.environ.get("R4D_THREADS", 8)))

ROOT = os.environ.get("R4D_ACDC", "data/raw/acdc")
SPLITS = load_patient_splits(
    os.environ.get("R4D_SPLITS", "results/benchmark/acdc_patient_splits.json"))
CHECKPOINT = Path(os.environ.get("R4D_SEGMENTER", "results/21900728/E31_segmenter.pt"))
OUTPUT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / "E36_image_input_chamber.json"
N_RINGS, N_ANGLES, CAVITY = 8, 32, 3

net = CompactUNet()
state = torch.load(CHECKPOINT, map_location="cpu")
net.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
net.eval()
print("loaded %s" % CHECKPOINT, flush=True)

import nibabel as nib

FACES = lv_topology_faces(N_RINGS, N_ANGLES)


def predict_volume(image_path: Path, reference_shape, spacing):
    """Segment a short-axis stack and return it on the original grid.

    Identical to E31's routine; the prediction has to land on the grid the
    fitter expects rather than on the network's patch.
    """

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
        resampled = functional.interpolate(tensor, size=size, mode="bilinear",
                                           align_corners=False)
        canvas = torch.zeros((1, 1, PATCH, PATCH))
        rows_, cols = min(PATCH, size[0]), min(PATCH, size[1])
        sr, sc = (size[0] - rows_) // 2, (size[1] - cols) // 2
        dr, dc = (PATCH - rows_) // 2, (PATCH - cols) // 2
        canvas[0, 0, dr:dr + rows_, dc:dc + cols] = resampled[0, 0, sr:sr + rows_, sc:sc + cols]
        with torch.no_grad():
            logits = net(canvas)
        patch = torch.zeros((1, 4, size[0], size[1]))
        patch[0, :, sr:sr + rows_, sc:sc + cols] = logits[0, :, dr:dr + rows_, dc:dc + cols]
        back = functional.interpolate(patch, size=plane.shape, mode="bilinear",
                                      align_corners=False)
        output[:, :, index] = back.argmax(1)[0].numpy()
    return output


def column(values):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    return {"n": int(v.size), "median": float(np.median(v)),
            "mean": float(v.mean()), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}


start = time.time()
rows = []
for gt_path in sorted(Path(ROOT).rglob("*_gt.nii.gz")):
    patient = gt_path.parent.name
    if SPLITS.get(patient) != "test":
        continue
    image_path = Path(str(gt_path).replace("_gt.nii.gz", ".nii.gz"))
    if not image_path.exists():
        continue
    handle = nib.load(str(gt_path))
    truth = np.asarray(handle.dataobj)
    zooms = tuple(float(z) for z in handle.header.get_zooms()[:3])
    frame = int(gt_path.name.split("_frame")[-1].split("_")[0])
    predicted = predict_volume(image_path, truth.shape, zooms[:2])

    # The reference surface is the same target panel A scores against, so the
    # only difference between the two rows is where the labels came from.
    reference_points = cavity_boundary_points(truth, zooms, cavity_label=CAVITY)
    entry = {"patient": patient, "frame": frame}
    for tag, labels in (("label_input", truth), ("image_input", predicted)):
        fit = fit_lv_surface(labels, zooms, n_rings=N_RINGS, n_angles=N_ANGLES,
                             patient=patient, frame=frame)
        if fit is None:
            entry[tag] = None
            continue
        entry[tag] = symmetric_surface_distance(reference_points, fit.vertices, FACES)
    rows.append(entry)
    print("  %s frame %d  label %.3f  image %.3f" % (
        patient, frame,
        entry["label_input"]["symmetric_mean_mm"] if entry["label_input"] else float("nan"),
        entry["image_input"]["symmetric_mean_mm"] if entry["image_input"] else float("nan"),
    ), flush=True)

good = [r for r in rows if r["label_input"] and r["image_input"]]
summary = {}
for tag in ("label_input", "image_input"):
    summary[tag] = {
        "symmetric_mm": column([r[tag]["symmetric_mean_mm"] for r in good]),
        "hausdorff_mm": column([r[tag]["hausdorff_mm"] for r in good]),
    }
paired = np.array([r["image_input"]["symmetric_mean_mm"] - r["label_input"]["symmetric_mean_mm"]
                   for r in good])
summary["image_minus_label_symmetric_mm"] = {
    "median": float(np.median(paired)), "mean": float(paired.mean()),
    "n": int(paired.size),
}

result = {
    "experiment": "E36_image_input_chamber",
    "checkpoint": str(CHECKPOINT),
    "reference": "ACDC provider labels, cavity boundary points",
    "note": "both rows are scored against the same reference surface; they differ "
            "only in whether the fit consumed reference labels or predicted ones",
    "n_frames": len(good),
    "n_patients": len({r["patient"] for r in good}),
    "wall_seconds": time.time() - start,
    "summary": summary,
    "rows": rows,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(result, indent=2) + "\n")

print("\nscored against the same reference surface, %d frames / %d patients:"
      % (result["n_frames"], result["n_patients"]))
for tag in ("label_input", "image_input"):
    s = summary[tag]
    print("  %-12s symmetric %.3f (p95 %.3f)   hausdorff %.3f"
          % (tag, s["symmetric_mm"]["median"], s["symmetric_mm"]["p95"],
             s["hausdorff_mm"]["median"]))
print("  image minus label, median %.3f mm" % summary["image_minus_label_symmetric_mm"]["median"])
print("wrote %s" % OUTPUT)
