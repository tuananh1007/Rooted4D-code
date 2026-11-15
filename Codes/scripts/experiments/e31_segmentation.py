"""E31: a segmentation front end, and what its errors do to the fit.

Everything downstream in this project consumes segmentations, and until now it
has consumed the provider's ground-truth labels, which assumes away the first
stage of any real pipeline. This trains a compact network on the ACDC people the
frozen split lock assigns to training and measures two different things on the
people it assigns to test.

Dice is the standard number and is reported. It is not the number that matters
here. A segmentation is an input, so the question is what its errors do to the
surface fitted from it and to the ejection fraction read off that surface. A
network can lose Dice on a thin structure while barely moving a volume, or win
Dice while shifting a boundary systematically; only the downstream comparison
tells those apart.

CPU only, on the machine owner's instruction.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from rooted4d.active.acdc import fit_lv_surface, lv_topology_faces
from rooted4d.metrics.topology import signed_enclosed_volume
from rooted4d.segmentation import (
    ACDCSlices,
    CompactUNet,
    dice_per_class,
    load_patient_splits,
)
from rooted4d.segmentation.data import PATCH, TARGET_SPACING_MM

torch.manual_seed(0)
torch.set_num_threads(int(os.environ.get("R4D_THREADS", 8)))
SPLITS = load_patient_splits(os.environ.get("R4D_SPLITS", "results/benchmark/acdc_patient_splits.json"))
ROOT = "data/raw/acdc"
EPOCHS = int(os.environ.get("R4D_EPOCHS", 12))
BATCH = int(os.environ.get("R4D_BATCH", 16))
TIME_BUDGET_S = int(os.environ.get("R4D_BUDGET_S", 2400))

print("loading slices...", flush=True)
t0 = time.time()
train = ACDCSlices(ROOT, SPLITS, "train")
val = ACDCSlices(ROOT, SPLITS, "val")
print("train %d slices / %d patients | val %d / %d  (%.0f s)"
      % (len(train), train.n_patients, len(val), val.n_patients, time.time() - t0), flush=True)

loader = torch.utils.data.DataLoader(train, batch_size=BATCH, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH)
net = CompactUNet()
optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)
# Class weights: background dominates by an order of magnitude and an unweighted
# cross-entropy spends its capacity predicting it.
weights = torch.tensor([0.2, 1.0, 1.0, 1.0], dtype=torch.float32)

best = {"dice": -1.0, "state": None, "epoch": -1}
history = []
start = time.time()
for epoch in range(EPOCHS):
    net.train()
    total = 0.0
    for images, labels in loader:
        optimizer.zero_grad()
        logits = net(images)
        loss = functional.cross_entropy(logits, labels, weight=weights)
        loss.backward()
        optimizer.step()
        total += float(loss) * len(images)
    net.eval()
    scores = []
    with torch.no_grad():
        for images, labels in val_loader:
            scores.append(dice_per_class(net(images).argmax(1), labels))
    dice = torch.nanmean(torch.stack(scores), dim=0)
    mean_fg = float(torch.nanmean(dice[1:]))
    history.append({"epoch": epoch, "loss": total / len(train),
                    "val_dice_per_class": [None if torch.isnan(v) else float(v) for v in dice],
                    "val_dice_foreground": mean_fg})
    print("epoch %2d  loss %.4f  val dice fg %.4f  (rv %.3f myo %.3f lv %.3f)  %.0fs"
          % (epoch, total / len(train), mean_fg, dice[1], dice[2], dice[3],
             time.time() - start), flush=True)
    if mean_fg > best["dice"]:
        best = {"dice": mean_fg, "state": {k: v.clone() for k, v in net.state_dict().items()},
                "epoch": epoch}
    if time.time() - start > TIME_BUDGET_S:
        print("time budget reached", flush=True)
        break

net.load_state_dict(best["state"])
net.eval()
print("\nbest epoch %d, val foreground dice %.4f" % (best["epoch"], best["dice"]), flush=True)
torch.save(best["state"], os.environ.get("R4D_OUT", "results/benchmark") + "/E31_segmenter.pt")

# ---- test Dice -------------------------------------------------------------
test = ACDCSlices(ROOT, SPLITS, "test")
scores = []
with torch.no_grad():
    for images, labels in torch.utils.data.DataLoader(test, batch_size=BATCH):
        scores.append(dice_per_class(net(images).argmax(1), labels))
test_dice = torch.nanmean(torch.stack(scores), dim=0)
print("test dice: rv %.3f  myo %.3f  lv cavity %.3f  (%d slices / %d patients)"
      % (test_dice[1], test_dice[2], test_dice[3], len(test), test.n_patients), flush=True)

# ---- downstream: fit from predicted labels vs from true labels -------------
import nibabel as nib

faces = lv_topology_faces(8, 32)


def predict_volume(image_path: Path, reference_shape, spacing):
    """Segment a whole short-axis stack and return it on the original grid."""

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
        rows = min(PATCH, size[0]); cols = min(PATCH, size[1])
        sr = (size[0] - rows) // 2; sc = (size[1] - cols) // 2
        dr = (PATCH - rows) // 2; dc = (PATCH - cols) // 2
        canvas[0, 0, dr:dr + rows, dc:dc + cols] = resampled[0, 0, sr:sr + rows, sc:sc + cols]
        with torch.no_grad():
            logits = net(canvas)
        # Undo the crop and the resampling, so the prediction lands on the grid
        # the fitter expects rather than on the network's patch.
        patch = torch.zeros((1, 4, size[0], size[1]))
        patch[0, :, sr:sr + rows, sc:sc + cols] = logits[0, :, dr:dr + rows, dc:dc + cols]
        back = functional.interpolate(patch, size=plane.shape, mode="bilinear",
                                      align_corners=False)
        output[:, :, index] = back.argmax(1)[0].numpy()
    return output


rows = []
seen = set()
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
    predicted = predict_volume(image_path, truth.shape, zooms[:2])
    frame = int(gt_path.name.split("_frame")[-1].split("_")[0])
    fits = {}
    for tag, labels in (("truth", truth), ("predicted", predicted)):
        fit = fit_lv_surface(labels, zooms, n_rings=8, n_angles=32,
                             patient=patient, frame=frame)
        fits[tag] = fit
    if fits["truth"] is None or fits["predicted"] is None:
        rows.append({"patient": patient, "frame": frame, "fit_failed": True,
                     "predicted_fit_ok": fits["predicted"] is not None})
        continue
    displacement = np.linalg.norm(fits["predicted"].vertices - fits["truth"].vertices, axis=1)
    rows.append({
        "patient": patient, "frame": frame, "fit_failed": False,
        "cavity_volume_truth_ml": float((truth == 3).sum() * np.prod(zooms) / 1000.0),
        "cavity_volume_pred_ml": float((predicted == 3).sum() * np.prod(zooms) / 1000.0),
        "mesh_volume_truth_ml": abs(signed_enclosed_volume(fits["truth"].vertices, faces)) / 1000.0,
        "mesh_volume_pred_ml": abs(signed_enclosed_volume(fits["predicted"].vertices, faces)) / 1000.0,
        "surface_displacement_median_mm": float(np.median(displacement)),
        "surface_displacement_p95_mm": float(np.percentile(displacement, 95)),
    })
    seen.add(patient)

good = [r for r in rows if not r["fit_failed"]]
failed = [r for r in rows if r["fit_failed"]]
by_patient = {}
for r in good:
    by_patient.setdefault(r["patient"], []).append(r)
ef_rows = []
for patient, frames in by_patient.items():
    if len(frames) != 2:
        continue
    for tag in ("truth", "pred"):
        volumes = sorted(f["mesh_volume_%s_ml" % tag] for f in frames)
        ef = 100.0 * (volumes[1] - volumes[0]) / max(volumes[1], 1e-9)
        frames[0].setdefault("ef", {})[tag] = ef
    ef_rows.append({"patient": patient,
                    "ef_truth": frames[0]["ef"]["truth"],
                    "ef_pred": frames[0]["ef"]["pred"]})

disp = np.array([r["surface_displacement_median_mm"] for r in good])
ef_error = np.array([r["ef_pred"] - r["ef_truth"] for r in ef_rows])
report = {
    "experiment": "E31_segmentation_front_end",
    "device": "cpu",
    "split_source": "GLOBAL_SPLIT_LOCK.json via results/benchmark/acdc_patient_splits.json",
    "n_train_slices": len(train), "n_train_patients": train.n_patients,
    "n_test_slices": len(test), "n_test_patients": test.n_patients,
    "best_epoch": best["epoch"],
    "val_dice_foreground": best["dice"],
    "test_dice": {"rv": float(test_dice[1]), "myocardium": float(test_dice[2]),
                  "lv_cavity": float(test_dice[3])},
    "downstream": {
        "frames_fitted": len(good), "frames_failed": len(failed),
        "surface_displacement_mm": {
            "median": float(np.median(disp)), "p95": float(np.percentile(disp, 95)),
            "max": float(disp.max())},
        "ef_error_points": {
            "n": len(ef_rows), "bias": float(ef_error.mean()),
            "median_absolute": float(np.median(np.abs(ef_error))),
            "limits_of_agreement": [float(ef_error.mean() - 1.96 * ef_error.std(ddof=1)),
                                    float(ef_error.mean() + 1.96 * ef_error.std(ddof=1))]},
    },
    "history": history, "rows": rows, "ef_rows": ef_rows,
}
json.dump(report, open(os.environ.get("R4D_OUT", "results/benchmark") + "/E31_segmentation.json", "w"), indent=2)
print("\ndownstream, fitting from predicted labels instead of true ones:", flush=True)
print("  frames fitted %d, failed %d" % (len(good), len(failed)), flush=True)
print("  surface displacement: median %.2f mm, p95 %.2f, max %.2f"
      % (np.median(disp), np.percentile(disp, 95), disp.max()), flush=True)
print("  EF error: bias %+.2f points, median abs %.2f, LoA [%+.2f, %+.2f] (n=%d)"
      % (ef_error.mean(), np.median(np.abs(ef_error)),
         ef_error.mean() - 1.96 * ef_error.std(ddof=1),
         ef_error.mean() + 1.96 * ef_error.std(ddof=1), len(ef_rows)), flush=True)
print("done", flush=True)
