"""E49: score TetHeart's predicted LV cavity against the ACDC labels.

Runs after `tools/test.py --show --show-dir` has written predicted meshes, and
scores them with the same metric and against the same reference points as
panel A, so the number is comparable to every other row.

Two facts here were established by reading their code and are then re-checked at
runtime, because both would be silent if wrong.

*Which class is the cavity.* `LoadMeshesFromFile` does not use the alphabetical
order the glob produced. It reorders to ``['myo', 'lv', 'rv']``, so the cavity is
class index 1 and lands in ``*_2_pred.obj``, not ``*_1_pred.obj``. Taking the
first file would score the myocardium against cavity labels and produce a
plausible, wrong number. The choice is verified per case by comparing physical
extents against the cavity and the cavity-plus-myocardium labels, and a case
whose evidence contradicts the expected index is reported rather than scored.

*The frame, which is three frames.* Predictions arrive in the tetrahedral
grid's normalized box, not in millimetres, and returning them takes two steps
rather than one:

1. ``geometry_mixins.generate_mesh_from_gt`` normalizes the ground truth by
   ``2 * ((v + 0.5) / patch_size - 0.5)``, so the model is trained in, and
   predicts in, a box of roughly [-1, 1]. Inverting that gives preprocessed
   voxel indices.
2. E50's per-frame permutation, flips, scale and offset then return those
   indices to index-times-spacing millimetres.

Missing either step is silent. With step 1 omitted the predicted meshes measured
2 to 4 mm across where the anatomy spans 30 to 90, and the class check rejected
every case rather than reporting a wrong distance -- which is the only reason
that omission did not become a number.
"""

import json
import os
import re
from pathlib import Path

import numpy as np

PRED_DIR = Path(os.environ["R4D_PRED_DIR"])
# Per-frame permutation, flips, scale and offset recorded by E50 when it moved
# the targets into the preprocessed frame. Required: without it the predictions
# cannot be returned to millimetres and nothing here is comparable to panel A.
TRANSFORMS = Path(os.environ["R4D_TET_FRAME_REPORT"])
ACDC = Path(os.environ.get("R4D_ACDC", "data/raw/acdc"))
OUTPUT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / os.environ.get(
    "R4D_OUTNAME", "E49_tetheart_eval.json")
CAVITY_LABEL, MYO_LABEL = 3, 2
#: The model's patch size, which sets the normalization the network is trained
#: in. Must match the training configuration or the denormalization is wrong.
PATCH_SIZE = np.asarray(
    [float(v) for v in os.environ.get("R4D_PATCH_SIZE", "64,64,64").split(",")])
EXPECTED_CAVITY_CLASS = 2  # file suffix _2_pred.obj, i.e. class index 1


def boundary_points(labels, zooms, label, drop_end_slices):
    occupied = labels == label
    padded = np.pad(occupied, 1)
    interior = (padded[1:-1, 1:-1, 1:-1]
                & padded[:-2, 1:-1, 1:-1] & padded[2:, 1:-1, 1:-1]
                & padded[1:-1, :-2, 1:-1] & padded[1:-1, 2:, 1:-1]
                & padded[1:-1, 1:-1, :-2] & padded[1:-1, 1:-1, 2:])
    rows, columns, slices = np.nonzero(occupied & ~interior)
    if drop_end_slices:
        present = np.unique(np.argwhere(occupied)[:, 2])
        keep = (slices > present.min()) & (slices < present.max())
        rows, columns, slices = rows[keep], columns[keep], slices[keep]
    return np.stack([rows * zooms[0], columns * zooms[1], slices * zooms[2]], axis=1)


def main() -> int:
    import nibabel as nib
    import trimesh
    from rooted4d.experiments.component_eval import symmetric_surface_distance

    transforms = {}
    for row in json.loads(TRANSFORMS.read_text())["rows"]:
        if row.get("status") == "written":
            transforms[(row["case"], row["frame"])] = row

    def denormalize(vertices):
        """Undo the network's own normalization, back to voxel indices.

        Forward: ``2 * ((v + 0.5) / patch_size - 0.5)``.
        """
        v = np.asarray(vertices, dtype=np.float64)
        return (v / 2.0 + 0.5) * PATCH_SIZE - 0.5

    def to_millimetres(vertices, record):
        """Undo the network normalization, then E50's forward map."""
        vertices = denormalize(vertices)
        perm = np.asarray(record["permutation"])
        flips = np.asarray(record["flips"], dtype=float)
        scale = np.asarray(record["scale"], dtype=float)
        offset = np.asarray(record["offset"], dtype=float)
        permuted = (np.asarray(vertices, dtype=np.float64) - offset) / scale / flips
        restored = np.empty_like(permuted)
        restored[:, perm] = permuted
        return restored

    def load_mesh(path):
        """Load an OBJ as a single mesh.

        trimesh returns a Scene when a file carries several object groups, and a
        Scene has no vertices, so ask for a mesh explicitly and concatenate if
        one still comes back.
        """
        mesh = trimesh.load(str(path), process=False, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            parts = list(mesh.geometry.values())
            if not parts:
                return None
            mesh = trimesh.util.concatenate(parts)
        if not hasattr(mesh, "vertices"):
            return None
        return mesh

    labels_by_key = {}
    for path in ACDC.rglob("*_frame??_gt.nii.gz"):
        labels_by_key[(path.parent.name, int(path.name.split("_frame")[1][:2]))] = path

    groups = {}
    for pred in sorted(PRED_DIR.glob("*_pred.obj")):
        match = re.match(r"(.+?)_(\d+)_pred\.obj$", pred.name)
        if not match:
            continue
        groups.setdefault(match.group(1), {})[int(match.group(2))] = pred

    rows = []
    for name, parts in sorted(groups.items()):
        patient_match = re.search(r"(patient\d+)", name)
        frame_match = re.search(r"frame(\d+)", name)
        if not patient_match or not frame_match:
            rows.append({"name": name, "status": "unparsed_name"})
            continue
        patient, frame = patient_match.group(1), int(frame_match.group(1))
        source = labels_by_key.get((patient, frame))
        if source is None or EXPECTED_CAVITY_CLASS not in parts:
            rows.append({"name": name, "status": "missing_label_or_mesh"})
            continue

        handle = nib.load(str(source))
        labels = np.asarray(handle.dataobj)
        zooms = tuple(float(z) for z in handle.header.get_zooms()[:3])

        cavity_extent = np.ptp(boundary_points(labels, zooms, CAVITY_LABEL, False), axis=0)
        wall = np.isin(labels, [CAVITY_LABEL, MYO_LABEL])
        wall_extent = np.ptp(np.argwhere(wall) * np.asarray(zooms), axis=0)

        record_transform = transforms.get((patient, frame))
        if record_transform is None:
            rows.append({"name": name, "patient": patient, "frame": frame,
                         "status": "no_recorded_transform"})
            print("  %-12s f%02d  WITHHELD: no E50 transform, cannot return to millimetres"
                  % (patient, frame), flush=True)
            continue

        # Which predicted class actually looks like the cavity?
        scores = {}
        for index, path in parts.items():
            mesh = load_mesh(path)
            if mesh is None or len(mesh.vertices) == 0:
                continue
            extent = np.ptp(to_millimetres(mesh.vertices, record_transform), axis=0)
            scores[index] = {
                "to_cavity": float(np.abs(extent - cavity_extent).sum()),
                "to_wall": float(np.abs(extent - wall_extent).sum()),
            }
        if not scores:
            rows.append({"name": name, "status": "all_meshes_empty"})
            continue
        best = min(scores, key=lambda i: scores[i]["to_cavity"])
        agrees = best == EXPECTED_CAVITY_CLASS

        mesh = load_mesh(parts[EXPECTED_CAVITY_CLASS])
        vertices = (to_millimetres(mesh.vertices, record_transform)
                    if mesh is not None else np.empty((0, 3)))
        faces = np.asarray(mesh.faces) if mesh is not None else np.empty((0, 3), dtype=int)
        if len(vertices) == 0 or len(faces) == 0:
            rows.append({"name": name, "status": "empty_cavity_mesh",
                         "extent_pick_agrees": bool(agrees)})
            continue

        record = {"name": name, "patient": patient, "frame": frame,
                  "extent_pick": int(best), "extent_pick_agrees": bool(agrees),
                  "verts": int(vertices.shape[0])}
        # The docstring says a case whose extent evidence contradicts the
        # expected class is withheld. It previously said so and scored it anyway.
        if not agrees:
            record["status"] = "withheld_extent_disagreement"
            rows.append(record)
            print("  %-12s f%02d  WITHHELD: extent evidence picks class %d, not %d"
                  % (patient, frame, best, EXPECTED_CAVITY_CLASS), flush=True)
            continue

        record["status"] = "scored"
        for drop, tag in ((False, "with_endcaps"), (True, "no_endcaps")):
            reference = boundary_points(labels, zooms, CAVITY_LABEL, drop)
            if reference.shape[0] < 50:
                continue
            sample = reference[:: max(1, len(reference) // 2000)]
            # The same metric panel A reports, from the same implementation:
            # symmetric surface distance and Hausdorff, not a one-way distance.
            score = symmetric_surface_distance(sample, vertices, faces, sample_faces=6)
            record["symmetric_mm_" + tag] = float(score["symmetric_mean_mm"])
            record["hausdorff_mm_" + tag] = float(score["hausdorff_mm"])
            record["points_to_surface_mm_" + tag] = float(score["mean_points_to_surface_mm"])
            record["surface_to_points_mm_" + tag] = float(score["mean_surface_to_points_mm"])
        rows.append(record)
        print("  %-12s f%02d  symmetric %.3f / %.3f  hausdorff %.3f / %.3f  (with/without end-caps)"
              % (patient, frame,
                 record.get("symmetric_mm_with_endcaps", float("nan")),
                 record.get("symmetric_mm_no_endcaps", float("nan")),
                 record.get("hausdorff_mm_with_endcaps", float("nan")),
                 record.get("hausdorff_mm_no_endcaps", float("nan"))), flush=True)

    scored = [r for r in rows if r["status"] == "scored"]
    summary = {"groups": len(groups), "scored": len(scored),
               "status_counts": {s: sum(1 for r in rows if r["status"] == s)
                                 for s in sorted({r["status"] for r in rows})},
               "extent_pick_disagreements": sum(1 for r in scored
                                                if not r["extent_pick_agrees"])}
    for metric in ("symmetric_mm", "hausdorff_mm"):
        for tag in ("with_endcaps", "no_endcaps"):
            key = "%s_%s" % (metric, tag)
            values = np.array([r[key] for r in scored if key in r])
            if values.size:
                summary[key] = {
                    "median": float(np.median(values)), "mean": float(values.mean()),
                    "p95": float(np.percentile(values, 95)), "max": float(values.max())}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"experiment": "E49_tetheart_eval",
                                  "predictions": str(PRED_DIR),
                                  "cavity_class_file_suffix": EXPECTED_CAVITY_CLASS,
                                  "frame": "predictions mapped back to index-times-spacing "
                                           "millimetres through the inverse of E50's "
                                           "per-frame transform",
                                  "metric": "symmetric surface distance and Hausdorff, from "
                                            "the same implementation panel A uses",
                                  "summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
