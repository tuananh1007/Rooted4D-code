"""E50: put the ground-truth meshes in the frame TetHeart actually consumes.

The first training run produced empty meshes at every checkpoint. The cause was
not the model: our ground truth was in the wrong coordinate frame, so the signed
distance field had no zero crossing inside the tetrahedral grid and marching
tetrahedra returned nothing. Training still looked healthy because the
segmentation losses dominate and they were unaffected.

TetHeart's meshes live in the voxel-index frame of its preprocessed 64^3 volume;
ours were in index-times-spacing millimetres on the original grid, about four
times out of range. Their own sample is the reference: its LV mesh sits 0.70
voxels from the matching label.

Rather than reverse-engineer their resize, the transform is recovered per case
from the data. Whatever the preprocessing does, it maps one frame to the other
by an axis-aligned affine, and the same anatomy appears in both, so matching the
foreground bounding boxes gives the axis permutation, the flips and the per-axis
scale and offset. Searching permutations and flips and keeping the arrangement
that best matches the label recovers 0.704 voxels on the reference case, which
is their own figure.

Every converted case is then checked against its preprocessed label, and one
that fails is reported rather than written, because a silently misplaced mesh
would train the model on nothing and look like a model failure.
"""

import itertools
import json
import os
from pathlib import Path

import numpy as np

GT = Path(os.environ["R4D_TETHEART_GT"])
PREP = Path(os.environ["R4D_TETHEART_DATA"]) / "914_3d"
ACDC = Path(os.environ.get("R4D_ACDC", "data/raw/acdc"))
DEST = Path(os.environ["R4D_TET_FRAME_GT"])
OUTPUT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / "E50_tet_frame.json"
# Their sample sits at 0.70 voxels; accept up to two, which tolerates the
# resampling of a ten-slice stack without admitting a misplaced mesh.
GATE_VOXELS = float(os.environ.get("R4D_TET_GATE", "2.0"))
# preprocessed label values, established by matching structure volume fractions
# against the raw ACDC labels: 1 = myocardium, 2 = LV cavity, 3 = RV.
STRUCTURE_LABEL = {"myo": 1, "lv": 2, "rv": 3}


def boundary(occupied):
    padded = np.pad(occupied, 1)
    interior = (padded[1:-1, 1:-1, 1:-1]
                & padded[:-2, 1:-1, 1:-1] & padded[2:, 1:-1, 1:-1]
                & padded[1:-1, :-2, 1:-1] & padded[1:-1, 2:, 1:-1]
                & padded[1:-1, 1:-1, :-2] & padded[1:-1, 1:-1, 2:])
    return np.argwhere(occupied & ~interior).astype(float)


def write_obj(path, vertices, faces):
    with open(path, "w") as handle:
        for v in vertices:
            handle.write("v %.6f %.6f %.6f\n" % (v[0], v[1], v[2]))
        for f in faces + 1:
            handle.write("f %d %d %d\n" % (f[0], f[1], f[2]))


def main() -> int:
    import nibabel as nib
    import trimesh
    from scipy.spatial import cKDTree

    raw_labels = {}
    for path in ACDC.rglob("*_frame??_gt.nii.gz"):
        raw_labels[(path.parent.name, int(path.name.split("_frame")[1][:2]))] = path

    rows = []
    for case_dir in sorted(GT.glob("ACDC_patient*")):
        patient = case_dir.name.replace("ACDC_", "")
        for lv_path in sorted(case_dir.glob("*_mesh_lv.obj")):
            stem = lv_path.name.replace("_mesh_lv.obj", "")
            frame = int(stem.split("_frame")[1][:2])
            prep = PREP / case_dir.name / ("%s_y.npy" % stem)
            raw = raw_labels.get((patient, frame))
            if not prep.is_file() or raw is None:
                rows.append({"case": patient, "frame": frame, "status": "missing_input"})
                continue

            target = np.load(prep)
            target = target[0] if target.ndim == 4 else target
            handle = nib.load(str(raw))
            labels = np.asarray(handle.dataobj)
            zooms = np.asarray([float(z) for z in handle.header.get_zooms()[:3]])

            source_points = np.argwhere(labels > 0).astype(float) * zooms
            dest_points = np.argwhere(target > 0).astype(float)
            if source_points.size == 0 or dest_points.size == 0:
                rows.append({"case": patient, "frame": frame, "status": "empty_label"})
                continue

            reference = boundary(target == STRUCTURE_LABEL["lv"])
            tree_reference = cKDTree(reference) if reference.size else None
            lv_vertices = np.asarray(trimesh.load(str(lv_path), process=False).vertices)

            best = None
            low_d, high_d = dest_points.min(0), dest_points.max(0)
            for perm in itertools.permutations(range(3)):
                for flips in itertools.product([1, -1], repeat=3):
                    oriented = source_points[:, list(perm)] * np.asarray(flips)
                    low_s, high_s = oriented.min(0), oriented.max(0)
                    scale = (high_d - low_d) / np.maximum(high_s - low_s, 1e-9)
                    offset = low_d - low_s * scale
                    moved = lv_vertices[:, list(perm)] * np.asarray(flips) * scale + offset
                    score = float(np.median(cKDTree(moved).query(reference)[0]))
                    if best is None or score < best[0]:
                        best = (score, perm, flips, scale, offset)

            score, perm, flips, scale, offset = best
            if score > GATE_VOXELS:
                rows.append({"case": patient, "frame": frame, "status": "rejected_gate",
                             "median_voxels": score})
                print("  %-12s f%02d  REJECTED at %.3f voxels" % (patient, frame, score), flush=True)
                continue

            out_dir = DEST / case_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            written = {}
            for structure in ("lv", "myo", "rv"):
                source_mesh = case_dir / ("%s_mesh_%s.obj" % (stem, structure))
                if not source_mesh.is_file():
                    continue
                mesh = trimesh.load(str(source_mesh), process=False)
                v = np.asarray(mesh.vertices)
                moved = v[:, list(perm)] * np.asarray(flips) * scale + offset
                write_obj(out_dir / source_mesh.name, moved,
                          np.asarray(mesh.faces))
                ref = boundary(target == STRUCTURE_LABEL[structure])
                written[structure] = (float(np.median(cKDTree(moved).query(ref)[0]))
                                      if ref.size else None)

            rows.append({"case": patient, "frame": frame, "status": "written",
                         "median_voxels_lv": score,
                         "per_structure_voxels": written,
                         "permutation": list(perm), "flips": list(flips),
                         "scale": scale.tolist(), "offset": offset.tolist()})
            print("  %-12s f%02d  lv %.3f  myo %s  rv %s voxels"
                  % (patient, frame, score,
                     "%.3f" % written.get("myo") if written.get("myo") is not None else "-",
                     "%.3f" % written.get("rv") if written.get("rv") is not None else "-"),
                  flush=True)

    written_rows = [r for r in rows if r["status"] == "written"]
    summary = {"frames": len(rows), "written": len(written_rows),
               "gate_voxels": GATE_VOXELS,
               "status_counts": {s: sum(1 for r in rows if r["status"] == s)
                                 for s in sorted({r["status"] for r in rows})}}
    if written_rows:
        for key in ("lv", "myo", "rv"):
            v = np.array([r["per_structure_voxels"][key] for r in written_rows
                          if r["per_structure_voxels"].get(key) is not None])
            if v.size:
                summary["voxels_" + key] = {"median": float(np.median(v)),
                                            "p95": float(np.percentile(v, 95)),
                                            "max": float(v.max())}
        perms = {tuple(r["permutation"]) for r in written_rows}
        flips = {tuple(r["flips"]) for r in written_rows}
        summary["distinct_permutations"] = [list(p) for p in perms]
        summary["distinct_flips"] = [list(f) for f in flips]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"experiment": "E50_tet_frame",
                                  "summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
