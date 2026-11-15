"""E49d: rescore TetHeart with a class check that the geometry cannot fool, and
report the prediction both as it comes and clipped to the reference's extent.

This supersedes E49's scoring for two reasons, both established by measurement
rather than by preference.

*The class check was invalid.* E49 verifies which predicted class is the cavity
by comparing full three-axis physical extents against the cavity label. TetHeart
reconstructs complete anatomy while the ACDC label stops at the basal and apical
slices, so on the test split every predicted class overshoots the label along the
stack axis by roughly 35 mm. That common term dominates the three-axis sum and
the winner is decided by noise: the full-extent test chose the cavity in 37 of 44
cases and the RV in the other 7, which E49 then withheld. Scoring those 7 anyway
gives 2.4 to 5.4 mm, as good as the cases it kept, so they were not failures.
The in-plane extent carries no overhang term and chooses the cavity in 44 of 44,
by a wide margin -- typically 1 to 13 mm for the cavity against 20 to 75 mm for
the other two classes. It agrees with what reading their code already said: the
loader reorders to ``['myo', 'lv', 'rv']``, so the cavity is ``*_2_pred.obj``.
The in-plane test is kept as a guard, and a case is withheld only if it fires.

*The symmetric mean over the raw prediction is not a like-for-like quantity.*
The same overhang means the surface-to-points direction is charged for anatomy
the reference never annotated. On the test split the two directions differ by
about a factor of six. Both the raw and the extent-matched numbers are computed
here and both are reported; the clipped one is the comparable one, and the raw
one stays on the record.

Cases are scored in a process pool, because serially a case takes about three
minutes and there are 276 of them. The metric is deterministic -- it seeds its
own generator -- so the parallel result is identical to the serial one.

The pool uses the ``spawn`` start method, not the default ``fork``. A forked run
of this deadlocked after 29 cases with 31 of 32 workers parked in
``futex_wait_queue``, the node 92.6% idle and iowait at zero: the children had
inherited a lock held by a native library the parent had already initialised,
which no amount of waiting resolves. ``spawn`` starts each worker from a fresh
interpreter, so there is no inherited lock to wait on.
"""

import json
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PRED_DIR = Path(os.environ["R4D_PRED_DIR"])
TRANSFORMS = Path(os.environ["R4D_TET_FRAME_REPORT"])
ACDC = Path(os.environ.get("R4D_ACDC", "data/raw/acdc"))
OUTPUT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / os.environ.get(
    "R4D_OUTNAME", "E49d_tetheart_rescore.json")
CAVITY_LABEL, MYO_LABEL = 3, 2
PATCH_SIZE = np.asarray(
    [float(v) for v in os.environ.get("R4D_PATCH_SIZE", "64,64,64").split(",")])
EXPECTED_CAVITY_CLASS = 2  # file suffix _2_pred.obj, i.e. class index 1
WORKERS = int(os.environ.get("R4D_WORKERS", "16"))
#: Score exactly as E36 scores our own fits: the whole cavity boundary set, and
#: the metric's default samples per face. E49f measured what the difference is
#: worth -- samples per face is irrelevant (max 0.0007 mm) and the reference
#: stride moves 3 cases in 12 by up to 0.19 mm, in either direction -- so this
#: exists to remove the discrepancy rather than to document it.
MATCH_E36 = os.environ.get("R4D_MATCH_E36", "0") == "1"


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


def to_millimetres(vertices, record):
    """Undo the network's own normalization, then E50's forward map."""
    vertices = (np.asarray(vertices, dtype=np.float64) / 2.0 + 0.5) * PATCH_SIZE - 0.5
    perm = np.asarray(record["permutation"])
    flips = np.asarray(record["flips"], dtype=float)
    scale = np.asarray(record["scale"], dtype=float)
    offset = np.asarray(record["offset"], dtype=float)
    permuted = (vertices - offset) / scale / flips
    restored = np.empty_like(permuted)
    restored[:, perm] = permuted
    return restored


def load_mesh(path):
    import trimesh
    mesh = trimesh.load(str(path), process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        parts = list(mesh.geometry.values())
        if not parts:
            return None
        mesh = trimesh.util.concatenate(parts)
    return mesh if hasattr(mesh, "vertices") else None


def clip_to_z(verts, faces, zlo, zhi):
    """Clip to a z-range; exact plane cut, or a face mask if shapely is absent."""
    import trimesh
    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        for normal, origin in (([0, 0, 1.0], [0, 0, zlo]), ([0, 0, -1.0], [0, 0, zhi])):
            if len(mesh.faces) == 0:
                break
            mesh = trimesh.intersections.slice_mesh_plane(
                mesh, plane_normal=np.array(normal, float),
                plane_origin=np.array(origin, float), cap=False)
        if mesh is not None and len(mesh.faces):
            return np.asarray(mesh.vertices), np.asarray(mesh.faces), "plane_slice"
    except Exception:
        pass
    z = verts[:, 2]
    keep = ((z >= zlo) & (z <= zhi))[faces].all(axis=1)
    kept = faces[keep]
    if len(kept) == 0:
        return verts[:0], kept, "face_mask"
    used = np.unique(kept)
    remap = -np.ones(len(verts), dtype=int)
    remap[used] = np.arange(len(used))
    return verts[used], remap[kept], "face_mask"


def score_case(task):
    """One frame: pick the cavity class, then score it raw and clipped."""
    import nibabel as nib
    from rooted4d.experiments.component_eval import symmetric_surface_distance

    name = task["name"]
    patient, frame = task["patient"], task["frame"]
    transform = task["transform"]
    parts = task["parts"]

    handle = nib.load(str(task["source"]))
    labels = np.asarray(handle.dataobj)
    zooms = tuple(float(z) for z in handle.header.get_zooms()[:3])
    cavity_extent = np.ptp(boundary_points(labels, zooms, CAVITY_LABEL, False), axis=0)

    # The guard: in-plane extent only, which the stack-axis overhang cannot move.
    in_plane = {}
    for index, path in parts.items():
        mesh = load_mesh(path)
        if mesh is None or len(mesh.vertices) == 0:
            continue
        extent = np.ptp(to_millimetres(mesh.vertices, transform), axis=0)
        in_plane[index] = float(np.abs(extent[:2] - cavity_extent[:2]).sum())
    if not in_plane:
        return {"name": name, "patient": patient, "frame": frame,
                "status": "all_meshes_empty"}, None

    best = min(in_plane, key=in_plane.get)
    runner_up = sorted(in_plane.values())[1] if len(in_plane) > 1 else float("inf")
    record = {"name": name, "patient": patient, "frame": frame,
              "inplane_pick": int(best),
              "inplane_pick_agrees": bool(best == EXPECTED_CAVITY_CLASS),
              "inplane_margin_mm": float(runner_up - in_plane[best])}
    if best != EXPECTED_CAVITY_CLASS:
        record["status"] = "withheld_inplane_disagreement"
        return record, ("  %-12s f%02d  WITHHELD: in-plane extent picks class %d, not %d"
                        % (patient, frame, best, EXPECTED_CAVITY_CLASS))

    mesh = load_mesh(parts[EXPECTED_CAVITY_CLASS])
    vertices = to_millimetres(mesh.vertices, transform)
    faces = np.asarray(mesh.faces)
    if len(vertices) == 0 or len(faces) == 0:
        record["status"] = "empty_cavity_mesh"
        return record, None

    record["status"] = "scored"
    record["verts"] = int(vertices.shape[0])
    for drop, caps in ((False, "with_endcaps"), (True, "no_endcaps")):
        reference = boundary_points(labels, zooms, CAVITY_LABEL, drop)
        if reference.shape[0] < 50:
            continue
        sample = reference if MATCH_E36 else reference[:: max(1, len(reference) // 2000)]
        for variant in ("raw", "clipped"):
            if variant == "raw":
                v, f = vertices, faces
            else:
                zlo, zhi = reference[:, 2].min(), reference[:, 2].max()
                v, f, how = clip_to_z(vertices, faces, zlo, zhi)
                record["clip_method"] = how
                if len(f) == 0:
                    continue
            score = (
                symmetric_surface_distance(sample, v, f)
                if MATCH_E36
                else symmetric_surface_distance(sample, v, f, sample_faces=6)
            )
            suffix = "%s_%s" % (variant, caps)
            record["symmetric_mm_" + suffix] = float(score["symmetric_mean_mm"])
            record["hausdorff_mm_" + suffix] = float(score["hausdorff_mm"])
            record["points_to_surface_mm_" + suffix] = float(
                score["mean_points_to_surface_mm"])
            record["surface_to_points_mm_" + suffix] = float(
                score["mean_surface_to_points_mm"])

    line = ("  %-12s f%02d  raw %.3f  clipped %.3f  (end-caps in) | raw %.3f  clipped %.3f (out)"
            % (patient, frame,
               record.get("symmetric_mm_raw_with_endcaps", float("nan")),
               record.get("symmetric_mm_clipped_with_endcaps", float("nan")),
               record.get("symmetric_mm_raw_no_endcaps", float("nan")),
               record.get("symmetric_mm_clipped_no_endcaps", float("nan"))))
    return record, line


def main() -> int:
    transforms = {}
    for row in json.loads(TRANSFORMS.read_text())["rows"]:
        if row.get("status") == "written":
            transforms[(row["case"], row["frame"])] = row

    labels_by_key = {}
    for path in ACDC.rglob("*_frame??_gt.nii.gz"):
        labels_by_key[(path.parent.name, int(path.name.split("_frame")[1][:2]))] = path

    groups = {}
    for pred in sorted(PRED_DIR.glob("*_pred.obj")):
        match = re.match(r"(.+?)_(\d+)_pred\.obj$", pred.name)
        if match:
            groups.setdefault(match.group(1), {})[int(match.group(2))] = pred

    rows, tasks = [], []
    for name, parts in sorted(groups.items()):
        patient_match = re.search(r"(patient\d+)", name)
        frame_match = re.search(r"frame(\d+)", name)
        if not patient_match or not frame_match:
            rows.append({"name": name, "status": "unparsed_name"})
            continue
        patient, frame = patient_match.group(1), int(frame_match.group(1))
        source = labels_by_key.get((patient, frame))
        transform = transforms.get((patient, frame))
        if source is None or EXPECTED_CAVITY_CLASS not in parts:
            rows.append({"name": name, "status": "missing_label_or_mesh"})
            continue
        if transform is None:
            rows.append({"name": name, "patient": patient, "frame": frame,
                         "status": "no_recorded_transform"})
            continue
        tasks.append({"name": name, "patient": patient, "frame": frame,
                      "source": source, "parts": parts, "transform": transform})

    print("scoring %d frames on %d spawned workers" % (len(tasks), WORKERS), flush=True)
    context = mp.get_context("spawn")
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, WORKERS), mp_context=context) as pool:
        futures = [pool.submit(score_case, task) for task in tasks]
        for future in as_completed(futures):
            record, line = future.result()
            rows.append(record)
            done += 1
            if line:
                print("%s   [%d/%d]" % (line, done, len(tasks)), flush=True)

    rows.sort(key=lambda r: (r.get("patient", ""), r.get("frame", 0)))
    scored = [r for r in rows if r["status"] == "scored"]
    summary = {"groups": len(groups), "scored": len(scored),
               "patients": len({r["patient"] for r in scored}),
               "status_counts": {s: sum(1 for r in rows if r["status"] == s)
                                 for s in sorted({r["status"] for r in rows})},
               "inplane_margin_mm_min": (float(min(r["inplane_margin_mm"] for r in scored))
                                         if scored else None)}
    for metric in ("symmetric_mm", "hausdorff_mm",
                   "points_to_surface_mm", "surface_to_points_mm"):
        for variant in ("raw", "clipped"):
            for caps in ("with_endcaps", "no_endcaps"):
                key = "%s_%s_%s" % (metric, variant, caps)
                values = np.array([r[key] for r in scored if key in r])
                if values.size:
                    summary[key] = {"median": float(np.median(values)),
                                    "mean": float(values.mean()),
                                    "p95": float(np.percentile(values, 95)),
                                    "max": float(values.max())}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"experiment": "E49d_tetheart_rescore",
                                  "predictions": str(PRED_DIR),
                                  "class_check": "in-plane extent, which the stack-axis "
                                                 "overhang cannot move",
                                  "summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
