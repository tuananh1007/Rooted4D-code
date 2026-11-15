"""E44: validate 4DMR's meshes against ACDC, then write TetHeart's .obj layout.

This is the quality gate for the whole TetHeart exercise. Ground truth we
generate ourselves is only usable if we can show it agrees with the labels, so
every case is scored before it is accepted, and the distribution is reported
rather than a pass/fail count.

Three facts about 4DMR's output, each established by measurement and each a
trap if assumed the other way:

* `surface_id` 0 is the **epicardium** and 1 is the **endocardium**. Matching
  physical extents settles it: surface 0 spans the cavity-plus-myocardium
  extent, surface 1 spans the cavity extent. Taking 0 as endocardium because it
  is listed first would exchange the two structures silently.
* The meshes are already in the **index-times-spacing** frame, the same one our
  metric uses. No registration is needed. ACDC's NIfTI headers report
  "unexpected scales in sform", so the affine must not be used here.
* 4DMR continues the ventricle past the first and last labelled slice, by 18 to
  23 mm. Those label boundary voxels are flat caps where the imaging stack
  stops, not anatomical surface, so the gate excludes them. Including them
  charges the mesh about 19 mm at the 95th percentile for being right.

Writes `<case>_frame<NN>_mesh_lv.obj` and `..._mesh_myo.obj`, which is
TetHeart's naming. The right ventricle is not written: 4DMR does not produce it,
and panel A scores the LV cavity, so the configuration drops that class. That
is a deviation from TetHeart's setup and is recorded in the manifest.
"""

import json
import os
from pathlib import Path

import numpy as np

OUTPUTS = Path(os.environ["R4D_4DMR_OUTPUTS"])
ACDC = Path(os.environ.get("R4D_ACDC", "data/raw/acdc"))
DEST = Path(os.environ["R4D_TETHEART_GT"])
REPORT = Path(os.environ.get("R4D_OUT", "results/benchmark")) / "E44_gt_validation.json"
MAPPING = Path(os.environ["R4D_4DMR_INPUTS"]) / "frame_mapping.json"
# Accept a case when the median distance on genuine surface is within this many
# millimetres. One voxel in plane is about 1.5 mm on ACDC, so 2.5 mm allows
# discretization and a little smoothing while rejecting a gross misplacement.
GATE_MM = float(os.environ.get("R4D_GT_GATE_MM", "2.5"))
ENDO_SURFACE_ID = 1
EPI_SURFACE_ID = 0


def surface(mesh, surface_id):
    import numpy as _np
    ids = _np.asarray(mesh.cell_data["surface_id"])
    part = mesh.extract_cells(_np.where(ids == surface_id)[0]).extract_surface().triangulate()
    vertices = _np.asarray(part.points, dtype=_np.float64)
    faces = _np.asarray(part.faces).reshape(-1, 4)[:, 1:]
    return vertices, faces


def boundary_points(labels, zooms, label, drop_end_slices):
    """Boundary voxel centres, optionally without the stack's end caps."""
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


def write_obj(path, vertices, faces):
    with open(path, "w") as handle:
        for v in vertices:
            handle.write("v %.6f %.6f %.6f\n" % (v[0], v[1], v[2]))
        for f in faces + 1:
            handle.write("f %d %d %d\n" % (f[0], f[1], f[2]))


def main() -> int:
    import nibabel as nib
    import pyvista as pv
    from rooted4d.experiments.component_eval import point_to_mesh_mm

    mapping = json.loads(MAPPING.read_text())
    label_paths = {p.parent.name: sorted(p.parent.glob("*_frame??_gt.nii.gz"))
                   for p in ACDC.rglob("*_frame??_gt.nii.gz")}

    rows, accepted = [], 0
    for case, frames in sorted(mapping.items()):
        paths = {int(p.name.split("_frame")[1][:2]): p for p in label_paths.get(case, [])}
        for entry in frames:
            index, acdc_frame = entry["index"], entry["acdc_frame"]
            vtk = OUTPUTS / case / "vtk" / ("frame_%02d.vtk" % index)
            if not vtk.is_file() or acdc_frame not in paths:
                rows.append({"case": case, "frame": acdc_frame, "status": "missing_output"})
                continue
            handle = nib.load(str(paths[acdc_frame]))
            labels = np.asarray(handle.dataobj)
            zooms = tuple(float(z) for z in handle.header.get_zooms()[:3])

            mesh = pv.read(str(vtk))
            endo_v, endo_f = surface(mesh, ENDO_SURFACE_ID)
            epi_v, epi_f = surface(mesh, EPI_SURFACE_ID)
            if endo_f.size == 0 or epi_f.size == 0:
                rows.append({"case": case, "frame": acdc_frame, "status": "empty_surface"})
                continue

            gated = boundary_points(labels, zooms, 3, drop_end_slices=True)
            full = boundary_points(labels, zooms, 3, drop_end_slices=False)
            if gated.shape[0] < 50:
                rows.append({"case": case, "frame": acdc_frame, "status": "too_few_points"})
                continue
            d_gated = point_to_mesh_mm(gated, endo_v, endo_f)
            d_full = point_to_mesh_mm(full, endo_v, endo_f)
            median = float(np.median(d_gated))
            status = "accepted" if median <= GATE_MM else "rejected_gate"

            record = {"case": case, "frame": acdc_frame, "status": status,
                      "median_mm_no_endcaps": median,
                      "mean_mm_no_endcaps": float(d_gated.mean()),
                      "p95_mm_no_endcaps": float(np.percentile(d_gated, 95)),
                      "median_mm_with_endcaps": float(np.median(d_full)),
                      "endo_verts": int(endo_v.shape[0]), "epi_verts": int(epi_v.shape[0])}
            rows.append(record)

            if status == "accepted":
                accepted += 1
                case_dir = DEST / ("ACDC_%s" % case)
                case_dir.mkdir(parents=True, exist_ok=True)
                stem = "ACDC_%s_frame%02d" % (case, acdc_frame)
                write_obj(case_dir / ("%s_mesh_lv.obj" % stem), endo_v, endo_f)
                write_obj(case_dir / ("%s_mesh_myo.obj" % stem), epi_v, epi_f)
            print("  %-12s frame %02d  median %6.3f mm  %s"
                  % (case, acdc_frame, median, status), flush=True)

    scored = [r for r in rows if "median_mm_no_endcaps" in r]
    summary = {"cases": len({r["case"] for r in rows}), "frames": len(rows),
               "accepted": accepted, "scored": len(scored),
               "gate_mm": GATE_MM,
               "status_counts": {s: sum(1 for r in rows if r["status"] == s)
                                 for s in sorted({r["status"] for r in rows})}}
    if scored:
        v = np.array([r["median_mm_no_endcaps"] for r in scored])
        summary["median_mm_no_endcaps"] = {
            "median": float(np.median(v)), "mean": float(v.mean()),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())}
        w = np.array([r["median_mm_with_endcaps"] for r in scored])
        summary["median_mm_with_endcaps"] = {"median": float(np.median(w))}

    result = {"experiment": "E44_gt_validation",
              "surface_id_convention": {"0": "epicardium", "1": "endocardium",
                                        "established_by": "matching physical extents "
                                                          "against the cavity and "
                                                          "cavity-plus-myocardium labels"},
              "frame": "index times spacing; no registration applied; ACDC affine not used "
                       "because its headers report unexpected sform scales",
              "right_ventricle": "not written; 4DMR does not produce it and panel A scores "
                                 "the LV cavity. This is a deviation from TetHeart's setup.",
              "summary": summary, "rows": rows}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2) + "\n")
    print("\n%s" % json.dumps(summary, indent=2))
    print("wrote %s and %d accepted frames under %s" % (REPORT, accepted, DEST))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
